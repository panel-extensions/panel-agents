"""
Drive a pydantic-ai agent and translate its run into panel-agents events.

``AgentRunner`` streams a run through ``_PanelEventStream``, a
``pydantic_ai.ui.UIEventStream`` subclass, and applies every resulting
event to a ``RunState``. A run-scoped capability reports per-response
usage and resolves tool approvals inline, so a run that pauses for a
human decision stays one run in the UI.

Tools publish artifacts either by awaiting ``publish_artifact(ctx, ...)``
or by returning ``artifact_return(...)``, which shows the model a short
summary while the UI receives the full value.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import param
from pydantic_ai import CancellationToken, CapabilityEvent, CustomEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelAPIError, ModelRetry, RunCancelled, ToolFailed
from pydantic_ai.messages import (
    DeferredToolRequestsEvent,
    FunctionToolResultEvent,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai.toolsets import AbstractToolset, ApprovalRequiredToolset, ToolsetTool
from pydantic_ai.ui import UIEventStream
from pydantic_core import to_jsonable_python

from .. import events as ev
from ..actions import Action, ActionRegistry
from ..policy import ApprovalPolicy
from ..recording import _dump_event
from .cassette import Cassette, CassetteError, Divergence
from .model_trace import (
    _METADATA_KEY,
    ModelCall,
    ModelTraceRecorder,
    _response_cost,
    fingerprint,
    load_messages,
    load_model_calls,
)

_log = logging.getLogger(__name__)
from ..state import ApprovalRequest, RunState, _new_id

__all__ = [
    "ActionToolset",
    "AgentRunner",
    "Cassette",
    "CassetteError",
    "Divergence",
    "ModelCall",
    "ModelTraceRecorder",
    "artifact_return",
    "fingerprint",
    "load_messages",
    "load_model_calls",
    "publish_artifact",
]

_ARTIFACTS_KEY = "panel_agents.artifacts"

_DENIED_MESSAGE = "The user denied this tool call."


@dataclass(kw_only=True)
class _ArtifactEvent(CustomEvent, name="panel_agents.artifact"):
    """Publishes a version of an artifact; emitted from tools via ``publish_artifact``."""

    artifact_id: str
    value: Any = None
    title: str | None = None
    kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    applied: bool = False


@dataclass(kw_only=True)
class _UsageEvent(CapabilityEvent, namespace="panel_agents"):
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    duration: float | None = None


async def publish_artifact(
    ctx: RunContext[Any],
    artifact_id: str,
    value: Any,
    *,
    title: str | None = None,
    kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    applied: bool = False,
) -> None:
    """
    Publish a new version of an artifact from inside an async tool.

    The event is stamped with the calling tool's ``tool_call_id`` and
    reaches the UI in stream order, before the tool's result. Pass
    ``applied=True`` when the tool edited a host document in place and
    ``value`` is what the document now holds.
    """
    await ctx.emit(_ArtifactEvent(
        artifact_id=artifact_id, value=value, title=title, kind=kind, metadata=metadata or {}, applied=applied,
    ))


def artifact_return(
    return_value: Any,
    artifact_id: str,
    value: Any,
    *,
    title: str | None = None,
    kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    applied: bool = False,
) -> ToolReturn:
    """
    Return ``return_value`` to the model and publish ``value`` as an artifact.

    Works from sync and async tools. The artifact travels in the tool
    return's metadata, which pydantic-ai never sends to the model. See
    ``publish_artifact`` for ``applied``.
    """
    artifact = _ArtifactEvent(
        artifact_id=artifact_id, value=value, title=title, kind=kind, metadata=metadata or {}, applied=applied,
    )
    return ToolReturn(return_value=return_value, metadata={_ARTIFACTS_KEY: [artifact]})


@dataclass
class _RecordingApprovalToolset(ApprovalRequiredToolset[Any]):
    """Applies the approval policy and keeps each tool's validator, to check edited arguments with it."""

    runner: Any = None

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: Any) -> Any:
        self.runner._validators[name] = tool.args_validator
        # The response that called the tool is complete in the history by now.
        self.runner._trace(ctx.messages)
        return await super().call_tool(name, tool_args, ctx, tool)


_URL_TAIL = re.compile(r"(https?://[^\s?#]+)[?#]\S*")


def _error_message(error: BaseException, limit: int = 300) -> str:
    """A short message for a failed run: a provider's refusal is reduced to its own message and status."""
    text = str(error) or type(error).__name__
    if isinstance(error, ModelAPIError):
        body = error.body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body = body["error"]
        message = body.get("message") if isinstance(body, dict) else None
        status = getattr(error, "status_code", None)
        text = "The model provider refused the request" + (f" ({status})" if status else "")
        if isinstance(message, str) and message.strip():
            text += f": {message.strip()}"
    text = _URL_TAIL.sub(r"\1", " ".join(text.split()))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _shorten(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _with_defaults(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    # Edited arguments are checked as the tool will see them, with omitted defaults filled in.
    properties = (schema or {}).get("properties", {})
    return {**{k: p["default"] for k, p in properties.items() if isinstance(p, dict) and "default" in p}, **args}


@dataclass
class _ActionArgsValidator:

    action: Action

    def validate_json(self, input: str | bytes | bytearray, *, allow_partial: Any = False, **kwargs: Any) -> dict[str, Any]:
        return self.action.validate_json(input)

    def validate_python(self, input: Any, *, allow_partial: Any = False, **kwargs: Any) -> dict[str, Any]:
        return self.action.validate(input)


class ActionToolset(AbstractToolset[Any]):
    """
    Exposes the actions of an ``ActionRegistry`` as pydantic-ai tools.

    Each tool has the action's name, description and JSON schema, and
    runs the action in the registry's Panel session. Actions registered
    or removed during a session are picked up on the next model request.
    An exception raised by an action is reported to the model as a
    failed tool call instead of ending the run.
    """

    def __init__(self, registry: ActionRegistry, *, id: str | None = None, max_retries: int = 1):
        self.registry = registry
        self.max_retries = max_retries
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {
            action.name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(name=action.name, description=action.description, parameters_json_schema=action.parameters),
                max_retries=self.max_retries,
                args_validator=_ActionArgsValidator(action),
            )
            for action in self.registry
        }

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]) -> Any:
        action = self.registry.actions.get(name)
        if action is None:
            raise ToolFailed(f"The action {name!r} is no longer available.")
        try:
            return await self.registry._invoke_validated(action, tool_args)
        except (ApprovalRequired, CallDeferred, ModelRetry, ToolFailed):
            raise
        except Exception as e:
            raise ToolFailed(f"{type(e).__name__}: {e}") from e


@dataclass
class _RunCapability(AbstractCapability[Any]):
    """Run-scoped hooks that need the run context rather than the event stream."""

    runner: AgentRunner | None = None
    request: dict[str, Any] | None = None
    messages: list[Any] = field(default_factory=list)

    async def prepare_tools(self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        self.runner._tool_defs.update({td.name: td for td in tool_defs})
        return tool_defs

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
        runner = self.runner

        def needs_approval(ctx: RunContext[Any], tool_def: ToolDefinition, args: dict[str, Any]) -> bool:
            # The validated arguments, defaults included, are what approvals are shown and checked against.
            validated = to_jsonable_python(args, fallback=str)
            if ctx.tool_call_id:
                runner._validated[ctx.tool_call_id] = validated
            policy = runner.approval_policy
            return policy is not None and policy.requires_approval(tool_def.name, validated)

        return _RecordingApprovalToolset(toolset, needs_approval, runner=runner)

    async def before_model_request(self, ctx: RunContext[Any], request_context: Any) -> Any:
        self.messages = ctx.messages
        self.request = {
            "model": request_context.model.model_name, "started": time.perf_counter(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.runner._trace(ctx.messages)
        return request_context

    async def after_model_request(self, ctx: RunContext[Any], *, request_context: Any, response: ModelResponse) -> ModelResponse:
        request, self.request = self.request, None
        duration = time.perf_counter() - request["started"] if request else None
        stamp = {"duration": duration, "fingerprint": fingerprint(request_context.model_request_parameters, ctx.messages)}
        response = replace(response, metadata={**(response.metadata or {}), _METADATA_KEY: stamp})
        usage = response.usage
        await ctx.emit(_UsageEvent(
            model=response.model_name, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cached_tokens=usage.cache_read_tokens, cost_usd=_response_cost(response), duration=duration,
        ))
        return response

    def failed_request(self) -> dict[str, Any] | None:
        """The model request that was in flight when the run failed, if any."""
        if self.request is None:
            return None
        request = dict(self.request)
        request["duration"] = time.perf_counter() - request.pop("started")
        return request

    async def handle_deferred_tool_calls(self, ctx: RunContext[Any], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # External calls (CallDeferred) are left unresolved so the run ends with them as its output.
        if not requests.approvals:
            return None
        runner = self.runner
        decisions = await asyncio.gather(*(runner._wait_for_decision(call.tool_call_id) for call in requests.approvals))
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for call, decision in zip(requests.approvals, decisions, strict=True):
            if not decision.approved:
                approvals[call.tool_call_id] = ToolDenied(decision.reason or _DENIED_MESSAGE)
                continue
            proposed = runner._validated.get(call.tool_call_id, call.args_as_dict())
            # The state only accepts permitted decisions; this guards against a policy changed in between.
            request = runner.state.approvals.get(call.tool_call_id)
            if request is not None and (message := runner._refusal(request, decision.user, decision.args, True)):
                runner.state.apply(ev.ToolCallResult(tool_call_id=call.tool_call_id, content=message, is_error=True, run_id=runner.run_id))
                approvals[call.tool_call_id] = ToolDenied(message)
                continue
            edited = decision.args is not None and decision.args != proposed
            approvals[call.tool_call_id] = ToolApproved(override_args=decision.args if edited else None)
        return DeferredToolResults(approvals=approvals)


@dataclass
class _PanelEventStream(UIEventStream[None, ev.Event, Any, Any]):
    """Translates pydantic-ai stream events into panel-agents events."""

    run_id: str = field(default_factory=lambda: _new_id("run-"))
    thread_id: str | None = None
    agent_name: str | None = None
    input: Any = None
    tool_defs: dict[str, ToolDefinition] = field(default_factory=dict)
    validated_args: dict[str, dict[str, Any]] = field(default_factory=dict)

    error: Exception | None = None
    _text_id: str | None = None
    _thinking_id: str | None = None
    _response_text_id: str | None = None

    def encode_event(self, event: ev.Event) -> str:
        return f"data: {_dump_event(event)}\n\n"

    def _stamp(self, event: ev.Event) -> ev.Event:
        event.run_id = self.run_id
        return event

    async def handle_event(self, event: Any) -> AsyncIterator[ev.Event]:
        async for e in super().handle_event(event):
            yield self._stamp(e)

    async def before_stream(self) -> AsyncIterator[ev.Event]:
        yield ev.RunStarted(run_id=self.run_id, thread_id=self.thread_id, agent_name=self.agent_name, input=self.input)

    async def after_stream(self) -> AsyncIterator[ev.Event]:
        if self.error is None and self._result is not None:
            yield ev.RunFinished(run_id=self.run_id, result=self._result.output)

    async def on_error(self, error: Exception) -> AsyncIterator[ev.Event]:
        self.error = error
        code = "cancelled" if isinstance(error, RunCancelled) else type(error).__name__
        yield ev.RunError(run_id=self.run_id, message=_error_message(error), code=code)

    async def before_response(self) -> AsyncIterator[ev.Event]:
        self._response_text_id = None
        return
        yield

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[ev.Event]:
        if not follows_text or self._text_id is None:
            self._text_id = self._response_text_id = _new_id("msg-")
            yield ev.TextMessageStart(message_id=self._text_id)
        if part.content:
            yield ev.TextMessageContent(message_id=self._text_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[ev.Event]:
        if delta.content_delta and self._text_id:
            yield ev.TextMessageContent(message_id=self._text_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[ev.Event]:
        if not followed_by_text and self._text_id:
            yield ev.TextMessageEnd(message_id=self._text_id)
            self._text_id = None

    async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[ev.Event]:
        if not follows_thinking or self._thinking_id is None:
            self._thinking_id = _new_id("think-")
            yield ev.ThinkingStart(message_id=self._thinking_id)
        if part.content:
            yield ev.ThinkingContent(message_id=self._thinking_id, delta=part.content)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[ev.Event]:
        if delta.content_delta and self._thinking_id:
            yield ev.ThinkingContent(message_id=self._thinking_id, delta=delta.content_delta)

    async def handle_thinking_end(self, part: ThinkingPart, followed_by_thinking: bool = False) -> AsyncIterator[ev.Event]:
        if not followed_by_thinking and self._thinking_id:
            yield ev.ThinkingEnd(message_id=self._thinking_id)
            self._thinking_id = None

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[ev.Event]:
        yield ev.ToolCallStart(tool_call_id=part.tool_call_id, tool_name=part.tool_name, parent_message_id=self._response_text_id)
        if part.args:
            # A str may be a partial JSON fragment completed by later deltas, so pass it through raw.
            yield ev.ToolCallArgs(tool_call_id=part.tool_call_id, delta=part.args if isinstance(part.args, str) else part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[ev.Event]:
        tool_call_id = delta.tool_call_id or (self._open_part.tool_call_id if isinstance(self._open_part, ToolCallPart) else None)
        if delta.args_delta is None or tool_call_id is None:
            return
        args = delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta)
        yield ev.ToolCallArgs(tool_call_id=tool_call_id, delta=args)

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[ev.Event]:
        yield ev.ToolCallEnd(tool_call_id=part.tool_call_id)

    async def handle_deferred_tool_requests(self, event: DeferredToolRequestsEvent) -> AsyncIterator[ev.Event]:
        for call in event.requests.approvals:
            tool_def = self.tool_defs.get(call.tool_name)
            yield ev.ApprovalRequested(
                tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                args=self.validated_args.get(call.tool_call_id, call.args_as_dict()),
                schema=tool_def.parameters_json_schema if tool_def else None,
                description=tool_def.description if tool_def else None,
            )

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[ev.Event]:
        part = event.part
        metadata = part.metadata if isinstance(part.metadata, dict) else {}
        for artifact in metadata.get(_ARTIFACTS_KEY, []):
            yield self._artifact(artifact, part.tool_call_id)
        # A denial is already recorded by the ApprovalResolved event that caused it.
        if part.outcome == "denied":
            return
        yield self._tool_result(part)

    def _tool_result(self, part: ToolReturnPart) -> ev.ToolCallResult:
        if part.outcome in ("failed", "retried", "interrupted"):
            return ev.ToolCallResult(tool_call_id=part.tool_call_id, content=part.model_response_str(wrap_if_error=False), is_error=True)
        return ev.ToolCallResult(tool_call_id=part.tool_call_id, content=part.content)

    def _artifact(self, artifact: _ArtifactEvent, tool_call_id: str | None) -> ev.ArtifactPublished:
        return ev.ArtifactPublished(
            artifact_id=artifact.artifact_id, value=artifact.value, kind=artifact.kind, title=artifact.title,
            tool_call_id=artifact.tool_call_id or tool_call_id, metadata=dict(artifact.metadata),
            applied=artifact.applied,
        )

    async def handle_custom_event(self, event: CustomEvent) -> AsyncIterator[ev.Event]:
        if isinstance(event, _ArtifactEvent):
            yield self._artifact(event, event.tool_call_id)
        else:
            yield ev.Custom(name=event.name, value=event.to_payload())

    async def handle_capability_event(self, event: CapabilityEvent) -> AsyncIterator[ev.Event]:
        if isinstance(event, _UsageEvent):
            yield ev.Usage(
                model=event.model, input_tokens=event.input_tokens, output_tokens=event.output_tokens,
                cached_tokens=event.cached_tokens, cost_usd=event.cost_usd, duration=event.duration,
            )


class AgentRunner(param.Parameterized):
    """
    Run a pydantic-ai agent and mirror the run into a ``RunState``.

    Each call to ``run`` adds the prompt as a user message, streams the
    run into the state and keeps the message history so the next call
    continues the conversation. Tool calls that need approval pause
    the run until every pending request is resolved through
    ``RunState.resolve_approval``; the run then continues as the same
    run. Tool calls deferred for external execution (``CallDeferred``)
    end the run with ``DeferredToolRequests`` as its output, which
    requires the agent's ``output_type`` to include it; continue with
    ``run(deferred_tool_results=...)``.

    Must be awaited on the event loop that owns the state, e.g. from a
    Panel callback.
    """

    agent = param.Parameter(doc="The pydantic-ai Agent to run.")

    state = param.ClassSelector(class_=RunState, doc="RunState the run is applied to.")

    deps = param.Parameter(default=None, doc="Dependencies passed to every run.")

    messages = param.List(default=[], doc="Message history carried across runs.")

    thread_id = param.String(default=None, allow_None=True, doc="Thread id recorded on each run.")

    raise_errors = param.Boolean(default=False, doc="""
        Whether exceptions raised by a run propagate out of ``run`` after
        the RunError event is applied. When False, ``run`` returns None
        and the exception is available as ``error``. Either way a failed
        run is logged with its traceback and shows as an error message in
        the timeline.""")

    run_kwargs = param.Dict(default={}, doc="Extra keyword arguments passed to ``Agent.run_stream_events``.")

    approval_policy = param.ClassSelector(class_=ApprovalPolicy, default=None, allow_None=True, doc="""
        Rules deciding which tool calls need approval and who may give
        it. Applies to every tool the agent can call during a run.""")

    running = param.Boolean(default=False, constant=True)

    run_id = param.String(default=None, allow_None=True, constant=True, doc="Id of the current or last run.")

    error = param.Parameter(default=None, constant=True, doc="Exception raised by the last run, if any.")

    deferred = param.Parameter(default=None, constant=True, doc="""
        DeferredToolRequests the last run ended with, if any.""")

    def __init__(self, agent=None, state: RunState | None = None, deps: Any = None, **params):
        params["agent"] = agent
        params["state"] = state if state is not None else RunState()
        params["deps"] = deps
        super().__init__(**params)
        self._tool_defs: dict[str, ToolDefinition] = {}
        self._validated: dict[str, dict[str, Any]] = {}
        self._validators: dict[str, Any] = {}
        self._notes: list[str] = []
        self._waiters: dict[str, asyncio.Future[ev.ApprovalResolved]] = {}
        self._early: dict[str, ev.ApprovalResolved] = {}
        self._token: CancellationToken | None = None
        self._trace_listeners: list[ModelTraceRecorder] = []
        self._unsubscribe = self.state.subscribe(self._on_event)
        self.state._decision_check = self._refusal

    def _validate_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Validate ``args`` for ``tool_name`` the way the tool will, returning
        them as JSON-compatible values with every default filled in.
        """
        validator = self._validators.get(tool_name)
        if validator is None:
            tool_def = self._tool_defs.get(tool_name)
            return _with_defaults(args, tool_def.parameters_json_schema if tool_def else None)
        return to_jsonable_python(validator.validate_python(dict(args)), fallback=str)

    def _refusal(self, request: ApprovalRequest, user: str | None, args: dict[str, Any] | None, approved: bool) -> str | None:
        policy = self.approval_policy
        if policy is None:
            return None
        who = user or "This user"
        if not approved:
            return None if policy.can_approve(request, user) else f"{who} may not decide this {request.tool_name} call."
        final = None
        if args is not None:
            try:
                final = self._validate_args(request.tool_name, args)
            except Exception as e:
                return f"Invalid arguments for {request.tool_name}: {e}"
        return None if policy.can_decide(request, user, final) else f"{who} may not approve this {request.tool_name} call."

    def _on_event(self, event: ev.Event):
        if isinstance(event, (ev.ArtifactApplied, ev.ArtifactPublished)):
            self._note_artifact_change(event)
            return
        if not isinstance(event, ev.ApprovalResolved):
            return
        # Only the decision the state accepted counts.
        request = self.state.approvals.get(event.tool_call_id)
        if request is None or getattr(request, "_decision", None) is not event:
            return
        future = self._waiters.get(event.tool_call_id)
        if future is None:
            # The run may not have reached its approval handler yet.
            if self.running and event.run_id == self.run_id:
                self._early.setdefault(event.tool_call_id, event)
        elif not future.done():
            future.set_result(event)

    def _note_artifact_change(self, event: ev.ArtifactApplied | ev.ArtifactPublished):
        # Changes made outside the agent's tools are reported with the next prompt, so the model's view stays current.
        artifact = self.state.artifacts.artifacts.get(event.artifact_id)
        # Titles may come from the model or a tool, so they are quoted and shortened rather than pasted in.
        name = repr(_shorten(event.artifact_id))
        if artifact is not None and artifact.title and artifact.title != event.artifact_id:
            name = f"{_shorten(artifact.title)!r} ({name})"
        if isinstance(event, ev.ArtifactApplied):
            who = f"{_shorten(event.user)!r} " if event.user else "The user "
            self._notes.append(f"{who}applied version {event.version} of the artifact {name}; the app now holds that version.")
        elif not self.running and artifact is not None:
            held = " and now holds it" if event.applied else ""
            self._notes.append(f"The app published version {len(artifact.versions)} of the artifact {name}{held}.")

    async def _wait_for_decision(self, tool_call_id: str) -> ev.ApprovalResolved:
        if tool_call_id in self._early:
            return self._early.pop(tool_call_id)
        future = self._waiters[tool_call_id] = asyncio.get_running_loop().create_future()
        try:
            return await future
        finally:
            self._waiters.pop(tool_call_id, None)

    async def run(
        self,
        prompt: str | None = None,
        *,
        deferred_tool_results: DeferredToolResults | None = None,
        **kwargs,
    ) -> Any:
        """Run the agent on ``prompt`` and return its output."""
        if self.running:
            raise RuntimeError("AgentRunner is already running; await the current run or cancel it first.")
        run_id = _new_id("run-")
        with param.edit_constant(self):
            self.param.update(running=True, run_id=run_id, error=None, deferred=None)
        if isinstance(prompt, str) and prompt:
            self.state.add_user_message(prompt, run_id=run_id)
        self._token = CancellationToken()
        stream = _PanelEventStream(
            run_id=run_id, thread_id=self.thread_id, agent_name=self.agent.name,
            input=prompt, tool_defs=self._tool_defs, validated_args=self._validated,
        )
        model_prompt = prompt
        if self._notes and isinstance(prompt, str):
            model_prompt = "\n".join(f"[{note}]" for note in self._notes) + "\n\n" + prompt
            self._notes.clear()
        run_kwargs = {"run_id": run_id, **self.run_kwargs, **kwargs}
        capability = _RunCapability(runner=self)
        capabilities = [*run_kwargs.pop("capabilities", []), capability]
        for listener in self._trace_listeners:
            listener._run_started(self.messages)
        try:
            async with self.agent.run_stream_events(
                model_prompt, message_history=self.messages or None, deferred_tool_results=deferred_tool_results,
                deps=self.deps, capabilities=capabilities, cancellation_token=self._token, **run_kwargs,
            ) as events:
                async for event in stream.transform_stream(events):
                    self._apply(stream, event)
        except asyncio.CancelledError as e:
            self._close(stream, ev.RunError(run_id=run_id, message="Run cancelled", code="cancelled"))
            self._trace_failure(run_id, capability, e)
            raise
        except Exception as e:
            # Errors raised outside the event stream, e.g. invalid run arguments.
            stream.error = e
            self._close(stream, ev.RunError(run_id=run_id, message=_error_message(e), code=type(e).__name__))
        finally:
            self._token = None
            self._early.clear()
            with param.edit_constant(self):
                self.running = False
        return self._finish(stream, capability)

    def _apply(self, stream: _PanelEventStream, event: ev.Event):
        if isinstance(event, ev.RunError):
            self._close(stream, event)
        else:
            self.state.apply(event)

    def _close(self, stream: _PanelEventStream, error: ev.RunError):
        # Approvals of a dead run can no longer be acted on, so close them before the error.
        for request in list(self.state.pending_approvals):
            if request.run_id == stream.run_id:
                self.state.apply(ev.ApprovalResolved(
                    tool_call_id=request.tool_call_id, approved=False, reason=error.message, run_id=stream.run_id,
                ))
        self.state.apply(error)

    def _trace(self, messages: list[Any]):
        for listener in self._trace_listeners:
            listener._messages(messages)

    def _trace_failure(self, run_id: str, capability: _RunCapability, error: BaseException):
        # The messages of a failed run stay out of the conversation, but the trace keeps what was sent.
        self._trace(capability.messages)
        request = capability.failed_request()
        for listener in self._trace_listeners:
            listener._run_failed(run_id, error, request)

    def _finish(self, stream: _PanelEventStream, capability: _RunCapability) -> Any:
        result: AgentRunResult | None = stream._result
        if stream.cancelled is not None:
            self.messages = list(stream.cancelled.all_messages())
        elif result is not None:
            self.messages = list(result.all_messages())
        if stream.cancelled is not None or result is not None:
            self._trace(self.messages)
        elif stream.error is not None:
            self._trace_failure(stream.run_id, capability, stream.error)
        with param.edit_constant(self):
            self.error = stream.error
            if result is not None and isinstance(result.output, DeferredToolRequests):
                self.deferred = result.output
        if stream.error is not None:
            if not isinstance(stream.error, RunCancelled):
                _log.error("Agent run %s failed", stream.run_id, exc_info=stream.error)
            if self.raise_errors:
                raise stream.error
            return None
        return result.output if result is not None else None

    def cancel(self):
        """Cancel the current run; it ends with a RunError coded ``cancelled``."""
        if self._token is not None:
            self._token.cancel()

    def clear(self):
        """Forget the message history so the next run starts a new conversation."""
        self.messages = []

    def close(self):
        self.cancel()
        self._unsubscribe()
