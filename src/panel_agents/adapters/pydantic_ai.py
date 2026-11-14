"""
Drive a pydantic-ai agent and translate its run into panel-agents events.

``AgentRunner`` streams a run through ``PanelEventStream``, a
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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import param
from pydantic_ai import CancellationToken, CapabilityEvent, CustomEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import RunCancelled
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
from pydantic_ai.ui import UIEventStream

from .. import events as ev
from ..recording import dump_event
from ..state import RunState, new_id

__all__ = [
    "AgentRunner",
    "ArtifactEvent",
    "PanelEventStream",
    "artifact_return",
    "publish_artifact",
]

_ARTIFACTS_KEY = "panel_agents.artifacts"

DENIED_MESSAGE = "The user denied this tool call."


@dataclass(kw_only=True)
class ArtifactEvent(CustomEvent, name="panel_agents.artifact"):
    """Publishes a version of an artifact; emitted from tools via ``publish_artifact``."""

    artifact_id: str
    value: Any = None
    title: str | None = None
    kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class UsageEvent(CapabilityEvent, namespace="panel_agents"):
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None


async def publish_artifact(
    ctx: RunContext[Any],
    artifact_id: str,
    value: Any,
    *,
    title: str | None = None,
    kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Publish a new version of an artifact from inside an async tool.

    The event is stamped with the calling tool's ``tool_call_id`` and
    reaches the UI in stream order, before the tool's result.
    """
    await ctx.emit(ArtifactEvent(artifact_id=artifact_id, value=value, title=title, kind=kind, metadata=metadata or {}))


def artifact_return(
    return_value: Any,
    artifact_id: str,
    value: Any,
    *,
    title: str | None = None,
    kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolReturn:
    """
    Return ``return_value`` to the model and publish ``value`` as an artifact.

    Works from sync and async tools. The artifact travels in the tool
    return's metadata, which pydantic-ai never sends to the model.
    """
    artifact = ArtifactEvent(artifact_id=artifact_id, value=value, title=title, kind=kind, metadata=metadata or {})
    return ToolReturn(return_value=return_value, metadata={_ARTIFACTS_KEY: [artifact]})


def _response_cost(response: ModelResponse) -> float | None:
    # Recent pydantic-ai versions price the usage already; older ones only offer the genai-prices lookup.
    if (cost := getattr(response.usage, "cost", None)) is not None:
        return float(cost)
    try:
        return float(response.cost().total_price)
    except Exception:
        return None


@dataclass
class _RunCapability(AbstractCapability[Any]):
    """Run-scoped hooks that need the run context rather than the event stream."""

    runner: AgentRunner | None = None

    async def prepare_tools(self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        self.runner._tool_defs.update({td.name: td for td in tool_defs})
        return tool_defs

    async def after_model_request(self, ctx: RunContext[Any], *, request_context: Any, response: ModelResponse) -> ModelResponse:
        usage = response.usage
        await ctx.emit(UsageEvent(
            model=response.model_name, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cached_tokens=usage.cache_read_tokens, cost_usd=_response_cost(response),
        ))
        return response

    async def handle_deferred_tool_calls(self, ctx: RunContext[Any], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # External calls (CallDeferred) are left unresolved so the run ends with them as its output.
        if not requests.approvals:
            return None
        decisions = await asyncio.gather(*(self.runner._wait_for_decision(call.tool_call_id) for call in requests.approvals))
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for call, decision in zip(requests.approvals, decisions, strict=True):
            if decision.approved:
                edited = decision.args is not None and decision.args != call.args_as_dict()
                approvals[call.tool_call_id] = ToolApproved(override_args=decision.args if edited else None)
            else:
                approvals[call.tool_call_id] = ToolDenied(decision.reason or DENIED_MESSAGE)
        return DeferredToolResults(approvals=approvals)


@dataclass
class PanelEventStream(UIEventStream[None, ev.Event, Any, Any]):
    """Translates pydantic-ai stream events into panel-agents events."""

    run_id: str = field(default_factory=lambda: new_id("run-"))
    thread_id: str | None = None
    agent_name: str | None = None
    input: Any = None
    tool_defs: dict[str, ToolDefinition] = field(default_factory=dict)

    error: Exception | None = None
    _text_id: str | None = None
    _thinking_id: str | None = None
    _response_text_id: str | None = None

    def encode_event(self, event: ev.Event) -> str:
        return f"data: {dump_event(event)}\n\n"

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
        yield ev.RunError(run_id=self.run_id, message=str(error) or type(error).__name__, code=code)

    async def before_response(self) -> AsyncIterator[ev.Event]:
        self._response_text_id = None
        return
        yield

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[ev.Event]:
        if not follows_text or self._text_id is None:
            self._text_id = self._response_text_id = new_id("msg-")
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
            self._thinking_id = new_id("think-")
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
                tool_call_id=call.tool_call_id, tool_name=call.tool_name, args=call.args_as_dict(),
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

    def _artifact(self, artifact: ArtifactEvent, tool_call_id: str | None) -> ev.ArtifactPublished:
        return ev.ArtifactPublished(
            artifact_id=artifact.artifact_id, value=artifact.value, kind=artifact.kind, title=artifact.title,
            tool_call_id=artifact.tool_call_id or tool_call_id, metadata=dict(artifact.metadata),
        )

    async def handle_custom_event(self, event: CustomEvent) -> AsyncIterator[ev.Event]:
        if isinstance(event, ArtifactEvent):
            yield self._artifact(event, event.tool_call_id)
        else:
            yield ev.Custom(name=event.name, value=event.to_payload())

    async def handle_capability_event(self, event: CapabilityEvent) -> AsyncIterator[ev.Event]:
        if isinstance(event, UsageEvent):
            yield ev.Usage(
                model=event.model, input_tokens=event.input_tokens, output_tokens=event.output_tokens,
                cached_tokens=event.cached_tokens, cost_usd=event.cost_usd,
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
        and the exception is available as ``error``.""")

    run_kwargs = param.Dict(default={}, doc="Extra keyword arguments passed to ``Agent.run_stream_events``.")

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
        self._waiters: dict[str, asyncio.Future[ev.ApprovalResolved]] = {}
        self._early: dict[str, ev.ApprovalResolved] = {}
        self._token: CancellationToken | None = None
        self._unsubscribe = self.state.subscribe(self._on_event)

    def _on_event(self, event: ev.Event):
        if not isinstance(event, ev.ApprovalResolved):
            return
        future = self._waiters.get(event.tool_call_id)
        if future is None:
            # The run may not have reached its approval handler yet.
            if self.running and event.run_id == self.run_id:
                self._early[event.tool_call_id] = event
        elif not future.done():
            future.set_result(event)

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
        run_id = new_id("run-")
        with param.edit_constant(self):
            self.param.update(running=True, run_id=run_id, error=None, deferred=None)
        if isinstance(prompt, str) and prompt:
            self.state.add_user_message(prompt, run_id=run_id)
        self._token = CancellationToken()
        stream = PanelEventStream(
            run_id=run_id, thread_id=self.thread_id, agent_name=self.agent.name,
            input=prompt, tool_defs=self._tool_defs,
        )
        run_kwargs = {**self.run_kwargs, **kwargs}
        capabilities = [*run_kwargs.pop("capabilities", []), _RunCapability(runner=self)]
        try:
            async with self.agent.run_stream_events(
                prompt, message_history=self.messages or None, deferred_tool_results=deferred_tool_results,
                deps=self.deps, capabilities=capabilities, cancellation_token=self._token, **run_kwargs,
            ) as events:
                async for event in stream.transform_stream(events):
                    self._apply(stream, event)
        except asyncio.CancelledError:
            self._close(stream, ev.RunError(run_id=run_id, message="Run cancelled", code="cancelled"))
            raise
        except Exception as e:
            # Errors raised outside the event stream, e.g. invalid run arguments.
            stream.error = e
            self._close(stream, ev.RunError(run_id=run_id, message=str(e) or type(e).__name__, code=type(e).__name__))
        finally:
            self._token = None
            self._early.clear()
            with param.edit_constant(self):
                self.running = False
        return self._finish(stream)

    def _apply(self, stream: PanelEventStream, event: ev.Event):
        if isinstance(event, ev.RunError):
            self._close(stream, event)
        else:
            self.state.apply(event)

    def _close(self, stream: PanelEventStream, error: ev.RunError):
        # Approvals of a dead run can no longer be acted on, so close them before the error.
        for request in list(self.state.pending_approvals):
            if request.run_id == stream.run_id:
                self.state.apply(ev.ApprovalResolved(
                    tool_call_id=request.tool_call_id, approved=False, reason=error.message, run_id=stream.run_id,
                ))
        self.state.apply(error)

    def _finish(self, stream: PanelEventStream) -> Any:
        result: AgentRunResult | None = stream._result
        if stream.cancelled is not None:
            self.messages = list(stream.cancelled.all_messages())
        elif result is not None:
            self.messages = list(result.all_messages())
        with param.edit_constant(self):
            self.error = stream.error
            if result is not None and isinstance(result.output, DeferredToolRequests):
                self.deferred = result.output
        if stream.error is not None:
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
