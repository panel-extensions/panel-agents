"""
Reactive state of agent runs, built by applying normalized events.

Components bind to the parameterized objects defined here and never
see framework-specific objects. Because the state is derived purely
from events, replaying a recording reproduces the same state.
"""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import param
from pydantic_core import from_json

from . import events as ev
from .artifacts import ArtifactStore


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def parse_partial_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = from_json(text, allow_partial="trailing-strings")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class MessageState(param.Parameterized):

    message_id = param.String(constant=True)

    role = param.Selector(default="assistant", objects=["assistant", "user", "system", "developer"])

    kind = param.Selector(default="text", objects=["text", "thinking"])

    content = param.String(default="")

    finished = param.Boolean(default=False)

    run_id = param.String(default=None, allow_None=True)

    created_at = param.Number(default=None, allow_None=True)


class ApprovalRequest(param.Parameterized):

    tool_call_id = param.String(constant=True)

    tool_name = param.String(constant=True)

    args = param.Dict(default={}, doc="Arguments the agent proposed.")

    schema = param.Dict(default=None, allow_None=True, doc="JSON schema of the tool arguments.")

    description = param.String(default=None, allow_None=True)

    status = param.Selector(default="pending", objects=["pending", "approved", "denied"])

    final_args = param.Dict(default=None, allow_None=True, doc="Arguments after user edits, if approved.")

    reason = param.String(default=None, allow_None=True)

    user = param.String(default=None, allow_None=True)

    requested_at = param.Number(default=None, allow_None=True)

    resolved_at = param.Number(default=None, allow_None=True)

    run_id = param.String(default=None, allow_None=True)

    @property
    def edited(self) -> bool:
        return self.final_args is not None and self.final_args != self.args


class ToolCallState(param.Parameterized):

    tool_call_id = param.String(constant=True)

    tool_name = param.String(default="")

    status = param.Selector(
        default="pending",
        objects=["pending", "awaiting_approval", "running", "success", "failed", "denied"],
        doc="""
        pending: arguments are still streaming.
        awaiting_approval: the call is blocked on a human decision.
        running: arguments are complete and the tool is executing.
        success / failed: the tool returned or raised.
        denied: a human rejected the call.""",
    )

    args_text = param.String(default="", doc="Raw JSON-encoded arguments as streamed.")

    args = param.Dict(default={}, doc="Best-effort parse of the (possibly partial) arguments.")

    result = param.Parameter(default=None)

    error = param.String(default=None, allow_None=True)

    approval = param.ClassSelector(class_=ApprovalRequest, default=None, allow_None=True)

    started_at = param.Number(default=None, allow_None=True)

    ended_at = param.Number(default=None, allow_None=True)

    run_id = param.String(default=None, allow_None=True)

    parent_message_id = param.String(default=None, allow_None=True)

    @property
    def duration(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


class RunInfo(param.Parameterized):

    run_id = param.String(constant=True)

    parent_run_id = param.String(default=None, allow_None=True)

    thread_id = param.String(default=None, allow_None=True)

    agent_name = param.String(default=None, allow_None=True)

    input = param.Parameter(default=None)

    result = param.Parameter(default=None)

    status = param.Selector(default="running", objects=["running", "finished", "error"])

    error = param.String(default=None, allow_None=True)

    started_at = param.Number(default=None, allow_None=True)

    ended_at = param.Number(default=None, allow_None=True)

    steps = param.List(default=[], doc="Names of steps in the order they started.")

    usage = param.List(default=[], item_type=ev.Usage)

    guardrails = param.List(default=[], item_type=ev.GuardrailFired)

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usage)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usage)

    @property
    def cost_usd(self) -> float | None:
        costs = [u.cost_usd for u in self.usage if u.cost_usd is not None]
        return sum(costs) if costs else None

    @property
    def duration(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


_DELTA_EVENTS = (ev.TextMessageContent, ev.ThinkingContent, ev.ToolCallArgs)


class RunState(param.Parameterized):
    """
    Reactive store for one session's agent runs.

    Adapters call ``apply`` with normalized events; components watch
    the parameters and the per-entity parameterized objects. All
    methods must be called from the session's event loop.
    """

    items = param.List(default=[], doc="""
        Messages and tool calls in the order they started, the
        timeline a chat view renders.""")

    messages = param.Dict(default={}, doc="Mapping of message id to MessageState.")

    tool_calls = param.Dict(default={}, doc="Mapping of tool call id to ToolCallState.")

    runs = param.Dict(default={}, doc="Mapping of run id to RunInfo.")

    approvals = param.Dict(default={}, doc="Mapping of tool call id to ApprovalRequest.")

    pending_approvals = param.List(default=[], item_type=ApprovalRequest)

    agent_state = param.Dict(default={}, doc="Shared state synchronized with the agent.")

    artifacts = param.ClassSelector(class_=ArtifactStore)

    running = param.Boolean(default=False, doc="Whether any top-level run is in progress.")

    flush_interval = param.Number(default=0.05, bounds=(0, None), doc="""
        Minimum seconds between UI updates while streaming deltas.
        Deltas arriving faster are coalesced. Zero disables coalescing.""")

    def __init__(self, **params):
        params.setdefault("artifacts", ArtifactStore())
        super().__init__(**params)
        self._events: list[ev.Event] = []
        self._subscribers: list[Callable[[ev.Event], Any]] = []
        self._buffers: dict[tuple[type, str], list[str]] = {}
        self._last_flush = 0.0

    @classmethod
    def from_events(cls, events: Iterable[ev.Event], **params) -> RunState:
        params.setdefault("flush_interval", 0)
        state = cls(**params)
        state.apply_many(events)
        return state

    @property
    def events(self) -> list[ev.Event]:
        return list(self._events)

    def subscribe(self, callback: Callable[[ev.Event], Any]) -> Callable[[], None]:
        """Call ``callback`` with every applied event; returns an unsubscribe function."""
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def reset(self):
        self._events.clear()
        self._buffers.clear()
        self.artifacts = ArtifactStore()
        self.param.update(
            items=[], messages={}, tool_calls={}, runs={}, approvals={},
            pending_approvals=[], agent_state={}, running=False,
        )

    def apply_many(self, events: Iterable[ev.Event]):
        for event in events:
            self.apply(event, notify=True)
        self.flush()

    def apply(self, event: ev.Event, notify: bool = True):
        self._events.append(event)
        if isinstance(event, _DELTA_EVENTS):
            key = (type(event), event.tool_call_id if isinstance(event, ev.ToolCallArgs) else event.message_id)
            self._buffers.setdefault(key, []).append(event.delta)
            now = time.monotonic()
            if self.flush_interval == 0 or now - self._last_flush >= self.flush_interval:
                self.flush()
        else:
            self.flush()
            handler = getattr(self, f"_on_{event.type.lower()}", None)
            if handler is not None:
                handler(event)
        if notify:
            for callback in list(self._subscribers):
                callback(event)

    def flush(self):
        """Apply all buffered streaming deltas."""
        self._last_flush = time.monotonic()
        buffers, self._buffers = self._buffers, {}
        for (event_type, key), chunks in buffers.items():
            delta = "".join(chunks)
            if event_type is ev.ToolCallArgs:
                call = self.tool_calls.get(key)
                if call is not None:
                    text = call.args_text + delta
                    call.param.update(args_text=text, args=parse_partial_json(text))
            else:
                message = self.messages.get(key)
                if message is not None:
                    message.content += delta

    # Convenience API for UIs

    def add_user_message(self, content: str, run_id: str | None = None) -> str:
        message_id = new_id("msg-")
        self.apply(ev.TextMessageStart(message_id=message_id, role="user", run_id=run_id))
        self.apply(ev.TextMessageContent(message_id=message_id, delta=content, run_id=run_id))
        self.apply(ev.TextMessageEnd(message_id=message_id, run_id=run_id))
        return message_id

    def resolve_approval(
        self,
        tool_call_id: str,
        approved: bool,
        args: dict[str, Any] | None = None,
        reason: str | None = None,
        user: str | None = None,
    ):
        request = self.approvals[tool_call_id]
        self.apply(ev.ApprovalResolved(
            tool_call_id=tool_call_id, approved=approved, args=args,
            reason=reason, user=user, run_id=request.run_id,
        ))

    # Event handlers

    def _add_item(self, item):
        self.items = [*self.items, item]

    def _update_running(self):
        self.running = any(r.status == "running" and r.parent_run_id is None for r in self.runs.values())

    def _on_run_started(self, event: ev.RunStarted):
        run = RunInfo(
            run_id=event.run_id, parent_run_id=event.parent_run_id, thread_id=event.thread_id,
            agent_name=event.agent_name, input=event.input, started_at=event.timestamp,
        )
        self.runs = {**self.runs, event.run_id: run}
        self._update_running()

    def _on_run_finished(self, event: ev.RunFinished):
        if run := self.runs.get(event.run_id):
            run.param.update(status="finished", result=event.result, ended_at=event.timestamp)
        self._update_running()

    def _on_run_error(self, event: ev.RunError):
        if run := self.runs.get(event.run_id):
            run.param.update(status="error", error=event.message, ended_at=event.timestamp)
        for call in self.tool_calls.values():
            if call.run_id == event.run_id and call.status in ("pending", "running"):
                call.param.update(status="failed", error=call.error or event.message, ended_at=event.timestamp)
        self._update_running()

    def _on_step_started(self, event: ev.StepStarted):
        if run := self.runs.get(event.run_id):
            run.steps = [*run.steps, event.step_name]

    def _start_message(self, event, kind: str, role: str = "assistant"):
        if event.message_id in self.messages:
            return
        message = MessageState(
            message_id=event.message_id, role=role, kind=kind,
            run_id=event.run_id, created_at=event.timestamp,
        )
        self.messages = {**self.messages, event.message_id: message}
        self._add_item(message)

    def _end_message(self, event):
        if message := self.messages.get(event.message_id):
            message.finished = True

    def _on_text_message_start(self, event: ev.TextMessageStart):
        self._start_message(event, "text", event.role)

    def _on_text_message_end(self, event: ev.TextMessageEnd):
        self._end_message(event)

    def _on_thinking_start(self, event: ev.ThinkingStart):
        self._start_message(event, "thinking")

    def _on_thinking_end(self, event: ev.ThinkingEnd):
        self._end_message(event)

    def _on_tool_call_start(self, event: ev.ToolCallStart):
        if event.tool_call_id in self.tool_calls:
            return
        call = ToolCallState(
            tool_call_id=event.tool_call_id, tool_name=event.tool_name, run_id=event.run_id,
            parent_message_id=event.parent_message_id, started_at=event.timestamp,
        )
        self.tool_calls = {**self.tool_calls, event.tool_call_id: call}
        self._add_item(call)

    def _on_tool_call_end(self, event: ev.ToolCallEnd):
        call = self.tool_calls.get(event.tool_call_id)
        if call is not None and call.status == "pending":
            call.param.update(status="running", args=parse_partial_json(call.args_text))

    def _on_tool_call_result(self, event: ev.ToolCallResult):
        call = self.tool_calls.get(event.tool_call_id)
        if call is None:
            return
        if event.is_error:
            call.param.update(status="failed", error=str(event.content), result=event.content, ended_at=event.timestamp)
        else:
            call.param.update(status="success", result=event.content, ended_at=event.timestamp)

    def _on_state_snapshot(self, event: ev.StateSnapshot):
        self.agent_state = copy.deepcopy(event.snapshot)

    def _on_state_delta(self, event: ev.StateDelta):
        self.agent_state = apply_json_patch(self.agent_state, event.delta)

    def _on_artifact_published(self, event: ev.ArtifactPublished):
        self.artifacts.publish(
            event.artifact_id, event.value, kind=event.kind, title=event.title,
            tool_call_id=event.tool_call_id, metadata=event.metadata, created_at=event.timestamp,
        )

    def _on_approval_requested(self, event: ev.ApprovalRequested):
        request = ApprovalRequest(
            tool_call_id=event.tool_call_id, tool_name=event.tool_name, args=dict(event.args),
            schema=event.schema, description=event.description,
            requested_at=event.timestamp, run_id=event.run_id,
        )
        self.approvals = {**self.approvals, event.tool_call_id: request}
        self.pending_approvals = [*self.pending_approvals, request]
        call = self.tool_calls.get(event.tool_call_id)
        if call is None:
            call = ToolCallState(
                tool_call_id=event.tool_call_id, tool_name=event.tool_name,
                run_id=event.run_id, started_at=event.timestamp,
            )
            self.tool_calls = {**self.tool_calls, event.tool_call_id: call}
            self._add_item(call)
        call.param.update(status="awaiting_approval", approval=request, args=dict(event.args))

    def _on_approval_resolved(self, event: ev.ApprovalResolved):
        request = self.approvals.get(event.tool_call_id)
        if request is None:
            return
        request.param.update(
            status="approved" if event.approved else "denied",
            final_args=event.args if event.approved else None,
            reason=event.reason, user=event.user, resolved_at=event.timestamp,
        )
        self.pending_approvals = [r for r in self.pending_approvals if r is not request]
        if call := self.tool_calls.get(event.tool_call_id):
            if event.approved:
                call.param.update(status="running", args=event.args if event.args is not None else call.args)
            else:
                call.param.update(status="denied", error=event.reason, ended_at=event.timestamp)

    def _on_usage(self, event: ev.Usage):
        if run := self.runs.get(event.run_id):
            run.usage = [*run.usage, event]

    def _on_guardrail_fired(self, event: ev.GuardrailFired):
        if run := self.runs.get(event.run_id):
            run.guardrails = [*run.guardrails, event]


def _split_pointer(path: str) -> list[str]:
    if not path:
        return []
    return [p.replace("~1", "/").replace("~0", "~") for p in path.lstrip("/").split("/")]


def _resolve(doc, parts):
    for part in parts:
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def apply_json_patch(document: dict[str, Any], patch: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply an RFC 6902 patch, returning a new document."""
    doc = copy.deepcopy(document)
    for op in patch:
        parts = _split_pointer(op["path"])
        kind = op["op"]
        if kind in ("move", "copy"):
            src = _split_pointer(op["from"])
            value = copy.deepcopy(_resolve(doc, src))
            if kind == "move":
                parent = _resolve(doc, src[:-1])
                del parent[int(src[-1]) if isinstance(parent, list) else src[-1]]
            kind, op = "add", {"value": value}
        if kind == "test":
            if _resolve(doc, parts) != op["value"]:
                raise ValueError(f"JSON patch test failed at {op['path']!r}")
            continue
        if not parts:
            if kind in ("add", "replace"):
                doc = copy.deepcopy(op["value"])
            continue
        parent, key = _resolve(doc, parts[:-1]), parts[-1]
        if isinstance(parent, list):
            index = len(parent) if key == "-" else int(key)
            if kind == "add":
                parent.insert(index, op["value"])
            elif kind == "replace":
                parent[index] = op["value"]
            elif kind == "remove":
                del parent[index]
        elif kind in ("add", "replace"):
            parent[key] = op["value"]
        elif kind == "remove":
            del parent[key]
    return doc
