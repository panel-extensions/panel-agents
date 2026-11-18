"""
Normalized agent run events.

The taxonomy mirrors the AG-UI protocol so that framework adapters
(pydantic-ai, AG-UI, ...) all translate into the same small set of
events and components never depend on a specific agent framework.
Kit-specific events (artifacts, approvals, usage, guardrails) map onto
AG-UI ``CUSTOM`` events when exchanged over that protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

_EVENT_TYPES: dict[str, type[Event]] = {}


@dataclass(kw_only=True)
class Event:
    """Base class for all run events."""

    type: ClassVar[str] = "EVENT"

    run_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("type"):
            _EVENT_TYPES[cls.type] = cls

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["type"] = self.type
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        data = dict(data)
        event_type = _EVENT_TYPES[data.pop("type")]
        names = {f.name for f in fields(event_type)}
        return event_type(**{k: v for k, v in data.items() if k in names})


# Lifecycle


@dataclass(kw_only=True)
class RunStarted(Event):
    type: ClassVar[str] = "RUN_STARTED"

    run_id: str
    thread_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    input: Any = None


@dataclass(kw_only=True)
class RunFinished(Event):
    type: ClassVar[str] = "RUN_FINISHED"

    run_id: str
    result: Any = None


@dataclass(kw_only=True)
class RunError(Event):
    type: ClassVar[str] = "RUN_ERROR"

    message: str
    code: str | None = None


@dataclass(kw_only=True)
class StepStarted(Event):
    type: ClassVar[str] = "STEP_STARTED"

    step_name: str


@dataclass(kw_only=True)
class StepFinished(Event):
    type: ClassVar[str] = "STEP_FINISHED"

    step_name: str


# Text messages


@dataclass(kw_only=True)
class TextMessageStart(Event):
    type: ClassVar[str] = "TEXT_MESSAGE_START"

    message_id: str
    role: Literal["assistant", "user", "system", "developer"] = "assistant"


@dataclass(kw_only=True)
class TextMessageContent(Event):
    type: ClassVar[str] = "TEXT_MESSAGE_CONTENT"

    message_id: str
    delta: str


@dataclass(kw_only=True)
class TextMessageEnd(Event):
    type: ClassVar[str] = "TEXT_MESSAGE_END"

    message_id: str


@dataclass(kw_only=True)
class ThinkingStart(Event):
    type: ClassVar[str] = "THINKING_START"

    message_id: str


@dataclass(kw_only=True)
class ThinkingContent(Event):
    type: ClassVar[str] = "THINKING_CONTENT"

    message_id: str
    delta: str


@dataclass(kw_only=True)
class ThinkingEnd(Event):
    type: ClassVar[str] = "THINKING_END"

    message_id: str


# Tool calls


@dataclass(kw_only=True)
class ToolCallStart(Event):
    type: ClassVar[str] = "TOOL_CALL_START"

    tool_call_id: str
    tool_name: str
    parent_message_id: str | None = None


@dataclass(kw_only=True)
class ToolCallArgs(Event):
    """A chunk of the JSON-encoded tool call arguments."""

    type: ClassVar[str] = "TOOL_CALL_ARGS"

    tool_call_id: str
    delta: str


@dataclass(kw_only=True)
class ToolCallEnd(Event):
    """The tool call arguments are complete and the tool begins executing."""

    type: ClassVar[str] = "TOOL_CALL_END"

    tool_call_id: str


@dataclass(kw_only=True)
class ToolCallResult(Event):
    type: ClassVar[str] = "TOOL_CALL_RESULT"

    tool_call_id: str
    content: Any = None
    is_error: bool = False
    message_id: str | None = None


# Shared state


@dataclass(kw_only=True)
class StateSnapshot(Event):
    type: ClassVar[str] = "STATE_SNAPSHOT"

    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class StateDelta(Event):
    """An RFC 6902 JSON patch applied to the shared agent state."""

    type: ClassVar[str] = "STATE_DELTA"

    delta: list[dict[str, Any]] = field(default_factory=list)


@dataclass(kw_only=True)
class Custom(Event):
    type: ClassVar[str] = "CUSTOM"

    name: str
    value: Any = None


# Kit-specific events, exchanged as CUSTOM events over AG-UI


@dataclass(kw_only=True)
class ArtifactPublished(Event):
    type: ClassVar[str] = "ARTIFACT_PUBLISHED"

    artifact_id: str
    value: Any
    kind: str | None = None
    title: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    """Whether the host document already holds this version, e.g. because
    the tool edited the document in place."""
    actor: str | None = None
    """Who made the version, e.g. ``"agent"`` or ``"user"``."""
    user: str | None = None
    """Name of the person who made the version, if a person did."""


@dataclass(kw_only=True)
class ArtifactApplied(Event):
    """A person applied a version of a document artifact to the host document."""

    type: ClassVar[str] = "ARTIFACT_APPLIED"

    artifact_id: str
    version: int
    """The applied ``ArtifactVersion.version``, counting from 1."""
    user: str | None = None
    action: str | None = None
    """``"undo"`` or ``"redo"`` when the version was applied by undoing or
    redoing; None for any other apply, which is a change that can itself
    be undone."""


@dataclass(kw_only=True)
class ArtifactVersionsDropped(Event):
    """The values of a document's older versions were dropped to bound memory."""

    type: ClassVar[str] = "ARTIFACT_VERSIONS_DROPPED"

    artifact_id: str
    before: int
    """Versions numbered below this lose their value, except the applied one."""

@dataclass(kw_only=True)
class ApprovalRequested(Event):
    type: ClassVar[str] = "APPROVAL_REQUESTED"

    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] | None = None
    description: str | None = None
    labels: dict[str, str] | None = None


@dataclass(kw_only=True)
class ApprovalResolved(Event):
    type: ClassVar[str] = "APPROVAL_RESOLVED"

    tool_call_id: str
    approved: bool
    args: dict[str, Any] | None = None
    reason: str | None = None
    user: str | None = None


@dataclass(kw_only=True)
class ModelRequestStarted(Event):
    """
    A request to the model began. Its ``Usage`` event, with the same
    ``request_id``, completes it once the response is in.
    """

    type: ClassVar[str] = "MODEL_REQUEST_STARTED"

    request_id: str
    model: str | None = None


@dataclass(kw_only=True)
class Usage(Event):
    """
    The outcome of one model request: its tokens, cost and duration.

    Events without a ``request_id``, as written before model requests were
    announced by ``ModelRequestStarted``, still describe one request each.
    """

    type: ClassVar[str] = "USAGE"

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    duration: float | None = None
    message_id: str | None = None
    request_id: str | None = None
    tool_call_ids: list[str] = field(default_factory=list)
    """The tool calls the response issued."""
    error: str | None = None
    """Why the request failed, None if it succeeded."""
    finish_reason: str | None = None
    provider_response_id: str | None = None
    attempt_of: str | None = None
    """For a failed attempt that a retry or fallback recovered from, the
    ``request_id`` of the request it belongs to."""


@dataclass(kw_only=True)
class GuardrailFired(Event):
    type: ClassVar[str] = "GUARDRAIL_FIRED"

    name: str
    message: str = ""
    severity: Literal["info", "warning", "block"] = "warning"
    tool_call_id: str | None = None
    message_id: str | None = None


_KIT_EVENTS = (
    ArtifactPublished, ArtifactApplied, ArtifactVersionsDropped, ApprovalRequested, ApprovalResolved, ModelRequestStarted, Usage, GuardrailFired,
)

__all__ = [name for name, obj in list(globals().items()) if isinstance(obj, type) and issubclass(obj, Event)]
