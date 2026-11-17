"""
Play a recorded model trace back to an agent instead of calling a model.

A ``Cassette`` is a pydantic-ai ``FunctionModel`` that answers each
model request with the next recorded ``ModelResponse``. The agent loop,
its tools, approvals and the UI run for real with no provider call,
which makes recorded sessions usable as fast, offline tests: a tool
refactor must produce the same tool results, and a UI change can be
driven through a full agent run without an API key.

The cassette compares what the agent sends back with the recording and
collects every difference as a ``Divergence``. It raises
``CassetteError`` when a recorded response calls a tool the agent no
longer offers, when the agent asks for more responses than were
recorded and, by default, when the instructions or tools differ from
the ones the recording was made with.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic_ai import messages as _messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_core import to_jsonable_python

from .model_trace import _METADATA_KEY, _read, fingerprint, load_messages

__all__ = ["Cassette", "CassetteError", "Divergence"]

# Output validator retries are RetryPromptParts without a tool in older pydantic-ai releases.
_RetryFeedbackPart = getattr(_messages, "RetryFeedbackPart", None)


class CassetteError(RuntimeError):
    """Playback cannot continue: the agent no longer matches the recording."""


@dataclass(frozen=True)
class Divergence:
    """A difference between what the agent sent the model and what was recorded."""

    request: int
    """The 1-based number of the model request it was found in."""
    kind: str
    """``"tool_return"``, ``"prompt"``, ``"retry"`` or ``"fingerprint"``."""
    expected: Any
    actual: Any
    tool_name: str | None = None
    tool_call_id: str | None = None

    def __str__(self) -> str:
        what = f"{self.kind} of {self.tool_name!r} ({self.tool_call_id})" if self.tool_call_id else self.kind
        return f"request {self.request}: {what} differs, recorded {self.expected!r}, got {self.actual!r}"


def _normalize(value: Any) -> Any:
    return to_jsonable_python(value, fallback=repr)


def _tool_parts(message: ModelMessage | None) -> dict[str, ToolReturnPart | RetryPromptPart]:
    if not isinstance(message, ModelRequest):
        return {}
    return {
        part.tool_call_id: part for part in message.parts
        if isinstance(part, ToolReturnPart) or (isinstance(part, RetryPromptPart) and part.tool_name)
    }


def _prompts(message: ModelMessage | None) -> list[Any]:
    if not isinstance(message, ModelRequest):
        return []
    return [_normalize(part.content) for part in message.parts if isinstance(part, UserPromptPart)]


def _retries(message: ModelMessage | None) -> list[Any]:
    if not isinstance(message, ModelRequest):
        return []
    return [
        _normalize(part.content) for part in message.parts
        if (isinstance(part, RetryPromptPart) and not part.tool_name)
        or (_RetryFeedbackPart is not None and isinstance(part, _RetryFeedbackPart))
    ]


class Cassette(FunctionModel):
    """
    A model that answers with the responses of a recorded message history, in order.

    Pass it as the agent's model, e.g. ``AgentRunner(agent, run_kwargs={"model":
    Cassette.from_trace(path)})``, or with ``agent.override(model=...)``.
    Streamed and non-streamed runs are both supported. Played responses
    keep their recorded parts, model name and usage, but take the run and
    conversation ids of the playback.

    Every tool result, user prompt and output retry the agent sends is
    compared with the recording; differences are collected in
    ``divergences``, or raised as ``CassetteError`` with ``strict=True``.
    ``check_fingerprint`` compares the instructions and tools offered with
    the fingerprint stored on each recorded response by ``AgentRunner``, so
    a cassette recorded against another version of the agent fails instead
    of passing silently; turn it off for agents whose instructions change
    from run to run.
    """

    def __init__(self, messages: Sequence[ModelMessage], *, strict: bool = False, check_fingerprint: bool = True):
        super().__init__(self._respond, model_name="cassette")
        self.messages = list(messages)
        self.strict = strict
        self.check_fingerprint = check_fingerprint
        self.divergences: list[Divergence] = []
        self.played = 0
        self._responses: list[ModelResponse] = []
        self._requests: list[list[ModelRequest]] = []
        self._unprepared: ModelRequestParameters | None = None
        pending: list[ModelRequest] = []
        for message in self.messages:
            if isinstance(message, ModelResponse):
                self._responses.append(message)
                self._requests.append(pending)
                pending = []
            else:
                pending.append(message)
        self._returns = {
            call_id: part for message in self.messages for call_id, part in _tool_parts(message).items()
        }

    @classmethod
    def from_trace(cls, source: str | os.PathLike | Iterable[dict[str, Any]], **kwargs) -> Cassette:
        """
        Load the conversation of a model trace written by ``ModelTraceRecorder``,
        from a file or, where the trace loaders accept them, from the records
        a callable sink received.
        """
        if not isinstance(source, (str, os.PathLike)):
            source = list(source)
        if not _read(source).include_content:
            name = source if isinstance(source, (str, os.PathLike)) else "The model trace"
            raise ValueError(f"{name} was recorded with include_content=False and has nothing to play back.")
        return cls(load_messages(source), **kwargs)

    @property
    def responses(self) -> list[ModelResponse]:
        """The recorded responses, in the order they are played."""
        return list(self._responses)

    @property
    def remaining(self) -> int:
        """How many recorded responses have not been played yet."""
        return len(self._responses) - self.played

    def _diverge(self, divergence: Divergence):
        if self.strict:
            raise CassetteError(f"Playback diverged from the recording at {divergence}")
        self.divergences.append(divergence)

    def _compare(self, number: int, messages: list[ModelMessage], params: ModelRequestParameters):
        recorded = self._responses[number - 1]
        stamp = (recorded.metadata or {}).get(_METADATA_KEY, {})
        if (expected := stamp.get("fingerprint")) and (actual := fingerprint(params, messages)) != expected:
            if self.check_fingerprint:
                raise CassetteError(
                    f"The cassette was recorded with other instructions or tools than the agent offers now "
                    f"(fingerprint {expected}, now {actual}); record it again, or pass check_fingerprint=False."
                )
            self._diverge(Divergence(number, "fingerprint", expected, actual))
        request = messages[-1] if messages else None
        recorded_requests = self._requests[number - 1]
        for kind, extract in (("prompt", _prompts), ("retry", _retries)):
            expected_values = [value for message in recorded_requests for value in extract(message)]
            if (actual_values := extract(request)) != expected_values:
                self._diverge(Divergence(number, kind, expected_values, actual_values))
        expected_parts = {k: v for message in recorded_requests for k, v in _tool_parts(message).items()}
        actual_parts = _tool_parts(request)
        for call_id in dict.fromkeys([*expected_parts, *actual_parts]):
            part, expected_part = actual_parts.get(call_id), expected_parts.get(call_id) or self._returns.get(call_id)
            actual_value = _normalize(part.content) if part else None
            expected_value = _normalize(expected_part.content) if expected_part else None
            if actual_value != expected_value or type(part) is not type(expected_part):
                name = (part or expected_part).tool_name
                self._diverge(Divergence(number, "tool_return", expected_value, actual_value, name, call_id))

    def _play(self, messages: list[ModelMessage], params: ModelRequestParameters) -> ModelResponse:
        if self.played >= len(self._responses):
            raise CassetteError(
                f"The cassette has no response left: it recorded {len(self._responses)} model requests "
                f"and the agent made another."
            )
        number = self.played + 1
        # Compared before the model prepares them, as AgentRunner fingerprints them when recording.
        self._compare(number, messages, params)
        recorded = self._responses[number - 1]
        offered = {tool.name for tool in [*params.function_tools, *params.output_tools]}
        for part in recorded.parts:
            if isinstance(part, ToolCallPart) and part.tool_name not in offered:
                raise CassetteError(
                    f"Recorded response {number} calls the tool {part.tool_name!r}, which the agent no longer "
                    f"offers; it offers {sorted(offered)}."
                )
        self.played = number
        # The run that plays the response stamps its own ids, so traces of a playback join with its events.
        return dataclasses.replace(copy.deepcopy(recorded), run_id=None, conversation_id=None)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response = self._play(messages, self._unprepared or info.model_request_parameters)
        self._model_name_played = response.model_name
        return response

    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse:
        self._unprepared = model_request_parameters
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        finally:
            self._unprepared = None
        # FunctionModel names every response after itself; the recorded name is the one to keep.
        response.model_name = self._model_name_played
        return response

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None) -> AsyncIterator[StreamedResponse]:
        # The run's history is what was recorded; the messages sent to the model may render some parts differently.
        history = list(run_context.messages) if run_context is not None and run_context.messages else messages
        recorded = self._play(history, model_request_parameters)
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        yield CompletedStreamedResponse(recorded, model_request_parameters=model_request_parameters, replay_events=True)
