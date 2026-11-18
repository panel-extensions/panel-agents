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

from ..recording import _describe
from .model_trace import _FAILED_KEY, _METADATA_KEY, _REWIND_KEY, _conversation, _read, fingerprint

__all__ = ["Cassette", "CassetteError", "Divergence", "RecordedFailure"]

# Output validator retries are RetryPromptParts without a tool in older pydantic-ai releases.
_RetryFeedbackPart = getattr(_messages, "RetryFeedbackPart", None)

try:
    # No public API gives a model the run's history in a non-streamed request.
    from pydantic_ai._run_context import get_current_run_context as _current_run_context
except ImportError:
    def _current_run_context() -> Any:
        return None


def _history(messages: list[ModelMessage], run_context: Any) -> list[ModelMessage]:
    """The run's history, which is what was recorded; the messages sent to the model may render some parts differently."""
    history = getattr(run_context, "messages", None)
    return list(history) if history else messages


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


class RecordedFailure(RuntimeError):
    """
    A recorded model request that failed, raised again where playback
    reaches it. Its message is the error as the trace holds it, prefixed
    with the exception type, e.g. ``"RuntimeError: provider refused"``.
    It is not a ``CassetteError``: the playback reproduced the recording.
    """


@dataclass(frozen=True)
class _Failure:
    error: str


def _call_key(response: ModelResponse) -> Any:
    # A history processor may rewrite a response that was already recorded, so it is matched as load_model_calls does.
    stamp = (response.metadata or {}).get(_METADATA_KEY) or {}
    return stamp.get("request_id") or (response.run_id, response.timestamp, response.model_name, response.provider_response_id)


def _sent(messages: list[ModelMessage], leftovers: set[int]) -> list[ModelRequest]:
    """
    The requests at the end of a history, after its last response: what the
    next model request sent. Cut-off responses in *leftovers* (by ``id``)
    were never answered, so they do not end the search.
    """
    sent: list[ModelRequest] = []
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            if id(message) in leftovers:
                continue
            break
        sent.insert(0, message)
    return sent


def _recorded_calls(trace: Any) -> list[tuple[ModelResponse | _Failure, list[ModelRequest]]] | None:
    """
    Every model call a trace recorded, once each and in order: the response
    as first written, or the failure of a request that failed, with the
    requests it sent. Those are the requests after the last response in the
    history as it stood when the call was first written, rewinds applied:
    the recorder writes the history at the start of every model request,
    after a history processor has run. Copies of a response that a rewind
    made the trace write again are not new calls. A cut-off response that a
    failed request left behind is replaced by the failure that follows it.
    None for a trace whose responses carry no request id, written before
    ``AgentRunner`` stamped one.
    """
    entries = trace.entries
    if not any(
        isinstance(e, ModelResponse) and ((e.metadata or {}).get(_METADATA_KEY) or {}).get("request_id") for e in entries
    ):
        return None
    calls: list[tuple[ModelResponse | _Failure, list[ModelRequest]]] = []
    messages: list[ModelMessage] = []
    seen: set[Any] = set()
    leftovers: set[int] = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            if _REWIND_KEY in entry:
                del messages[entry[_REWIND_KEY]:]
            elif _FAILED_KEY in entry and entry.get("request"):
                calls.append((_Failure(entry.get("error") or "The recorded request failed."), _sent(messages, leftovers)))
            continue
        if isinstance(entry, ModelResponse) and (key := _call_key(entry)) not in seen:
            seen.add(key)
            if getattr(entry, "state", None) == "interrupted" and _failure_follows(entries, index):
                leftovers.add(id(entry))
            else:
                calls.append((entry, _sent(messages, leftovers)))
        messages.append(entry)
    return calls


def _failure_follows(entries: list[Any], index: int) -> bool:
    """Whether the run of a cut-off response at *index* failed with a request in flight, before any later call."""
    for entry in entries[index + 1:]:
        if isinstance(entry, ModelResponse):
            return False
        if isinstance(entry, dict) and _FAILED_KEY in entry:
            return bool(entry.get("request"))
    return False


def _normalize(value: Any) -> Any:
    return to_jsonable_python(value, fallback=repr)


def _tool_parts(message: ModelMessage | None) -> dict[str, ToolReturnPart | RetryPromptPart]:
    if not isinstance(message, ModelRequest):
        return {}
    return {
        part.tool_call_id: part for part in message.parts
        if isinstance(part, ToolReturnPart) or (isinstance(part, RetryPromptPart) and part.tool_name)
    }


def _request_parts(message: ModelMessage | None) -> Sequence[Any]:
    return message.parts if isinstance(message, ModelRequest) else ()


def _prompts(message: ModelMessage | None) -> list[Any]:
    return [_normalize(part.content) for part in _request_parts(message) if isinstance(part, UserPromptPart)]


def _retries(message: ModelMessage | None) -> list[Any]:
    return [
        _normalize(part.content) for part in _request_parts(message)
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
        self._run_history: list[ModelMessage] | None = None
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

    def _replay_calls(self, calls: list[tuple[ModelResponse | _Failure, list[ModelRequest]]]):
        self._responses = [response for response, _ in calls]
        self._requests = [list(requests) for _, requests in calls]
        self._returns = {
            call_id: part for requests in self._requests for message in requests
            for call_id, part in _tool_parts(message).items()
        }

    @classmethod
    def from_trace(cls, source: str | os.PathLike | Iterable[dict[str, Any]], **kwargs) -> Cassette:
        """
        Play back every model call a trace written by ``ModelTraceRecorder``
        recorded, once each and in the order they were made, each with its
        response as first written and compared with the requests as first
        sent. That holds when the recorded agent rewrote its history, e.g.
        with a history processor that shortens earlier tool results,
        rewrites earlier responses or keeps only the last messages, and
        after ``AgentRunner.clear()``. ``messages`` is the history the trace
        ends with.

        A request that failed, e.g. because the provider refused it, raises
        ``RecordedFailure`` with the recorded error where playback reaches
        it, so the run fails as it did; a tool that raised does so again on
        its own. A trace whose responses carry no request id, written by an
        earlier panel-agents, is played as the history it ends with.
        """
        trace = _read(source if isinstance(source, (str, os.PathLike)) else list(source))
        if not trace.include_content:
            raise ValueError(f"{_describe(source)} was recorded with include_content=False and has nothing to play back.")
        cassette = cls(_conversation(trace)[0], **kwargs)
        if (calls := _recorded_calls(trace)) is not None:
            cassette._replay_calls(calls)
        return cassette

    @property
    def responses(self) -> list[ModelResponse]:
        """The recorded responses, in the order they are played; a recorded failure is not one."""
        return [response for response in self._responses if isinstance(response, ModelResponse)]

    @property
    def remaining(self) -> int:
        """How many recorded calls, responses and failures, have not been played yet."""
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
        self._compare_requests(number, messages)

    def _compare_requests(self, number: int, messages: list[ModelMessage]):
        # Both sides are every request after the last response, as the recorder holds them.
        sent = _sent(messages, set())
        recorded_requests = self._requests[number - 1]
        for kind, extract in (("prompt", _prompts), ("retry", _retries)):
            expected_values = [value for message in recorded_requests for value in extract(message)]
            if (actual_values := [value for message in sent for value in extract(message)]) != expected_values:
                self._diverge(Divergence(number, kind, expected_values, actual_values))
        expected_parts = {k: v for message in recorded_requests for k, v in _tool_parts(message).items()}
        actual_parts = {k: v for message in sent for k, v in _tool_parts(message).items()}
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
        recorded = self._responses[number - 1]
        if isinstance(recorded, _Failure):
            self._compare_requests(number, messages)
            self.played = number
            raise RecordedFailure(recorded.error)
        # Compared before the model prepares them, as AgentRunner fingerprints them when recording.
        self._compare(number, messages, params)
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
        response = self._play(self._run_history or messages, self._unprepared or info.model_request_parameters)
        self._model_name_played = response.model_name
        return response

    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse:
        self._unprepared = model_request_parameters
        self._run_history = _history(messages, _current_run_context())
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        finally:
            self._unprepared = self._run_history = None
        # FunctionModel names every response after itself; the recorded name is the one to keep.
        response.model_name = self._model_name_played
        return response

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None) -> AsyncIterator[StreamedResponse]:
        recorded = self._play(_history(messages, run_context or _current_run_context()), model_request_parameters)
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        yield CompletedStreamedResponse(recorded, model_request_parameters=model_request_parameters, replay_events=True)
