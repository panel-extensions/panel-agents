"""
Record the model calls of an ``AgentRunner`` as JSON Lines and load them back.

The model trace is pydantic-ai's own message history. After a header
line, each line is one ``ModelMessage`` in the format of
``ModelMessagesTypeAdapter``, or one of two kinds of record:

- ``{"rewind": n}`` says the history was rewritten, e.g. by a history
  processor, ``AgentRunner.clear()`` or a failed run whose messages do
  not join the conversation, and keeps only its first ``n`` messages;
  the messages that follow continue from there.
- ``{"failed_run": run_id, "error": ..., "request": ...}`` says a run
  failed, with the model request that was in flight, if any.

``AgentRunner`` stamps every ``ModelResponse`` with the wall-clock
duration of its request and a fingerprint of the instructions and tools
the model was offered, under ``metadata["panel_agents"]``.

``load_messages`` applies the rewinds and returns the history the
runner holds, e.g. to continue the conversation with
``runner.messages = load_messages(path)``. ``load_model_calls``
summarizes every model request that was ever recorded.

Like the event ``Recorder``, ``ModelTraceRecorder`` writes to a file, a
text stream or a callable receiving each record as a dict, and the
loaders read a file or an iterable of those dicts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
)
from pydantic_core import to_jsonable_python

from ..recording import _describe, _JsonLinesSink, _read_records, _Sink, _Source

__all__ = [
    "FORMAT_VERSION",
    "ModelRequestSummary",
    "ModelTraceRecorder",
    "fingerprint",
    "load_messages",
    "load_model_calls",
]

FORMAT_VERSION = 1

_HEADER_KEY = "panel_agents_model_trace"

_FAILED_KEY = "failed_run"

_REWIND_KEY = "rewind"

_METADATA_KEY = "panel_agents"

_ARTIFACTS_KEY = "panel_agents.artifacts"

_INTERRUPTED = "Interrupted"


def fingerprint(model_request_parameters: Any, messages: Sequence[ModelMessage] = ()) -> str:
    """
    A short hash of what the model was offered: the instructions, any
    system prompts in ``messages`` and the name, description and JSON
    schema of each function tool.

    ``AgentRunner`` stores it on every response it records, and a
    ``Cassette`` compares it with the agent it plays back to, so a
    recording made against other instructions or tools is detected.
    Instructions that change from run to run, e.g. ones including the
    date, change the fingerprint too.
    """
    params = model_request_parameters
    tools = getattr(params, "declared_function_tools", None)
    if tools is None:
        tools = params.function_tools
    surface = {
        "instructions": [part.content for part in params.instruction_parts or []],
        "system": [
            part.content for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, SystemPromptPart)
        ],
        "tools": sorted(
            ({"name": t.name, "description": t.description, "parameters": t.parameters_json_schema} for t in tools),
            key=lambda t: t["name"],
        ),
    }
    data = json.dumps(to_jsonable_python(surface, fallback=str), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _error_text(error: BaseException, include_content: bool = True) -> str:
    if not include_content or not str(error):
        return type(error).__name__
    return f"{type(error).__name__}: {error}"


def _response_cost(response: ModelResponse) -> float | None:
    # Recent pydantic-ai versions price the usage already; older ones only offer the genai-prices lookup.
    if (cost := getattr(response.usage, "cost", None)) is not None:
        return float(cost)
    try:
        return float(response.cost().total_price)
    except Exception:
        return None


def _without_artifacts(message: ModelMessage) -> ModelMessage:
    # Artifacts reach the UI through the tool return's metadata; the model never sees them and the conversation recording keeps them.
    if not isinstance(message, ModelRequest):
        return message
    parts = []
    for part in message.parts:
        metadata = getattr(part, "metadata", None)
        if isinstance(metadata, dict) and _ARTIFACTS_KEY in metadata:
            part = dataclasses.replace(part, metadata={k: v for k, v in metadata.items() if k != _ARTIFACTS_KEY} or None)
        parts.append(part)
    return dataclasses.replace(message, parts=parts)


_CONTENT_FIELDS = {"args": None, "signature": None, "metadata": None, "provider_details": None}


def _redact_part(part: dict[str, Any]) -> dict[str, Any]:
    part = {k: (_CONTENT_FIELDS[k] if k in _CONTENT_FIELDS else v) for k, v in part.items()}
    if "content" in part:
        content = part["content"]
        if part.get("part_kind") == "file" and isinstance(content, dict):
            # A file part must hold binary content to validate; its identifier is derived from the data.
            part["content"] = {**content, "data": "", "identifier": "redacted", "vendor_metadata": None}
        else:
            part["content"] = ""
    return part


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the content of a dumped message, keeping its parts, tool names and ids, outcomes, usage and timing."""
    data = dict(data)
    data["instructions"] = None
    data["provider_details"] = None
    data["history_digest"] = None
    if isinstance(data.get("metadata"), dict):
        data["metadata"] = {k: v for k, v in data["metadata"].items() if k == _METADATA_KEY} or None
    if data.get("failed_attempts"):
        # Error messages can quote the request, so only the exception type is kept.
        data["failed_attempts"] = [
            {**attempt, "error": attempt["error"].split(":", 1)[0] if attempt.get("error") else None}
            for attempt in data["failed_attempts"]
        ]
    data["parts"] = [_redact_part(part) for part in data.get("parts", [])]
    return {k: v for k, v in data.items() if k in ("parts", "kind") or v is not None}


def _message_data(message: ModelMessage, include_content: bool) -> dict[str, Any]:
    (data,) = ModelMessagesTypeAdapter.dump_python([_without_artifacts(message)], mode="json", fallback=repr)
    return data if include_content else _redact(data)


def _header(include_content: bool) -> dict[str, Any]:
    return {_HEADER_KEY: FORMAT_VERSION, "include_content": include_content}


@dataclasses.dataclass
class _Trace:
    include_content: bool = True
    """False if any part of the trace was recorded without content."""
    entries: list[ModelMessage | dict[str, Any]] = dataclasses.field(default_factory=list)
    """Messages, failed_run and rewind records, in the order they were written."""
    data: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    """The dumped form of the messages in ``entries``, in the same order."""


def _read(source: _Source) -> _Trace:
    trace = _Trace()
    for record in _read_records(source):
        if _HEADER_KEY in record:
            version = int(record[_HEADER_KEY])
            if version > FORMAT_VERSION:
                raise ValueError(
                    f"{_describe(source)} is a version {version} model trace; this panel-agents reads up to version {FORMAT_VERSION}."
                )
            trace.include_content = trace.include_content and record.get("include_content", True)
        elif _FAILED_KEY in record or _REWIND_KEY in record:
            trace.entries.append(record)
        else:
            trace.entries.extend(ModelMessagesTypeAdapter.validate_python([record]))
            trace.data.append(record)
    return trace


def _conversation(trace: _Trace, include_failed: bool = False) -> tuple[list[ModelMessage], list[dict[str, Any]]]:
    """The messages a trace ends with once its rewinds are applied, and their dumped form."""
    messages: list[ModelMessage] = []
    data: list[dict[str, Any]] = []
    stored = iter(trace.data)
    after_failure = False
    for entry in trace.entries:
        if isinstance(entry, dict):
            if _REWIND_KEY in entry and not (include_failed and after_failure):
                del messages[entry[_REWIND_KEY]:]
                del data[entry[_REWIND_KEY]:]
            after_failure = _FAILED_KEY in entry
        else:
            messages.append(entry)
            data.append(next(stored))
            after_failure = False
    return messages, data


def load_messages(source: _Source, *, include_failed: bool = False) -> list[ModelMessage]:
    """
    Read the message history of a model trace as pydantic-ai messages,
    from a JSON Lines file or the records a callable sink received.

    The result is the history the recorded ``AgentRunner`` held, so
    assigning it to ``AgentRunner.messages`` continues the conversation
    where it was. With ``include_failed``, the messages of a failed run,
    which the runner does not keep, are kept. A trace recorded with
    ``include_content=False`` loads, but has no content to continue from.
    """
    return _conversation(_read(source), include_failed)[0]


@dataclasses.dataclass
class ModelRequestSummary:
    """One request to a model, as summarized by ``load_model_calls``."""

    run_id: str | None
    model: str | None
    timestamp: datetime | None = None
    duration: float | None = None
    """Wall-clock seconds the request took, if recorded."""
    tool_calls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    """The ``(tool_name, tool_call_id)`` of each tool call in the response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    """Why the request failed, ``"Interrupted"`` for a response cut off by a cancelled run, None if it succeeded."""
    provider_response_id: str | None = None


def _summaries(response: ModelResponse) -> list[ModelRequestSummary]:
    summaries = []
    for attempt in getattr(response, "failed_attempts", None) or []:
        usage = attempt.usage
        summaries.append(ModelRequestSummary(
            run_id=response.run_id, model=attempt.model_name, timestamp=attempt.timestamp,
            duration=attempt.duration.total_seconds(), error=attempt.error or "Response rejected",
            input_tokens=usage.input_tokens if usage else 0, output_tokens=usage.output_tokens if usage else 0,
            cached_tokens=usage.cache_read_tokens if usage else 0,
            cost_usd=float(usage.cost) if usage is not None and getattr(usage, "cost", None) is not None else None,
        ))
    metadata = (response.metadata or {}).get(_METADATA_KEY, {})
    usage = response.usage
    summaries.append(ModelRequestSummary(
        run_id=response.run_id, model=response.model_name, timestamp=response.timestamp,
        duration=metadata.get("duration"),
        tool_calls=[(p.tool_name, p.tool_call_id) for p in response.parts if isinstance(p, ToolCallPart)],
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, cached_tokens=usage.cache_read_tokens,
        cost_usd=_response_cost(response), provider_response_id=response.provider_response_id,
        error=_INTERRUPTED if getattr(response, "state", None) == "interrupted" else None,
    ))
    return summaries


def load_model_calls(source: _Source) -> list[ModelRequestSummary]:
    """
    Summarize every model request in a model trace, in order: the model,
    run, tool calls, tokens, cost, duration and, for failed requests,
    the error. Failed attempts that a later attempt recovered from come
    before the response they belong to, and requests that ended a run
    follow the run's last response. Responses that a rewind dropped from
    the history are still counted, once.
    """
    summaries: list[ModelRequestSummary] = []
    seen = set()
    for entry in _read(source).entries:
        if isinstance(entry, ModelResponse):
            # A history processor may rewrite a response that was already recorded, so it is matched by request id.
            stamp = (entry.metadata or {}).get(_METADATA_KEY, {})
            key = stamp.get("request_id") or (entry.run_id, entry.timestamp, entry.model_name, entry.provider_response_id)
            if key not in seen:
                seen.add(key)
                summaries.extend(_summaries(entry))
        elif isinstance(entry, dict) and (request := entry.get("request")):
            last = summaries[-1] if summaries else None
            if last is not None and last.run_id == entry[_FAILED_KEY] and last.error == _INTERRUPTED:
                # The request in flight when the run failed is the interrupted response itself.
                last.error = entry.get("error") or last.error
                if last.duration is None:
                    last.duration = request.get("duration")
                continue
            summaries.append(ModelRequestSummary(
                run_id=entry[_FAILED_KEY], model=request.get("model"),
                timestamp=datetime.fromisoformat(request["timestamp"]) if request.get("timestamp") else None,
                duration=request.get("duration"), error=entry.get("error"),
            ))
    return summaries


class ModelTraceRecorder:
    """
    Record the message history of an ``AgentRunner``.

    ``sink`` is a JSON Lines file to append to, a text stream, or a
    callable receiving a copy of each record as the JSON-ready dict a line
    would hold, with redaction already applied. A callable runs
    synchronously on the event loop, so one that stores records slowly
    blocks the app and should hand them to a queue or thread.

    The history is synchronized with the file at the start of every model
    request, before the tools of a response run and when a run ends, so a
    response is on disk before the tools it called run. Runs of the same
    conversation append to the same file. When the runner's history no
    longer extends what the file holds, e.g. after a history processor
    rewrote it, ``AgentRunner.clear()`` or a run that failed, the
    recorder writes a ``rewind`` record and the history from there on,
    so ``load_messages`` always returns what the runner holds.

    Attaching writes the runner's current history: to an existing file
    only where it differs from the conversation the file ends with, so
    ``runner.messages = load_messages(path)`` followed by a recorder on
    the same file only appends. A stream or callable cannot be read back,
    so the recorder starts it over with a ``rewind`` to 0 and the full
    history.

    A failing sink is logged with its traceback and the agent run goes
    on. The recorder then forgets what it wrote and, at the next sync the
    sink accepts, starts over with a ``rewind`` to 0 and the full history,
    so the store heals itself. Pass ``on_error="raise"`` to let the error
    propagate into the run instead.

    Pass ``include_content=False`` to drop prompts, instructions, text,
    thinking, tool arguments and results and error messages, keeping the
    parts, tool names, ids and outcomes, the usage and the timing.

    Usable as a context manager; ``close`` detaches it from the runner.
    """

    def __init__(self, runner: Any, sink: _Sink, *, include_content: bool = True, on_error: str = "log"):
        self.runner = runner
        self.include_content = include_content
        self._written: list[dict[str, Any]] = []
        self._objects: list[Any] = []
        self._restart = False
        self._sink = _JsonLinesSink(sink, on_error=on_error)
        if self._sink.path is not None and not self._sink.new:
            trace = _read(self._sink.path)
            if trace.include_content != include_content:
                self._sink.close()
                raise ValueError(f"{sink} was recorded with include_content={not include_content}; record to another file.")
            self._written = _conversation(trace)[1]
            self._objects = [None] * len(self._written)
        if self._sink.new:
            self._emit(_header(include_content))
        # A stream or callable may hold an earlier conversation, or this one; either way it starts over.
        self._restart = self._sink.path is None
        self._sync(list(runner.messages))
        runner._trace_listeners.append(self)

    def _emit(self, record: dict[str, Any]) -> bool:
        if self._sink.write(record):
            return True
        # What the sink holds is unknown now, so the next sync writes the whole history again.
        self._written.clear()
        self._objects.clear()
        self._restart = True
        return False

    def _sync(self, messages: Sequence[ModelMessage]):
        if self._restart:
            if not self._emit({_REWIND_KEY: 0}):
                return
            self._restart = False
        written, objects = self._written, self._objects
        prefix, limit = 0, min(len(messages), len(written))
        mismatch = None
        while prefix < limit:
            message = messages[prefix]
            # Only the last message written may still change in place, e.g. an interrupted request repaired by the next run.
            if objects[prefix] is message and prefix < len(written) - 1:
                prefix += 1
                continue
            data = _message_data(message, self.include_content)
            if data != written[prefix]:
                mismatch = data
                break
            objects[prefix] = message
            prefix += 1
        if prefix < len(written):
            if not self._emit({_REWIND_KEY: prefix}):
                return
            del written[prefix:], objects[prefix:]
        for message in messages[prefix:]:
            data = mismatch if mismatch is not None else _message_data(message, self.include_content)
            mismatch = None
            if not self._emit(data):
                return
            written.append(data)
            objects.append(message)

    def _messages(self, messages: Sequence[ModelMessage]):
        self._sync(messages)

    def _run_failed(self, run_id: str, error: BaseException, request: dict[str, Any] | None):
        self._emit({_FAILED_KEY: run_id, "error": _error_text(error, self.include_content), "request": request})

    def close(self):
        if self in self.runner._trace_listeners:
            self.runner._trace_listeners.remove(self)
        self._sink.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
