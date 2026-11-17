"""
Record the model calls of an ``AgentRunner`` as JSON Lines and load them back.

The model trace is pydantic-ai's own message history: after a header
line, each line is one ``ModelMessage`` in the format of
``ModelMessagesTypeAdapter``. ``AgentRunner`` stamps every
``ModelResponse`` with the wall-clock duration of its request and a
fingerprint of the instructions and tools the model was offered, under
``metadata["panel_agents"]``. A run that ends without its messages
joining the conversation, e.g. because the model request failed, is
followed by a ``failed_run`` line.

Load the history with ``load_messages``, e.g. to continue the
conversation with ``runner.messages = load_messages(path)``, and a
per-request summary with ``load_model_calls``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
)
from pydantic_core import to_jsonable_python

__all__ = [
    "FORMAT_VERSION",
    "ModelCall",
    "ModelTraceRecorder",
    "fingerprint",
    "load_messages",
    "load_model_calls",
]

FORMAT_VERSION = 1

_HEADER_KEY = "panel_agents_model_trace"

_FAILED_KEY = "failed_run"

_METADATA_KEY = "panel_agents"

_ARTIFACTS_KEY = "panel_agents.artifacts"


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


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


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


_CONTENT_FIELDS = {"content": "", "args": None, "signature": None, "metadata": None, "provider_details": None}


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Drop the content of a dumped message, keeping its parts, tool names and ids, usage and timing."""
    data = dict(data)
    data["instructions"] = None
    data["provider_details"] = None
    data["history_digest"] = None
    if isinstance(data.get("metadata"), dict):
        data["metadata"] = {k: v for k, v in data["metadata"].items() if k == _METADATA_KEY} or None
    parts = []
    for part in data.get("parts", []):
        # Binary content, e.g. a file part, has no empty form that still validates.
        if isinstance(part.get("content"), dict):
            continue
        parts.append({k: (_CONTENT_FIELDS[k] if k in _CONTENT_FIELDS else v) for k, v in part.items()})
    data["parts"] = parts
    return {k: v for k, v in data.items() if k in ("parts", "kind") or v is not None}


def _dump_message(message: ModelMessage, include_content: bool) -> str:
    (data,) = ModelMessagesTypeAdapter.dump_python([_without_artifacts(message)], mode="json", fallback=repr)
    if not include_content:
        data = _redact(data)
    return json.dumps(data)


def _header(include_content: bool) -> str:
    return json.dumps({_HEADER_KEY: FORMAT_VERSION, "include_content": include_content})


@dataclasses.dataclass
class _Trace:
    include_content: bool = True
    entries: list[ModelMessage | dict[str, Any]] = dataclasses.field(default_factory=list)


def _read(path: str | os.PathLike) -> _Trace:
    trace = _Trace()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if _HEADER_KEY in data:
                version = int(data[_HEADER_KEY])
                if version > FORMAT_VERSION:
                    raise ValueError(f"{path} is a version {version} model trace; this panel-agents reads up to version {FORMAT_VERSION}.")
                trace.include_content = trace.include_content and data.get("include_content", True)
            elif _FAILED_KEY in data:
                trace.entries.append(data)
            else:
                trace.entries.extend(ModelMessagesTypeAdapter.validate_python([data]))
    return trace


def load_messages(path: str | os.PathLike, *, include_failed: bool = False) -> list[ModelMessage]:
    """
    Read the message history of a model trace as pydantic-ai messages.

    Messages of failed runs are left out by default, as ``AgentRunner``
    leaves them out of the conversation, so assigning the result to
    ``AgentRunner.messages`` continues the conversation where it was.
    A trace recorded with ``include_content=False`` loads, but has no
    content to continue from.
    """
    trace = _read(path)
    failed = set() if include_failed else {entry[_FAILED_KEY] for entry in trace.entries if isinstance(entry, dict)}
    return [
        entry for entry in trace.entries
        if not isinstance(entry, dict) and (entry.run_id is None or entry.run_id not in failed)
    ]


@dataclasses.dataclass
class ModelCall:
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
    """Why the request failed; None if it produced the response."""
    provider_response_id: str | None = None


def _calls_of(response: ModelResponse) -> list[ModelCall]:
    calls = []
    for attempt in getattr(response, "failed_attempts", None) or []:
        usage = attempt.usage
        calls.append(ModelCall(
            run_id=response.run_id, model=attempt.model_name, timestamp=attempt.timestamp,
            duration=attempt.duration.total_seconds(), error=attempt.error or "Response rejected",
            input_tokens=usage.input_tokens if usage else 0, output_tokens=usage.output_tokens if usage else 0,
            cached_tokens=usage.cache_read_tokens if usage else 0,
            cost_usd=float(usage.cost) if usage is not None and getattr(usage, "cost", None) is not None else None,
        ))
    metadata = (response.metadata or {}).get(_METADATA_KEY, {})
    usage = response.usage
    calls.append(ModelCall(
        run_id=response.run_id, model=response.model_name, timestamp=response.timestamp,
        duration=metadata.get("duration"),
        tool_calls=[(p.tool_name, p.tool_call_id) for p in response.parts if isinstance(p, ToolCallPart)],
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, cached_tokens=usage.cache_read_tokens,
        cost_usd=_response_cost(response), provider_response_id=response.provider_response_id,
    ))
    return calls


def load_model_calls(path: str | os.PathLike) -> list[ModelCall]:
    """
    Summarize every model request in a model trace, in order: the model,
    run, tool calls, tokens, cost, duration and, for failed requests,
    the error. Failed attempts that a later attempt recovered from and
    requests that ended a run come before the response they belong to.
    """
    calls: list[ModelCall] = []
    for entry in _read(path).entries:
        if isinstance(entry, ModelResponse):
            calls.extend(_calls_of(entry))
        elif isinstance(entry, dict) and (request := entry.get("request")):
            calls.append(ModelCall(
                run_id=entry[_FAILED_KEY], model=request.get("model"),
                timestamp=datetime.fromisoformat(request["timestamp"]) if request.get("timestamp") else None,
                duration=request.get("duration"), error=entry.get("error"),
            ))
    return calls


class ModelTraceRecorder:
    """
    Append the model calls of an ``AgentRunner`` to a JSON Lines file.

    Every message is written as soon as the run produced it, and runs
    of the same conversation append to the same file. When the file is
    new, the runner's current history is written first, so a recording
    started on a resumed conversation is complete on its own. Pass
    ``include_content=False`` to drop prompts, instructions, text,
    tool arguments and results, keeping which tools were called, the
    usage and the timing.

    Usable as a context manager; ``close`` detaches it from the runner.
    """

    def __init__(self, runner: Any, path: str | os.PathLike | IO[str], *, include_content: bool = True):
        self.runner = runner
        self.include_content = include_content
        if isinstance(path, (str, os.PathLike)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            new = not Path(path).exists() or Path(path).stat().st_size == 0
            if not new and _read(path).include_content != include_content:
                raise ValueError(f"{path} was recorded with include_content={not include_content}; record to another file.")
            self._file = open(path, "a")
            self._owned = True
        else:
            self._file = path
            self._owned = False
            new = True
        self._written = 0
        if new:
            self._file.write(_header(include_content) + "\n")
            self._write(list(runner.messages))
        runner._trace_listeners.append(self)

    def _write(self, messages: Sequence[ModelMessage]):
        for message in messages:
            self._file.write(_dump_message(message, self.include_content) + "\n")
        self._file.flush()

    def _run_started(self, history: Sequence[ModelMessage]):
        self._written = len(history)

    def _messages(self, messages: Sequence[ModelMessage]):
        if len(messages) > self._written:
            self._write(messages[self._written:])
            self._written = len(messages)

    def _run_failed(self, run_id: str, error: BaseException, request: dict[str, Any] | None):
        record = {_FAILED_KEY: run_id, "error": _error_text(error), "request": request}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        if self in self.runner._trace_listeners:
            self.runner._trace_listeners.remove(self)
        if self._owned:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
