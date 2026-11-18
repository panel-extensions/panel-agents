"""
Record runs as JSON Lines and load them back for replay.

Each line is one event. Values that are not JSON serializable are
encoded by a registered codec so they round-trip, e.g. DataFrames come
back as DataFrames and can be rendered by the same components.

Recorders write to a sink: a file path, a text stream, or a callable
that receives each record as the JSON-ready dict a line would hold, so
a host can store recordings in its own database. Loaders read a path
or any iterable of those dicts.
"""

from __future__ import annotations

import base64
import copy
import inspect
import io
import json
import logging
import os
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .events import ArtifactApplied, ArtifactPublished, Event, RunError, RunFinished, RunStarted
from .state import RunState

_log = logging.getLogger(__name__)

_CODEC_KEY = "__codec__"


@dataclass
class Codec:
    name: str
    applies: Callable[[Any], bool]
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]


_CODECS: list[Codec] = []


def register_codec(codec: Codec, first: bool = True):
    """Register a codec; codecs registered later take precedence by default."""
    if first:
        _CODECS.insert(0, codec)
    else:
        _CODECS.append(codec)


def _is_pandas_frame(value) -> bool:
    return type(value).__name__ == "DataFrame" and type(value).__module__.startswith("pandas")


def _encode_frame(df) -> dict:
    buf = io.BytesIO()
    try:
        df.to_parquet(buf)
    except Exception:
        return {"format": "json", "data": df.to_json(orient="split", date_format="iso")}
    return {"format": "parquet", "data": base64.b64encode(buf.getvalue()).decode()}


def _decode_frame(payload):
    import pandas as pd

    if payload["format"] == "parquet":
        return pd.read_parquet(io.BytesIO(base64.b64decode(payload["data"])))
    return pd.read_json(io.StringIO(payload["data"]), orient="split")


def _is_pydantic(value) -> bool:
    return hasattr(value, "model_dump") and hasattr(type(value), "model_fields")


register_codec(Codec("pydantic", _is_pydantic, lambda v: v.model_dump(mode="json"), lambda v: v), first=False)
register_codec(Codec("dataframe", _is_pandas_frame, _encode_frame, _decode_frame), first=False)
register_codec(Codec("bytes", lambda v: isinstance(v, bytes), lambda v: base64.b64encode(v).decode(), base64.b64decode), first=False)


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _encode_value(v) for k, v in value.items()}
    for codec in _CODECS:
        if codec.applies(value):
            return {_CODEC_KEY: codec.name, "value": codec.encode(value)}
    return {_CODEC_KEY: "repr", "value": repr(value)}


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    if isinstance(value, dict):
        if _CODEC_KEY in value:
            name = value[_CODEC_KEY]
            if name == "repr":
                return value["value"]
            for codec in _CODECS:
                if codec.name == name:
                    return codec.decode(value["value"])
            return value["value"]
        return {k: _decode_value(v) for k, v in value.items()}
    return value


# Version of the recording format, written as the first line of every recording.
FORMAT_VERSION = 1

_HEADER_KEY = "panel_agents_recording"


def _header() -> str:
    return json.dumps({_HEADER_KEY: FORMAT_VERSION})


def _dump_event(event: Event) -> str:
    return json.dumps(_encode_value(event.to_dict()))


# Where a recorder writes: a JSON Lines file, a text stream, or a callable receiving each record.
_Sink = str | os.PathLike | IO[str] | Callable[[dict[str, Any]], Any]

# What a loader reads: a JSON Lines file, or the records a callable sink received, in order.
_Source = str | os.PathLike | Iterable[dict[str, Any]]


def _is_async_callable(obj: Any) -> bool:
    return inspect.iscoroutinefunction(obj) or inspect.iscoroutinefunction(type(obj).__call__)


def _end_cleanly(path: Path):
    """Complete or drop a last line cut short, e.g. by a crash mid-write, so appending starts on a new line."""
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except ValueError:
            _log.warning("Dropping the incomplete last line of %s.", path)
            f.truncate(start)
        else:
            f.write(b"\n")


class _JsonLinesSink:
    """
    Writes JSON-ready records to a file, a text stream or a callable, one
    line or call per record. A sink that fails is logged and marked as
    ``failed``, unless ``on_error="raise"``, so recording never ends an
    agent run on its own.
    """

    def __init__(self, sink: _Sink, on_error: str = "log"):
        if on_error not in ("log", "raise"):
            raise ValueError(f"on_error must be 'log' or 'raise', not {on_error!r}.")
        self.on_error = on_error
        self.failed = False
        self.path: Path | None = None
        self._file: IO[str] | None = None
        self._callable: Callable[[dict[str, Any]], Any] | None = None
        self._owned = False
        if isinstance(sink, (str, os.PathLike)):
            self.path = Path(sink)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.new = not self.path.exists() or self.path.stat().st_size == 0
            if not self.new:
                _end_cleanly(self.path)
            self._file = open(self.path, "a")
            self._owned = True
        elif hasattr(sink, "write"):
            self._file = sink
            try:
                self.new = sink.tell() == 0
            except (OSError, AttributeError):
                self.new = True
        elif callable(sink):
            if _is_async_callable(sink):
                raise TypeError(
                    "A recording sink must be a plain callable; recorders call it synchronously, so an async "
                    "function would never be awaited. Hand records to a queue or thread from a plain function instead."
                )
            # A callable cannot be asked what it already holds, so every recorder starts with a header.
            self._callable = sink
            self.new = True
        else:
            raise TypeError(f"A recording sink must be a path, a text stream or a callable, not {type(sink).__name__}.")

    def write(self, record: dict[str, Any]) -> bool:
        """Write ``record``; returns False if the sink failed and the failure was logged."""
        try:
            if self._callable is not None:
                # A copy, so a host annotating the record it received cannot change what the recorder compares against.
                self._callable(copy.deepcopy(record))
            else:
                self._file.write(json.dumps(record) + "\n")
                self._file.flush()
        except Exception:
            if self.on_error == "raise":
                raise
            _log.exception("A recording sink failed; this record is lost.")
            self.failed = True
            return False
        return True

    def close(self):
        if self._owned:
            self._file.close()


def _read_records(source: _Source) -> Iterator[dict[str, Any]]:
    if isinstance(source, (str, os.PathLike)):
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    # A last line without its newline was cut short while it was written.
                    if line.endswith("\n"):
                        raise
                    _log.warning("Ignoring the incomplete last line of %s.", source)
    else:
        yield from source


def _deprecated_path(value: Any, path: Any, name: str) -> Any:
    if path is None:
        return value
    warnings.warn(f"The path argument is deprecated; use {name} instead.", DeprecationWarning, stacklevel=3)
    if value is not None:
        raise TypeError(f"Pass either {name} or path, not both.")
    return path


def _describe(source: _Source) -> str:
    return str(source) if isinstance(source, (str, os.PathLike)) else "The recording"


def dump_events(events: Iterable[Event], path: str | os.PathLike):
    """Write events to a JSON Lines recording, starting with the format version."""
    with open(path, "w") as f:
        f.write(_header() + "\n")
        for event in events:
            f.write(_dump_event(event) + "\n")


def load_events(source: _Source | None = None, *, path: str | os.PathLike | None = None) -> list[Event]:
    """
    Read the events of a recording from a JSON Lines file or from the
    records a callable sink received. Recordings without a format line,
    written before it existed, are read as version 1. A recording from a
    newer format version raises ``ValueError``. ``path`` is a deprecated
    name for ``source``.
    """
    source = _deprecated_path(source, path, "source")
    events = []
    for record in _read_records(source):
        if isinstance(record, dict) and _HEADER_KEY in record:
            if (version := int(record[_HEADER_KEY])) > FORMAT_VERSION:
                raise ValueError(
                    f"{_describe(source)} is a version {version} recording; this panel-agents reads up to version {FORMAT_VERSION}."
                )
        else:
            events.append(Event.from_dict(_decode_value(record)))
    return events


def load_state(source: _Source | None = None, *, path: str | os.PathLike | None = None, **params) -> RunState:
    """Replay a recording into a new ``RunState``. ``path`` is a deprecated name for ``source``."""
    return RunState.from_events(load_events(_deprecated_path(source, path, "source")), **params)


class Recorder:
    """
    Record every event applied to a RunState.

    ``sink`` is a JSON Lines file to append to, a text stream, or a
    callable receiving a copy of each record as the JSON-ready dict a line
    would hold: first the format record, then one per event. A callable
    runs synchronously on the event loop, so one that stores records
    slowly blocks the app and should hand them to a queue or thread.
    ``load_events`` and ``load_state`` read either back. ``path`` is a
    deprecated name for ``sink``.

    A failing sink is logged with its traceback and the run goes on;
    pass ``on_error="raise"`` to let the error propagate instead.

    Usable as a context manager; ``close`` detaches it from the state.
    """

    def __init__(self, state: RunState, sink: _Sink | None = None, *, path: _Sink | None = None, on_error: str = "log"):
        self.state = state
        self._sink = _JsonLinesSink(_deprecated_path(sink, path, "sink"), on_error=on_error)
        if self._sink.new:
            self._sink.write({_HEADER_KEY: FORMAT_VERSION})
        self._unsubscribe = state.subscribe(self._write)

    def _write(self, event: Event):
        self._sink.write(_encode_value(event.to_dict()))

    def close(self):
        self._unsubscribe()
        self._sink.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HistoryRecorder(Recorder):
    """
    Record the version history of artifacts, e.g. to keep a document's
    history across sessions.

    Writes the ``ArtifactPublished`` and ``ArtifactApplied`` events of the
    artifacts in ``artifact_ids`` (default: all) and, for each run that
    published one of those versions, its ``RunStarted`` before the first
    such version and its ``RunFinished`` or ``RunError``, so the history
    keeps its grouping by run and the runs' inputs. Messages, tool calls
    and runs that published nothing are left out. ``sink`` and
    ``on_error`` work as for ``Recorder``; ``load_history`` restores
    what it wrote.
    """

    def __init__(
        self, state: RunState, sink: _Sink, *, artifact_ids: Iterable[str] | None = None, on_error: str = "log",
    ):
        self.artifact_ids = None if artifact_ids is None else set(artifact_ids)
        self._started: dict[str, RunStarted] = {}
        self._written: set[str] = set()
        super().__init__(state, sink, on_error=on_error)

    def _write(self, event: Event):
        if isinstance(event, RunStarted):
            self._started[event.run_id] = event
            return
        if isinstance(event, (RunFinished, RunError)):
            self._started.pop(event.run_id, None)
            if event.run_id in self._written:
                self._written.discard(event.run_id)
                super()._write(event)
            return
        if not isinstance(event, (ArtifactPublished, ArtifactApplied)):
            return
        if self.artifact_ids is not None and event.artifact_id not in self.artifact_ids:
            return
        if (start := self._started.pop(event.run_id, None)) is not None:
            self._written.add(event.run_id)
            super()._write(start)
        super()._write(event)


def load_history(source: _Source, state: RunState | None = None) -> RunState:
    """
    Restore a history written by ``HistoryRecorder`` (or any recording)
    into ``state``, or into a new ``RunState``, and return the state.

    The events are applied without notifying the state's subscribers, so
    a recorder attached to it does not write them again and an
    ``AgentRunner`` does not report them to the model as new versions. A
    run that the stored session never finished is finished, since it can
    no longer continue.
    """
    state = state if state is not None else RunState(flush_interval=0)
    events = load_events(source)
    state.apply_many(events, notify=False)
    open_runs = {e.run_id: e.timestamp for e in events if isinstance(e, RunStarted)}
    for event in events:
        if isinstance(event, (RunFinished, RunError)):
            open_runs.pop(event.run_id, None)
    for run_id in open_runs:
        if (run := state.runs.get(run_id)) is not None and run.status == "running":
            state.apply(RunFinished(run_id=run_id), notify=False)
    return state


__all__ = [
    "FORMAT_VERSION",
    "Codec",
    "HistoryRecorder",
    "Recorder",
    "dump_events",
    "load_events",
    "load_history",
    "load_state",
    "register_codec",
]
