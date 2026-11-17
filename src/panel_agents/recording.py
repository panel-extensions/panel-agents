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
import io
import json
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .events import Event
from .state import RunState

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


class _JsonLinesSink:
    """Writes JSON-ready records to a file, a text stream or a callable, one line or call per record."""

    def __init__(self, sink: _Sink):
        self.path: Path | None = None
        self._file: IO[str] | None = None
        self._callable: Callable[[dict[str, Any]], Any] | None = None
        self._owned = False
        if isinstance(sink, (str, os.PathLike)):
            self.path = Path(sink)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.new = not self.path.exists() or self.path.stat().st_size == 0
            self._file = open(self.path, "a")
            self._owned = True
        elif hasattr(sink, "write"):
            self._file = sink
            try:
                self.new = sink.tell() == 0
            except (OSError, AttributeError):
                self.new = True
        elif callable(sink):
            # A callable cannot be asked what it already holds, so every recorder starts with a header.
            self._callable = sink
            self.new = True
        else:
            raise TypeError(f"A recording sink must be a path, a text stream or a callable, not {type(sink).__name__}.")

    def write(self, record: dict[str, Any]):
        if self._callable is not None:
            self._callable(record)
        else:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def close(self):
        if self._owned:
            self._file.close()


def _read_records(source: _Source) -> Iterator[dict[str, Any]]:
    if isinstance(source, (str, os.PathLike)):
        with open(source) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from source


def _describe(source: _Source) -> str:
    return str(source) if isinstance(source, (str, os.PathLike)) else "The recording"


def dump_events(events: Iterable[Event], path: str | os.PathLike):
    """Write events to a JSON Lines recording, starting with the format version."""
    with open(path, "w") as f:
        f.write(_header() + "\n")
        for event in events:
            f.write(_dump_event(event) + "\n")


def load_events(source: _Source) -> list[Event]:
    """
    Read the events of a recording from a JSON Lines file or from the
    records a callable sink received. Recordings without a format line,
    written before it existed, are read as version 1. A recording from a
    newer format version raises ``ValueError``.
    """
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


def load_state(source: _Source, **params) -> RunState:
    return RunState.from_events(load_events(source), **params)


class Recorder:
    """
    Record every event applied to a RunState.

    ``sink`` is a JSON Lines file to append to, a text stream, or a
    callable receiving each record as the JSON-ready dict a line would
    hold: first the format record, then one per event. ``load_events``
    and ``load_state`` read either back.

    Usable as a context manager; ``close`` detaches it from the state.
    """

    def __init__(self, state: RunState, sink: _Sink):
        self.state = state
        self._sink = _JsonLinesSink(sink)
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


__all__ = [
    "FORMAT_VERSION",
    "Codec",
    "Recorder",
    "dump_events",
    "load_events",
    "load_state",
    "register_codec",
]
