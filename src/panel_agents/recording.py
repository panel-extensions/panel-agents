"""
Record runs as JSON Lines and load them back for replay.

Each line is one event. Values that are not JSON serializable are
encoded by a registered codec so they round-trip, e.g. DataFrames come
back as DataFrames and can be rendered by the same components.
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .events import Event
from .state import RunState

CODEC_KEY = "__codec__"


@dataclass
class Codec:
    name: str
    applies: Callable[[Any], bool]
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]


CODECS: list[Codec] = []


def register_codec(codec: Codec, first: bool = True):
    """Register a codec; codecs registered later take precedence by default."""
    if first:
        CODECS.insert(0, codec)
    else:
        CODECS.append(codec)


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


def encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: encode_value(v) for k, v in value.items()}
    for codec in CODECS:
        if codec.applies(value):
            return {CODEC_KEY: codec.name, "value": codec.encode(value)}
    return {CODEC_KEY: "repr", "value": repr(value)}


def decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if isinstance(value, dict):
        if CODEC_KEY in value:
            name = value[CODEC_KEY]
            if name == "repr":
                return value["value"]
            for codec in CODECS:
                if codec.name == name:
                    return codec.decode(value["value"])
            return value["value"]
        return {k: decode_value(v) for k, v in value.items()}
    return value


def dump_event(event: Event) -> str:
    return json.dumps(encode_value(event.to_dict()))


def load_event(line: str) -> Event:
    return Event.from_dict(decode_value(json.loads(line)))


def dump_events(events: Iterable[Event], path: str | os.PathLike):
    with open(path, "w") as f:
        for event in events:
            f.write(dump_event(event) + "\n")


def load_events(path: str | os.PathLike) -> list[Event]:
    with open(path) as f:
        return [load_event(line) for line in f if line.strip()]


def load_state(path: str | os.PathLike, **params) -> RunState:
    return RunState.from_events(load_events(path), **params)


class Recorder:
    """
    Append every event applied to a RunState to a JSON Lines file.

    Usable as a context manager; ``close`` detaches it from the state.
    """

    def __init__(self, state: RunState, path: str | os.PathLike | IO[str]):
        self.state = state
        if isinstance(path, (str, os.PathLike)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a")
            self._owned = True
        else:
            self._file = path
            self._owned = False
        self._unsubscribe = state.subscribe(self._write)

    def _write(self, event: Event):
        self._file.write(dump_event(event) + "\n")
        self._file.flush()

    def close(self):
        self._unsubscribe()
        if self._owned:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
