"""
Audit trail of human approval decisions.

``AuditLog`` subscribes to a ``RunState`` and writes one record per
``ApprovalResolved`` event to a sink: a JSON Lines file or any callable.
It depends on neither Panel nor an agent framework.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import events as ev
from .state import RunState

_Sink = str | os.PathLike | Callable[[dict[str, Any]], Any]


class AuditLog:
    """
    Records every approval decision made on one or more RunStates.

    Each record holds the timestamp (epoch seconds and ISO 8601 UTC),
    run id, tool call id, tool name, the original and final arguments,
    the decision, whether the arguments were edited, the reason and the
    user. Records are also kept in memory on ``records``.

    >>> log = AuditLog(state, "approvals.jsonl")
    """

    def __init__(self, state: RunState | None = None, sink: _Sink | None = None, include_existing: bool = False):
        self.sink = sink
        self.records: list[dict[str, Any]] = []
        self._unsubscribers: dict[int, Callable[[], None]] = {}
        if state is not None:
            self.attach(state, include_existing=include_existing)

    def attach(self, state: RunState, include_existing: bool = False):
        """
        Start recording decisions on ``state``. With ``include_existing``,
        decisions already applied (e.g. from a loaded recording) are
        recorded too.
        """
        if id(state) in self._unsubscribers:
            return
        if include_existing:
            for event in state.events:
                if isinstance(event, ev.ApprovalResolved):
                    self._record(state, event)
        self._unsubscribers[id(state)] = state.subscribe(lambda event: self._on_event(state, event))

    def detach(self, state: RunState | None = None):
        """Stop recording ``state``, or every attached state."""
        keys = list(self._unsubscribers) if state is None else [id(state)]
        for key in keys:
            if unsubscribe := self._unsubscribers.pop(key, None):
                unsubscribe()

    close = detach

    def _on_event(self, state: RunState, event: ev.Event):
        if isinstance(event, ev.ApprovalResolved):
            self._record(state, event)

    def _record(self, state: RunState, event: ev.ApprovalResolved):
        request = state.approvals.get(event.tool_call_id)
        original = copy.deepcopy(request.args) if request is not None else None
        final = copy.deepcopy(event.args if event.args is not None else original) if event.approved else None
        record = {
            "timestamp": event.timestamp,
            "time": dt.datetime.fromtimestamp(event.timestamp, dt.timezone.utc).isoformat(),
            "run_id": event.run_id or (request.run_id if request is not None else None),
            "tool_call_id": event.tool_call_id,
            "tool_name": request.tool_name if request is not None else None,
            "original_args": original,
            "final_args": final,
            "decision": "approved" if event.approved else "denied",
            "edited": final is not None and original is not None and final != original,
            "reason": event.reason,
            "user": event.user,
        }
        self.records.append(record)
        self.write(record)

    def write(self, record: dict[str, Any]):
        sink = self.sink
        if sink is None:
            return
        if callable(sink):
            sink(record)
            return
        path = Path(sink)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def read(path: str | os.PathLike) -> list[dict[str, Any]]:
        """Read the records of a JSON Lines audit file."""
        with Path(path).open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


__all__ = ["AuditLog"]
