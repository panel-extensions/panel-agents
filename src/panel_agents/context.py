"""
The run and tool call the current code executes in.

An ``AgentRunner`` sets them while a run is in progress and around each
tool call, and ``RunState.user_run`` sets them for a batch of a
person's edits, so host code called from a tool, such as a document
mutator, can tell which run and tool call it belongs to without being
handed a pydantic-ai ``RunContext``. The values live in a context
variable, so they follow the code into awaited coroutines, tasks it
starts and worker threads pydantic-ai runs sync tools in, but not into
unrelated callbacks: a widget callback that fires while an agent runs
sees no run, and its edits count as the person's.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class _Scope:

    run_id: str | None = None
    tool_call_id: str | None = None
    actor: str | None = None
    user: str | None = None


_SCOPE: ContextVar[_Scope | None] = ContextVar("panel_agents_scope", default=None)


def current_run_id() -> str | None:
    """Id of the run the calling code executes in, or None outside a run."""
    scope = _SCOPE.get()
    return scope.run_id if scope else None


def current_tool_call_id() -> str | None:
    """Id of the tool call the calling code executes in, or None outside a tool."""
    scope = _SCOPE.get()
    return scope.tool_call_id if scope else None


def _current_scope() -> _Scope | None:
    return _SCOPE.get()


@contextmanager
def _scope(**fields) -> Iterator[_Scope]:
    scope = _Scope(**fields)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


__all__ = ["current_run_id", "current_tool_call_id"]
