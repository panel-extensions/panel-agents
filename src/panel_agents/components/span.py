"""
Span view of one run: what was asked, every model request and tool call
it made, the artifact versions it published and what went wrong.
"""

from __future__ import annotations

import datetime as dt
from html import escape
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..security import _escape_markdown
from ..state import MessageState, ModelCallState, RunState, ToolCallState, _RunTreeTotals, _span_errors
from ._timeline import version_detail, version_label
from .tool_call import _ARTIFACT_ICONS, _format_duration
from .trace import RunTrace, _release_with_session

_HEADER_CSS = """
:host { width: 100%; }
.sv-head { font-family: var(--mui-font-family, inherit); color: var(--mui-palette-text-primary, inherit); }
.sv-question {
  font-size: 1rem; font-weight: 500; line-height: 1.5; margin: 0 0 2px 0;
  overflow-wrap: anywhere; white-space: pre-wrap; max-height: 9em; overflow-y: auto;
}
.sv-question.muted { font-style: italic; font-weight: 400; color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); }
.sv-meta {
  display: flex; flex-wrap: wrap; gap: 0.25em 0.75em; font-size: 0.75rem; line-height: 1.5;
  font-variant-numeric: tabular-nums; color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
.sv-status.error { color: var(--mui-palette-error-main, #d32f2f); font-weight: 500; }
.sv-status.cancelled { font-weight: 500; }
.sv-status.running { color: var(--mui-palette-primary-main, #1976d2); font-weight: 500; }
.sv-problems { color: var(--mui-palette-warning-main, #ed6c02); font-weight: 500; }
"""

_SECTION_CSS = """
:host { width: 100%; }
.sv-section {
  font-family: var(--mui-font-family, inherit); font-size: 0.75rem; font-weight: 500; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); margin-top: 8px;
}
"""

_PROBLEMS = {
    "run": ("sub-run failed", "sub-runs failed"),
    "cancelled": ("sub-run cancelled", "sub-runs cancelled"),
    "model_call": ("request failed", "requests failed"),
    "attempt": ("request retried", "requests retried"),
    "tool_call": ("tool call failed", "tool calls failed"),
    "guardrail": ("step blocked", "steps blocked"),
}


def _time(timestamp: float | None) -> str:
    if not timestamp:
        return ""
    moment = dt.datetime.fromtimestamp(timestamp)
    if moment.date() == dt.datetime.now().date():
        return moment.strftime("%H:%M:%S")
    return moment.strftime("%b %d, %H:%M")


class SpanView(Viewer):
    """
    Everything one run did, read-only: the question, the run's totals,
    every model request with the tool calls it issued on one time axis
    (a ``RunTrace``), the artifact versions it published and the problems
    it ran into. Child runs count as part of the run. The view follows a
    run still in progress until ``release`` is called.

    Built for provenance, e.g. a host's "How was this made?" on something
    an agent produced: ``RunState.span`` gives the same information as
    data. With ``redacted=True`` it shows the run's shape without its
    content, for people who may not see the data.
    """

    state = param.ClassSelector(class_=RunState, constant=True)

    run_id = param.String(default=None, allow_None=True, doc="Id of the run to show.")

    redacted = param.Boolean(default=False, constant=True, doc="""
        Whether to show which tools ran, the models, tokens, cost, timings
        and the titles and numbers of the versions published, but not the
        question, the agent's text, tool arguments and results, approval
        descriptions, error messages or version details.""")

    row_limit = param.Integer(default=50, allow_None=True, bounds=(1, None), constant=True, doc="""
        Number of steps built when the view opens, with a button building
        the next as many; None builds them all.""")

    on_artifact = param.Callable(default=None, allow_None=True, doc="""
        Called with an artifact id and a version number when a published
        version is clicked, e.g. to open that version.""")

    def __init__(self, state: RunState | None = None, **params):
        if state is not None:
            params["state"] = state
        super().__init__(**params)
        self._header = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 0, 8, 0), stylesheets=[_HEADER_CSS])
        self._missing = pmui.Alert(
            object="This run is not in the session.", severity="info", variant="outlined",
            sizing_mode="stretch_width", visible=False, margin=(4, 0),
        )
        self._unrecorded = pmui.Alert(
            object="The steps of this run were not recorded, only what it published.", severity="info",
            variant="outlined", sizing_mode="stretch_width", visible=False, margin=(4, 0),
        )
        self._trace = RunTrace(
            self.state, run_id=self.run_id, redacted=self.redacted, row_limit=self.row_limit, include_children=True,
            _hide_question=True,
        )
        self._problems = pmui.Alert(
            object="", title="Problems", severity="warning", variant="outlined",
            sizing_mode="stretch_width", visible=False, margin=(4, 0),
        )
        self._published_title = pn.pane.HTML(
            '<div class="sv-section">Published</div>', margin=0, stylesheets=[_SECTION_CSS], visible=False,
        )
        self._published = pmui.MenuList(items=[], dense=True, sizing_mode="stretch_width", margin=0, visible=False)
        self._published.on_click(self._on_published_click)
        self._body = pn.Column(
            self._unrecorded, self._problems, self._trace, self._published_title, self._published,
            sizing_mode="stretch_width", margin=0,
        )
        self._layout = pn.Column(
            self._header, self._missing, self._body, sizing_mode="stretch_width", css_classes=["span-view"],
        )
        self._store_watcher = None
        self._released = False
        _release_with_session(self)
        self._state_watchers = [
            self.state.param.watch(self._on_runs, "runs"),
            self.state.param.watch(self._watch_store, "artifacts"),
        ]
        self._watch_store()
        self._trace._on_refresh = self._refresh_header
        self._bind()

    def __panel__(self):
        return self._layout

    def release(self):
        """
        Stop following the state, e.g. once the view is closed for good. A
        view that is not released keeps updating for as long as the state
        lives.
        """
        if self._released:
            return
        self._released = True
        for watcher in self._state_watchers:
            self.state.param.unwatch(watcher)
        self._state_watchers = []
        if self._store_watcher is not None:
            self._store_watcher[0].param.unwatch(self._store_watcher[1])
            self._store_watcher = None
        self._trace._on_refresh = None
        self._trace.release()

    @property
    def _run(self):
        return self.state.runs.get(self.run_id) if self.run_id else None

    @param.depends("run_id", watch=True)
    def _bind(self):
        if self._released:
            return
        self._trace.run_id = self.run_id
        self._refresh()

    def _on_runs(self, *_):
        self._refresh()

    def _watch_store(self, *_):
        if self._store_watcher is not None:
            self._store_watcher[0].param.unwatch(self._store_watcher[1])
        store = self.state.artifacts
        self._store_watcher = (store, store.param.watch(self._refresh_published, "artifacts"))
        self._refresh_published()

    def _refresh(self, *_):
        self._refresh_header()
        self._refresh_published()

    def _tree(self) -> list[Any]:
        return [r for r, _ in self._trace._watched_runs]

    def _refresh_header(self):
        if self._released:
            return
        run = self._run
        self._missing.visible = run is None
        self._body.visible = run is not None
        if run is None:
            self._header.object = ""
            return
        items = self._trace._own_items
        runs = self._tree() or [run]
        errors = _span_errors(
            runs, [i for i in items if isinstance(i, ModelCallState)], [i for i in items if isinstance(i, ToolCallState)],
        )
        question = run.input if isinstance(run.input, str) and run.input.strip() else next((
            i.content for i in items if isinstance(i, MessageState) and i.role == "user" and i.content
        ), None)
        self._header.object = self._header_html(run, runs, question, errors)
        self._unrecorded.visible = not self._trace._step_count and run.status != "running"
        # Errors a step already shows: failed requests, retries and tool calls, and the error message of a top-level run.
        # The run's own failure is a step and its cancellation is in the header.
        unshown = [e for e in errors if e.kind == "guardrail" or (e.kind in ("run", "cancelled") and e.item is not run)]
        lines = []
        for e in unshown:
            if e.kind == "guardrail":
                message = e.message if e.message.lower().startswith("blocked") else f"Blocked: {e.message}"
                text = "A guardrail blocked a step." if self.redacted else message
            else:
                name = e.item.agent_name or e.item.run_id
                verb = "was cancelled" if e.kind == "cancelled" else "failed"
                text = f"Sub-run {name} {verb}." if self.redacted else f"Sub-run {name} {verb}: {e.message}"
            lines.append(_escape_markdown(text))
        self._problems.param.update(object="\n\n".join(lines), visible=bool(lines))

    def _refresh_published(self, *_):
        if self._released or self._run is None:
            return
        published = self.state._published_in(self.state._run_tree(self.run_id))
        items = [self._published_item(artifact, version) for artifact, version in published]
        self._published.items = items
        self._published.visible = self._published_title.visible = bool(items)

    def _header_html(self, run, runs, question, errors) -> str:
        if self.redacted:
            question_html = '<div class="sv-question muted">The question is hidden.</div>'
        elif question:
            question_html = f'<div class="sv-question">{escape(question)}</div>'
        else:
            question_html = '<div class="sv-question muted">No question recorded.</div>'
        meta = []
        if run.agent_name:
            meta.append(f"<span>{escape(run.agent_name)}</span>")
        status = {"running": "Running", "finished": "Finished", "error": "Failed", "cancelled": "Cancelled"}[run.status]
        if run.status == "cancelled" and run.error and not self.redacted:
            status = f"Cancelled: {run.error}"
        meta.append(f'<span class="sv-status {run.status}">{escape(status)}</span>')
        if when := _time(run.started_at):
            meta.append(f"<span>{escape(when)}</span>")
        if run.duration is not None:
            meta.append(f"<span>{escape(_format_duration(run.duration))}</span>")
        totals = _RunTreeTotals(tuple(runs))
        if totals.cached_share:
            meta.append(f"<span>{totals.cached_share:.0%} of input from cache</span>")
        counts: dict[str, int] = {}
        for error in errors:
            # The run's own failure or cancellation is its status.
            if error.item is not run:
                counts[error.kind] = counts.get(error.kind, 0) + 1
        problems = [f"{n} {_PROBLEMS[kind][n != 1]}" for kind, n in counts.items() if kind in _PROBLEMS]
        if problems:
            meta.append(f'<span class="sv-problems">{escape(", ".join(problems))}</span>')
        return f'<div class="sv-head">{question_html}<div class="sv-meta">{"".join(meta)}</div></div>'

    def _published_item(self, artifact, version) -> dict[str, Any]:
        label = version_label(version)
        title = artifact.title or artifact.artifact_id
        secondary = [f"v{version.version}"]
        if artifact.applied == version.version:
            secondary.append("applied")
        if version.dropped:
            secondary.append("dropped")
        if not self.redacted and label != f"Version {version.version}":
            secondary.append(label)
        item = {
            "id": f"{artifact.artifact_id}@{version.version}", "label": title, "secondary": " · ".join(secondary),
            "icon": _ARTIFACT_ICONS.get(artifact.kind, "description"), "artifact_id": artifact.artifact_id,
            "version": version.version,
        }
        if not self.redacted and (detail := version_detail(version)):
            item["tooltip"] = detail
        return item

    def _on_published_click(self, item: dict[str, Any]):
        artifact_id, version = item.get("artifact_id"), item.get("version")
        if artifact_id is not None and self.on_artifact is not None:
            self.on_artifact(artifact_id, version)


__all__ = ["SpanView"]
