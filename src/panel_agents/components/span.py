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
from ..state import RunState
from ._timeline import version_detail, version_label
from .tool_call import _ARTIFACT_ICONS, _format_duration
from .trace import RunTrace

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
    "run": ("run failed", "runs failed"),
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
    run still in progress.

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
        question, the agent's text, tool arguments and results, error
        messages or version details.""")

    row_limit = param.Integer(default=50, allow_None=True, bounds=(1, None), constant=True, doc="""
        Number of steps built when the view opens, with a button building
        the next as many; None builds them all.""")

    on_artifact = param.Callable(default=None, allow_None=True, doc="""
        Called with an artifact id and a version number when a published
        version is clicked. When None, the artifact becomes the state's
        active one.""")

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
            self.state, run_id=self.run_id, redacted=self.redacted, row_limit=self.row_limit, show_question=False,
            include_children=True,
        )
        self._guardrails = pmui.Alert(
            object="", title="Blocked", severity="warning", variant="outlined",
            sizing_mode="stretch_width", visible=False, margin=(4, 0),
        )
        self._published_title = pn.pane.HTML(
            '<div class="sv-section">Published</div>', margin=0, stylesheets=[_SECTION_CSS], visible=False,
        )
        self._published = pmui.MenuList(items=[], dense=True, sizing_mode="stretch_width", margin=0, visible=False)
        self._published.on_click(self._on_published_click)
        self._body = pn.Column(
            self._unrecorded, self._trace, self._guardrails, self._published_title, self._published,
            sizing_mode="stretch_width", margin=0,
        )
        self._layout = pn.Column(
            self._header, self._missing, self._body, sizing_mode="stretch_width", css_classes=["span-view"],
        )
        self._watched: list[tuple[param.Parameterized, Any]] = []
        self._store_watcher = None
        self.state.param.watch(self._on_runs, "runs")
        self.state.param.watch(self._watch_store, "artifacts")
        self._trace._rows.param.watch(self._refresh, "objects")
        self._watch_store()
        self._bind()

    def __panel__(self):
        return self._layout

    @property
    def run(self):
        return self.state.runs.get(self.run_id) if self.run_id else None

    @property
    def trace(self) -> RunTrace:
        return self._trace

    @param.depends("run_id", watch=True)
    def _bind(self):
        self._trace.run_id = self.run_id
        self._watch_runs()
        self._refresh()

    def _on_runs(self, *_):
        self._watch_runs()
        self._refresh()

    def _watch_runs(self):
        runs = [self.state.runs[r] for r in self.state.run_tree(self.run_id)] if self.run is not None else []
        if [obj for obj, _ in self._watched] == runs:
            return
        for obj, watcher in self._watched:
            obj.param.unwatch(watcher)
        self._watched = [(r, r.param.watch(self._refresh, ["status", "ended_at", "input", "guardrails", "error"])) for r in runs]

    def _watch_store(self, *_):
        if self._store_watcher is not None:
            self._store_watcher[0].param.unwatch(self._store_watcher[1])
        store = self.state.artifacts
        self._store_watcher = (store, store.param.watch(self._refresh, "artifacts"))
        self._refresh()

    def _refresh(self, *_):
        run = self.run
        self._missing.visible = run is None
        self._body.visible = run is not None
        if run is None:
            self._header.object = ""
            return
        span = self.state.span(self.run_id)
        self._header.object = self._header_html(span)
        self._unrecorded.visible = not span.steps and run.status != "running"
        blocks = [e for e in span.errors if e.kind == "guardrail"]
        self._guardrails.param.update(
            object="\n\n".join(
                "A guardrail blocked a step." if self.redacted else _escape_markdown(e.message) for e in blocks
            ),
            visible=bool(blocks),
        )
        items = [self._published_item(artifact, version) for artifact, version in span.published]
        self._published.items = items
        self._published.visible = self._published_title.visible = bool(items)

    def _header_html(self, span) -> str:
        run = span.run
        if self.redacted:
            question = '<div class="sv-question muted">The question is hidden.</div>'
        elif span.question:
            question = f'<div class="sv-question">{escape(span.question)}</div>'
        else:
            question = '<div class="sv-question muted">No question recorded.</div>'
        meta = []
        if run.agent_name:
            meta.append(f"<span>{escape(run.agent_name)}</span>")
        status = {"running": "Running", "finished": "Finished", "error": "Failed"}[run.status]
        meta.append(f'<span class="sv-status {run.status}">{status}</span>')
        if when := _time(run.started_at):
            meta.append(f"<span>{escape(when)}</span>")
        if run.duration is not None:
            meta.append(f"<span>{escape(_format_duration(run.duration))}</span>")
        if span.cached_share:
            meta.append(f"<span>{span.cached_share:.0%} of input from cache</span>")
        counts: dict[str, int] = {}
        for error in span.errors:
            counts[error.kind] = counts.get(error.kind, 0) + 1
        problems = [
            f"{n} {_PROBLEMS[kind][n != 1]}" for kind, n in counts.items() if kind in _PROBLEMS and kind != "run"
        ]
        if problems:
            meta.append(f'<span class="sv-problems">{escape(", ".join(problems))}</span>')
        return f'<div class="sv-head">{question}<div class="sv-meta">{"".join(meta)}</div></div>'

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
        if artifact_id is None:
            return
        if self.on_artifact is None:
            self.state.artifacts.active = artifact_id
            return
        self.on_artifact(artifact_id, version)


__all__ = ["SpanView"]
