"""
Span view of one agent run: its model requests with the tool calls each
one issued, on a shared time axis, under the run's usage totals.
"""

from __future__ import annotations

import time
from html import escape
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..security import _escape_markdown, untrusted_markdown
from ..state import MessageState, ModelCallState, RunState, ToolCallState
from .model_call import ModelCallCard, format_cost, format_tokens
from .tool_call import ToolCallCard, _format_duration

_TOTALS_CSS = """
:host { width: 100%; }
.rt-totals {
  display: flex; flex-wrap: wrap; gap: 1px; overflow: hidden;
  border: 1px solid var(--mui-palette-divider, rgba(127, 127, 127, 0.3));
  border-radius: var(--mui-shape-borderRadius, 4px);
  background: var(--mui-palette-divider, rgba(127, 127, 127, 0.3));
}
.rt-stat {
  flex: 1 1 120px; min-width: 0; padding: 8px 12px;
  background: var(--mui-palette-background-paper, #fff);
  font-family: var(--mui-font-family, inherit);
}
.rt-label, .rt-sub {
  font-size: 0.75rem; line-height: 1.5; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
.rt-value {
  font-size: 1.25rem; line-height: 1.4; font-variant-numeric: tabular-nums;
  color: var(--mui-palette-text-primary, inherit);
}
"""

_AXIS_CSS = """
:host { width: 100%; }
.rt-axis {
  display: flex; align-items: baseline; gap: 12px; margin-top: 4px;
  font-family: var(--mui-font-family, inherit); color: var(--mui-palette-text-primary, inherit);
}
.rt-question {
  flex: 1 1 auto; min-width: 0; font-size: 0.9375rem; font-weight: 500; line-height: 1.5;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.rt-ticks {
  flex: 0 0 24%; min-width: 56px; display: flex; justify-content: space-between;
  font-size: 0.75rem; font-variant-numeric: tabular-nums;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
"""

_BAR_CSS = """
:host { flex: 0 0 24%; min-width: 56px; align-self: flex-start; margin: 0 !important; }
.rt-track {
  position: relative; height: 10px; margin-top: 19px; border-radius: 2px;
  background: rgba(var(--mui-palette-text-primaryChannel, 127 127 127) / 0.06);
}
.rt-bar {
  position: absolute; top: 0; bottom: 0; min-width: 2px; border-radius: 2px;
  background: var(--mui-palette-primary-main, #1976d2);
}
.rt-bar.tool { background: var(--mui-palette-success-main, #2e7d32); opacity: 0.75; }
.rt-bar.failed { background: var(--mui-palette-error-main, #d32f2f); }
"""

_TEXT_CSS = """
:host { width: 100%; }
.markdown { font-size: 0.875rem; }
"""


def _stat(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="rt-sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="rt-stat"><div class="rt-label">{escape(label)}</div>'
        f'<div class="rt-value">{escape(value)}</div>{sub_html}</div>'
    )


class RunTrace(Viewer):
    """
    Every step of one run, unfolded: each model request with the tool
    calls it issued nested under it, a bar per step on the run's time
    axis, and the run's totals (model requests, input tokens with those
    read from the cache, output tokens, cost and wall time) on top.

    Expanding a model request shows its details. The view is readonly
    and follows the run while it is in progress, so a host can show the
    provenance of any run by its ``run_id``, e.g. in a dialog.
    """

    state = param.ClassSelector(class_=RunState, constant=True)

    run_id = param.String(default=None, allow_None=True, doc="Id of the run to show.")

    show_totals = param.Boolean(default=True, doc="Whether to show the run's usage totals above the steps.")

    def __init__(self, state: RunState | None = None, **params):
        if state is not None:
            params["state"] = state
        super().__init__(**params)
        self._totals = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 0, 8, 0), stylesheets=[_TOTALS_CSS])
        self._axis = pn.pane.HTML(sizing_mode="stretch_width", margin=0, stylesheets=[_AXIS_CSS])
        self._rows = pn.Column(sizing_mode="stretch_width", margin=0)
        self._layout = pn.Column(self._totals, self._axis, self._rows, sizing_mode="stretch_width", css_classes=["run-trace"])
        self._cache: dict[int, tuple[Any, pn.pane.HTML | None]] = {}
        self._cards: dict[int, Any] = {}
        self._shown: list[Any] = []
        self._watchers: list[tuple[param.Parameterized, Any]] = []
        self.state.param.watch(self._update_rows, "items")
        self.state.param.watch(self._watch_run, "runs")
        self._watch_run()

    def __panel__(self):
        return self._layout

    @property
    def run(self):
        return self.state.runs.get(self.run_id) if self.run_id else None

    def _unwatch(self):
        for obj, watcher in self._watchers:
            obj.param.unwatch(watcher)
        self._watchers = []

    @param.depends("run_id", watch=True)
    def _watch_run(self, *_):
        self._unwatch()
        self._cache, self._cards, self._shown = {}, {}, []
        self._rows.objects = []
        if (run := self.run) is not None:
            self._watchers.append((run, run.param.watch(self._refresh, ["status", "ended_at", "usage", "input"])))
        self._update_rows()

    def _items(self) -> list[Any]:
        return [i for i in self.state.items if i.run_id == self.run_id] if self.run_id else []

    def _nested(self) -> list[tuple[Any, int]]:
        """The run's items with their depth: tool calls sit under the model request that issued them."""
        items = self._items()
        issued = {t: m for m in items if isinstance(m, ModelCallState) for t in m.tool_call_ids}
        children: dict[int, list[ToolCallState]] = {}
        order: list[Any] = []
        current = None
        for item in items:
            if isinstance(item, ModelCallState):
                current = item
                order.append(item)
            elif isinstance(item, ToolCallState):
                parent = issued.get(item.tool_call_id, current)
                if parent is None:
                    order.append(item)
                else:
                    children.setdefault(id(parent), []).append(item)
            elif isinstance(item, MessageState) and item.role != "user" and item.kind in ("text", "error"):
                order.append(item)
        rows = []
        for item in order:
            rows.append((item, 0))
            rows += [(child, 1) for child in children.get(id(item), [])]
        return rows

    def _row(self, item: Any, depth: int) -> tuple[Any, pn.pane.HTML | None]:
        if isinstance(item, MessageState):
            if item.kind == "error":
                view = pmui.Alert(
                    object=_escape_markdown(item.content), severity="error", variant="outlined",
                    sizing_mode="stretch_width", margin=(4, 0),
                )
            else:
                view = untrusted_markdown(item.content, sizing_mode="stretch_width", margin=(4, 0), stylesheets=[_TEXT_CSS])
                self._watchers.append((item, item.param.watch(lambda e, p=view: setattr(p, "object", e.new), "content")))
            return view, None
        if isinstance(item, ModelCallState):
            card = ModelCallCard(item, state=self.state, show_tool_count=True)
        else:
            card = ToolCallCard(item, state=self.state)
        self._cards[id(item)] = card
        self._watchers.append((item, item.param.watch(self._refresh, ["status", "ended_at"])))
        bar = pn.pane.HTML(margin=0, stylesheets=[_BAR_CSS], css_classes=["rt-timing"])
        # Only the card is indented, so the bars of every row share one time axis.
        body = pn.Column(card, sizing_mode="stretch_width", margin=(0, 0, 0, 24 * depth))
        return pn.Row(body, bar, sizing_mode="stretch_width", margin=0, styles={"gap": "12px"}), bar

    def _update_rows(self, *_):
        rows = self._nested()
        if [r for r, _ in rows] != self._shown:
            views = []
            for item, depth in rows:
                key = id(item)
                if key not in self._cache:
                    self._cache[key] = self._row(item, depth)
                views.append(self._cache[key][0])
            self._shown = [r for r, _ in rows]
            self._rows.objects = views
        self._refresh()

    def _window(self) -> tuple[float, float]:
        run = self.run
        items = [i for i in self._shown if isinstance(i, (ModelCallState, ToolCallState))]
        starts = [i.started_at for i in items if i.started_at is not None]
        ends = [i.ended_at for i in items if i.ended_at is not None]
        start = run.started_at if run is not None and run.started_at is not None else min(starts, default=0.0)
        if run is not None and run.ended_at is not None:
            end = run.ended_at
        elif run is not None and run.status == "running":
            end = time.time()
        else:
            end = max(ends, default=start)
        return start, max(end, start + 1e-6)

    def _refresh(self, *_):
        run = self.run
        start, end = self._window()
        span = end - start
        for item in self._shown:
            bar = self._cache.get(id(item), (None, None))[1]
            if bar is None:
                continue
            begin = item.started_at if item.started_at is not None else start
            finish = item.ended_at if item.ended_at is not None else (begin + item.duration if item.duration else end)
            if isinstance(item, ModelCallState) and item.duration is not None and item.ended_at is not None:
                begin = item.ended_at - item.duration
            left = max(0.0, min(1.0, (begin - start) / span))
            width = max(0.0, min(1.0 - left, (finish - begin) / span))
            kind = "tool" if isinstance(item, ToolCallState) else "model"
            if item.status in ("failed", "denied"):
                kind += " failed"
            label = escape(_format_duration(finish - begin), quote=True)
            bar.object = (
                f'<div class="rt-track" title="{label}">'
                f'<div class="rt-bar {kind}" style="left:{left * 100:.2f}%;width:{width * 100:.2f}%"></div></div>'
            )
        self._totals.visible = self.show_totals and run is not None
        if run is not None:
            self._totals.object = self._totals_html(run)
        self._axis.object = self._axis_html(run, span)

    @param.depends("show_totals", watch=True)
    def _toggle_totals(self):
        self._refresh()

    def _totals_html(self, run) -> str:
        models = ", ".join(run.models)
        cached = f"{format_tokens(run.cached_tokens)} from cache" if run.cached_tokens else ""
        model_time = f"{_format_duration(run.model_duration)} in the model" if run.model_duration is not None else ""
        wall = run.duration if run.duration is not None else None
        return '<div class="rt-totals">' + "".join([
            _stat("Model requests", str(run.requests), models),
            _stat("Input tokens", format_tokens(run.input_tokens), cached),
            _stat("Output tokens", format_tokens(run.output_tokens)),
            _stat("Cost", format_cost(run.cost_usd) or "n/a"),
            _stat("Wall time", _format_duration(wall) if wall is not None else "running", model_time),
        ]) + "</div>"

    def _axis_html(self, run, span: float) -> str:
        question = ""
        if run is not None and isinstance(run.input, str):
            question = run.input
        else:
            users = [i for i in self.state.items if isinstance(i, MessageState) and i.role == "user" and i.run_id == self.run_id]
            question = users[0].content if users else ""
        ticks = "".join(f"<span>{_format_duration(span * f) if f else '0s'}</span>" for f in (0, 0.5, 1))
        return (
            f'<div class="rt-axis"><span class="rt-question" title="{escape(question, quote=True)}">{escape(question)}</span>'
            f'<span class="rt-ticks">{ticks}</span></div>'
        )


__all__ = ["RunTrace"]
