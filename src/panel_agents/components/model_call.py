"""
Card showing one model request: the model, tokens, cost and latency,
with the details of the response in its body.
"""

from __future__ import annotations

import datetime as dt
from html import escape

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..security import _escape_markdown
from ..state import ModelCallState, RunInfo, RunState
from .tool_call import _HEADER_CSS, _format_duration

_STEP_STATUS = {"running": "running", "success": "success", "failed": "failed"}

_MODEL_HEADER_CSS = _HEADER_CSS + """
:host { container-type: inline-size; }
.mc-model { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mc-meta {
  display: flex; align-items: center; gap: 0.6em; flex: none;
  font-size: 0.75rem; font-variant-numeric: tabular-nums; white-space: nowrap;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
.mc-meta b { font-weight: 500; color: var(--mui-palette-text-primary, inherit); }
.mc-tools {
  padding: 0 0.55em; border-radius: 999px; font-size: 0.7rem; font-weight: 500; line-height: 1.6;
  color: var(--mui-palette-primary-main, #1976d2);
  background: rgba(var(--mui-palette-primary-mainChannel, 25 118 210) / 0.1);
}
@container (max-width: 420px) { .mc-tools { display: none; } }
@container (max-width: 360px) { .mc-cost { display: none; } }
@container (max-width: 320px) { .mc-tokens { display: none; } }
"""

_DETAILS_CSS = """
:host { width: 100%; }
dl {
  display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 0.2em 1.2em; margin: 0.25em 0;
  font-size: 0.8rem; line-height: 1.5;
}
dt { color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); }
dd {
  margin: 0; min-width: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.75rem;
}
"""

USAGE_CSS = """
.usage {
  font-size: 0.75rem; line-height: 1.5; font-variant-numeric: tabular-nums;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
"""


def format_tokens(n: int, compact: bool = False) -> str:
    if compact and n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if compact and n >= 100_000:
        return f"{n / 1000:.0f}k"
    if compact and n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def format_cost(cost: float | None) -> str:
    if cost is None:
        return ""
    if cost == 0:
        return "$0"
    return f"${cost:.4f}" if cost < 0.1 else f"${cost:.2f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def usage_summary(run: RunInfo) -> str:
    """One line of run totals: requests, tokens, cost and wall time."""
    parts = [_plural(run.requests, "request")]
    if run.input_tokens or run.output_tokens:
        parts.append(f"{format_tokens(run.input_tokens, True)} in")
        parts.append(f"{format_tokens(run.output_tokens, True)} out")
    if run.cost_usd is not None:
        parts.append(format_cost(run.cost_usd))
    if run.duration is not None:
        # "total" tells the run's wall time apart from a step group's duration, which leaves out the answer.
        parts.append(f"{_format_duration(run.duration)} total")
    return " · ".join(parts)


def usage_tooltip(run: RunInfo) -> str:
    lines = [f"{run.input_tokens:,} input tokens" + (f", {run.cached_tokens:,} from the cache" if run.cached_tokens else "")]
    lines.append(f"{run.output_tokens:,} output tokens")
    if run.model_duration is not None:
        lines.append(f"{_format_duration(run.model_duration)} waiting for the model")
    if run.models:
        lines.append(", ".join(run.models))
    return "\n".join(lines)


class ModelCallCard(Viewer):
    """
    Collapsible card for one model request, bound to a ``ModelCallState``.

    The header shows the model, the input and output tokens, the cost and
    the latency; the body lists the cached tokens, finish reason, response,
    request and run ids, the tool calls the response issued and any error.
    """

    call = param.ClassSelector(class_=ModelCallState, constant=True)

    state = param.ClassSelector(class_=RunState, default=None, allow_None=True, constant=True, doc="""
        The RunState the call belongs to, used to name the tool calls it issued.""")

    collapsed = param.Boolean(default=True, doc="Whether the card body is collapsed.")

    show_tool_count = param.Boolean(default=False, doc="""
        Whether the header counts the tool calls the response issued.""")

    def __init__(self, call: ModelCallState | None = None, **params):
        if call is not None:
            params["call"] = call
        super().__init__(**params)
        self._header = pn.pane.HTML(sizing_mode="stretch_width", margin=0, stylesheets=[_MODEL_HEADER_CSS])
        self._details = pn.pane.HTML(sizing_mode="stretch_width", margin=0, stylesheets=[_DETAILS_CSS])
        self._error = pmui.Alert(
            object="", severity="error", variant="outlined", sizing_mode="stretch_width", visible=False, margin=(5, 0),
        )
        self._step = pmui.ChatStep(
            self._details, self._error, status=_STEP_STATUS[self.call.status], collapsed=self.collapsed,
            collapsed_on_success=False, sizing_mode="stretch_width", margin=(4, 0), css_classes=["model-call-card"],
        )
        self._step.header.objects = [self._header]
        self._step.param.watch(lambda e: setattr(self, "collapsed", e.new), "collapsed")
        self.call.param.watch(self._update, list(ModelCallState.param))
        self._update()

    def __panel__(self):
        return self._step

    @param.depends("collapsed", watch=True)
    def _sync_collapsed(self):
        self._step.collapsed = self.collapsed

    @param.depends("show_tool_count", watch=True)
    def _update(self, *_):
        call = self.call
        self._step.status = _STEP_STATUS[call.status]
        model = escape(call.model or "Model")
        meta = []
        if call.status == "failed":
            meta.append('<span class="tc-badge error">Failed</span>')
        elif call.attempts:
            n = len(call.attempts)
            meta.append(f'<span class="tc-badge warning">{n} {"retry" if n == 1 else "retries"}</span>')
        if self.show_tool_count and call.tool_call_ids:
            meta.append(f'<span class="mc-tools">{_plural(len(call.tool_call_ids), "tool call")}</span>')
        if call.status == "success" or call.input_tokens or call.output_tokens:
            meta.append(
                f'<span class="mc-tokens"><b>{format_tokens(call.input_tokens)}</b> in · '
                f'<b>{format_tokens(call.output_tokens)}</b> out</span>'
            )
        if call.cost_usd is not None:
            meta.append(f'<span class="mc-cost">{format_cost(call.cost_usd)}</span>')
        if call.duration is not None and call.status != "running":
            meta.append(f'<span>{_format_duration(call.duration)}</span>')
        tooltip = escape(self._tooltip(), quote=True)
        self._header.object = (
            f'<div class="tc-head" title="{tooltip}"><span class="mc-model">{model}</span>'
            f'<span class="mc-meta">{"".join(meta)}</span></div>'
        )
        self._details.object = self._details_html()
        self._error.param.update(object=_escape_markdown(call.error or ""), visible=bool(call.error))

    def _tooltip(self) -> str:
        call = self.call
        lines = [call.model or "Model"]
        if call.cached_tokens:
            lines.append(f"{call.cached_tokens:,} of {call.input_tokens:,} input tokens from the cache")
        if call.error:
            lines.append(call.error)
        lines += [f"Attempt {i} failed: {a.error}" for i, a in enumerate(call.attempts, start=1)]
        return "\n".join(lines)

    def _tool_names(self) -> list[str]:
        calls = self.state.tool_calls if self.state is not None else {}
        return [f"{calls[i].tool_name} · {i}" if i in calls else i for i in self.call.tool_call_ids]

    def _details_html(self) -> str:
        call = self.call
        rows = [("model", call.model)]
        if call.started_at is not None:
            rows.append(("started", dt.datetime.fromtimestamp(call.started_at).strftime("%H:%M:%S.%f")[:-4]))
        if call.duration is not None:
            rows.append(("latency", _format_duration(call.duration)))
        if call.status != "running":
            cached = f", {call.cached_tokens:,} cached" if call.cached_tokens else ""
            rows += [("input", f"{call.input_tokens:,} tokens{cached}"), ("output", f"{call.output_tokens:,} tokens")]
        rows += [
            ("cost", format_cost(call.cost_usd)), ("finish", call.finish_reason),
            ("tool calls", ", ".join(self._tool_names())),
            ("response id", call.provider_response_id), ("request id", call.request_id), ("run id", call.run_id),
        ]
        rows += [
            (f"attempt {i}", f"{a.error} ({_format_duration(a.duration)})" if a.duration is not None else a.error)
            for i, a in enumerate(call.attempts, start=1)
        ]
        items = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in rows if v)
        return f"<dl>{items}</dl>"


__all__ = ["ModelCallCard"]
