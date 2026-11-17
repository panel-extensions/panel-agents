"""
Card showing a single tool call: its arguments, status, result and any
artifacts it published.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from html import escape
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..artifacts.model import infer_kind
from ..security import _escape_markdown
from ..state import ApprovalRequest, RunState, ToolCallState
from .approval import ApprovalCard
from .renderers import render

# ChatStep only knows four statuses; the extra ones are conveyed by a badge.
_STEP_STATUS = {
    "pending": "pending",
    "awaiting_approval": "pending",
    "running": "running",
    "success": "success",
    "failed": "failed",
    "denied": "failed",
}

_STATUS_BADGES = {
    "awaiting_approval": ("Needs approval", "warning"),
    "failed": ("Failed", "error"),
    "denied": ("Denied", "error"),
}

_EXPAND_ON = ("failed", "awaiting_approval")

_ARTIFACT_ICONS = {
    "table": "table_chart",
    "sql": "code",
    "code": "code",
    "python": "code",
    "json": "data_object",
    "text": "article",
    "plot": "insert_chart",
}

_HEADER_CSS = """
:host { width: 100%; min-width: 0; }
.tc-head {
  display: flex; align-items: center; gap: 0.6em; min-width: 0;
  font-size: 0.875rem; line-height: 1.5;
}
.tc-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-weight: 600; white-space: nowrap;
}
.tc-preview {
  flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.8rem;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
.tc-meta { display: flex; align-items: center; gap: 0.5em; flex: none; }
.tc-badge {
  padding: 0 0.55em; border-radius: 999px; font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.02em; white-space: nowrap; line-height: 1.6;
}
.tc-badge.warning {
  color: var(--mui-palette-warning-dark, #b26a00);
  background: rgba(var(--mui-palette-warning-mainChannel, 237 108 2) / 0.14);
}
.tc-badge.error {
  color: var(--mui-palette-error-main, #d32f2f);
  background: rgba(var(--mui-palette-error-mainChannel, 211 47 47) / 0.12);
}
.tc-duration {
  font-size: 0.75rem; font-variant-numeric: tabular-nums;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
}
"""

_CODE_CSS = """
:host { width: 100%; }
.tc-arg { margin-bottom: 0.5em; }
.tc-key {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.75rem;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); margin-bottom: 2px;
}
pre {
  margin: 0; padding: 0.6em 0.8em; border-radius: 6px; max-height: 320px; overflow: auto;
  font-size: 0.8rem; line-height: 1.45; white-space: pre-wrap; word-break: break-word;
  background: rgba(127, 127, 127, 0.08);
}
"""

_LABEL_CSS = """
:host { width: 100%; }
.tc-label {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); margin-top: 0.4em;
}
.tc-hint { font-size: 0.75rem; font-style: italic; color: var(--mui-palette-text-secondary, gray); }
"""


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m {seconds}s"


def _preview_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(" ".join(value.split()))
    try:
        return json.dumps(value, default=repr)
    except (TypeError, ValueError):
        return repr(value)


def _args_preview(args: dict[str, Any], max_length: int = 120) -> str:
    """One-line ``key=value`` summary of tool call arguments."""
    text = ", ".join(f"{key}={_preview_value(value)}" for key, value in args.items())
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def _format_args(args: dict[str, Any], args_text: str = "") -> str:
    """HTML listing each argument, with strings shown verbatim so code stays readable."""
    if not args:
        return f"<pre>{escape(args_text)}</pre>" if args_text else ""
    blocks = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, indent=2, default=repr, ensure_ascii=False)
        blocks.append(f'<div class="tc-arg"><div class="tc-key">{escape(str(key))}</div><pre>{escape(text)}</pre></div>')
    return "".join(blocks)


def _truncate(value: Any, max_chars: int = 20_000, max_items: int = 1_000) -> tuple[Any, str | None]:
    """Shorten large values, returning the value to display and a hint describing what was cut."""
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars], f"Showing the first {max_chars:,} of {len(value):,} characters."
        return value, None
    if isinstance(value, (list, tuple)):
        if len(value) > max_items:
            return value[:max_items], f"Showing the first {max_items:,} of {len(value):,} items."
        return value, None
    if isinstance(value, dict):
        if len(value) > max_items:
            head = dict(list(value.items())[:max_items])
            return head, f"Showing the first {max_items:,} of {len(value):,} keys."
        return value, None
    if hasattr(value, "head") and hasattr(value, "__len__"):
        try:
            n = len(value)
        except TypeError:
            return value, None
        if n > max_items:
            return value.head(max_items), f"Showing the first {max_items:,} of {n:,} rows."
    return value, None


class ToolCallCard(Viewer):
    """
    Collapsible card for one tool call, bound to a ``ToolCallState``.

    The header summarizes the call (tool name, arguments, status and
    duration); the body shows the arguments as they stream, the rendered
    result or error, a pending approval and the artifacts the call
    published. Collapsed by default, it expands when the call fails or
    needs approval.
    """

    call = param.ClassSelector(class_=ToolCallState, constant=True)

    state = param.ClassSelector(class_=RunState, default=None, allow_None=True, constant=True, doc="""
        The RunState the call belongs to, used to find published artifacts.""")

    approval_factory = param.Callable(default=None, doc="""
        Callable ``(ApprovalRequest, RunState) -> Viewable`` producing the
        widget shown while the call awaits approval. When None a short
        notice is shown instead.""")

    max_chars = param.Integer(default=20_000, bounds=(1, None), doc="""
        Maximum number of characters of a string result to display.""")

    max_items = param.Integer(default=1_000, bounds=(1, None), doc="""
        Maximum number of items or rows of a result to display.""")

    collapsed = param.Boolean(default=True, doc="Whether the card body is collapsed.")

    def __init__(self, call: ToolCallState | None = None, **params):
        if call is not None:
            params["call"] = call
        super().__init__(**params)
        self._header = pn.pane.HTML(sizing_mode="stretch_width", margin=0, stylesheets=[_HEADER_CSS])
        self._args = pn.pane.HTML(margin=(2, 0), sizing_mode="stretch_width", stylesheets=[_CODE_CSS], css_classes=["tc-args"])
        self._result = pn.Column(sizing_mode="stretch_width", margin=0)
        self._hint = pn.pane.HTML(margin=0, stylesheets=[_LABEL_CSS])
        # Visibility is toggled on layouts only: Panel's HTML view errors if hidden before it renders.
        self._args_section = pn.Column(
            self._label("Arguments"), self._args, sizing_mode="stretch_width", margin=0, visible=False,
        )
        self._result_section = pn.Column(
            self._label("Result"), self._result, self._hint, sizing_mode="stretch_width", margin=0, visible=False,
        )
        self._error = pmui.Alert(severity="error", variant="outlined", sizing_mode="stretch_width", visible=False, margin=(5, 0))
        self._approval = pn.Column(sizing_mode="stretch_width", margin=0, visible=False)
        self._artifacts = pmui.Row(sizing_mode="stretch_width", margin=(5, 0), visible=False)
        self._step = pmui.ChatStep(
            self._approval, self._args_section, self._result_section, self._error, self._artifacts,
            status=_STEP_STATUS[self.call.status], collapsed=self.collapsed, collapsed_on_success=False,
            sizing_mode="stretch_width", margin=(4, 0), css_classes=["tool-call-card"],
        )
        self._step.header.objects = [self._header]
        self._step.param.watch(lambda e: setattr(self, "collapsed", e.new), "collapsed")
        self._approval_watcher = None
        self._rendered_result = None
        self._artifact_watchers: list = []
        self._chips: dict[str, pmui.Chip] = {}

        self.call.param.watch(self._update_header, ["tool_name", "args", "status", "ended_at", "approval"])
        self.call.param.watch(self._update_args, ["args", "args_text"])
        self.call.param.watch(self._update_status, "status")
        self.call.param.watch(self._update_result, ["result", "error", "status"])
        self.call.param.watch(self._watch_approval, "approval")
        if self.state is not None:
            self.state.param.watch(self._watch_artifacts, "artifacts")
            self._watch_artifacts()

        self._update_header()
        self._update_args()
        self._update_result()
        self._watch_approval()
        if self.call.status in _EXPAND_ON:
            self.collapsed = False

    def __panel__(self):
        return self._step

    @staticmethod
    def _label(text: str) -> pn.pane.HTML:
        return pn.pane.HTML(f'<div class="tc-label">{text}</div>', margin=0, stylesheets=[_LABEL_CSS])

    @param.depends("collapsed", watch=True)
    def _sync_collapsed(self):
        self._step.collapsed = self.collapsed

    @param.depends("approval_factory", watch=True)
    def _refresh_approval(self):
        self._update_approval()

    def _update_header(self, *_):
        call = self.call
        parts = [f'<span class="tc-name">{escape(call.tool_name)}</span>']
        preview = _args_preview(call.args)
        parts.append(f'<span class="tc-preview" title="{escape(preview)}">{escape(preview)}</span>')
        meta = []
        if badge := _STATUS_BADGES.get(call.status):
            label, kind = badge
            meta.append(f'<span class="tc-badge {kind}">{label}</span>')
        elif call.approval is not None and call.approval.status == "approved" and call.approval.edited:
            meta.append('<span class="tc-badge warning">Edited</span>')
        if call.duration is not None:
            meta.append(f'<span class="tc-duration">{_format_duration(call.duration)}</span>')
        parts.append(f'<span class="tc-meta">{"".join(meta)}</span>')
        self._header.object = f'<div class="tc-head">{"".join(parts)}</div>'

    def _update_args(self, *_):
        html = _format_args(self.call.args, self.call.args_text)
        if html:
            self._args.object = html
        # A pending approval card already lists the arguments, editable.
        self._args_section.visible = bool(html) and not self._awaiting_approval()

    def _awaiting_approval(self) -> bool:
        request = self.call.approval
        return request is not None and request.status == "pending" and self.approval_factory is not None

    def _update_status(self, *_):
        status = self.call.status
        self._step.status = _STEP_STATUS[status]
        if status in _EXPAND_ON:
            self.collapsed = False

    def _update_result(self, *_):
        call = self.call
        if call.status in ("failed", "denied"):
            denied = call.status == "denied"
            message = call.error or ("The call was denied." if denied else "The call failed.")
            self._error.param.update(
                object=_escape_markdown(message), visible=True, severity="warning" if denied else "error",
                title="Denied" if denied else "Error",
            )
            self._show_result(None)
            return
        self._error.visible = False
        self._show_result(call.result)

    def _show_result(self, result):
        if result is not None and result is self._rendered_result and self._result.objects:
            return
        self._rendered_result = result
        if result is None:
            self._result.objects = []
            self._result_section.visible = False
            return
        value, hint = _truncate(result, self.max_chars, self.max_items)
        self._result.objects = [render(value, kind="dataframe" if infer_kind(value) == "table" else None)]
        self._hint.object = f'<div class="tc-hint">{escape(hint)}</div>' if hint else ""
        self._result_section.visible = True

    def _watch_approval(self, *_):
        if self._approval_watcher is not None:
            self._approval_watcher[0].param.unwatch(self._approval_watcher[1])
            self._approval_watcher = None
        request = self.call.approval
        if request is not None:
            watcher = request.param.watch(self._update_approval, ["status", "final_args"])
            self._approval_watcher = (request, watcher)
        self._update_approval()

    def _update_approval(self, *_):
        request: ApprovalRequest | None = self.call.approval
        self._update_args()
        if request is None or request.status != "pending":
            self._approval.param.update(objects=[], visible=False)
            if request is not None:
                self._update_header()
            return
        if self.approval_factory is not None:
            widget = self.approval_factory(request, self.state)
            if isinstance(widget, ApprovalCard):
                # The card's own header already names the tool.
                widget._title.visible = False
        else:
            widget = pmui.Alert(
                object=_escape_markdown(request.description or "Waiting for a decision before this tool runs."),
                title="Approval required", severity="warning", variant="outlined",
                sizing_mode="stretch_width", margin=(5, 0),
            )
        self._approval.param.update(objects=[widget], visible=True)

    def _watch_artifacts(self, *_):
        for store, watcher in self._artifact_watchers:
            store.param.unwatch(watcher)
        store = self.state.artifacts
        self._artifact_watchers = [(store, store.param.watch(self._update_artifacts, "artifacts"))]
        self._update_artifacts()

    def published_artifacts(self) -> list[tuple[Any, Any]]:
        """Artifacts published by this call, as (artifact, latest matching version) pairs."""
        if self.state is None:
            return []
        call_id = self.call.tool_call_id
        found = []
        for artifact in self.state.artifacts:
            versions = [v for v in artifact.versions if v.tool_call_id == call_id]
            if versions:
                found.append((artifact, versions[-1]))
        return found

    def _update_artifacts(self, *_):
        published = self.published_artifacts()
        chips = {}
        for artifact, version in published:
            label = artifact.title or artifact.artifact_id
            if len(artifact.versions) > 1:
                label = f"{label} · v{version.version}"
            chip = self._chips.get(artifact.artifact_id)
            if chip is None:
                chip = pmui.Chip(
                    label=label, icon=_ARTIFACT_ICONS.get(artifact.kind, "description"),
                    size="small", variant="outlined", color="primary", margin=(0, 5, 0, 0),
                )
                chip.on_click(lambda _, aid=artifact.artifact_id: self._activate(aid))
            else:
                chip.label = label
            chips[artifact.artifact_id] = chip
        self._chips = chips
        self._artifacts.param.update(objects=list(chips.values()), visible=bool(chips))

    def _activate(self, artifact_id: str):
        self.state.artifacts.active = artifact_id


_ApprovalFactory = Callable[[ApprovalRequest, RunState], Any]

__all__ = ["ToolCallCard"]
