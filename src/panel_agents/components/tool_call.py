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

from .._labels import _lower, approval_labels
from ..security import _escape_markdown
from ..state import ApprovalRequest, RunState, ToolCallState
from .approval import ApprovalCard
from .renderers import _code_fence, _dump_json, _output_markdown, output_format, render_output

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
.tc-label-text {
  flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tc-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
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

_ARGS_CSS = """
:host p {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.75rem;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); margin: 0.5em 0 2px;
}
:host pre:has(> code > .codehilite) { margin-top: 0; }
:host .codehilite pre { max-height: 320px; overflow: auto; }
"""

_ERROR_CSS = """
:host { width: 100%; min-width: 0; }
div { box-sizing: border-box; width: 100%; min-width: 0; max-width: 100%; }
pre {
  margin: 0; max-height: 320px; overflow: auto; white-space: pre;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.8rem; line-height: 1.45;
}
"""

_LABEL_CSS = """
:host { width: 100%; }
.tc-label {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9)); margin-top: 0.4em;
}
.tc-tool { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.8rem; }
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
    """
    Markdown listing the arguments: one JSON block for the short ones and a
    labelled block per multi-line string, so code passed as an argument stays readable.
    """
    if not args:
        return _code_fence(args_text, "json") if args_text else ""
    multiline = {key for key, value in args.items() if isinstance(value, str) and "\n" in value.strip()}
    blocks = []
    if inline := {key: value for key, value in args.items() if key not in multiline}:
        blocks.append(_code_fence(_dump_json(inline), "json"))
    for key in (key for key in args if key in multiline):
        fmt, text = output_format(args[key])
        blocks += [_escape_markdown(key), _code_fence(text, "" if fmt == "text" else fmt)]
    return "\n\n".join(blocks)


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

    max_lines = param.Integer(default=30, bounds=(1, None), doc="""
        Number of lines of a code or JSON result shown before a
        "Show all" button expands it.""")

    collapsed = param.Boolean(default=True, doc="Whether the card body is collapsed.")

    label = param.Parameter(default=None, doc="""
        What the call does in the user's words, e.g. "Read the dashboard",
        shown in the header instead of the tool name and arguments, which
        move into the body. A callable is called with the call's arguments;
        if it raises or returns anything but a non-empty string the header
        shows the tool name and arguments as without a label.""")

    defer_result = param.Boolean(default=False, constant=True, doc="""
        Whether the result is rendered only once the card is opened, so a
        long list of collapsed cards stays cheap.""")

    redacted = param.Boolean(default=False, constant=True, doc="""
        Whether to show which tool ran, its status and duration without
        its arguments, result or error message, e.g. for a viewer who may
        not see the data. A callable ``label`` is not used, as it reads
        the arguments.""")

    def __init__(self, call: ToolCallState | None = None, **params):
        if call is not None:
            params["call"] = call
        super().__init__(**params)
        self._header = pn.pane.HTML(sizing_mode="stretch_width", margin=0, stylesheets=[_HEADER_CSS])
        self._args = _output_markdown(margin=(2, 0), stylesheets=[_ARGS_CSS], css_classes=["tc-args"])
        self._result = pn.Column(sizing_mode="stretch_width", margin=0)
        self._hint = pn.pane.HTML(margin=0, stylesheets=[_LABEL_CSS])
        # Visibility is toggled on layouts only: Panel's HTML view errors if hidden before it renders.
        self._tool = pn.pane.HTML(margin=0, stylesheets=[_LABEL_CSS])
        self._tool_section = pn.Column(self._tool, sizing_mode="stretch_width", margin=0, visible=False)
        self._args_section = pn.Column(
            self._label("Arguments"), self._args, sizing_mode="stretch_width", margin=0, visible=False,
        )
        self._result_section = pn.Column(
            self._label("Result"), self._result, self._hint, sizing_mode="stretch_width", margin=0, visible=False,
        )
        self._error_text = pn.pane.HTML(margin=0, sizing_mode="stretch_width", stylesheets=[_ERROR_CSS])
        self._error = pmui.Alert(
            self._error_text, severity="error", variant="outlined", sizing_mode="stretch_width", visible=False, margin=(5, 0),
        )
        self._approval = pn.Column(sizing_mode="stretch_width", margin=0, visible=False)
        self._artifacts = pmui.Row(sizing_mode="stretch_width", margin=(5, 0), visible=False)
        self._step = pmui.ChatStep(
            self._tool_section, self._approval, self._args_section, self._result_section, self._error, self._artifacts,
            status=_STEP_STATUS[self.call.status], collapsed=self.collapsed, collapsed_on_success=False,
            sizing_mode="stretch_width", margin=(4, 0), css_classes=["tool-call-card"],
        )
        self._step.header.objects = [self._header]
        self._step.param.watch(lambda e: setattr(self, "collapsed", e.new), "collapsed")
        self._approval_watcher = None
        self._rendered_result = None
        self._deferred = False
        self._unwatch_artifacts: Callable[[], None] | None = None
        self._state_watcher = None
        self._title_watchers: list = []
        self._chips: dict[str, pmui.Chip] = {}

        self._call_watchers = [
            self.call.param.watch(self._update_header, ["tool_name", "args", "status", "ended_at", "approval"]),
            self.call.param.watch(self._update_args, ["args", "args_text"]),
            self.call.param.watch(self._update_status, "status"),
            self.call.param.watch(self._update_result, ["result", "error", "status"]),
            self.call.param.watch(self._watch_approval, "approval"),
        ]
        if self.state is not None:
            self._state_watcher = self.state.param.watch(self._watch_artifacts, "artifacts")
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
        if not self.collapsed and self._deferred:
            self._update_result()

    @param.depends("approval_factory", watch=True)
    def _refresh_approval(self):
        self._update_approval()

    def resolved_label(self) -> str | None:
        """The label of the call, or None when there is none or it could not be made."""
        label = self.label
        if callable(label) and self.redacted:
            return None
        if callable(label):
            try:
                label = label(dict(self.call.args))
            except Exception:
                return None
        return label if isinstance(label, str) and label.strip() else None

    @param.depends("label", watch=True)
    def _update_header(self, *_):
        call = self.call
        label = self.resolved_label()
        name = escape(call.tool_name)
        if label is not None:
            # The tool stays in the accessible name and in the body, so a label cannot hide what ran.
            parts = [f'<span class="tc-label-text">{escape(label)}<span class="tc-sr"> ({name})</span></span>']
            self._tool.object = f'<div class="tc-label">Tool</div><div class="tc-tool">{name}</div>'
        else:
            parts = [f'<span class="tc-name">{name}</span>']
            preview = "" if self.redacted else _args_preview(call.args)
            parts.append(f'<span class="tc-preview" title="{escape(preview)}">{escape(preview)}</span>')
        self._tool_section.visible = label is not None
        meta = []
        if badge := _STATUS_BADGES.get(call.status):
            label, kind = badge
            if call.status == "denied":
                label = escape(self._approval_labels()["denied"])
            meta.append(f'<span class="tc-badge {kind}">{label}</span>')
        elif call.approval is not None and call.approval.status == "approved" and call.approval.edited:
            meta.append('<span class="tc-badge warning">Edited</span>')
        if call.duration is not None:
            meta.append(f'<span class="tc-duration">{_format_duration(call.duration)}</span>')
        parts.append(f'<span class="tc-meta">{"".join(meta)}</span>')
        self._header.object = f'<div class="tc-head">{"".join(parts)}</div>'

    def _approval_labels(self) -> dict[str, str]:
        request = self.call.approval
        return approval_labels(request.labels if request is not None else None)

    def _update_args(self, *_):
        text = "" if self.redacted else _format_args(self.call.args, self.call.args_text)
        if text:
            self._args.object = text
        # A pending approval card already lists the arguments, editable.
        self._args_section.visible = bool(text) and not self._awaiting_approval()

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
            denied_label = self._approval_labels()["denied"]
            message = call.error or (f"The call was {_lower(denied_label)}." if denied else "The call failed.")
            if self.redacted:
                message = f"The call was {_lower(denied_label)}." if denied else "The call failed."
            # Denial reasons are prose; errors are often tracebacks or driver messages, kept verbatim.
            self._error_text.object = "" if denied else f"<pre>{escape(message)}</pre>"
            self._error.param.update(
                object=_escape_markdown(message) if denied else "", visible=True,
                severity="warning" if denied else "error", title=denied_label if denied else "Error",
            )
            self._show_result(None)
            return
        self._error.visible = False
        self._show_result(None if self.redacted else call.result)

    def _show_result(self, result):
        if result is not None and result is self._rendered_result and self._result.objects:
            return
        # A result is rendered once the card is opened, so long runs of collapsed cards stay cheap.
        self._deferred = result is not None and self.collapsed and self.defer_result
        if self._deferred:
            self._result.objects = []
            self._result_section.visible = False
            self._rendered_result = None
            return
        self._rendered_result = result
        if result is None:
            self._result.objects = []
            self._result_section.visible = False
            return
        value, hint = _truncate(result, self.max_chars, self.max_items)
        self._result.objects = [render_output(value, max_lines=self.max_lines)]
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
            description = None if self.redacted else request.description
            widget = pmui.Alert(
                object=_escape_markdown(description or "Waiting for a decision before this tool runs."),
                title="Approval required", severity="warning", variant="outlined",
                sizing_mode="stretch_width", margin=(5, 0),
            )
        self._approval.param.update(objects=[widget], visible=True)

    def _watch_artifacts(self, *_):
        if self._unwatch_artifacts is not None:
            self._unwatch_artifacts()
        # Only publishes concerning this call's artifacts notify the card, so publishing stays cheap in long chats.
        self._unwatch_artifacts = self.state.artifacts._watch_call(self.call.tool_call_id, self._update_artifacts)
        self._update_artifacts()

    def published_artifacts(self) -> list[tuple[Any, Any]]:
        """Artifacts published by this call, as (artifact, latest matching version) pairs."""
        if self.state is None:
            return []
        return self.state.artifacts.published_by(self.call.tool_call_id)

    def _release(self):
        """Stop following the call and the state's artifacts, e.g. when the chat showing the card is cleared."""
        for watcher in self._call_watchers:
            self.call.param.unwatch(watcher)
        self._call_watchers = []
        if self._approval_watcher is not None:
            self._approval_watcher[0].param.unwatch(self._approval_watcher[1])
            self._approval_watcher = None
        if self._state_watcher is not None:
            self.state.param.unwatch(self._state_watcher)
            self._state_watcher = None
        if self._unwatch_artifacts is not None:
            self._unwatch_artifacts()
            self._unwatch_artifacts = None
        self._unwatch_titles()

    def _unwatch_titles(self):
        for artifact, watcher in self._title_watchers:
            artifact.param.unwatch(watcher)
        self._title_watchers = []

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
        # Titles can change without a publish, which is the only thing the store tells the card about.
        self._unwatch_titles()
        self._title_watchers = [(a, a.param.watch(self._update_artifacts, "title")) for a, _ in published]
        self._artifacts.param.update(objects=list(chips.values()), visible=bool(chips))

    def _activate(self, artifact_id: str):
        self.state.artifacts.active = artifact_id


_ApprovalFactory = Callable[[ApprovalRequest, RunState], Any]

__all__ = ["ToolCallCard"]
