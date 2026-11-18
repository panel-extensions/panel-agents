"""
Human-in-the-loop approval components.

An adapter emits ``ApprovalRequested`` and waits for the matching
``ApprovalResolved``, which these components produce through
``RunState.resolve_approval``. Everything a resolved card displays is
read from the ``ApprovalRequest``, so replayed sessions render the same.
"""

from __future__ import annotations

import copy
import datetime as dt
import html
import json
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from .._labels import APPROVAL_LABELS, Labels, _is_default, _lower, approval_labels
from ..security import _escape_markdown, _plain_typography, inert_markdown
from ..state import ApprovalRequest, RunState
from ._schema_form import SchemaForm, infer_schema, resolve_schema

_MISSING = object()

_EMPTY_VALUES = (None, "", [], {}, False)

_STYLESHEET = """
:host { width: 100%; }
.pa-grid {
  display: grid;
  grid-template-columns: minmax(80px, max-content) 1fr;
  gap: 6px 16px;
  font-size: 0.875rem;
  line-height: 1.45;
}
.pa-key {
  color: var(--mui-palette-text-secondary, #6b6b6b);
  font-family: var(--mui-font-family-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
  font-size: 0.8rem;
  padding-top: 2px;
}
.pa-val { min-width: 0; color: var(--mui-palette-text-primary, inherit); }
.pa-val code, .pa-val pre, .pa-diff code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8rem;
}
.pa-val pre {
  margin: 0;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--mui-palette-action-hover, rgba(127, 127, 127, 0.08));
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 240px;
  overflow: auto;
}
.pa-val .pa-num { color: var(--mui-palette-info-main, #0277bd); }
.pa-val .pa-bool { color: var(--mui-palette-secondary-main, #7b1fa2); }
.pa-val .pa-null { color: var(--mui-palette-text-disabled, #9e9e9e); font-style: italic; }
.pa-diff {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.85rem;
}
.pa-diff-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px; }
.pa-diff-key {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8rem;
  color: var(--mui-palette-text-secondary, #6b6b6b);
  min-width: 80px;
}
.pa-old {
  text-decoration: line-through;
  color: var(--mui-palette-error-main, #c62828);
  background: rgba(198, 40, 40, 0.08);
  padding: 0 4px;
  border-radius: 4px;
}
.pa-new {
  color: var(--mui-palette-success-dark, #1b5e20);
  background: rgba(46, 125, 50, 0.1);
  padding: 0 4px;
  border-radius: 4px;
}
.pa-arrow { color: var(--mui-palette-text-disabled, #9e9e9e); }
.pa-empty { color: var(--mui-palette-text-disabled, #9e9e9e); font-style: italic; font-size: 0.85rem; }
"""


_TEXT_TOGGLE_SX = {
    "width": "auto",
    "border": 0,
    "borderRadius": "var(--mui-shape-borderRadius, 4px)",
    "padding": "4px 5px",
    "gap": "4px",
    "fontSize": "0.8125rem",
    "fontWeight": 500,
    "lineHeight": 1.75,
    "letterSpacing": "0.02857em",
    "textTransform": "none",
    "color": "primary.main",
    "&.Mui-selected, &.Mui-selected:hover": {"color": "primary.main", "backgroundColor": "action.selected"},
    "&.Mui-disabled": {"border": 0},
}


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _value_html(value: Any) -> str:
    if value is None:
        return '<span class="pa-null">null</span>'
    if isinstance(value, bool):
        return f'<span class="pa-bool">{str(value).lower()}</span>'
    if isinstance(value, (int, float)):
        return f'<span class="pa-num">{value}</span>'
    if isinstance(value, str):
        if "\n" in value or len(value) > 60:
            return f"<pre>{html.escape(value)}</pre>"
        return f"<code>{html.escape(value)}</code>"
    text = json.dumps(value, indent=2, default=str)
    if len(text) <= 60:
        return f"<code>{html.escape(json.dumps(value, default=str))}</code>"
    return f"<pre>{html.escape(text)}</pre>"


def _args_html(args: dict[str, Any]) -> str:
    if not args:
        return '<div class="pa-empty">No arguments</div>'
    rows = "".join(
        f'<div class="pa-key">{html.escape(str(k))}</div><div class="pa-val">{_value_html(v)}</div>'
        for k, v in args.items()
    )
    return f'<div class="pa-grid">{rows}</div>'


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: value}


def _diff_args(original: dict[str, Any], edited: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return ``(path, old, new)`` for each changed leaf; absent values are ``None``-like sentinels."""
    old, new = _flatten(original or {}), _flatten(edited or {})
    changes = []
    for path in [*old, *(p for p in new if p not in old)]:
        a, b = old.get(path, _MISSING), new.get(path, _MISSING)
        if a != b:
            changes.append((path, a, b))
    return changes


def _diff_html(changes: list[tuple[str, Any, Any]]) -> str:
    rows = []
    for path, old, new in changes:
        parts = [f'<span class="pa-diff-key">{html.escape(path)}</span>']
        if old is not _MISSING:
            parts.append(f'<span class="pa-old">{html.escape(_short(old))}</span>')
        if old is not _MISSING and new is not _MISSING:
            parts.append('<span class="pa-arrow">→</span>')
        if new is not _MISSING:
            parts.append(f'<span class="pa-new">{html.escape(_short(new))}</span>')
        else:
            parts.append('<span class="pa-empty">removed</span>')
        rows.append(f'<div class="pa-diff-row">{"".join(parts)}</div>')
    return f'<div class="pa-diff">{"".join(rows)}</div>'


def _prune_args(original: dict[str, Any], edited: dict[str, Any], schema: dict[str, Any] | None, root: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Drop keys the form filled in that the agent never set and whose value
    is the schema default or empty, so they do not count as edits.
    """
    root = root if root is not None else (schema or {})
    resolved, _ = resolve_schema(schema or {}, root)
    props = resolved.get("properties", {})
    pruned = {}
    for key, value in edited.items():
        sub = props.get(key, {})
        sub_resolved, _ = resolve_schema(sub, root)
        if key not in original:
            if value in _EMPTY_VALUES or ("default" in sub_resolved and sub_resolved["default"] == value):
                continue
            if isinstance(value, dict):
                value = _prune_args({}, value, sub, root)
                if not value:
                    continue
        elif isinstance(value, dict) and isinstance(original[key], dict):
            value = _prune_args(original[key], value, sub, root)
        pruned[key] = value
    return pruned


def _current_user() -> str | None:
    try:
        return pn.state.user
    except Exception:
        return None


def _format_time(ts: float | None) -> str:
    if ts is None:
        return ""
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


class ApprovalCard(Viewer):
    """
    Review card for a single tool call awaiting human approval.

    Shows the tool, its description and the proposed arguments. The
    arguments can be edited in a form generated from the tool's JSON
    schema, with a field-level diff of the edits. Approving or denying
    calls ``RunState.resolve_approval``; once resolved the card collapses
    into a compact summary.
    """

    request = param.ClassSelector(class_=ApprovalRequest, doc="The approval request to review.")

    state = param.ClassSelector(class_=RunState, doc="The RunState the request belongs to.")

    can_approve = param.Callable(default=None, allow_None=True, doc="""
        Policy hook called as can_approve(request, user). When it returns
        False the decision buttons are disabled and policy_message is shown.""")

    policy_message = param.String(default=None, allow_None=True, doc="""
        Explanation shown when can_approve rejects the current user.
        Defaults to a sentence using the action labels.""")

    labels = Labels(default=None, allow_None=True, doc="""
        Wording of the actions and outcomes, e.g. {"deny": "Remove",
        "denied": "Removed"} for a plan step. Keys: approve, deny,
        approved, denied, edit, approve_edited, approved_edited,
        approve_all and deny_all. A changed approve or deny needs its
        past tense too; the "with edits" and "all" wording is derived
        from them. Labels on the request, declared with
        ApprovalRule(labels=...), take precedence.""")

    require_reason = param.Selector(default="never", objects=["never", "deny", "edit", "always"], doc="""
        When a reason must be given: never, when denying, when approving
        with edits, or always.""")

    user = param.String(default=None, allow_None=True, doc="""
        User recorded on the decision. Defaults to pn.state.user.""")

    editing = param.Boolean(default=False, doc="Whether the arguments editor is shown.")

    edited_args = param.Dict(default=None, allow_None=True, doc="""
        Arguments as edited in the form, or None when not editing.""")

    error = param.String(default="", doc="Message explaining why the last action was rejected.")

    def __init__(self, request: ApprovalRequest | None = None, state: RunState | None = None, **params):
        if request is not None:
            params["request"] = request
        if state is not None:
            params["state"] = state
        super().__init__(**params)
        self._form: SchemaForm | None = None
        self._args_pane = pn.pane.HTML(_args_html(self.request.args), stylesheets=[_STYLESHEET], sizing_mode="stretch_width", margin=(0, 4))
        self._diff_pane = pn.pane.HTML("", stylesheets=[_STYLESHEET], sizing_mode="stretch_width", margin=(0, 4))
        self._diff_section = pmui.Column(
            pmui.Typography("Changes", variant="overline", color="text.secondary", margin=(4, 4, 0, 4)),
            self._diff_pane, visible=False, sizing_mode="stretch_width", margin=0,
        )
        self._form_holder = pmui.Column(sizing_mode="stretch_width", margin=0, visible=False)
        # A ToggleButton keeps aria-pressed; restyled to read like pmui's small text Button.
        self._edit_toggle = pmui.Toggle(
            icon="edit", icon_size="18px", variant="text", size="small", color="primary", margin=(0, 4),
            sx=_TEXT_TOGGLE_SX,
            visible=bool(self.request.args or (self.request.schema or {}).get("properties")),
        )
        self._edit_toggle.param.watch(lambda e: setattr(self, "editing", e.new), "value")
        self._reason = pmui.TextAreaInput(
            label="Reason", placeholder="Optional", rows=1, auto_grow=True, max_rows=6,
            sizing_mode="stretch_width", margin=(8, 4, 4, 4), size="small",
        )
        self._approve = pmui.Button(
            color="primary", variant="contained", size="small",
            disable_elevation=True, margin=(4, 4), on_click=lambda e: self.approve(),
        )
        self._deny = pmui.Button(
            color="error", variant="text", size="small",
            margin=(4, 4), on_click=lambda e: self.deny(),
        )
        self._policy = pmui.Alert(object="", severity="info", variant="outlined", visible=False, sizing_mode="stretch_width", margin=(4, 4))
        self._error = pmui.Alert(object="", severity="error", visible=False, sizing_mode="stretch_width", margin=(4, 4))
        self._title = _plain_typography(
            self.request.tool_name, variant="subtitle1", margin=(0, 8, 0, 4),
            sx={"fontWeight": 600, "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace"},
        )
        # Wraps the edit toggle below the title when the card is narrow, e.g. in a drawer.
        header = pmui.FlexBox(
            self._title,
            self._edit_toggle,
            flex_wrap="wrap", justify_content="space-between", align_items="center", gap="4px",
            sizing_mode="stretch_width", margin=0,
        )
        body = [header]
        if self.request.description:
            body.append(_plain_typography(self.request.description, variant="body2", color="text.secondary", margin=(0, 4, 8, 4)))
        body += [
            self._args_pane, self._form_holder, self._diff_section, self._reason,
            self._policy, self._error,
            pmui.Row(pn.Spacer(sizing_mode="stretch_width"), self._deny, self._approve, sizing_mode="stretch_width", margin=0),
        ]
        self._pending_view = pmui.Column(*body, sizing_mode="stretch_width", margin=0)
        self._layout = pmui.Paper(
            self._pending_view, variant="outlined", sizing_mode="stretch_width", margin=(6, 0),
            sx={"padding": "12px 14px", "borderLeft": "4px solid", "borderLeftColor": "warning.main", "borderRadius": "8px"},
        )
        self.request.param.watch(self._on_status, "status")
        self._reason.param.watch(lambda e: self._update_controls(), "value_input")
        self._update_controls()
        self._on_status()

    def __panel__(self):
        return self._layout

    @property
    def form(self) -> SchemaForm | None:
        """The arguments form, created the first time editing is enabled."""
        return self._form

    @property
    def resolved_labels(self) -> dict[str, str]:
        """The action labels in effect: defaults, then ``labels``, then the request's own."""
        return approval_labels(self.labels, self.request.labels)

    @property
    def current_user(self) -> str | None:
        return self.user if self.user is not None else _current_user()

    @property
    def permitted(self) -> bool:
        return self._allows(self.request.args)

    def _allows(self, args: dict[str, Any]) -> bool:
        request = self.request
        check = getattr(self.state, "_decision_check", None)
        if args is not request.args and check is not None and check(request, self.current_user, args, True):
            # The runner validates edits with the tool's own validator before the policy sees them.
            return False
        if self.can_approve is None:
            return True
        if args is not request.args:
            request = ApprovalRequest(
                tool_call_id=request.tool_call_id, tool_name=request.tool_name, args=dict(args),
                schema=request.schema, description=request.description, run_id=request.run_id,
                labels=request.labels,
            )
        try:
            return bool(self.can_approve(request, self.current_user))
        except Exception:
            return False

    @property
    def is_edited(self) -> bool:
        return self.editing and self.edited_args is not None and bool(_diff_args(self.request.args, self.edited_args))

    @param.depends("editing", watch=True)
    def _on_editing(self):
        self._edit_toggle.value = self.editing
        if self.editing and self._form is None:
            schema = self.request.schema or infer_schema(self.request.args)
            self._form = SchemaForm(schema=schema, value=copy.deepcopy(self.request.args), live_validation=True)
            self._form.param.watch(lambda e: self._sync_edits(), "value")
            self._form_holder[:] = [self._form]
        elif not self.editing and self._form is not None:
            self._form.value = copy.deepcopy(self.request.args)
        self._args_pane.visible = not self.editing
        self._form_holder.visible = self.editing
        self._sync_edits()

    def _sync_edits(self):
        if self.editing and self._form is not None:
            self.edited_args = _prune_args(self.request.args, self._form.value, self._form.schema)
        else:
            self.edited_args = None
        changes = _diff_args(self.request.args, self.edited_args) if self.edited_args is not None else []
        self._diff_pane.object = _diff_html(changes) if changes else ""
        self._diff_section.visible = bool(changes)
        self._update_controls()

    @param.depends("labels", watch=True)
    def _on_labels(self):
        self._update_controls()
        self._on_status()

    def _policy_text(self, labels: dict[str, str], markdown: bool = False) -> str:
        """The policy message; ``markdown`` escapes the labels for the alert, while errors are escaped as a whole."""
        if self.policy_message is not None:
            return self.policy_message
        verbs = f"{_lower(labels['approve'])} or {_lower(labels['deny'])}"
        return f"You are not permitted to {_escape_markdown(verbs) if markdown else verbs} this tool call."

    @param.depends("require_reason", "can_approve", "user", "policy_message", "error", watch=True)
    def _update_controls(self):
        edited = self.is_edited
        labels = self.resolved_labels
        self._approve.label = labels["approve_edited"] if edited else labels["approve"]
        self._deny.label = labels["deny"]
        self._edit_toggle.param.update(
            label=labels["edit"],
            description="Edit the arguments before approving" if _is_default(labels, "approve") else "Edit the arguments first",
        )
        permitted = self.permitted
        self._approve.disabled = self._deny.disabled = not permitted
        self._edit_toggle.disabled = not permitted
        self._reason.disabled = not permitted
        self._policy.param.update(object=self._policy_text(labels, markdown=True), visible=not permitted)
        self._error.param.update(object=_escape_markdown(self.error), visible=bool(self.error))
        approve_edited = _lower(labels["approve_edited"])
        hint = {
            "never": "",
            "deny": f"Required to {_lower(labels['deny'])}",
            "edit": "Required for edited approvals" if edited and labels["approve"] == "Approve" else f"Required to {approve_edited}",
            "always": "Required",
        }[self.require_reason]
        self._reason.label = "Reason *" if self._reason_required_now() else "Reason"
        if not self._reason.error_state:
            self._reason.helper_text = hint

    def _reason_required_now(self) -> bool:
        return self.require_reason == "always" or (self.require_reason == "edit" and self.is_edited)

    def _reason_text(self) -> str:
        return (self._reason.value_input or self._reason.value or "").strip()

    def _check(self, approved: bool) -> str | None:
        if self.request.status != "pending":
            return "This request has already been resolved."
        if not self.permitted:
            return self._policy_text(self.resolved_labels)
        mode = self.require_reason
        needs_reason = mode == "always" or (mode == "deny" and not approved) or (mode == "edit" and approved and self.is_edited)
        if needs_reason and not self._reason_text():
            self._reason.param.update(error_state=True, helper_text="Please give a reason.")
            return "A reason is required for this decision."
        self._reason.error_state = False
        if approved and self.editing and self._form is not None and self._form.validate():
            return "Fix the highlighted arguments before approving." if _is_default(self.resolved_labels, "approve") else "Fix the highlighted arguments first."
        if approved and self.is_edited and not self._allows(self.edited_args):
            return f"You are not permitted to {_lower(self.resolved_labels['approve'])} the call with these arguments."
        return None

    def approve(self) -> bool:
        """Approve the call, with edits if any. Returns whether it was resolved."""
        return self._resolve(True)

    def deny(self) -> bool:
        """Deny the call. Returns whether it was resolved."""
        return self._resolve(False)

    def _resolve(self, approved: bool) -> bool:
        if error := self._check(approved):
            self.error = error
            return False
        self.error = ""
        args = None
        if approved:
            args = copy.deepcopy(self.edited_args if self.is_edited else self.request.args)
        try:
            self.state.resolve_approval(
                self.request.tool_call_id, approved, args=args,
                reason=self._reason_text() or None, user=self.current_user,
            )
        except (PermissionError, ValueError) as e:
            self.error = str(e)
            return False
        return True

    def _on_status(self, *events):
        request = self.request
        if request.status == "pending":
            return
        approved = request.status == "approved"
        edits = _diff_args(request.args, request.final_args) if approved and request.final_args is not None else []
        labels = self.resolved_labels
        verdict = (labels["approved_edited"] if edits else labels["approved"]) if approved else labels["denied"]
        who = f" by **{_escape_markdown(request.user)}**" if request.user else ""
        when = f" at {_format_time(request.resolved_at)}" if request.resolved_at else ""
        items = [
            pmui.Row(
                pmui.Chip(
                    label=verdict, color="success" if approved else "error", size="small",
                    icon="check_circle" if approved else "cancel", margin=(0, 8, 0, 0),
                ),
                pmui.Typography(f"**{_escape_markdown(request.tool_name)}**{who}{when}", variant="body2", margin=0, plugins=[inert_markdown], disable_math=True),
                sizing_mode="stretch_width", margin=0, align="center",
            )
        ]
        if request.reason:
            items.append(_plain_typography(f"“{request.reason}”", variant="body2", color="text.secondary", margin=(6, 0, 0, 2), sx={"fontStyle": "italic"}))
        if edits:
            items.append(pn.pane.HTML(_diff_html(edits), stylesheets=[_STYLESHEET], sizing_mode="stretch_width", margin=(6, 0, 0, 2)))
        self._layout.objects = [pmui.Column(*items, sizing_mode="stretch_width", margin=0)]
        self._layout.sx = {
            "padding": "8px 12px", "borderLeft": "4px solid", "borderRadius": "8px",
            "borderLeftColor": "success.main" if approved else "error.main",
        }


def approval_factory(request: ApprovalRequest, state: RunState) -> ApprovalCard:
    """Build an ApprovalCard, matching the ``Callable[[ApprovalRequest, RunState], Viewable]`` hook."""
    return ApprovalCard(request=request, state=state)


class ApprovalQueue(Viewer):
    """
    Lists the pending approvals of a RunState as ApprovalCards, with
    approve-all / deny-all actions for batches.
    """

    state = param.ClassSelector(class_=RunState, doc="The RunState whose pending approvals are listed.")

    can_approve = param.Callable(default=None, allow_None=True, doc="Policy hook passed to each card.")

    require_reason = param.Selector(default="never", objects=["never", "deny", "edit", "always"], doc="""
        Reason requirement passed to each card.""")

    user = param.String(default=None, allow_None=True, doc="User recorded on decisions, defaults to pn.state.user.")

    show_resolved = param.Boolean(default=False, doc="Whether resolved requests stay listed as summaries.")

    title = param.String(default="Pending approvals")

    card_factory = param.Callable(default=None, allow_None=True, doc="""
        Optional callable(request, state, **params) returning an ApprovalCard.""")

    labels = Labels(default=None, allow_None=True, doc="""
        Action labels passed to each card, see ApprovalCard.labels. The
        batch buttons use them too, unless every pending request carries
        the same labels of its own.""")

    message = param.String(default="", doc="Outcome of the last batch action.")

    def __init__(self, state: RunState | None = None, **params):
        if state is not None:
            params["state"] = state
        super().__init__(**params)
        self._cards: dict[str, ApprovalCard] = {}
        self._ordered: list[ApprovalCard] = []
        self._count = pmui.Chip(label="0", size="small", color="warning", margin=(0, 8))
        self._approve_all = pmui.Button(
            icon="done_all", color="success", variant="outlined", size="small",
            margin=(0, 4), sx={"whiteSpace": "nowrap"}, on_click=lambda e: self.approve_all(),
        )
        self._deny_all = pmui.Button(
            icon="block", color="error", variant="text", size="small",
            margin=(0, 4), sx={"whiteSpace": "nowrap"}, on_click=lambda e: self.deny_all(),
        )
        self._message = pmui.Alert(object="", severity="warning", visible=False, sizing_mode="stretch_width", margin=(4, 0), closeable=True)
        self._empty = pmui.Typography("Nothing waiting for review.", variant="body2", color="text.secondary", margin=(8, 4))
        self._list = pmui.Column(sizing_mode="stretch_width", margin=0)
        header = pmui.Row(
            pmui.Typography(self.title, variant="h6", margin=(0, 4), sx={"whiteSpace": "nowrap"}), self._count,
            pn.Spacer(sizing_mode="stretch_width"), self._deny_all, self._approve_all,
            sizing_mode="stretch_width", margin=(0, 0, 4, 0), align="center",
        )
        self._layout = pmui.Column(header, self._message, self._empty, self._list, sizing_mode="stretch_width")
        self._refresh()

    def __panel__(self):
        return self._layout

    @property
    def cards(self) -> list[ApprovalCard]:
        return list(self._ordered)

    def _make_card(self, request: ApprovalRequest) -> ApprovalCard:
        params = dict(can_approve=self.can_approve, require_reason=self.require_reason, user=self.user)
        if self.labels is not None:
            params["labels"] = self.labels
        if self.card_factory is not None:
            return self.card_factory(request, self.state, **params)
        return ApprovalCard(request=request, state=self.state, **params)

    @param.depends("state", "state.pending_approvals", "state.approvals", "show_resolved", watch=True)
    def _refresh(self):
        requests = list(self.state.approvals.values()) if self.show_resolved else list(self.state.pending_approvals)
        ids = {r.tool_call_id for r in requests}
        self._cards = {k: c for k, c in self._cards.items() if k in ids and c.request is self.state.approvals.get(k)}
        cards = []
        for request in requests:
            card = self._cards.get(request.tool_call_id)
            if card is None:
                card = self._cards[request.tool_call_id] = self._make_card(request)
            cards.append(card)
        self._ordered = cards
        self._list.objects = cards
        pending = len(self.state.pending_approvals)
        self._count.label = str(pending)
        self._count.visible = pending > 0
        self._approve_all.visible = self._deny_all.visible = pending > 1
        self._empty.visible = not cards
        self._update_batch_labels()

    @property
    def batch_labels(self) -> dict[str, str]:
        """The labels of the batch actions: the queue's, or those all pending requests share."""
        own = [r.labels for r in self.state.pending_approvals]
        shared = own[0] if own and all(labels == own[0] for labels in own) else None
        return approval_labels(self.labels, shared)

    def _update_batch_labels(self):
        labels = self.batch_labels
        self._approve_all.label = labels["approve_all"]
        self._deny_all.label = labels["deny_all"]

    @param.depends("can_approve", "require_reason", "user", "labels", watch=True)
    def _propagate(self):
        for card in self._cards.values():
            params = dict(can_approve=self.can_approve, require_reason=self.require_reason, user=self.user)
            if "labels" in card.param:
                params["labels"] = self.labels
            card.param.update(**params)
        self._update_batch_labels()

    @param.depends("message", watch=True)
    def _show_message(self):
        self._message.param.update(object=_escape_markdown(self.message), visible=bool(self.message), closed=False)

    def _batch(self, approved: bool) -> int:
        labels = self.batch_labels
        pending = [self._cards[r.tool_call_id] for r in list(self.state.pending_approvals) if r.tool_call_id in self._cards]
        done = sum(card.approve() if approved else card.deny() for card in pending)
        skipped = len(pending) - done
        verb = _lower(labels["approved"] if approved else labels["denied"])
        self.message = f"{done} of {len(pending)} {verb}; {skipped} need attention." if skipped else ""
        return done

    def approve_all(self) -> int:
        """Approve every pending request whose card allows it. Returns the number resolved."""
        return self._batch(True)

    def deny_all(self) -> int:
        """Deny every pending request whose card allows it. Returns the number resolved."""
        return self._batch(False)


__all__ = ["APPROVAL_LABELS", "ApprovalCard", "ApprovalQueue", "approval_factory", "approval_labels"]
