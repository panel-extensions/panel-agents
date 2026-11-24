"""
An agent panel that docks beside an existing app.
"""

from __future__ import annotations

from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewable, Viewer

from ..state import RunState
from ._toggle import DrawerResizeHandle, DrawerResizeNotifier, DrawerToggle
from .approval import ApprovalQueue
from .artifacts import ArtifactPane
from .chat import AgentChat
from .span import SpanView

_TABS = ("chat", "artifacts", "approvals")


class AgentDrawer(Viewer):
    """
    A collapsible right-hand drawer holding an ``AgentChat``, with views
    of the session's artifacts and pending approvals.

    Given ``main``, the drawer renders beside that content and, in the
    persistent variant, narrows it while open. Place ``toggle`` in the
    app's header to open and close it; its badge counts pending
    approvals. To use the drawer as the contextbar of a ``pmui.Page``
    instead, put ``content`` in the page's ``contextbar``.

    >>> drawer = AgentDrawer(runner, main=dashboard, title="Assistant")
    >>> pmui.Page(header=[drawer.toggle], main=[drawer])
    """

    runner = param.Parameter(default=None, doc="""
        Object with an (async) ``run(text)`` method, such as an
        ``AgentRunner``. Its ``state`` and ``approval_policy`` are used
        unless given explicitly.""")

    state = param.ClassSelector(class_=RunState, constant=True, doc="The RunState to display.")

    main = param.Parameter(default=None, doc="""
        The app content the drawer docks beside. Without it the drawer
        renders on its own, e.g. as the last item of an existing layout.""")

    open = param.Boolean(default=False, doc="Whether the drawer is open.")

    width = param.Integer(default=420, bounds=(240, None), doc="Width of the open drawer in pixels.")

    resizable = param.Boolean(default=True, doc="""
        Whether the drawer's left edge can be dragged to resize it, from
        the smallest ``width`` to 60% of the window. A resize sets
        ``width``, which a host may store to open the drawer at that width
        next time.""")

    title = param.String(default="Assistant", doc="Title shown at the top of the drawer.")

    variant = param.Selector(default="persistent", objects=["persistent", "temporary"], doc="""
        persistent: the drawer sits beside the content and narrows it.
        temporary: the drawer overlays the content and closes on outside clicks.""")

    overlay_below = param.Integer(default=None, allow_None=True, bounds=(0, None), doc="""
        Window width in pixels below which a persistent drawer overlays the
        content like a temporary one instead of narrowing it, e.g. 1200 so
        a narrow window keeps its full width for the app. As in the
        temporary variant, a new approval then only updates the toggle's
        badge.""")

    tab = param.Selector(default="chat", objects=list(_TABS), doc="""
        The view shown in the drawer: the chat, the artifacts (opened from
        the header) or the pending approvals (opened from the bar above the
        chat).""")

    show_artifacts = param.Boolean(default=True, doc="Whether the artifacts view can be opened.")

    show_approvals = param.Boolean(default=True, doc="Whether the pending approvals bar and view are shown.")

    open_on_approval = param.Boolean(default=True, doc="""
        Whether a new pending approval opens a persistent drawer, so a
        paused run does not go unnoticed. A temporary drawer would cover
        the app, so it only updates the toggle's badge.""")

    can_approve = param.Callable(default=None, allow_None=True, doc="""
        Permission hook ``(request, user) -> bool`` for approvals.
        Defaults to the runner's ``approval_policy.can_approve``.""")

    on_apply = param.Callable(default=None, allow_None=True, doc="""
        Callback ``(artifact, version)`` applying a version of a document
        artifact to the host app; see ``ArtifactView.on_apply``.""")

    chat_params = param.Dict(default={}, doc="Additional parameters passed to the AgentChat.")

    redact_runs = param.Parameter(default=False, doc="""
        Whether the runs ``show_run`` opens are shown redacted, without
        their content (see ``SpanView.redacted``): a boolean, or a callable
        taking a run id and returning one, e.g. a host's check of whether
        the person viewing may see the data.""")

    def __init__(self, runner: Any = None, **params):
        if runner is not None:
            params["runner"] = runner
        runner = params.get("runner")
        if "state" not in params:
            params["state"] = runner.state if runner is not None else RunState()
        follow_policy = "can_approve" not in params and isinstance(runner, param.Parameterized) and "approval_policy" in runner.param
        if "can_approve" not in params and (policy := getattr(runner, "approval_policy", None)) is not None:
            params["can_approve"] = policy.can_approve
        super().__init__(**params)
        if follow_policy:
            runner.param.watch(lambda e: setattr(self, "can_approve", e.new.can_approve if e.new is not None else None), "approval_policy")
        self._pending_seen = len(self.state.pending_approvals)

        chat_params = dict(self.chat_params)
        # The drawer is the chat's frame, so the feed's own card is flat rather than a raised card inside it.
        feed_params = {
            "min_height": 0, "card_params": {"elevation": 0, "raised": False, "square": True},
            **chat_params.pop("feed_params", {}),
        }
        self._chat = AgentChat(
            state=self.state, runner=self.runner, can_approve=self.can_approve,
            **{"feed_params": feed_params, "on_show_run": self.show_run, **chat_params},
        )
        self._artifacts = ArtifactPane(state=self.state, on_apply=self.on_apply, show_list=False, on_show_run=self.show_run)
        self._span: SpanView | None = None
        self._span_dialog = pmui.Dialog(
            title="How this was made", open=False, show_close_button=True, close_on_click=True, width_option="md",
            css_classes=["span-dialog"],
        )
        self._artifact_select = pmui.Select(
            label="Artifact", options={}, size="small", sizing_mode="stretch_width", margin=(8, 8, 0, 8),
        )
        self._artifact_select.param.watch(self._on_artifact_select, "value")
        self._approvals = ApprovalQueue(self.state, can_approve=self.can_approve, title="Pending approvals")

        self._views = {
            "chat": pmui.Column(self._chat, sizing_mode="stretch_both", margin=0),
            "artifacts": pmui.Column(self._artifact_select, self._artifacts, sizing_mode="stretch_both", margin=0, scroll=True),
            "approvals": pmui.Column(self._approvals, sizing_mode="stretch_both", margin=(8, 8), scroll=True),
        }
        self._body = pmui.Column(sizing_mode="stretch_both", margin=0, styles={"min-height": "0"})

        # No tab strip: artifacts and approvals are views reached from the header
        # and from the pending bar, with a back button to the chat.
        self._back = pmui.IconButton(
            icon="arrow_back", description="Back to chat", size="small",
            on_click=lambda e: self.param.update(tab="chat"),
        )
        self._heading = pmui.Typography(variant="subtitle1", margin=(0, 8), sx={"fontWeight": 500})
        self._artifacts_button = pmui.IconButton(
            icon="history", description="Artifacts", size="small", on_click=lambda e: self.param.update(tab="artifacts"),
        )
        # The count must not swallow clicks meant for the button.
        self._artifacts_badge = pmui.Badge(
            self._artifacts_button, content=0, color="primary", margin=0,
            sx={"& .MuiBadge-badge": {"pointerEvents": "none", "top": "6px", "right": "6px"}},
        )
        self._pending_text = pmui.Typography(variant="body2", margin=(0, 8), sizing_mode="stretch_width")
        self._pending_bar = pmui.Row(
            self._pending_text,
            pmui.Button(label="Review", variant="text", size="small", margin=0, on_click=lambda e: self.param.update(tab="approvals")),
            sizing_mode="stretch_width", margin=(8, 12, 0, 12), align="center", visible=False,
            styles={
                "align-items": "center", "border-radius": "8px", "padding": "2px 4px 2px 6px",
                "border": "1px solid var(--mui-palette-warning-main)",
                "background": "rgba(var(--mui-palette-warning-mainChannel) / 0.12)",
            },
        )

        self._clear = pmui.IconButton(
            icon="edit_square", description="New conversation", size="small",
            on_click=lambda e: self._ask_new_conversation(), visible=hasattr(self.runner, "clear"),
            disabled=self.state.param.running,
        )
        self._confirm = pmui.Dialog(
            pmui.Typography(
                "Starting a new conversation clears the versions of the documents the agent edited, so they can no "
                "longer be compared or reverted. The app keeps its current state.", variant="body2",
            ),
            pmui.Row(
                pn.HSpacer(),
                pmui.Button(label="Cancel", variant="text", on_click=lambda e: self._confirm.param.update(open=False)),
                pmui.Button(label="Start new conversation", color="warning", on_click=lambda e: self._confirm_new_conversation()),
                sizing_mode="stretch_width",
            ),
            title="Start a new conversation?", open=False,
        )
        close = pmui.IconButton(icon="close", description="Close", size="small", on_click=lambda e: self._close_drawer())
        header = pmui.Row(
            self._back, self._heading, pn.HSpacer(), self._artifacts_badge, self._clear, close,
            sizing_mode="stretch_width", margin=0, align="center",
            styles={"align-items": "center", "padding": "6px 6px 6px 12px", "min-height": "52px",
                    "border-bottom": "1px solid var(--mui-palette-divider)"},
        )
        self._content = pmui.Column(
            header, self._pending_bar, self._body, self._confirm, self._span_dialog, sizing_mode="stretch_both", margin=0, styles={"height": "100%"},
        )
        # Along the drawer's left edge, beside its content.
        self._handle = DrawerResizeHandle(
            min_size=self.param.width.bounds[0], label=f"Resize {self.title}", visible=self.resizable,
            width=8, margin=0, sizing_mode="stretch_height", styles={"flex": "none"},
        )
        self._handle.param.watch(self._on_resized, "size")

        self._drawer = pmui.Drawer(
            pmui.Row(self._handle, self._content, sizing_mode="stretch_both", margin=0, styles={"height": "100%"}),
            anchor="right", open=self.open, size=self.width,
            variant=self.variant, inline=self.variant == "persistent",
            # The inline drawer sets its own width (0 when closed); the row must neither grow nor shrink it.
            styles={"flex": "0 0 auto"},
            # On a phone narrower than the drawer it takes the screen's width rather than running off it.
            sx={"maxWidth": "100vw"},
        )
        self._drawer.param.watch(self._on_drawer_open, "open")

        self._toggle = DrawerToggle(label=self.title, expanded=self.open, margin=0, align="center")
        self._toggle.param.watch(self._on_toggle, "expanded")

        self._layout = pmui.Row(sizing_mode="stretch_both", margin=0)
        self._resize = DrawerResizeNotifier(open=self.open, width=0, height=0, margin=0)
        self._resize.param.watch(lambda e: self._sync_variant(), "window_width")

        self.state.param.watch(self._on_pending, "pending_approvals")
        self._watch_store()
        self.state.param.watch(lambda e: self._watch_store(), "artifacts")
        self._sync_tabs()
        self._sync_layout()
        self._on_pending()

    def __panel__(self):
        return self._layout

    @property
    def chat(self) -> AgentChat:
        return self._chat

    @property
    def artifacts(self) -> ArtifactPane:
        return self._artifacts

    @property
    def approvals(self) -> ApprovalQueue:
        return self._approvals

    @property
    def toggle(self) -> Viewable:
        """
        A button opening and closing the drawer for the app's header. It
        shows the number of pending approvals, includes it in its
        accessible name and reports whether the drawer is open with
        ``aria-expanded``.
        """
        return self._toggle

    @property
    def content(self) -> pmui.Column:
        """The body of the drawer, for placing it in another container such as a page's contextbar."""
        return self._content

    def show_run(self, run_id: str, *, redacted: bool | None = None):
        """
        Open a dialog showing how the run ``run_id`` was made: a
        ``SpanView`` with its question, model requests, tool calls and the
        versions it published. Clicking a version opens it in the
        artifacts view. The chat's "Details" link and the history's "How
        was this made?" action call this. ``redacted`` defaults to what
        ``redact_runs`` says for the run.
        """
        if redacted is None:
            policy = self.redact_runs
            redacted = bool(policy(run_id) if callable(policy) else policy)
        span = self._span
        if span is None or span.state is not self.state or span.redacted != redacted:
            if span is not None:
                span.release()
            self._span = SpanView(self.state, run_id=run_id, redacted=redacted, on_artifact=self._open_version)
            self._span_dialog.objects = [self._span]
        else:
            span.run_id = run_id
        self._span_dialog.open = True

    def _open_version(self, artifact_id: str, version: int):
        self._span_dialog.open = False
        self.state.artifacts.active = artifact_id
        self.param.update(open=True, tab="artifacts")
        if (view := self._artifacts.views.get(artifact_id)) is not None:
            view.show_version(version)

    def _ask_new_conversation(self):
        if any(artifact.is_document for artifact in self.state.artifacts):
            self._confirm.open = True
        else:
            self.new_conversation()

    def _confirm_new_conversation(self):
        self._confirm.open = False
        self.new_conversation()

    def new_conversation(self):
        """
        Forget the runner's message history and clear the session's runs,
        artifacts and approvals. The button in the drawer asks first when
        there are document artifacts, whose versions would be lost.
        """
        if self.state.running:
            return
        if hasattr(self.runner, "clear"):
            self.runner.clear()
        self.state.reset()

    def _visible_tabs(self) -> list[str]:
        return [t for t in _TABS if t == "chat" or getattr(self, f"show_{t}")]

    @param.depends("show_artifacts", "show_approvals", watch=True)
    def _sync_tabs(self):
        if self.tab not in self._visible_tabs():
            self.tab = "chat"
        self._on_tab()

    @param.depends("tab", watch=True)
    def _on_tab(self):
        view = self._views[self.tab]
        if self._body.objects != [view]:
            self._body.objects = [view]
        self._update_labels()

    @param.depends("title", watch=True)
    def _update_labels(self, *events):
        chat = self.tab == "chat"
        self._back.visible = not chat
        self._heading.object = self.title if chat else self.tab.capitalize()
        n = len(self.state.artifacts)
        self._artifacts_badge.content = n
        self._artifacts_badge.visible = chat and self.show_artifacts and n > 0
        pending = len(self.state.pending_approvals)
        self._pending_bar.visible = chat and self.show_approvals and pending > 0
        self._pending_text.object = f"{pending} {'action waits' if pending == 1 else 'actions wait'} for you"

    def _on_resized(self, event):
        if event.new is not None:
            self.width = max(self.param.width.bounds[0], event.new)

    @param.depends("resizable", watch=True)
    def _sync_resizable(self):
        self._handle.visible = self.resizable

    @param.depends("open", "width", watch=True)
    def _sync_open(self):
        self._drawer.param.update(open=self.open, size=self.width)
        # A width set here replaces the one a drag left on the drawer in the page.
        self._handle.size = self.width
        self._toggle.expanded = self.open
        self._resize.open = self.open

    def _on_drawer_open(self, event):
        if self.open and not event.new:
            # Closed from inside the drawer, e.g. Escape on a temporary drawer: focus goes back to the toggle.
            self._toggle.focus()
        self.open = event.new

    def _on_toggle(self, event):
        if event.new == self.open:
            return
        self.open = event.new
        if event.new and (widget := getattr(self._chat.feed, "active_widget", None)) is not None and hasattr(widget, "focus"):
            widget.focus()

    def _close_drawer(self):
        self.open = False
        self._toggle.focus()

    @param.depends("title", watch=True)
    def _sync_title(self):
        self._toggle.label = self.title

    @property
    def _variant(self) -> str:
        """The variant in effect: a persistent drawer overlays the app in a narrow window."""
        width = self._resize.window_width
        if self.variant == "persistent" and self.overlay_below is not None and width is not None and width < self.overlay_below:
            return "temporary"
        return self.variant

    @param.depends("variant", "overlay_below", watch=True)
    def _sync_variant(self):
        variant = self._variant
        self._drawer.param.update(variant=variant, inline=variant == "persistent")

    @param.depends("main", watch=True)
    def _sync_layout(self):
        self._sync_variant()
        if self.main is None:
            self._layout.objects = [self._drawer, self._resize]
        else:
            main = pmui.Column(self.main, sizing_mode="stretch_both", margin=0, styles={"min-width": "0"})
            self._layout.objects = [main, self._drawer, self._resize]

    @param.depends("can_approve", "on_apply", watch=True)
    def _propagate(self):
        self._chat.can_approve = self.can_approve
        self._approvals.can_approve = self.can_approve
        self._artifacts.on_apply = self.on_apply

    def _on_pending(self, *events):
        pending = len(self.state.pending_approvals)
        self._toggle.count = pending
        if pending > self._pending_seen and self.open_on_approval and self._variant == "persistent":
            self.open = True
        self._pending_seen = pending
        self._update_labels()

    def _watch_store(self):
        store = self.state.artifacts
        if getattr(self, "_store", None) is store:
            return
        if (old := getattr(self, "_store", None)) is not None:
            for watcher in self._store_watchers:
                old.param.unwatch(watcher)
        self._store = store
        self._artifacts.store = store
        self._store_watchers = [
            store.param.watch(self._sync_artifact_select, "artifacts"),
            store.param.watch(self._sync_artifact_select, "active"),
        ]
        self._sync_artifact_select()

    def _sync_artifact_select(self, *events):
        store = self.state.artifacts
        options = {a.title or a.artifact_id: a.artifact_id for a in store}
        self._artifact_select.param.update(options=options, value=store.active, visible=len(options) > 1)
        self._update_labels()

    def _on_artifact_select(self, event):
        if event.new is not None and event.new != self.state.artifacts.active:
            self.state.artifacts.active = event.new


__all__ = ["AgentDrawer"]
