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
from ._toggle import DrawerToggle
from .approval import ApprovalQueue
from .artifacts import ArtifactPane
from .chat import AgentChat

TABS = ("chat", "artifacts", "approvals")


class AgentDrawer(Viewer):
    """
    A collapsible right-hand drawer holding an ``AgentChat`` with tabs
    for the session's artifacts and pending approvals.

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

    title = param.String(default="Assistant", doc="Title shown at the top of the drawer.")

    variant = param.Selector(default="persistent", objects=["persistent", "temporary"], doc="""
        persistent: the drawer sits beside the content and narrows it.
        temporary: the drawer overlays the content and closes on outside clicks.""")

    tab = param.Selector(default="chat", objects=list(TABS), doc="The tab shown in the drawer.")

    show_artifacts = param.Boolean(default=True, doc="Whether to show the Artifacts tab.")

    show_approvals = param.Boolean(default=True, doc="Whether to show the Approvals tab.")

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

        self._chat = AgentChat(
            state=self.state, runner=self.runner, can_approve=self.can_approve,
            **{"feed_params": {"min_height": 0}, **self.chat_params},
        )
        self._artifacts = ArtifactPane(state=self.state, on_apply=self.on_apply, show_list=False)
        self._artifact_select = pmui.Select(
            label="Artifact", options={}, size="small", sizing_mode="stretch_width", margin=(8, 8, 0, 8),
        )
        self._artifact_select.param.watch(self._on_artifact_select, "value")
        self._approvals = ApprovalQueue(self.state, can_approve=self.can_approve, title="Pending approvals")

        self._tab_items = {
            "chat": pmui.Column(self._chat, sizing_mode="stretch_both", margin=0),
            "artifacts": pmui.Column(self._artifact_select, self._artifacts, sizing_mode="stretch_both", margin=0, scroll=True),
            "approvals": pmui.Column(self._approvals, sizing_mode="stretch_both", margin=(8, 8), scroll=True),
        }
        self._tabs = pmui.Tabs(sizing_mode="stretch_both", margin=0, dynamic=False)
        self._tabs.param.watch(self._on_tab_click, "active")

        self._clear = pmui.IconButton(
            icon="add_comment", description="New conversation", size="small",
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
            pmui.Typography(self.param.title, variant="h6", margin=(0, 8)),
            pn.HSpacer(), self._clear, close,
            sizing_mode="stretch_width", margin=(8, 4, 0, 8), align="center", styles={"align-items": "center"},
        )
        self._content = pmui.Column(header, self._tabs, self._confirm, sizing_mode="stretch_both", margin=0, styles={"height": "100%"})

        self._drawer = pmui.Drawer(
            self._content, anchor="right", open=self.open, size=self.width,
            variant=self.variant, inline=self.variant == "persistent",
            # The inline drawer sets its own width (0 when closed); the row must neither grow nor shrink it.
            styles={"flex": "0 0 auto"},
        )
        self._drawer.param.watch(self._on_drawer_open, "open")

        self._toggle = DrawerToggle(label=self.title, expanded=self.open, margin=0, align="center")
        self._toggle.param.watch(self._on_toggle, "expanded")

        self._layout = pmui.Row(sizing_mode="stretch_both", margin=0)

        self.state.param.watch(self._on_pending, "pending_approvals")
        self.state.param.watch(self._update_labels, "items")
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
        return [t for t in TABS if t == "chat" or getattr(self, f"show_{t}")]

    def _label(self, tab: str) -> str:
        if tab == "artifacts":
            n = len(self.state.artifacts)
            return f"Artifacts ({n})" if n else "Artifacts"
        if tab == "approvals":
            n = len(self.state.pending_approvals)
            return f"Approvals ({n})" if n else "Approvals"
        return "Chat"

    @param.depends("show_artifacts", "show_approvals", watch=True)
    def _sync_tabs(self):
        tabs = self._visible_tabs()
        if self.tab not in tabs:
            self.tab = "chat"
        self._labels = None
        self._update_labels()
        self._tabs.active = tabs.index(self.tab)

    def _update_labels(self, *events):
        tabs = self._visible_tabs()
        labels = [self._label(t) for t in tabs]
        if labels != getattr(self, "_labels", None):
            self._labels = labels
            self._tabs[:] = [(label, self._tab_items[t]) for label, t in zip(labels, tabs, strict=True)]

    @param.depends("tab", watch=True)
    def _on_tab(self):
        tabs = self._visible_tabs()
        if self.tab in tabs and self._tabs.active != tabs.index(self.tab):
            self._tabs.active = tabs.index(self.tab)

    def _on_tab_click(self, event):
        tabs = self._visible_tabs()
        if event.new is not None and event.new < len(tabs):
            self.tab = tabs[event.new]

    @param.depends("open", "width", watch=True)
    def _sync_open(self):
        self._drawer.param.update(open=self.open, size=self.width)
        self._toggle.expanded = self.open

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

    @param.depends("variant", "main", watch=True)
    def _sync_layout(self):
        self._drawer.param.update(variant=self.variant, inline=self.variant == "persistent")
        if self.main is None:
            self._layout.objects = [self._drawer]
        else:
            main = pmui.Column(self.main, sizing_mode="stretch_both", margin=0, styles={"min-width": "0"})
            self._layout.objects = [main, self._drawer]

    @param.depends("can_approve", "on_apply", watch=True)
    def _propagate(self):
        self._chat.can_approve = self.can_approve
        self._approvals.can_approve = self.can_approve
        self._artifacts.on_apply = self.on_apply

    def _on_pending(self, *events):
        pending = len(self.state.pending_approvals)
        self._toggle.count = pending
        if pending > self._pending_seen and self.open_on_approval and self.variant == "persistent":
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
        self._artifact_select.param.update(options=options, value=store.active, visible=bool(options))
        self._update_labels()

    def _on_artifact_select(self, event):
        if event.new is not None and event.new != self.state.artifacts.active:
            self.state.artifacts.active = event.new


__all__ = ["AgentDrawer"]
