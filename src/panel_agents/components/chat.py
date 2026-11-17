"""
Chat view of a RunState timeline: messages, thinking and tool calls.
"""

from __future__ import annotations

import datetime as dt
import inspect
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..state import MessageState, RunState, ToolCallState
from .approval import ApprovalCard
from .approval import approval_factory as default_approval_factory
from .tool_call import ToolCallCard

THINKING_CSS = """
:host { width: 100%; }
.markdown { font-size: 0.85rem; opacity: 0.75; }
"""


class AgentChat(Viewer):
    """
    Renders the ``items`` of a ``RunState`` as a chat feed.

    User and assistant messages become chat messages whose content
    streams in place, thinking becomes a collapsed step and tool calls
    become ``ToolCallCard``s, all in timeline order. Submitted text is
    passed to ``runner.run`` (e.g. an ``AgentRunner``, which records the
    user message itself) or to a plain ``callback``. With ``readonly=True`` no input is shown, e.g. to replay a recording.
    """

    state = param.ClassSelector(class_=RunState, constant=True)

    runner = param.Parameter(default=None, doc="""
        Object with an (async) ``run(text)`` method that adds the user
        message to the state itself, such as an ``AgentRunner``. Its
        ``state`` is used when no state is given.""")

    callback = param.Callable(default=None, doc="""
        Callable, usually async, invoked with the submitted text when no
        runner is given.""")

    readonly = param.Boolean(default=False, constant=True, doc="""
        Whether to render the feed without an input, e.g. for replay.""")

    add_user_message = param.Boolean(default=True, doc="""
        Whether submitting adds the text to the state as a user message
        before invoking the callback. Disable it if the callback emits the
        user message itself. Ignored when a runner is given.""")

    approval_factory = param.Callable(default=None, doc="""
        Callable ``(ApprovalRequest, RunState) -> Viewable`` rendering
        pending approvals inside tool call cards. Defaults to the
        ApprovalCard factory; set to None to show a placeholder.""")

    can_approve = param.Callable(default=None, allow_None=True, doc="""
        Permission hook ``(request, user) -> bool`` set on the approval
        cards the default factory creates. Defaults to the runner's
        ``approval_policy.can_approve`` when it has one.""")

    user = param.String(default="User", doc="Name shown for user messages.")

    assistant = param.String(default="Assistant", doc="""
        Name shown for assistant messages when the run has no agent name.""")

    placeholder = param.String(default="Ask the agent…", doc="Placeholder of the input.")

    feed_params = param.Dict(default={}, doc="""
        Additional parameters passed to the underlying pmui ChatFeed or ChatInterface.""")

    def __init__(self, state: RunState | None = None, **params):
        params.setdefault("approval_factory", default_approval_factory)
        if state is not None:
            params["state"] = state
        elif params.get("runner") is not None and "state" not in params:
            params["state"] = params["runner"].state
        runner = params.get("runner")
        follow_policy = "can_approve" not in params and isinstance(runner, param.Parameterized) and "approval_policy" in runner.param
        if follow_policy and runner.approval_policy is not None:
            params["can_approve"] = runner.approval_policy.can_approve
        super().__init__(**params)
        if follow_policy:
            runner.param.watch(lambda e: setattr(self, "can_approve", e.new.can_approve if e.new is not None else None), "approval_policy")
        feed_params = dict(
            sizing_mode="stretch_both", show_activity_dot=False, min_height=300,
            **self.feed_params,
        )
        if self.readonly:
            self._feed = pmui.ChatFeed(**feed_params)
        else:
            self._feed = pmui.ChatInterface(
                on_submit=self._on_submit, show_rerun=False, show_undo=False, show_clear=False,
                input_params={"placeholder": self.placeholder}, **feed_params,
            )
        self._rendered: list[Any] = []
        self._views: dict[int, Any] = {}
        self._watchers: list[tuple[param.Parameterized, Any]] = []
        self.state.param.watch(self._update_items, "items")
        self.state.param.watch(self._update_running, "running")
        self._update_items()
        self._update_running()

    def __panel__(self):
        return self._feed

    @property
    def feed(self) -> pmui.ChatFeed:
        return self._feed

    def view_for(self, item: MessageState | ToolCallState) -> Any:
        """The component rendering a timeline item."""
        return self._views[id(item)]

    async def submit(self, text: str):
        """Submit text as the user would through the input."""
        text = text.strip()
        if not text or self.state.running:
            return
        if self.runner is not None:
            result = self.runner.run(text)
        elif self.callback is not None:
            if self.add_user_message:
                self.state.add_user_message(text)
            result = self.callback(text)
        else:
            result = None
            if self.add_user_message:
                self.state.add_user_message(text)
        if inspect.isawaitable(result):
            await result

    def _on_submit(self, event=None, instance=None):
        text = event.new if event is not None else ""
        if not text or not text.strip():
            return

        async def run():
            await self.submit(text)

        pn.state.execute(run)

    def _update_running(self, *_):
        if not self.readonly:
            self._feed.disabled = self.state.running

    def _update_items(self, *_):
        items = self.state.items
        n = len(self._rendered)
        if len(items) < n or any(a is not b for a, b in zip(items[:n], self._rendered, strict=True)):
            self._clear()
            n = 0
        for item in items[n:]:
            message = self._build(item)
            self._rendered.append(item)
            if message is not None:
                self._feed.send(message, respond=False)

    def _clear(self):
        for obj, watcher in self._watchers:
            obj.param.unwatch(watcher)
        self._watchers = []
        self._rendered = []
        self._views = {}
        self._feed.clear()

    def _watch(self, obj, callback, names):
        self._watchers.append((obj, obj.param.watch(callback, names)))

    def _agent_name(self, run_id: str | None) -> str:
        run = self.state.runs.get(run_id) if run_id else None
        return (run.agent_name if run is not None else None) or self.assistant

    def _message_params(self, user: str, created_at: float | None, compact: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {
            "user": user,
            "placement": "right" if user == self.user else "left",
            "show_reaction_icons": False,
            "show_copy_icon": False,
            # The hover timestamp row adds a gap that would space out runs of tool calls.
            "show_timestamp": not compact,
        }
        if created_at is not None:
            params["timestamp"] = dt.datetime.fromtimestamp(created_at)
        return params

    def _build(self, item) -> pmui.ChatMessage | None:
        if isinstance(item, ToolCallState):
            card = ToolCallCard(item, state=self.state, approval_factory=self._approval_factory())
            self._views[id(item)] = card
            self._watch(self, lambda e, c=card: setattr(c, "approval_factory", self._approval_factory()), ["approval_factory", "can_approve"])
            return pmui.ChatMessage(card, **self._message_params(self._agent_name(item.run_id), item.started_at, compact=True))
        if not isinstance(item, MessageState):
            return None
        if item.kind == "thinking":
            return self._build_thinking(item)
        user = self.user if item.role == "user" else self._agent_name(item.run_id)
        pane = pn.pane.Markdown(item.content)
        message = pmui.ChatMessage(pane, visible=bool(item.content), **self._message_params(user, item.created_at))
        self._views[id(item)] = message

        def stream(*_):
            pane.object = item.content
            message.visible = bool(item.content)
            self._scroll()

        self._watch(item, stream, "content")
        return message

    def _build_thinking(self, item: MessageState) -> pmui.ChatMessage:
        pane = pn.pane.Markdown(item.content, sizing_mode="stretch_width", margin=0, stylesheets=[THINKING_CSS], css_classes=["thinking"])
        step = pmui.ChatStep(
            pane, running_title="Thinking…", success_title="Thought process",
            status="success" if item.finished else "running",
            collapsed=True, sizing_mode="stretch_width", margin=(4, 0),
            css_classes=["thinking-step"],
        )
        self._views[id(item)] = step

        params = self._message_params(self._agent_name(item.run_id), item.created_at, compact=True)
        message = pmui.ChatMessage(step, visible=_show_thinking(item), **params)

        def update(*_):
            pane.object = item.content
            step.status = "success" if item.finished else "running"
            # Some providers only return encrypted reasoning, which leaves nothing to show.
            message.visible = _show_thinking(item)

        self._watch(item, update, ["content", "finished"])
        return message

    def _approval_factory(self):
        factory, can_approve = self.approval_factory, self.can_approve
        if factory is None or can_approve is None:
            return factory

        def build(request, state):
            view = factory(request, state)
            if isinstance(view, ApprovalCard) and view.can_approve is None:
                view.can_approve = can_approve
            return view
        return build

    def _scroll(self):
        # ChatFeed exposes no public scroll API; this mirrors what its own stream() does.
        self._feed._chat_log.scroll_to_latest(scroll_limit=self._feed.auto_scroll_limit)


def _show_thinking(item: MessageState) -> bool:
    return bool(item.content.strip()) or not item.finished


__all__ = ["AgentChat"]
