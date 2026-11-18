"""
Chat view of a RunState timeline: messages, thinking and tool calls.
"""

from __future__ import annotations

import datetime as dt
import inspect
from html import escape
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..security import _escape_markdown, untrusted_markdown
from ..state import MessageState, ModelCallState, RunState, ToolCallState
from .approval import ApprovalCard
from .approval import approval_factory as default_approval_factory
from .model_call import USAGE_CSS, ModelCallCard, usage_summary, usage_tooltip
from .tool_call import ToolCallCard, _format_duration

_THINKING_CSS = """
:host { width: 100%; }
.markdown { font-size: 0.85rem; opacity: 0.75; }
"""


class AgentChat(Viewer):
    """
    Renders the ``items`` of a ``RunState`` as a chat feed.

    User and assistant messages become chat messages whose content
    streams in place, thinking becomes a collapsed step, model requests
    become ``ModelCallCard``s, tool calls ``ToolCallCard``s and a failed
    run an error alert, all in timeline order. A finished run's usage
    totals are shown under its last message. Submitted text is
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

    tool_labels = param.Dict(default={}, doc="""
        Labels for tool calls in the user's words, keyed by tool name: a
        string, or a callable taking the call's arguments and returning
        one, e.g. ``{"get_report": "Read the report", "update_report":
        lambda args: f"Change {', '.join(args)}"}``. Imperative labels
        read right whether the call runs, succeeded, failed or waits. See
        ``ToolCallCard.label``.""")

    group_tool_calls = param.Boolean(default=True, doc="""
        Whether consecutive tool calls of a run that finished without
        trouble, and the model requests between them, fold into one row,
        "3 steps · 2.1s", that expands to show them. Calls that failed,
        were denied, wait for approval or published an artifact are always
        shown.""")

    show_model_calls = param.Boolean(default=True, doc="""
        Whether model requests are shown as steps. Like tool calls, the
        requests of a run fold into its step group once they succeed; a
        request with no step group to fold into is left out once its run
        finished, when ``show_usage`` reports the run's totals.""")

    show_usage = param.Boolean(default=True, doc="""
        Whether a finished run's model requests, tokens, cost and wall
        time are shown in a line under its last message.""")

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
        feed_params = {
            "sizing_mode": "stretch_both", "show_activity_dot": False, "min_height": 300,
            **self.feed_params,
        }
        if self.readonly:
            self._feed = pmui.ChatFeed(**feed_params)
        else:
            self._feed = pmui.ChatInterface(
                on_submit=self._on_submit, show_rerun=False, show_undo=False, show_clear=False,
                input_params={"placeholder": self.placeholder, "enable_upload": False}, **feed_params,
            )
        if not self.readonly:
            self._feed._widget.on_action("stop", self._on_stop)
        self._rendered: list[Any] = []
        self._views: dict[int, Any] = {}
        self._messages: dict[int, pmui.ChatMessage] = {}
        self._groups: dict[int, _StepGroup] = {}
        self._watchers: list[tuple[param.Parameterized, Any]] = []
        self._footers: dict[str, pmui.ChatMessage] = {}
        self._run_watchers: dict[str, tuple[param.Parameterized, Any]] = {}
        self.state.param.watch(self._update_items, "items")
        self.state.param.watch(self._watch_runs, "runs")
        self._watch_runs()
        self.state.param.watch(self._update_running, "running")
        self._update_items()
        self._update_running()

    def __panel__(self):
        return self._feed

    @property
    def feed(self) -> pmui.ChatFeed:
        return self._feed

    def view_for(self, item: MessageState | ModelCallState | ToolCallState) -> Any:
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

    def _on_stop(self, *_):
        cancel = getattr(self.runner, "cancel", None)
        if callable(cancel):
            cancel()

    def _update_running(self, *_):
        if self.readonly:
            return
        # Only sending waits for the run: a disabled input would lose its focus
        # and its text, and the user could not type the next message meanwhile.
        self._feed._widget.disabled_enter = self.state.running

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
                self._messages[id(item)] = message
                self._feed.send(message, respond=False)
        self._regroup()

    def _clear(self):
        for obj, watcher in self._watchers:
            obj.param.unwatch(watcher)
        self._watchers = []
        self._rendered = []
        self._views = {}
        self._messages = {}
        self._groups = {}
        self._footers = {}
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
            card = ToolCallCard(
                item, state=self.state, approval_factory=self._approval_factory(),
                label=self.tool_labels.get(item.tool_name),
            )
            self._views[id(item)] = card
            self._watch(self, lambda e, c=card: setattr(c, "approval_factory", self._approval_factory()), ["approval_factory", "can_approve"])
            self._watch(self, lambda e, c=card, i=item: setattr(c, "label", self.tool_labels.get(i.tool_name)), "tool_labels")
            self._watch(item, self._regroup, "status")
            return pmui.ChatMessage(card, **self._message_params(self._agent_name(item.run_id), item.started_at, compact=True))
        if isinstance(item, ModelCallState):
            card = ModelCallCard(item, state=self.state)
            self._views[id(item)] = card
            self._watch(item, self._regroup, "status")
            params = self._message_params(self._agent_name(item.run_id), item.started_at, compact=True)
            return pmui.ChatMessage(card, visible=self.show_model_calls, **params)
        if not isinstance(item, MessageState):
            return None
        if item.kind == "thinking":
            return self._build_thinking(item)
        if item.kind == "error":
            # Error messages may quote provider responses or tool output, so they are shown as plain text.
            alert = pmui.Alert(
                object=_escape_markdown(item.content), severity="error", variant="outlined",
                sizing_mode="stretch_width", margin=0, css_classes=["run-error"],
            )
            self._views[id(item)] = alert
            return pmui.ChatMessage(alert, **self._message_params(self._agent_name(item.run_id), item.created_at))
        user = self.user if item.role == "user" else self._agent_name(item.run_id)
        # Message text may carry injected content, so it is rendered inert: no images or HTML that load by themselves.
        pane = untrusted_markdown(item.content)
        message = pmui.ChatMessage(pane, visible=bool(item.content), **self._message_params(user, item.created_at))
        self._views[id(item)] = message

        def stream(*_):
            pane.object = item.content
            shown = bool(item.content)
            if message.visible != shown:
                message.visible = shown
                self._regroup()
            self._scroll()

        self._watch(item, stream, "content")
        return message

    def _build_thinking(self, item: MessageState) -> pmui.ChatMessage:
        pane = untrusted_markdown(
            item.content, sizing_mode="stretch_width", margin=0, stylesheets=[_THINKING_CSS], css_classes=["thinking"],
        )
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

    def _foldable(self, item: ToolCallState | ModelCallState) -> bool:
        if isinstance(item, ModelCallState):
            return item.status == "success"
        if item.status != "success":
            return False
        if item.approval is not None and item.approval.status != "approved":
            return False
        card = self._views.get(id(item))
        return card is not None and not card.published_artifacts()

    def _breaks_group(self, item) -> bool:
        message = self._messages.get(id(item))
        return message is not None and message.visible

    @param.depends("group_tool_calls", "show_model_calls", "show_usage", watch=True)
    def _regroup(self, *_):
        """Fold runs of two or more untroubled tool and model calls of one run into a step group."""
        runs: list[list[ToolCallState | ModelCallState]] = []
        current: list[ToolCallState | ModelCallState] = []
        for item in self._rendered:
            if isinstance(item, ModelCallState) and not self.show_model_calls:
                continue
            if isinstance(item, (ToolCallState, ModelCallState)):
                if self.group_tool_calls and self._foldable(item) and (not current or current[-1].run_id == item.run_id):
                    current.append(item)
                    continue
                runs.append(current)
                current = [item] if self.group_tool_calls and self._foldable(item) else []
            elif self._breaks_group(item):
                runs.append(current)
                current = []
        runs.append(current)
        wanted = {id(run[0]): run for run in runs if len(run) > 1}

        objects = list(self._feed.objects)
        changed = False
        for key in [k for k in self._groups if k not in wanted]:
            group = self._groups.pop(key)
            group.release()
            if group.message in objects:
                objects.remove(group.message)
                changed = True
        for key, members in wanted.items():
            group = self._groups.get(key)
            if group is None:
                first = self._messages[key]
                group = self._groups[key] = _StepGroup(self, first.user)
                if first in objects:
                    objects.insert(objects.index(first), group.message)
                    changed = True
            group.update([self._messages[id(m)] for m in members], members)
        grouped = {id(m) for members in wanted.values() for m in members}
        for item in self._rendered:
            if isinstance(item, ModelCallState) and id(item) not in grouped:
                self._messages[id(item)].visible = self._show_lone_model_call(item)
        if changed:
            self._feed.objects = objects
        self._update_usage()

    def _show_lone_model_call(self, item: ModelCallState) -> bool:
        if not self.show_model_calls:
            return False
        run = self.state.runs.get(item.run_id)
        # The usage line reports a finished run's requests, so one with no step group to fold into is left out.
        return not (self.show_usage and self._foldable(item) and run is not None and run.status != "running")

    def _watch_runs(self, *_):
        for run_id, run in self.state.runs.items():
            if run_id not in self._run_watchers:
                self._run_watchers[run_id] = (run, run.param.watch(self._regroup, ["status", "usage"]))
        for run_id in [r for r in self._run_watchers if r not in self.state.runs]:
            run, watcher = self._run_watchers.pop(run_id)
            run.param.unwatch(watcher)

    def _update_usage(self):
        """Show each finished run's usage totals under the last message of the run that is shown."""
        objects = self._feed.objects
        owners = {id(self._messages[id(item)]): item for item in self._rendered if id(item) in self._messages}
        groups = {id(group.message): group.calls[0] for group in self._groups.values() if group.calls}
        last: dict[str, pmui.ChatMessage] = {}
        for message in objects:
            item = owners.get(id(message)) or groups.get(id(message))
            if item is None or not message.visible or (isinstance(item, MessageState) and item.role == "user"):
                continue
            last[item.run_id] = message
        wanted = {}
        if self.show_usage:
            for run_id, message in last.items():
                run = self.state.runs.get(run_id)
                if run is not None and run.status != "running" and run.usage and run.parent_run_id is None:
                    wanted[run_id] = message
        for run_id, message in list(self._footers.items()):
            if wanted.get(run_id) is not message:
                message.footer_objects = []
                del self._footers[run_id]
        for run_id, message in wanted.items():
            run = self.state.runs[run_id]
            footer = (
                f'<div class="usage" title="{escape(usage_tooltip(run), quote=True)}">{escape(usage_summary(run))}</div>'
            )
            panes = message.footer_objects
            if panes and getattr(panes[0], "object", None) == footer:
                continue
            message.footer_objects = [pn.pane.HTML(footer, margin=(0, 10), stylesheets=[USAGE_CSS], css_classes=["run-usage"])]
            self._footers[run_id] = message

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


class _StepGroup:
    """The folded row standing in for a run of tool call messages."""

    def __init__(self, chat: AgentChat, user: str):
        self.expanded = False
        self.members: list[pmui.ChatMessage] = []
        self.button = pmui.Button(
            variant="text", size="small", icon="expand_more", margin=0,
            sx={"textTransform": "none", "fontWeight": 400, "justifyContent": "flex-start", "color": "text.secondary"},
            on_click=lambda e: self.toggle(),
        )
        self.message = pmui.ChatMessage(self.button, **chat._message_params(user, None, compact=True))
        self.calls: list[ToolCallState | ModelCallState] = []

    def update(self, members: list[pmui.ChatMessage], calls: list[ToolCallState | ModelCallState]):
        self.members, self.calls = members, calls
        self._sync()

    def toggle(self):
        self.expanded = not self.expanded
        self._sync()

    def release(self):
        for message in self.members:
            message.visible = True

    def _sync(self):
        # A group's steps are its tool calls; the model requests in it are counted by the run's usage line.
        n = sum(isinstance(c, ToolCallState) for c in self.calls) or len(self.calls)
        if any(isinstance(c, ModelCallState) for c in self.calls):
            # Tool calls stream within the model request that issues them, so durations overlap; the wall time is the span.
            starts = [c.started_at for c in self.calls if c.started_at is not None]
            ends = [c.ended_at for c in self.calls if c.ended_at is not None]
            total = max(ends) - min(starts) if starts and ends else 0
        else:
            total = sum(c.duration or 0 for c in self.calls)
        duration = _format_duration(total) if total >= 0.05 else ""
        steps = f"{n} step" if n == 1 else f"{n} steps"
        spoken = steps + (f", {duration}" if duration else "")
        self.button.param.update(
            label=steps + (f" · {duration}" if duration else ""),
            icon="expand_less" if self.expanded else "expand_more",
            # The tooltip is the button's accessible name.
            description=f"{spoken}, {'collapse' if self.expanded else 'expand'}",
        )
        for message in self.members:
            message.visible = self.expanded


def _show_thinking(item: MessageState) -> bool:
    return bool(item.content.strip()) or not item.finished


__all__ = ["AgentChat"]
