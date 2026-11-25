"""
Chat view of a RunState timeline: messages, thinking and tool calls.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from dataclasses import dataclass
from html import escape
from typing import Any

import panel as pn
import panel_material_ui as pmui
import param
from panel.io.state import set_curdoc
from panel.viewable import Viewer

from ..actions import ActionRegistry
from ..artifacts.model import Artifact, root_run_id
from ..security import _escape_markdown, untrusted_markdown
from ..state import MessageState, ModelCallState, RunState, ToolCallState
from ._mentions import MentionChips
from .approval import ApprovalCard
from .approval import approval_factory as default_approval_factory
from .artifact_cell import ArtifactAction, ArtifactCell
from .model_call import USAGE_CSS, ModelCallCard, usage_summary, usage_tooltip
from .model_picker import ModelPicker
from .tool_call import ToolCallCard, _format_duration

_Step = ToolCallState | ModelCallState

_STEP_TYPES = (ToolCallState, ModelCallState)

_THINKING_CSS = """
:host { width: 100%; }
.markdown { font-size: 0.85rem; opacity: 0.75; }
"""

# The model picker's row (the div after the textarea) sits right under the text, so the text drops
# its bottom padding, which pmui sets inline, hence !important.
_FOOTER_CSS = """
textarea:has(~ div) { padding-bottom: 0 !important; }
"""


@dataclass(frozen=True)
class Mention:
    """
    Something a person's next message is about, e.g. an item handed over
    to the chat from elsewhere in the app.

    It shows as a chip in the chat's input, which the person can remove,
    and sending a message writes its ``text`` at the start of the message,
    so the agent reads what the message is about and the conversation
    shows it. By default that is ``@`` and the ``label``, quoted when the
    label has spaces (``@"Sales by region"``) so the name stays one
    token; pass ``text`` to write something else, e.g. an id.
    """

    label: str
    icon: str | None = None
    text: str | None = None

    @property
    def written(self) -> str:
        """How the sent message names it."""
        if self.text:
            return self.text
        name = self.label.strip().lstrip("@")
        return f"@{json.dumps(name, ensure_ascii=False)}" if len(name.split()) != 1 else f"@{name}"


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

    With ``layout="document"`` the feed reads as a document: a wide centred
    column with every message on the left, and the artifacts each run
    published shown inline as ``ArtifactCell``s under the run's last
    message, with the ``artifact_actions`` that apply to them.
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

    avatars = param.Dict(default={}, doc="""
        Avatars of the people and agents in the feed, keyed by the name
        shown, e.g. ``{"Dashboard": "📊"}``: an emoji, text or image URL.
        Names left out get panel-material-ui's default for the name.""")

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

    on_show_run = param.Callable(default=None, allow_None=True, doc="""
        Called with a run id when a person asks how a finished run's
        answer was made. When set, every finished top-level run gets a
        "Details" link under its last message, after its usage line if
        one is shown, e.g. opening a ``SpanView`` of the run, as
        ``AgentDrawer`` does.""")

    layout = param.Selector(default="chat", objects=["chat", "document"], constant=True, doc="""
        chat: user messages on the right, artifacts reachable from the
        tool calls that published them. document: a wide centred column
        with every message on the left and the artifacts a run published
        shown inline under its last message.""")

    user_placement = param.Selector(default=None, objects=[None, "left", "right"], allow_None=True, doc="""
        Where the user's messages are placed. By default on the right in
        the chat layout, as bubbles, and on the left in the document
        layout; "right" makes them bubbles in the document layout too.
        Changing it builds the messages already shown again.""")

    message_params = param.Dict(default={}, doc="""
        Parameters given to every ``ChatMessage`` on top of the chat's
        own, e.g. ``{"show_avatar": False, "show_user": False}`` to leave
        out who sent each message. ``visible`` is left to the chat, which
        hides and shows messages as runs stream and steps fold, and
        ``css_classes`` are added to the chat's own. Changing it builds
        the messages already shown again.""")

    inline_artifacts = param.Callable(default=None, allow_None=True, doc="""
        Predicate ``(artifact) -> bool`` choosing the artifacts shown inline
        in the document layout. Defaults to every artifact that does not
        mirror a host document (``Artifact.is_document``).""")

    cell_renderer = param.Callable(default=None, allow_None=True, doc="""
        Callable ``(artifact) -> Viewable`` rendering the body of an inline
        artifact, e.g. the live object an artifact's value refers to.
        Defaults to the renderer registry.""")

    action_registry = param.ClassSelector(class_=ActionRegistry, default=None, allow_None=True, doc="""
        The registry running the ``artifact_actions`` of inline artifacts.""")

    artifact_actions = param.List(default=[], item_type=ArtifactAction, doc="""
        Actions offered on each inline artifact they apply to, e.g. "Add to
        dashboard"; see ``ArtifactAction``.""")

    max_cells = param.Integer(default=6, bounds=(1, None), doc="""
        How many inline artifacts of a run are shown before the rest wait
        behind a "Show more" button, so a run publishing many is quick to
        show.""")

    feed_params = param.Dict(default={}, doc="""
        Additional parameters passed to the underlying pmui ChatFeed or ChatInterface.""")

    models = param.List(default=[], doc="""
        Models a person may pick for the next request, in a small menu
        below the input (see ``ModelPicker``): model ids, dicts of
        ``ModelChoice`` fields or ``ModelChoice`` objects. The menu is
        shown when there is more than one, and sets the runner's ``model``.""")

    model_id = param.String(default=None, allow_None=True, doc="""
        The id of the picked model, one of the ``models``' ids. Defaults to the first usable one.""")

    remember_model = param.String(default=None, allow_None=True, doc="""
        A localStorage key under which the browser remembers the picked
        model for the next session, see ``ModelPicker.remember``.""")

    mentions = param.List(default=[], item_type=Mention, doc="""
        What the next message is about, e.g. an item a person handed over
        from elsewhere in the app, shown as chips in the input that they
        can remove. Sending a message writes them at its start, see
        ``Mention``, and clears them.""")

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
        if self.layout == "document":
            feed_params["css_classes"] = [*feed_params.get("css_classes", []), "agent-chat-document"]
        if self.readonly:
            self._feed = pmui.ChatFeed(**feed_params)
        else:
            self._feed = pmui.ChatInterface(
                on_submit=self._on_submit, show_rerun=False, show_undo=False, show_clear=False,
                input_params={"placeholder": self.placeholder, "enable_upload": False}, **feed_params,
            )
        self._model_picker: ModelPicker | None = None
        if not self.readonly:
            self._feed._widget.on_action("stop", self._on_stop)
            self._model_picker = ModelPicker(
                models=self.models, value=self.model_id, runner=self.runner, remember=self.remember_model,
            )
            # pmui pulls the footer into the input's 8px bottom padding, so the picker's text
            # needs 14px more (16px less its button's 2px padding) to sit as far from the
            # bottom edge as the text is from the top (8px padding and 8px in the textarea).
            self._model_picker.__panel__().margin = (0, 0, 14, 0)
            widget = self._feed._widget
            widget.stylesheets = [*widget.stylesheets, _FOOTER_CSS]
            self.model_id = self._model_picker.value
            self._model_picker.param.watch(lambda e: setattr(self, "model_id", e.new), "value")
            # Below the text like the model picker, whose 14px bottom margin it shares.
            self._mention_chips = MentionChips(margin=(0, 0, 14, 0), styles={"flex-shrink": "0"}, css_classes=["agent-chat-mention"])
            self._mention_chips.on_msg(self._on_mention_msg)
            self._sync_footer()
            self.param.watch(self._sync_model_picker, ["models", "model_id", "remember_model"])
            self.param.watch(lambda e: self._sync_footer(), "mentions")
        # Timeline items in order; only the ones shown get a view and a message.
        self._rendered: list[Any] = []
        self._index: dict[int, int] = {}
        self._run_items: dict[str | None, list[Any]] = {}
        self._views: dict[int, Any] = {}
        self._messages: dict[int, pmui.ChatMessage] = {}
        # Steps (tool and model calls) that are not shown are built once they are, e.g. when their group expands.
        self._unbuilt: dict[int, _Step] = {}
        # Messages shown or hidden since the feed was last laid out. Hidden messages are kept out of the
        # feed: its virtualization counts them, and once a window of them shifts the feed re-renders
        # every child, blanking the group rows and links that React renders for a moment.
        self._toggled: dict[int, tuple[pmui.ChatMessage, Any]] = {}
        # Hidden messages still in the feed, taken out once their run has finished, see _drop_settled.
        self._leaving: dict[int, tuple[pmui.ChatMessage, Any]] = {}
        self._hooked: set[int] = set()
        self._built_models: dict[int, ModelCallState] = {}
        # Step groups keyed by their first step, and the group of each folded step.
        self._groups: dict[int, _StepGroup] = {}
        self._group_of: dict[int, _StepGroup] = {}
        self._watchers: list[tuple[param.Parameterized, Any]] = []
        self._footers: dict[str, pmui.ChatMessage] = {}
        self._run_watchers: dict[str, tuple[param.Parameterized, Any]] = {}
        self._cells: dict[str, ArtifactCell] = {}
        self._cell_rows: dict[str, _CellRow] = {}
        self._usage_rows: dict[str, Any] = {}
        self._store_watcher: tuple[param.Parameterized, Any] | None = None
        self.state.param.watch(self._update_items, "items")
        self.state.param.watch(self._watch_runs, "runs")
        self._watch_runs()
        self.state.param.watch(self._update_running, "running")
        self._update_items()
        self._update_running()
        # After the items, so the rows of cells of a replayed state follow their runs' messages.
        self.state.param.watch(self._watch_store, "artifacts")
        self._watch_store()

    def __panel__(self):
        return self._feed

    @property
    def feed(self) -> pmui.ChatFeed:
        return self._feed

    @property
    def cells(self) -> dict[str, ArtifactCell]:
        """The inline artifact cells built so far, keyed by artifact id."""
        return dict(self._cells)

    def refresh_actions(self):
        """Evaluate the inline artifacts' actions again, e.g. after the host's state changed."""
        for cell in self._cells.values():
            cell.refresh()

    @property
    def model_picker(self) -> ModelPicker | None:
        """The menu of ``models`` below the input; None for a readonly chat."""
        return self._model_picker

    def _sync_model_picker(self, *events):
        picker = self._model_picker
        if picker is None:
            return
        names = {"models": "models", "model_id": "value", "remember_model": "remember"}
        picker.param.update({
            names[e.name]: e.new for e in events if getattr(picker, names[e.name]) is not e.new
        })
        self._sync_footer()

    def _sync_footer(self):
        chips = self._mention_chips
        mentions = [{"label": m.label, "icon": m.icon} for m in self._unique_mentions]
        if chips.mentions != mentions:
            chips.mentions = mentions
        # An empty footer row still takes space in the input and pushes the text off center.
        footer = [*([chips] if mentions else []), *([self._model_picker.__panel__()] if self._model_picker.choices else [])]
        if self._feed._widget.footer_objects != footer:
            self._feed._widget.footer_objects = footer

    @property
    def _unique_mentions(self) -> list[Mention]:
        return list(dict.fromkeys(self.mentions))

    def _on_mention_msg(self, event):
        msg, mentions = event.data, self._unique_mentions
        index = msg.get("index")
        if msg.get("type") == "remove" and isinstance(index, int) and 0 <= index < len(mentions):
            self.mentions = [m for m in self.mentions if m != mentions[index]]

    def view_for(self, item: MessageState | ModelCallState | ToolCallState) -> Any:
        """
        The component rendering a timeline item. Steps that are not shown,
        such as calls folded into a collapsed group, are only built when
        they are shown or asked for here.
        """
        if id(item) in self._unbuilt:
            self._materialize([item])
        return self._views[id(item)]

    async def submit(self, text: str):
        """Submit text as the user would through the input, after the ``mentions``, which it clears."""
        text = text.strip()
        if not text or self.state.running:
            return
        if self.mentions and not self.readonly:
            text = " ".join([*dict.fromkeys(mention.written for mention in self.mentions), text])
            self.mentions = []
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
        old = self._rendered
        if len(items) >= len(old) and all(a is b for a, b in zip(items, old, strict=False)):
            new = items[len(old):]
            tail_from = len(old)
            for item in new:
                self._index[id(item)] = len(self._rendered)
                self._rendered.append(item)
                self._run_items.setdefault(getattr(item, "run_id", None), []).append(item)
        else:
            # Model calls recorded after their response are inserted before it rather than appended.
            new, j = [], 0
            for item in items:
                if j < len(old) and item is old[j]:
                    j += 1
                else:
                    new.append(item)
            if j < len(old):
                self._clear()
                new = list(items)
                tail_from = 0
            else:
                tail_from = next(i for i in range(len(items) - 1, -1, -1) if items[i] is old[-1]) + 1
            self._rendered = list(items)
            self._index = {id(item): i for i, item in enumerate(items)}
            self._run_items = {}
            for item in items:
                self._run_items.setdefault(getattr(item, "run_id", None), []).append(item)
        for item in new:
            if isinstance(item, _STEP_TYPES):
                # Whether a step is built now depends on the grouping, which needs every new item.
                self._unbuilt[id(item)] = item
                self._watch(item, self._regroup, "status")
            elif (message := self._build(item)) is not None:
                self._own(item, message)
        self._regroup(new=new, tail_from=tail_from)
        if self._store_watcher is not None and not self._cell_rows:
            # A cleared feed lays out the rows of cells again.
            self._update_cells()

    def _clear(self):
        for obj, watcher in self._watchers:
            obj.param.unwatch(watcher)
        for view in self._views.values():
            if isinstance(view, ToolCallCard):
                view._release()
        self._watchers = []
        self._rendered = []
        self._index = {}
        self._run_items = {}
        self._views = {}
        self._messages = {}
        self._toggled = {}
        self._leaving = {}
        self._hooked = set()
        self._unbuilt = {}
        self._built_models = {}
        self._groups = {}
        self._group_of = {}
        self._footers = {}
        for row in self._cell_rows.values():
            row.release()
        self._cells = {}
        self._cell_rows = {}
        self._usage_rows = {}
        self._feed.clear()

    def _watch(self, obj, callback, names):
        self._watchers.append((obj, obj.param.watch(callback, names)))

    def _own(self, item, message: pmui.ChatMessage):
        self._messages[id(item)] = message

        def toggled(*_):
            self._toggled[id(message)] = (message, item)

        self._watch(message, toggled, "visible")

    def _agent_name(self, run_id: str | None) -> str:
        run = self.state.runs.get(run_id) if run_id else None
        return (run.agent_name if run is not None else None) or self.assistant

    def _message_params(self, user: str, created_at: float | None, compact: bool = False, **own) -> dict[str, Any]:
        """The parameters of a message: the chat's, then ``message_params``, then the message's ``own``."""
        placement = "left"
        if user == self.user:
            placement = self.user_placement or ("right" if self.layout == "chat" else "left")
        params: dict[str, Any] = {
            "user": user,
            "placement": placement,
            "show_reaction_icons": False,
            "show_copy_icon": False,
            # The hover timestamp row adds a gap that would space out runs of tool calls.
            "show_timestamp": not compact,
        }
        if created_at is not None:
            params["timestamp"] = dt.datetime.fromtimestamp(created_at)
        if user in self.avatars:
            params["avatar"] = self.avatars[user]
        if self.layout == "document" and placement == "left":
            params["sizing_mode"] = "stretch_width"
        extra = {k: v for k, v in self.message_params.items() if k != "visible"}
        if "css_classes" in own:
            own["css_classes"] = [*extra.get("css_classes", []), *own["css_classes"]]
        return {**params, **extra, **own}

    @param.depends("user_placement", "message_params", watch=True)
    def _rebuild(self):
        """Build the feed's messages again, so a change reaches the ones already shown."""
        self._clear()
        self._update_items()

    def _build_step(self, item: _Step) -> pmui.ChatMessage:
        params = self._message_params(
            self._agent_name(item.run_id), item.started_at, compact=True, visible=self._step_visible(item),
        )
        if isinstance(item, ModelCallState):
            card = ModelCallCard(item, state=self.state)
            self._built_models[id(item)] = item
        else:
            card = ToolCallCard(
                item, state=self.state, approval_factory=self._approval_factory(),
                label=self.tool_labels.get(item.tool_name), show_artifact=self._chip_shown,
            )
            self._watch(self, lambda e, c=card: setattr(c, "approval_factory", self._approval_factory()), ["approval_factory", "can_approve"])
            self._watch(self, lambda e, c=card, i=item: setattr(c, "label", self.tool_labels.get(i.tool_name)), "tool_labels")
        self._views[id(item)] = card
        message = pmui.ChatMessage(card, **params)
        self._own(item, message)
        self._unbuilt.pop(id(item), None)
        return message

    def _step_visible(self, item: _Step) -> bool:
        if (group := self._group_of.get(id(item))) is not None:
            return group.expanded
        if isinstance(item, ModelCallState):
            return self._show_lone_model_call(item)
        return True

    def _build(self, item) -> pmui.ChatMessage | None:
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
        message = pmui.ChatMessage(pane, **self._message_params(user, item.created_at, visible=bool(item.content)))
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

        params = self._message_params(self._agent_name(item.run_id), item.created_at, compact=True, visible=_show_thinking(item))
        message = pmui.ChatMessage(step, **params)

        def update(*_):
            pane.object = item.content
            step.status = "success" if item.finished else "running"
            # Some providers only return encrypted reasoning, which leaves nothing to show.
            shown = _show_thinking(item)
            if message.visible != shown:
                message.visible = shown
                self._regroup()

        self._watch(item, update, ["content", "finished"])
        return message

    def _foldable(self, item: _Step) -> bool:
        if isinstance(item, ModelCallState):
            return item.status == "success"
        if item.status != "success":
            return False
        if item.approval is not None and item.approval.status != "approved":
            return False
        # A call is shown for its artifacts' chips; artifacts shown inline as cells need none.
        return not any(self._chip_shown(a) for a, _ in self.state.artifacts.published_by(item.tool_call_id))

    def _chip_shown(self, artifact: Artifact) -> bool:
        return self.layout != "document" or not self._inline(artifact)

    def _breaks_group(self, item) -> bool:
        message = self._messages.get(id(item))
        return message is not None and message.visible

    def _fold_runs(self) -> dict[int, list[_Step]]:
        """Runs of two or more untroubled steps of one run, keyed by their first step."""
        runs: list[list[_Step]] = []
        current: list[_Step] = []
        group, models = self.group_tool_calls, self.show_model_calls
        for item in self._rendered:
            if isinstance(item, ModelCallState) and not models:
                continue
            if isinstance(item, _STEP_TYPES):
                foldable = group and self._foldable(item)
                if foldable and (not current or current[-1].run_id == item.run_id):
                    current.append(item)
                    continue
                runs.append(current)
                current = [item] if foldable else []
            elif self._breaks_group(item):
                runs.append(current)
                current = []
        runs.append(current)
        return {id(run[0]): run for run in runs if len(run) > 1}

    @param.depends("group_tool_calls", "show_model_calls", "show_usage", "on_show_run", watch=True)
    def _regroup(self, *_, new=(), tail_from=None):
        """Fold runs of two or more untroubled tool and model calls of one run into a step group."""
        wanted = self._fold_runs()
        stale = [key for key in self._groups if key not in wanted]
        for key in stale:
            self._groups[key].release()
        added, group_of = [], {}
        for key, calls in wanted.items():
            group = self._groups.get(key)
            if group is None:
                group = self._groups[key] = _StepGroup(self, self._agent_name(calls[0].run_id))
                added.append(group)
            for call in calls:
                group_of[id(call)] = group
        self._group_of = group_of
        for key, calls in wanted.items():
            self._groups[key].update(calls)
        for key, item in self._built_models.items():
            if key not in group_of:
                self._messages[key].visible = self._show_lone_model_call(item)
        shown = [item for item in self._unbuilt.values() if self._step_visible(item)]
        for item in shown:
            self._build_step(item)
        built = {id(item) for item in shown}
        placed = [item for item in new if id(item) in self._messages and id(item) not in built] + shown
        self._place(placed, tail_from, added, [self._groups.pop(key) for key in stale])
        self._drop_settled()
        self._update_usage()

    def _materialize(self, items: list[_Step]):
        """Build steps that are not shown yet, e.g. when their group expands."""
        for item in items:
            self._build_step(item)
        self._place(items, None, (), ())

    def _place(self, items, tail_from, added, removed):
        """
        Put the messages of newly built and newly shown items and new
        group rows into the feed and take removed rows out, assigning the
        feed's objects at most once. Hidden messages are noted for
        _drop_settled to take out. Items from
        ``tail_from`` on are appended, like ChatFeed.send, after anything
        else in the feed.
        """
        toggled, self._toggled = self._toggled, {}
        if not (items or added or removed or toggled):
            return
        objects = list(self._feed.objects)
        for key, entry in toggled.items():
            if entry[0].visible:
                self._leaving.pop(key, None)
            else:
                self._leaving[key] = entry
        changed = False
        present = {id(obj) for obj in objects}
        tail_from = len(self._rendered) if tail_from is None else tail_from
        # A running run's messages go in hidden as well, so they are in place before their cards change.
        shown = {id(item): item for item in items if self._messages[id(item)].visible or not self._settled(item)}
        for message, item in toggled.values():
            if message.visible and id(message) not in present:
                shown[id(item)] = item
        inserted = []
        for item in sorted(shown.values(), key=lambda i: self._index[id(i)]):
            message = self._messages[id(item)]
            index = self._index[id(item)]
            position = len(objects) if index >= tail_from else self._position(objects, present, index)
            # A run's row of cells ends the run, as it does when the run is replayed.
            row = self._cell_rows.get(root_run_id(self.state.runs, getattr(item, "run_id", None)))
            if row is not None and id(row.message) in present:
                position = min(position, objects.index(row.message))
            objects.insert(position, message)
            present.add(id(message))
            inserted.append(message)
            if not message.visible:
                self._leaving[id(message)] = (message, item)
        for group in added:
            first = self._messages.get(id(group.calls[0]))
            if first is not None and id(first) in present:
                position = objects.index(first)
            else:
                position = self._position(objects, present, self._index[id(group.calls[0])], own_row=False)
            row = self._cell_rows.get(root_run_id(self.state.runs, group.calls[0].run_id))
            if row is not None and id(row.message) in present:
                position = min(position, objects.index(row.message))
            objects.insert(position, group.message)
            present.add(id(group.message))
        for group in removed:
            if id(group.message) in present:
                objects.remove(group.message)
                changed = True
        if not (changed or inserted or added):
            return
        self._feed.objects = objects
        for message in inserted:
            if id(message) not in self._hooked:
                self._hooked.add(id(message))
                self._feed._run_post_hook(message)

    def _settled(self, item) -> bool:
        run = self.state.runs.get(root_run_id(self.state.runs, getattr(item, "run_id", None)))
        return run is None or run.status != "running"

    def _drop_settled(self):
        """
        Take the hidden messages of finished runs out of the feed. A running
        run's messages stay until it ends: their cards keep changing, and a
        change to a message's frontend model sent in the update that takes
        the message out reaches a frontend that no longer knows it. In a
        served session they are taken out in the next update for the same
        reason.
        """
        if not any(self._settled(item) for _, item in self._leaving.values()):
            return
        doc = pn.state.curdoc
        if doc is None or doc.session_context is None:
            self._drop_hidden()
        else:
            # Each drop takes out what is due when it runs, so one more than needed is harmless.
            def drop():
                with set_curdoc(doc):
                    self._drop_hidden()

            doc.add_next_tick_callback(drop)

    def _drop_hidden(self):
        drop = {
            key for key, (message, item) in self._leaving.items()
            if not message.visible and self._settled(item)
        }
        for key in drop:
            del self._leaving[key]
        objects = [obj for obj in self._feed.objects if id(obj) not in drop]
        if len(objects) != len(self._feed.objects):
            self._feed.objects = objects

    def _position(self, objects, present, index: int, own_row: bool = True) -> int:
        """Where a message for the item at ``index`` goes: after what earlier items put in the feed."""
        runs = self.state.runs
        own = root_run_id(runs, getattr(self._rendered[index], "run_id", None))
        for j in range(index, -1, -1):
            item = self._rendered[j]
            # An earlier run ends with its row of cells.
            if j < index and (run := root_run_id(runs, getattr(item, "run_id", None))) != own:
                row = self._cell_rows.get(run)
                if row is not None and id(row.message) in present:
                    return objects.index(row.message) + 1
            message = self._messages.get(id(item)) if j < index else None
            if message is not None and id(message) in present:
                return objects.index(message) + 1
            group = self._groups.get(id(item)) if j < index or own_row else None
            if group is not None and id(group.message) in present:
                return objects.index(group.message) + 1
        owned = {id(m) for m in self._messages.values()} | {id(g.message) for g in self._groups.values()}
        return next((k for k, obj in enumerate(objects) if id(obj) in owned), len(objects))

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

    def _watch_store(self, *_):
        if self._store_watcher is not None:
            store, watcher = self._store_watcher
            store.param.unwatch(watcher)
        store = self.state.artifacts
        self._store_watcher = (store, store.param.watch(self._update_cells, "artifacts"))
        self._update_cells()

    def _inline(self, artifact: Artifact) -> bool:
        if self.inline_artifacts is not None:
            return bool(self.inline_artifacts(artifact))
        return not artifact.is_document

    def _inline_by_run(self) -> dict[str, list[Artifact]]:
        """The inline artifacts of each top-level run, in the order they were first published."""
        runs = self.state.runs
        found: dict[str, list[Artifact]] = {}
        for artifact in self.state.artifacts:
            first = artifact.versions[0] if artifact.versions else None
            run_id = root_run_id(runs, first.run_id) if first is not None else None
            if run_id is not None and self._inline(artifact):
                found.setdefault(run_id, []).append(artifact)
        return found

    def _update_cells(self, *_):
        """
        Add the cells of newly published inline artifacts. A run's row of
        cells is put after the run's messages shown when its first artifact
        is published and stays there, while the run's later messages go in
        before it, so the cells end the run as they do in a replay. Later
        cells are appended to the row, so no cell is moved or rendered again.
        """
        if self.layout != "document":
            return
        for run_id, artifacts in self._inline_by_run().items():
            row = self._cell_rows.get(run_id)
            if row is None:
                row = self._cell_rows[run_id] = _CellRow(self, run_id)
                objects = list(self._feed.objects)
                objects.insert(self._row_position(objects, run_id), row.message)
                self._feed.objects = objects
            row.update(artifacts)

    def _row_position(self, objects: list, run_id: str) -> int:
        """Where a run's row of cells goes: after the last of the run's messages and group rows in the feed."""
        present = {id(obj): k for k, obj in enumerate(objects)}
        for item in reversed(self._run_items.get(run_id, ())):
            for message in (self._messages.get(id(item)), getattr(self._groups.get(id(item)), "message", None)):
                if message is not None and id(message) in present:
                    return present[id(message)] + 1
        return len(objects)

    def _make_cell(self, artifact: Artifact) -> ArtifactCell:
        cell = self._cells[artifact.artifact_id] = ArtifactCell(
            artifact, renderer=self.cell_renderer, actions=self.action_registry,
            artifact_actions=self.artifact_actions, styles={"flex": "1 1 420px", "min-width": "0"},
        )
        return cell

    @param.depends("cell_renderer", "action_registry", "artifact_actions", watch=True)
    def _propagate_cells(self):
        for cell in self._cells.values():
            cell.param.update(
                renderer=self.cell_renderer, actions=self.action_registry, artifact_actions=self.artifact_actions,
            )

    @param.depends("inline_artifacts", "max_cells", watch=True)
    def _update_inline(self):
        # Which artifacts are inline changed, so the rows are laid out anew.
        for row in self._cell_rows.values():
            row.release()
        self._cell_rows = {}
        self._cells = {}
        self._feed.objects = [o for o in self._feed.objects if "run-artifacts-message" not in o.css_classes]
        self._update_cells()

    def _update_usage(self):
        """Show each finished run's usage totals under the last message of the run that is shown."""
        wanted = {}
        linked = self.on_show_run is not None
        for run_id, run in self.state.runs.items():
            if run.status == "running" or run.parent_run_id is not None:
                continue
            if (self.show_usage and run.usage) or linked:
                if (message := self._last_shown(run_id)) is not None:
                    wanted[run_id] = message
        for run_id, message in list(self._footers.items()):
            if wanted.get(run_id) is not message:
                message.footer_objects = []
                del self._footers[run_id]
        for run_id, message in wanted.items():
            row = self._usage_row(run_id, linked)
            if not message.footer_objects or message.footer_objects[0] is not row:
                message.footer_objects = [row]
            self._footers[run_id] = message

    def _usage_row(self, run_id: str, linked: bool):
        run = self.state.runs[run_id]
        footer = (
            f'<div class="usage" title="{escape(usage_tooltip(run), quote=True)}">{escape(usage_summary(run))}</div>'
            if self.show_usage and run.usage else ""
        )
        row = self._usage_rows.get(run_id)
        if row is not None and row._usage == (footer, linked):
            return row
        usage = pn.pane.HTML(
            footer, margin=(0, 10), stylesheets=[USAGE_CSS], css_classes=["run-usage"], visible=bool(footer),
        )
        if linked:
            link = pmui.Button(
                label="Details", variant="text", size="small", margin=0, css_classes=["run-details"],
                description="How this answer was made: every model request and tool call of the run",
                sx={"minWidth": 0, "py": 0, "px": "4px", "fontSize": "0.75rem", "textTransform": "none"},
            )
            link.on_click(lambda _, rid=run_id: self._show_run(rid))
            row = pn.Row(usage, link, margin=0, styles={"align-items": "center"})
        else:
            row = usage
        row._usage = (footer, linked)
        self._usage_rows[run_id] = row
        return row

    def _show_run(self, run_id: str):
        if self.on_show_run is not None:
            self.on_show_run(run_id)

    def _last_shown(self, run_id: str) -> pmui.ChatMessage | None:
        """The run's last visible message or group row in the feed, leaving out user messages."""
        for item in reversed(self._run_items.get(run_id, ())):
            message = self._messages.get(id(item))
            if message is not None and message.visible and not (isinstance(item, MessageState) and item.role == "user"):
                return message
            group = self._groups.get(id(item))
            if group is not None and group.message.visible:
                return group.message
        return None

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
    """The folded row standing in for a run of tool and model call messages."""

    def __init__(self, chat: AgentChat, user: str):
        self.chat = chat
        self.expanded = False
        self.button = pmui.Button(
            variant="text", size="small", icon="expand_more", margin=0,
            sx={"textTransform": "none", "fontWeight": 400, "justifyContent": "flex-start", "color": "text.secondary"},
            on_click=lambda e: self.toggle(),
        )
        self.message = pmui.ChatMessage(self.button, **chat._message_params(user, None, compact=True))
        self.calls: list[_Step] = []

    @property
    def members(self) -> list[pmui.ChatMessage]:
        """The messages of the steps built so far."""
        return [m for c in self.calls if (m := self.chat._messages.get(id(c))) is not None]

    def update(self, calls: list[_Step]):
        if len(calls) == len(self.calls) and all(a is b for a, b in zip(calls, self.calls, strict=True)):
            return
        kept = {id(c) for c in calls}
        left = [c for c in self.calls if id(c) not in kept]
        self.calls = calls
        for call in left:
            if (message := self.chat._messages.get(id(call))) is not None:
                message.visible = self.chat._step_visible(call)
        self._sync()

    def toggle(self):
        self.expanded = not self.expanded
        built = [c for c in self.calls if id(c) in self.chat._unbuilt] if self.expanded else []
        for call in built:
            self.chat._build_step(call)
        self._sync()
        self.chat._place(built, None, (), ())
        self.chat._drop_settled()
        self.chat._update_usage()

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


class _CellRow:
    """
    The cells of one run's inline artifacts, in a message after the run's
    messages. Cells are only ever appended; past ``max_cells`` the rest
    wait behind a "Show more" button and are built when it is pressed.
    """

    def __init__(self, chat: AgentChat, run_id: str):
        self.chat = chat
        self.cells: list[ArtifactCell] = []
        self.pending: list[Artifact] = []
        self.expanded = False
        # The margin lines the cells up with the text of the messages.
        self.flex = pn.FlexBox(
            sizing_mode="stretch_width", margin=(0, 10), gap="12px", align_items="flex-start",
            css_classes=["run-artifacts"],
        )
        self.more = pmui.Button(
            variant="text", size="small", icon="expand_more", visible=False, margin=(4, 4),
            sx={"textTransform": "none"}, css_classes=["run-artifacts-more"],
        )
        self.more.on_click(lambda _: self.expand())
        self.message = pmui.ChatMessage(
            pn.Column(self.flex, self.more, sizing_mode="stretch_width", margin=0),
            **chat._message_params(chat._agent_name(run_id), None, compact=True, css_classes=["run-artifacts-message"]),
        )

    def update(self, artifacts: list[Artifact]):
        known = {cell.artifact.artifact_id for cell in self.cells} | {a.artifact_id for a in self.pending}
        new = [a for a in artifacts if a.artifact_id not in known]
        if not new:
            return
        built = []
        for artifact in new:
            if self.expanded or (not self.pending and len(self.cells) + len(built) < self.chat.max_cells):
                built.append(self.chat._make_cell(artifact))
            else:
                self.pending.append(artifact)
        self._add(built)

    def expand(self):
        self.expanded = True
        pending, self.pending = self.pending, []
        self._add([self.chat._make_cell(artifact) for artifact in pending])

    def _add(self, built: list[ArtifactCell]):
        if built:
            self.cells += built
            self.flex.extend([cell.__panel__() for cell in built])
        n = len(self.pending)
        self.more.param.update(label=f"Show {n} more", visible=bool(n))

    def release(self):
        for cell in self.cells:
            cell.release()


def _show_thinking(item: MessageState) -> bool:
    return bool(item.content.strip()) or not item.finished


__all__ = ["AgentChat"]
