import panel as pn
import panel_material_ui as pmui
import pytest
from panel.models.esm import DataEvent

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat, Mention
from panel_agents.components.model_call import ModelCallCard
from panel_agents.components.tool_call import ToolCallCard
from panel_agents.state import ModelCallState


def run_events():
    return [
        ev.RunStarted(run_id="r1", agent_name="SQL Analyst"),
        ev.TextMessageStart(message_id="u1", role="user", run_id="r1"),
        ev.TextMessageContent(message_id="u1", delta="Revenue by region?", run_id="r1"),
        ev.TextMessageEnd(message_id="u1", run_id="r1"),
        ev.ThinkingStart(message_id="th", run_id="r1"),
        ev.ThinkingContent(message_id="th", delta="Query the sales table.", run_id="r1"),
        ev.ThinkingEnd(message_id="th", run_id="r1"),
        ev.ToolCallStart(tool_call_id="t1", tool_name="run_query", run_id="r1"),
        ev.ToolCallArgs(tool_call_id="t1", delta='{"sql": "SELECT 1"}', run_id="r1"),
        ev.ToolCallEnd(tool_call_id="t1", run_id="r1"),
        ev.ToolCallResult(tool_call_id="t1", content=[[1]], run_id="r1"),
        ev.TextMessageStart(message_id="m1", run_id="r1"),
        ev.TextMessageContent(message_id="m1", delta="EMEA leads.", run_id="r1"),
        ev.TextMessageEnd(message_id="m1", run_id="r1"),
        ev.RunFinished(run_id="r1"),
    ]


def test_renders_items_in_timeline_order():
    state = RunState.from_events(run_events())
    chat = AgentChat(state)
    messages = chat.feed.objects
    assert len(messages) == 4
    user, thinking, tool, answer = messages
    assert user.user == "User" and user.placement == "right"
    assert user.object.object == "Revenue by region?"
    assert isinstance(thinking.object, pmui.ChatStep)
    assert thinking.object.collapsed
    assert thinking.object.status == "success"
    assert thinking.user == "SQL Analyst"
    assert tool.object is chat.view_for(state.tool_calls["t1"])
    assert isinstance(chat.view_for(state.tool_calls["t1"]), ToolCallCard)
    assert answer.object.object == "EMEA leads."
    assert answer.placement == "left"


def test_streams_into_existing_message():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    state.apply(ev.RunStarted(run_id="r1"))
    state.apply(ev.TextMessageStart(message_id="m1", run_id="r1"))
    message = chat.view_for(state.messages["m1"])
    pane = message.object
    # A running run's message goes in hidden, so it is in place when it starts streaming.
    assert not message.visible
    assert chat.feed.objects == [message]
    state.apply(ev.TextMessageContent(message_id="m1", delta="Hel", run_id="r1"))  # codespell:ignore
    assert message.visible
    state.apply(ev.TextMessageContent(message_id="m1", delta="lo", run_id="r1"))
    assert chat.feed.objects == [message]
    assert message.object is pane
    assert pane.object == "Hello"


def test_thinking_updates_status():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    state.apply(ev.ThinkingStart(message_id="th"))
    step = chat.feed.objects[0].object
    assert step.status == "running"
    assert step.title == "Thinking…"
    state.apply(ev.ThinkingContent(message_id="th", delta="hmm"))
    state.apply(ev.ThinkingEnd(message_id="th"))
    assert step.status == "success"
    assert step.objects[0].object == "hmm"


def test_appends_only_new_items():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    state.add_user_message("hi")
    first = chat.feed.objects[0]
    state.apply(ev.ToolCallStart(tool_call_id="t1", tool_name="f"))
    assert chat.feed.objects[0] is first
    assert len(chat.feed.objects) == 2


def test_reset_clears_feed():
    state = RunState.from_events(run_events())
    chat = AgentChat(state)
    state.reset()
    assert chat.feed.objects == []
    state.add_user_message("again")
    assert len(chat.feed.objects) == 1


def test_only_sending_is_disabled_while_running():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    assert isinstance(chat.feed, pmui.ChatInterface)
    widget = chat.feed._widget
    assert not widget.disabled_enter
    state.apply(ev.RunStarted(run_id="r1"))
    assert widget.disabled_enter
    assert not chat.feed.disabled and not widget.disabled
    state.apply(ev.RunFinished(run_id="r1"))
    assert not widget.disabled_enter


def test_readonly_has_no_input():
    chat = AgentChat(RunState.from_events(run_events()), readonly=True)
    assert type(chat.feed) is pmui.ChatFeed
    assert len(chat.feed.objects) == 4


async def test_submit_adds_message_and_invokes_callback():
    state = RunState(flush_interval=0)
    received = []

    async def callback(text):
        received.append(text)
        state.apply(ev.RunStarted(run_id="r1"))
        state.apply(ev.TextMessageStart(message_id="m1", run_id="r1"))
        state.apply(ev.TextMessageContent(message_id="m1", delta="Hi!", run_id="r1"))
        state.apply(ev.TextMessageEnd(message_id="m1", run_id="r1"))
        state.apply(ev.RunFinished(run_id="r1"))

    chat = AgentChat(state, callback=callback)
    await chat.submit("  hello ")
    assert received == ["hello"]
    assert [m.object.object for m in chat.feed.objects] == ["hello", "Hi!"]


async def test_submit_without_echo():
    state = RunState(flush_interval=0)
    received = []
    chat = AgentChat(state, callback=received.append, add_user_message=False)
    await chat.submit("hello")
    assert received == ["hello"]
    assert chat.feed.objects == []


async def test_submit_ignored_while_running():
    state = RunState(flush_interval=0)
    received = []
    chat = AgentChat(state, callback=received.append)
    state.apply(ev.RunStarted(run_id="r1"))
    await chat.submit("hello")
    assert received == []


def test_approval_factory_reaches_cards():
    state = RunState(flush_interval=0)
    chat = AgentChat(state, approval_factory=lambda request, run_state: pn.pane.Markdown(request.tool_name))
    state.apply(ev.ApprovalRequested(tool_call_id="t1", tool_name="drop_table"))
    card = chat.view_for(state.tool_calls["t1"])
    assert card._approval[0].object == "drop_table"
    chat.approval_factory = None
    assert isinstance(card._approval[0], pmui.Alert)


@pytest.mark.parametrize("readonly", [True, False])
def test_renders_to_model(readonly, document, comm):
    chat = AgentChat(RunState.from_events(run_events()), readonly=readonly)
    model = chat.__panel__().get_root(document, comm)
    assert model is not None


async def test_runner_adds_user_message_once():
    class Runner:
        def __init__(self):
            self.state = RunState(flush_interval=0)
            self.prompts = []

        async def run(self, text):
            self.prompts.append(text)
            self.state.add_user_message(text)

    runner = Runner()
    chat = AgentChat(runner=runner)
    assert chat.state is runner.state
    await chat.submit("hello")
    assert runner.prompts == ["hello"]
    assert [m.content for m in runner.state.items] == ["hello"]


def test_approval_card_is_default():
    from panel_agents.components.approval import approval_factory

    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    state.apply(ev.ApprovalRequested(tool_call_id="t1", tool_name="drop_table", args={"table": "a"}))
    card = chat.view_for(state.tool_calls["t1"])
    assert card.approval_factory is approval_factory
    assert card._approval.visible and len(card._approval) == 1


def test_empty_thinking_is_hidden_once_finished():
    state = RunState(flush_interval=0)
    chat = AgentChat(state, readonly=True)
    state.apply(ev.ThinkingStart(message_id="t"))
    message = chat.feed.objects[-1]
    assert message.visible
    state.apply(ev.ThinkingEnd(message_id="t"))
    assert not message.visible


def calls(state, *specs, run_id="r1"):
    for call_id, name, outcome in specs:
        state.apply_many([
            ev.ToolCallStart(tool_call_id=call_id, tool_name=name, run_id=run_id, timestamp=100.0),
            ev.ToolCallArgs(tool_call_id=call_id, delta='{"name": "kpi"}', run_id=run_id),
            ev.ToolCallEnd(tool_call_id=call_id, run_id=run_id),
        ])
        if outcome is not None:
            state.apply(ev.ToolCallResult(
                tool_call_id=call_id, content=outcome, is_error=outcome == "boom", run_id=run_id, timestamp=101.0,
            ))


def group_rows(chat):
    return [g for g in chat._groups.values() if g.message in chat.feed.objects]


def test_finished_calls_fold_into_one_row_that_expands():
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1"))
    chat = AgentChat(state)
    calls(state, ("a", "read", "ok"), ("b", "list", "ok"), ("c", "check", "ok"))
    (group,) = group_rows(chat)
    assert group.button.label == "3 steps · 3.0s"
    assert group.button.description == "3 steps, 3.0s, expand"
    members = [chat._messages[id(state.tool_calls[c])] for c in "abc"]
    assert not any(m.visible for m in members)
    # Folded calls leave the feed, whose virtualization would count them, once their run has ended.
    assert all(m in chat.feed.objects for m in members)
    state.apply(ev.RunFinished(run_id="r1"))
    assert not any(m in chat.feed.objects for m in members)
    group.toggle()
    assert all(m.visible for m in members)
    row = chat.feed.objects.index(group.message)
    assert chat.feed.objects[row + 1:row + 4] == members
    assert group.button.description.endswith("collapse")
    group.toggle()
    assert chat.feed.objects == [group.message]


def test_failed_running_and_artifact_calls_are_never_folded():
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1"))
    chat = AgentChat(state)
    calls(state, ("a", "read", "ok"), ("b", "write", "boom"), ("c", "read", "ok"), ("d", "list", None))
    assert group_rows(chat) == []
    state.apply(ev.ArtifactPublished(artifact_id="doc", value="x", tool_call_id="e", run_id="r1"))
    calls(state, ("e", "publish", "ok"), ("f", "read", "ok"))
    assert group_rows(chat) == []
    state.apply(ev.ToolCallResult(tool_call_id="d", content="ok", run_id="r1"))
    (group,) = group_rows(chat)
    assert group.calls == [state.tool_calls["c"], state.tool_calls["d"]]


def test_a_visible_message_splits_groups_and_grouping_can_be_off():
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1"))
    chat = AgentChat(state)
    calls(state, ("a", "read", "ok"), ("b", "read", "ok"))
    state.apply_many([
        ev.TextMessageStart(message_id="m", run_id="r1"),
        ev.TextMessageContent(message_id="m", delta="Halfway.", run_id="r1"),
    ])
    calls(state, ("c", "read", "ok"), ("d", "read", "ok"))
    assert len(group_rows(chat)) == 2
    chat.group_tool_calls = False
    assert group_rows(chat) == []
    assert all(chat._messages[id(state.tool_calls[c])].visible for c in "abcd")


def test_tool_labels_name_calls_in_the_users_words():
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1"))
    chat = AgentChat(state, tool_labels={
        "read": "Read the dashboard", "set": lambda args: f"Changed {args['name']}", "bad": lambda args: 1 / 0,
    })
    calls(state, ("a", "read", None), ("b", "set", None), ("c", "bad", None), ("d", "other", None))
    header = lambda c: chat.view_for(state.tool_calls[c])._header.object
    assert "Read the dashboard" in header("a") and "(read)" in header("a") and "tc-preview" not in header("a")
    assert "Changed kpi" in header("b")
    assert "tc-name" in header("c") and "tc-name" in header("d")
    assert chat.view_for(state.tool_calls["a"])._tool_section.visible
    chat.tool_labels = {}
    assert "tc-name" in header("a")


def test_stop_cancels_the_runner():
    class Runner:
        state = RunState(flush_interval=0)
        cancelled = 0

        def run(self, text):
            pass

        def cancel(self):
            self.cancelled += 1

    runner = Runner()
    chat = AgentChat(runner=runner)
    chat._on_stop()
    assert runner.cancelled == 1


def finished_run(run_id, n_calls=3, publish=True, usage=False, answer="EMEA leads."):
    """
    A finished run whose first calls fold into a group and whose last one
    publishes a table. With ``usage`` each call is followed by the Usage of
    the model request that issued it, which the state inserts as a model call.
    """
    events = [
        ev.RunStarted(run_id=run_id),
        ev.TextMessageStart(message_id=f"{run_id}-u", role="user", run_id=run_id),
        ev.TextMessageContent(message_id=f"{run_id}-u", delta="Revenue?", run_id=run_id),
        ev.ThinkingStart(message_id=f"{run_id}-t", run_id=run_id),
        ev.ThinkingContent(message_id=f"{run_id}-t", delta="Look it up.", run_id=run_id),
        ev.ThinkingEnd(message_id=f"{run_id}-t", run_id=run_id),
    ]
    for k in range(n_calls):
        call_id = f"{run_id}-c{k}"
        events += [
            ev.ToolCallStart(tool_call_id=call_id, tool_name="read" if k else "list", run_id=run_id, timestamp=100.0),
            ev.ToolCallArgs(tool_call_id=call_id, delta='{"name": "kpi"}', run_id=run_id),
            ev.ToolCallEnd(tool_call_id=call_id, run_id=run_id),
        ]
        if publish and k == n_calls - 1:
            events.append(ev.ArtifactPublished(artifact_id=f"{run_id}-a", value=[1], tool_call_id=call_id, run_id=run_id))
        events.append(ev.ToolCallResult(tool_call_id=call_id, content="ok", run_id=run_id, timestamp=100.5))
        if usage:
            events.append(ev.Usage(model="m", input_tokens=100, output_tokens=10, duration=0.2, run_id=run_id, timestamp=100.6))
    events += [
        ev.TextMessageStart(message_id=f"{run_id}-m", run_id=run_id),
        ev.TextMessageContent(message_id=f"{run_id}-m", delta=answer, run_id=run_id),
        ev.RunFinished(run_id=run_id),
    ]
    return events


def test_folded_calls_are_built_when_their_group_expands():
    state = RunState.from_events(finished_run("r1"))
    policy = lambda request, user: True
    chat = AgentChat(state, can_approve=policy)
    folded = [state.tool_calls[f"r1-c{k}"] for k in (0, 1)]
    (group,) = group_rows(chat)
    assert group.calls == folded
    assert not any(id(c) in chat._views for c in folded)
    chat.tool_labels = {"read": "Read the report"}
    group.toggle()
    members = [chat._messages[id(c)] for c in folded]
    objects = chat.feed.objects
    row = objects.index(group.message)
    assert objects[row + 1:row + 3] == members
    assert all(m.visible for m in members)
    card = chat.view_for(folded[1])
    assert "Read the report" in card._header.object
    chat.tool_labels = {"read": "Read it"}
    assert "Read it" in card._header.object
    chat.approval_factory = None
    assert card.approval_factory is None
    group.toggle()
    assert not any(m.visible for m in members)


def test_view_for_builds_a_folded_call_hidden_in_place():
    state = RunState.from_events(finished_run("r1"))
    chat = AgentChat(state)
    (group,) = group_rows(chat)
    second = state.tool_calls["r1-c1"]
    card = chat.view_for(second)
    assert isinstance(card, ToolCallCard)
    message = chat._messages[id(second)]
    assert not message.visible
    assert message not in chat.feed.objects
    group.toggle()
    objects = chat.feed.objects
    first = chat._messages[id(state.tool_calls["r1-c0"])]
    assert objects.index(group.message) + 1 == objects.index(first) == objects.index(message) - 1
    assert objects.index(message) < objects.index(chat._messages[id(state.tool_calls["r1-c2"])])


def test_ungrouping_builds_and_shows_folded_calls_in_order():
    state = RunState.from_events(finished_run("r1"))
    chat = AgentChat(state)
    chat.group_tool_calls = False
    assert group_rows(chat) == []
    calls = [chat._messages[id(state.tool_calls[f"r1-c{k}"])] for k in range(3)]
    assert all(m.visible for m in calls)
    objects = chat.feed.objects
    assert [objects.index(m) for m in calls] == [2, 3, 4]
    chat.group_tool_calls = True
    (group,) = group_rows(chat)
    assert chat.feed.objects == [*objects[:2], group.message, *objects[4:]]


class _Counted:
    """Counts ChatMessage constructions and changes to the messages the feed's log renders."""

    def __init__(self, monkeypatch):
        self.messages = 0
        self.assignments = 0
        counter = self

        class ChatMessage(pmui.ChatMessage):
            def __init__(self, *args, **params):
                counter.messages += 1
                super().__init__(*args, **params)

        class ChatFeed(pmui.ChatFeed):
            def __init__(self, *args, **params):
                super().__init__(*args, **params)
                self._chat_log.param.watch(lambda e: setattr(counter, "assignments", counter.assignments + 1), "objects")

        monkeypatch.setattr(pmui, "ChatMessage", ChatMessage)
        monkeypatch.setattr(pmui, "ChatFeed", ChatFeed)

    def reset(self):
        self.messages = self.assignments = 0


@pytest.mark.parametrize("n_runs", [1, 20])
@pytest.mark.parametrize("usage", [False, True])
def test_long_feeds_build_shown_items_once(monkeypatch, n_runs, usage):
    # Per run: user message, thinking, the row of the folded steps, the publishing call and the answer.
    counted = _Counted(monkeypatch)
    cards = []
    monkeypatch.setattr(ToolCallCard, "__init__", lambda self, *a, _init=ToolCallCard.__init__, **k: (cards.append(self), _init(self, *a, **k))[1])
    state = RunState.from_events([e for i in range(n_runs) for e in finished_run(f"r{i}", usage=usage)])
    chat = AgentChat(state, readonly=True)
    assert counted.messages == 5 * n_runs
    assert len(cards) == n_runs
    assert counted.assignments == 1
    assert len(chat.feed.objects) == 5 * n_runs

    counted.reset()
    state.add_user_message("One more")
    assert counted.messages == 1
    assert counted.assignments == 1

    counted.reset()
    first = chat.feed.objects[0]
    state.apply_many(finished_run("next", usage=usage))
    # Steps of a running run are shown, so a live run builds them; model calls inserted
    # before their response go into place without rebuilding the feed.
    assert chat.feed.objects[0] is first
    assert counted.messages == (11 if usage else 7)
    assert len(group_rows(chat)) == n_runs + 1


def test_a_folded_group_of_tool_and_model_calls_expands_and_moves_the_usage_line():
    state = RunState.from_events(finished_run("r1", publish=False, usage=True, answer=""))
    chat = AgentChat(state, readonly=True)
    (group,) = group_rows(chat)
    steps = [state.tool_calls["r1-c0"], *list(state.model_calls.values())[1:], state.tool_calls["r1-c1"], state.tool_calls["r1-c2"]]
    assert set(map(id, group.calls)) == set(map(id, steps))
    assert any(isinstance(c, ModelCallState) for c in group.calls)
    assert not any(id(c) in chat._messages for c in group.calls)
    # The run shows no answer, so its totals sit under the group row.
    assert group.message.footer_objects
    group.toggle()
    members = [chat._messages[id(c)] for c in group.calls]
    objects = chat.feed.objects
    row = objects.index(group.message)
    assert objects[row + 1:row + 1 + len(members)] == members
    assert all(m.visible for m in members)
    assert isinstance(chat.view_for(group.calls[1]), ModelCallCard)
    assert not group.message.footer_objects
    assert members[-1].footer_objects
    group.toggle()
    assert group.message.footer_objects and not members[-1].footer_objects


def test_hidden_model_calls_are_built_when_shown():
    state = RunState.from_events(finished_run("r1", n_calls=1, publish=False, usage=True))
    chat = AgentChat(state, show_model_calls=False)
    (call,) = state.model_calls.values()
    assert id(call) not in chat._messages
    chat.show_model_calls = True
    # A lone request of a finished run is reported by the usage line instead.
    assert id(call) not in chat._messages
    chat.show_usage = False
    assert chat._messages[id(call)].visible
    assert chat._messages[id(call)] in chat.feed.objects


def test_post_hook_runs_for_each_message_put_in_the_feed():
    state = RunState.from_events(finished_run("r1"))
    hooked = []
    chat = AgentChat(state, feed_params={"post_hook": lambda message, feed: hooked.append(message)})
    assert hooked == [m for m in chat.feed.objects if m not in [g.message for g in chat._groups.values()]]
    hooked.clear()
    state.add_user_message("more")
    assert hooked == [chat.feed.objects[-1]]


def test_reset_releases_cards_from_the_artifact_store():
    state = RunState.from_events(finished_run("r1"))
    chat = AgentChat(state)
    card = chat.view_for(state.tool_calls["r1-c2"])
    assert card._artifacts.visible
    state.reset()
    store = state.artifacts
    store.publish("r1-a", 1, tool_call_id="r1-c2")
    assert card._unwatch_artifacts is None
    assert "r1-c2" not in store._call_watchers


def test_replacing_the_artifacts_keeps_the_chat_working():
    state = RunState.from_events(finished_run("r1"))
    chat = AgentChat(state)
    state.artifacts.artifacts = {}
    state.add_user_message("more")
    assert len(group_rows(chat)) == 1 and chat.feed.objects[-1].object.object == "more"


def test_a_model_call_inserted_before_its_response_keeps_the_feed():
    state = RunState(flush_interval=0)
    chat = AgentChat(state, readonly=True)
    state.apply_many([
        ev.RunStarted(run_id="r1"),
        ev.TextMessageStart(message_id="u", role="user", run_id="r1"),
        ev.TextMessageContent(message_id="u", delta="Hi", run_id="r1"),
    ])
    state.apply(ev.Usage(model="m", input_tokens=1, output_tokens=1, run_id="r1"))
    state.apply_many([
        ev.TextMessageStart(message_id="a", run_id="r1"),
        ev.TextMessageContent(message_id="a", delta="Hello", run_id="r1"),
    ])
    views = {id(i): chat.view_for(i) for i in state.items}
    before = list(chat.feed.objects)
    # A Usage without a ModelRequestStarted goes before the response it paid for.
    state.apply(ev.Usage(model="m", input_tokens=2, output_tokens=2, run_id="r1"))
    second = list(state.model_calls.values())[-1]
    assert state.items.index(second) < state.items.index(state.messages["a"])
    assert all(chat.view_for(i) is v for i, v in ((i, views[id(i)]) for i in state.items if id(i) in views))
    objects = chat.feed.objects
    assert [o for o in objects if o in before] == before
    message = chat._messages[id(second)]
    assert objects.index(message) < objects.index(chat._messages[id(state.messages["a"])])


async def test_mentions_show_in_the_input_and_start_the_next_message():
    state = RunState(flush_interval=0)
    received = []
    chat = AgentChat(state, callback=received.append, models=["a", "b"])
    chat.mentions = [Mention("sales_by_region", icon="bar_chart"), Mention("Orders", text="@orders_table")]
    chips, picker = chat.feed._widget.footer_objects
    assert picker is chat.model_picker.__panel__()
    assert chips.mentions == [{"label": "sales_by_region", "icon": "bar_chart"}, {"label": "Orders", "icon": None}]
    await chat.submit("make it a bar chart")
    assert received == ["@sales_by_region @orders_table make it a bar chart"]
    assert chat.mentions == [] and chat.feed._widget.footer_objects == [chat.model_picker.__panel__()]
    assert chat.feed.objects[0].object.object == "@sales_by_region @orders_table make it a bar chart"


async def test_a_mention_is_kept_until_a_message_is_sent():
    state = RunState(flush_interval=0)
    received = []
    chat = AgentChat(state, callback=received.append)
    chat.mentions = [Mention("sales_by_region")]
    await chat.submit("   ")
    state.apply(ev.RunStarted(run_id="r1"))
    await chat.submit("hello")
    assert received == [] and chat.mentions == [Mention("sales_by_region")]


def remove_chip(chat, index):
    chat.feed._widget.footer_objects[0]._process_event(DataEvent(model=None, data={"type": "remove", "index": index}))


def test_removing_a_mention_drops_its_chip():
    chat = AgentChat(RunState(flush_interval=0), callback=lambda text: None)
    chat.mentions = [Mention("a"), Mention("b")]
    chips = chat.feed._widget.footer_objects[0]
    remove_chip(chat, 0)
    assert chat.mentions == [Mention("b")]
    assert chat.feed._widget.footer_objects == [chips] and chips.mentions == [{"label": "b", "icon": None}]
    remove_chip(chat, 0)
    assert chat.mentions == [] and chat.feed._widget.footer_objects == []


def test_removing_a_mention_ignores_an_index_out_of_range():
    chat = AgentChat(RunState(flush_interval=0), callback=lambda text: None)
    chat.mentions = [Mention("a")]
    remove_chip(chat, 1)
    remove_chip(chat, -1)
    assert chat.mentions == [Mention("a")]


async def test_a_repeated_mention_shows_and_is_written_once():
    received = []
    chat = AgentChat(RunState(flush_interval=0), callback=received.append)
    chat.mentions = [Mention("a"), Mention("a"), Mention("a", icon="table")]
    assert chat.feed._widget.footer_objects[0].mentions == [{"label": "a", "icon": None}, {"label": "a", "icon": "table"}]
    remove_chip(chat, 0)
    assert chat.mentions == [Mention("a", icon="table")]
    chat.mentions = [Mention("a"), Mention("a")]
    await chat.submit("x")
    assert received == ["@a x"]


@pytest.mark.parametrize(("mention", "written"), [
    (Mention("sales_by_region"), "@sales_by_region"),
    (Mention("Sales by region"), '@"Sales by region"'),
    (Mention("@odd"), "@odd"),
    (Mention(' say "hi" '), '@"say \\"hi\\""'),
    (Mention("Sales by region", text="@sales"), "@sales"),
])
def test_a_mention_is_written_as_one_name(mention, written):
    assert mention.written == written
