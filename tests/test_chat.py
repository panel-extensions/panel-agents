import panel as pn
import panel_material_ui as pmui
import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat
from panel_agents.components.tool_call import ToolCallCard


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
    (message,) = chat.feed.objects
    pane = message.object
    assert not message.visible
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
    assert chat.feed.objects.index(group.message) < chat.feed.objects.index(chat._messages[id(state.tool_calls["a"])])
    assert not any(chat._messages[id(state.tool_calls[c])].visible for c in "abc")
    group.toggle()
    assert all(chat._messages[id(state.tool_calls[c])].visible for c in "abc")
    assert group.button.description.endswith("collapse")


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
