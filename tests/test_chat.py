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


def test_input_disabled_while_running():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    assert isinstance(chat.feed, pmui.ChatInterface)
    assert not chat.feed.disabled
    state.apply(ev.RunStarted(run_id="r1"))
    assert chat.feed.disabled
    state.apply(ev.RunFinished(run_id="r1"))
    assert not chat.feed.disabled


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
