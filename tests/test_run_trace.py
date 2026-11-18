import pytest

from panel_agents import ModelCallCard, RunState, RunTrace, ToolCallCard
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat
from panel_agents.components.model_call import format_cost, format_tokens, usage_summary

from .test_model_calls import model_run, old_recording


def card_rows(trace):
    """(card type, depth) of each row with a time bar, i.e. each step."""
    rows = []
    for item, depth in trace._nested():
        view, bar = trace._cache[id(item)]
        assert view in trace._rows.objects
        if bar is not None:
            rows.append((type(trace._cards[id(item)]).__name__, depth))
    return rows


def test_trace_nests_tool_calls_under_the_request_that_issued_them():
    state = RunState.from_events(model_run())
    trace = RunTrace(state, run_id="r1")
    assert card_rows(trace) == [("ModelCallCard", 0), ("ToolCallCard", 1), ("ModelCallCard", 0)]
    assert isinstance(trace._cards[id(state.tool_calls["t1"])], ToolCallCard)
    totals = trace._totals.object
    for text in ("Model requests", ">2<", "2,100", "800 from cache", ">50<", "$0.0030", "2.0s", "1.6s in the model"):
        assert text in totals


def test_trace_nests_by_position_when_requests_do_not_name_their_tool_calls():
    state = RunState.from_events(old_recording(model_run()))
    trace = RunTrace(state, run_id="r1")
    assert card_rows(trace) == [("ModelCallCard", 0), ("ToolCallCard", 1), ("ModelCallCard", 0)]


def test_trace_bars_share_the_runs_time_axis():
    state = RunState.from_events(model_run())
    trace = RunTrace(state, run_id="r1")
    first_bar = trace._rows.objects[0].objects[1].object
    # The first request took 1.0s of the run's 2.0s, from its start.
    assert "left:0.00%;width:50.00%" in first_bar
    last_bar = trace._rows.objects[-2].objects[1].object
    assert "left:60.00%;width:30.00%" in last_bar


def test_trace_follows_a_run_in_progress_and_can_switch_runs():
    events = model_run()
    state = RunState(flush_interval=0)
    trace = RunTrace(state, run_id="r1")
    state.apply_many(events[:3])
    assert card_rows(trace) == [("ModelCallCard", 0)]
    assert "running" in trace._totals.object
    state.apply_many(events[3:])
    assert card_rows(trace) == [("ModelCallCard", 0), ("ToolCallCard", 1), ("ModelCallCard", 0)]
    state.apply_many([ev.RunStarted(run_id="r2"), ev.ModelRequestStarted(request_id="q3", model="m", run_id="r2")])
    trace.run_id = "r2"
    assert card_rows(trace) == [("ModelCallCard", 0)]
    trace.run_id = None
    assert trace._rows.objects == [] and not trace._totals.visible


def test_model_call_card_lists_the_details():
    state = RunState.from_events(model_run())
    card = ModelCallCard(state.model_calls["q1"], state=state, show_tool_count=True)
    assert "1,000</b> in" in card._header.object and "1 tool call" in card._header.object
    assert "800 of 1,000 input tokens from the cache" in card._header.object
    details = card._details.object
    for text in ("1,000 tokens, 800 cached", "tool_call", "lookup · t1", "resp-1", "q1", "r1"):
        assert text in details
    assert card._step.status == "success" and not card._error.visible


def test_model_call_card_shows_a_failure():
    state = RunState.from_events([
        ev.RunStarted(run_id="r1"),
        ev.ModelRequestStarted(request_id="q1", model="m", run_id="r1"),
    ])
    card = ModelCallCard(state.model_calls["q1"])
    assert card._step.status == "running"
    state.apply(ev.RunError(run_id="r1", message="provider *refused*"))
    assert card._step.status == "failed" and "Failed" in card._header.object
    assert card._error.visible and card._error.object == r"provider \*refused\*"


@pytest.mark.parametrize(("n", "compact", "text"), [(950, True, "950"), (18512, True, "18.5k"), (18512, False, "18,512"), (2_400_000, True, "2.4M")])
def test_format_tokens(n, compact, text):
    assert format_tokens(n, compact) == text


def test_format_cost():
    assert (format_cost(None), format_cost(0.0081), format_cost(1.234)) == ("", "$0.0081", "$1.23")


# AgentChat


def messages_of(chat, kind):
    return [chat._messages[id(i)] for i in chat.state.items if type(i).__name__ == kind]


def test_chat_folds_model_calls_with_the_tool_calls_and_shows_run_totals():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    events = model_run()
    state.apply_many(events[:-1])
    assert all(isinstance(chat.view_for(i), ModelCallCard) for i in state.model_calls.values())
    first, second = messages_of(chat, "ModelCallState")
    # The text before the tool call is shown, so the first request has no group to fold into.
    (group,) = chat._groups.values()
    assert group.calls == [state.tool_calls["t1"], state.model_calls["q2"]]
    # Steps are the tool calls; the span runs from the tool call's start to the request's end.
    assert group.button.label == "1 step · 1.0s"
    assert first.visible and not second.visible
    assert not any(m.footer_objects for m in chat.feed.objects)

    state.apply(events[-1])
    assert not first.visible
    answer = chat.view_for(state.messages["m2"])
    (footer,) = answer.footer_objects
    assert usage_summary(state.runs["r1"]) == "2 requests · 2.1k in · 50 out · $0.0030 · 2.0s"
    assert usage_summary(state.runs["r1"]) in footer.object
    chat.show_usage = False
    assert answer.footer_objects == []


def text_run(run_id="r1"):
    return [
        ev.RunStarted(run_id=run_id, timestamp=1.0),
        ev.ModelRequestStarted(request_id=f"{run_id}-q", model="m", run_id=run_id, timestamp=1.0),
        ev.TextMessageStart(message_id=f"{run_id}-m", run_id=run_id),
        ev.TextMessageContent(message_id=f"{run_id}-m", delta="Hello.", run_id=run_id),
        ev.TextMessageEnd(message_id=f"{run_id}-m", run_id=run_id),
        ev.Usage(request_id=f"{run_id}-q", model="m", input_tokens=10, output_tokens=2, duration=0.5, run_id=run_id, timestamp=1.5),
    ]


def test_a_lone_request_shows_while_running_and_folds_into_the_usage_line():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    state.apply_many(text_run())
    (request,) = messages_of(chat, "ModelCallState")
    assert request.visible
    state.apply(ev.RunFinished(run_id="r1", timestamp=2.0))
    assert not request.visible
    assert "1 request · 10 in · 2 out · 1.0s" in chat.view_for(state.messages["r1-m"]).footer_objects[0].object
    chat.show_usage = False
    assert request.visible


def test_model_calls_can_be_hidden():
    state = RunState.from_events(model_run())
    chat = AgentChat(state, show_model_calls=False)
    assert not any(m.visible for m in messages_of(chat, "ModelCallState"))
    assert chat._groups == {}
    chat.show_model_calls = True
    (group,) = chat._groups.values()
    assert group.calls == [state.tool_calls["t1"], state.model_calls["q2"]]


def test_failed_model_calls_are_never_folded():
    state = RunState.from_events([
        ev.RunStarted(run_id="r1"),
        ev.ModelRequestStarted(request_id="q1", model="m", run_id="r1"),
        ev.Usage(request_id="q1", model="m", error="overloaded", run_id="r1"),
        ev.ModelRequestStarted(request_id="q2", model="m", run_id="r1"),
        ev.ToolCallStart(tool_call_id="t1", tool_name="f", run_id="r1"),
        ev.ToolCallEnd(tool_call_id="t1", run_id="r1"),
        ev.Usage(request_id="q2", model="m", tool_call_ids=["t1"], run_id="r1"),
        ev.ToolCallResult(tool_call_id="t1", content="ok", run_id="r1"),
        ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state)
    failed, answered = messages_of(chat, "ModelCallState")
    (group,) = chat._groups.values()
    assert failed.visible and state.model_calls["q1"] not in group.calls
    assert group.calls == [state.model_calls["q2"], state.tool_calls["t1"]]
    # Without an answer, the totals go under the step group.
    assert group.message.footer_objects and not answered.visible
