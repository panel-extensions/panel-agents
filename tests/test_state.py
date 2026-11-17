import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.state import _apply_json_patch


def run_events(run_id="run-1"):
    return [
        ev.RunStarted(run_id=run_id, agent_name="analyst"),
        ev.TextMessageStart(message_id="m1", run_id=run_id),
        ev.TextMessageContent(message_id="m1", delta="Let me ", run_id=run_id),
        ev.TextMessageContent(message_id="m1", delta="check.", run_id=run_id),
        ev.TextMessageEnd(message_id="m1", run_id=run_id),
        ev.ToolCallStart(tool_call_id="t1", tool_name="run_query", run_id=run_id),
        ev.ToolCallArgs(tool_call_id="t1", delta='{"sql": "SELECT', run_id=run_id),
        ev.ToolCallArgs(tool_call_id="t1", delta=' 1"}', run_id=run_id),
        ev.ToolCallEnd(tool_call_id="t1", run_id=run_id),
        ev.ToolCallResult(tool_call_id="t1", content=[[1]], run_id=run_id),
        ev.Usage(model="m", input_tokens=10, output_tokens=5, cost_usd=0.01, run_id=run_id),
        ev.RunFinished(run_id=run_id, result="done"),
    ]


def test_message_and_tool_call_lifecycle():
    state = RunState.from_events(run_events())
    message, call = state.items
    assert message.content == "Let me check."
    assert message.finished
    assert call.status == "success"
    assert call.args == {"sql": "SELECT 1"}
    assert call.result == [[1]]
    run = state.runs["run-1"]
    assert run.status == "finished"
    assert (run.input_tokens, run.output_tokens, run.cost_usd) == (10, 5, 0.01)
    assert not state.running


def test_partial_args_are_parsed_while_streaming():
    state = RunState(flush_interval=0)
    state.apply(ev.ToolCallStart(tool_call_id="t1", tool_name="f"))
    state.apply(ev.ToolCallArgs(tool_call_id="t1", delta='{"sql": "SEL'))
    call = state.tool_calls["t1"]
    assert call.status == "pending"
    assert call.args == {"sql": "SEL"}


def test_deltas_are_coalesced_until_flush():
    state = RunState(flush_interval=3600)
    state.apply(ev.TextMessageStart(message_id="m1"))
    state.flush()
    state.apply(ev.TextMessageContent(message_id="m1", delta="a"))
    state.apply(ev.TextMessageContent(message_id="m1", delta="b"))
    assert state.messages["m1"].content == ""
    state.apply(ev.TextMessageEnd(message_id="m1"))
    assert state.messages["m1"].content == "ab"
    assert len(state.events) == 4


def test_failed_tool_call():
    state = RunState.from_events([
        ev.ToolCallStart(tool_call_id="t1", tool_name="f"),
        ev.ToolCallEnd(tool_call_id="t1"),
        ev.ToolCallResult(tool_call_id="t1", content="boom", is_error=True),
    ])
    call = state.tool_calls["t1"]
    assert call.status == "failed"
    assert call.error == "boom"


def test_run_error_fails_running_tool_calls():
    state = RunState.from_events([
        ev.RunStarted(run_id="r"),
        ev.ToolCallStart(tool_call_id="t1", tool_name="f", run_id="r"),
        ev.ToolCallEnd(tool_call_id="t1", run_id="r"),
        ev.RunError(run_id="r", message="crashed"),
    ])
    assert state.runs["r"].status == "error"
    assert state.tool_calls["t1"].status == "failed"


@pytest.mark.parametrize("approved", [True, False])
def test_approval_flow(approved):
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ToolCallStart(tool_call_id="t1", tool_name="drop", run_id="r"),
        ev.ToolCallArgs(tool_call_id="t1", delta='{"table": "a"}', run_id="r"),
        ev.ToolCallEnd(tool_call_id="t1", run_id="r"),
        ev.ApprovalRequested(tool_call_id="t1", tool_name="drop", args={"table": "a"}, run_id="r"),
    ])
    call = state.tool_calls["t1"]
    assert call.status == "awaiting_approval"
    assert len(state.pending_approvals) == 1
    seen = []
    state.subscribe(seen.append)
    state.resolve_approval("t1", approved, args={"table": "b"}, reason="typo", user="alice")
    request = state.approvals["t1"]
    assert state.pending_approvals == []
    assert isinstance(seen[0], ev.ApprovalResolved)
    if approved:
        assert call.status == "running"
        assert call.args == {"table": "b"}
        assert request.edited
    else:
        assert call.status == "denied"
        assert request.final_args is None
    assert request.user == "alice"


def test_artifact_versions():
    state = RunState.from_events([
        ev.ArtifactPublished(artifact_id="q", value="SELECT 1", kind="sql"),
        ev.ArtifactPublished(artifact_id="q", value="SELECT 2"),
    ])
    artifact = state.artifacts["q"]
    assert artifact.kind == "sql"
    assert [v.value for v in artifact.versions] == ["SELECT 1", "SELECT 2"]
    assert artifact.value == "SELECT 2"
    assert state.artifacts.active == "q"


def test_state_snapshot_and_delta():
    state = RunState.from_events([
        ev.StateSnapshot(snapshot={"filters": {"year": 2020}, "tags": ["a"]}),
        ev.StateDelta(delta=[
            {"op": "replace", "path": "/filters/year", "value": 2021},
            {"op": "add", "path": "/tags/-", "value": "b"},
        ]),
    ])
    assert state.agent_state == {"filters": {"year": 2021}, "tags": ["a", "b"]}


def test_json_patch_move_and_remove():
    doc = _apply_json_patch({"a": 1, "b": {"c": 2}}, [
        {"op": "move", "from": "/a", "path": "/b/a"},
        {"op": "remove", "path": "/b/c"},
    ])
    assert doc == {"b": {"a": 1}}


def test_add_user_message():
    state = RunState(flush_interval=0)
    state.add_user_message("hi")
    (message,) = state.items
    assert (message.role, message.content, message.finished) == ("user", "hi", True)


def test_reset():
    state = RunState.from_events(run_events())
    state.reset()
    assert state.items == [] and state.events == [] and len(state.artifacts) == 0
