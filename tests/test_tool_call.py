import pandas as pd
import panel as pn
import panel_material_ui as pmui
import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.approval import ApprovalCard, approval_factory
from panel_agents.components.renderers import _TABLE_CSS
from panel_agents.components.tool_call import (
    ToolCallCard,
    _args_preview,
    _format_duration,
    _truncate,
)


@pytest.fixture
def state():
    return RunState(flush_interval=0)


def start(state, call_id="t1", name="run_query"):
    state.apply(ev.ToolCallStart(tool_call_id=call_id, tool_name=name, timestamp=100.0))
    return state.tool_calls[call_id]


def finish(state, content, call_id="t1", is_error=False, timestamp=101.5):
    state.apply(ev.ToolCallEnd(tool_call_id=call_id))
    state.apply(ev.ToolCallResult(tool_call_id=call_id, content=content, is_error=is_error, timestamp=timestamp))


def test_card_header_and_default_collapsed(state):
    call = start(state)
    card = ToolCallCard(call, state=state)
    assert card.collapsed
    assert card._step.collapsed
    assert card._step.status == "pending"
    assert "run_query" in card._header.object
    assert not card._args_section.visible
    assert not card._result_section.visible


def test_args_stream_into_existing_pane(state):
    call = start(state)
    card = ToolCallCard(call, state=state)
    args_pane = card._args
    state.apply(ev.ToolCallArgs(tool_call_id="t1", delta='{"sql": "SEL'))
    assert card._args_section.visible
    assert "SEL" in args_pane.object
    assert 'sql=&quot;SEL&quot;' in card._header.object
    state.apply(ev.ToolCallArgs(tool_call_id="t1", delta='ECT 1"}'))
    assert card._args is args_pane
    assert "SELECT 1" in args_pane.object


def test_success_renders_result_and_duration(state):
    call = start(state)
    card = ToolCallCard(call, state=state)
    finish(state, {"rows": 3})
    assert card._step.status == "success"
    assert card.collapsed
    assert card._result_section.visible
    assert isinstance(card._result[0], pn.pane.JSON)
    assert "1.5s" in card._header.object
    rendered = card._result[0]
    call.param.trigger("status")
    assert card._result[0] is rendered


def test_failure_expands_and_shows_error(state):
    call = start(state)
    card = ToolCallCard(call, state=state)
    finish(state, "Binder Error: no column qtr", is_error=True)
    assert card._step.status == "failed"
    assert not card.collapsed
    assert not card._step.collapsed
    assert card._error.visible
    assert card._error.severity == "error"
    assert "Binder Error" in card._error.object
    assert not card._result_section.visible
    assert "Failed" in card._header.object


def test_card_built_after_failure_starts_expanded(state):
    call = start(state)
    finish(state, "boom", is_error=True)
    card = ToolCallCard(call, state=state)
    assert not card.collapsed


def test_awaiting_approval_placeholder(state):
    state.apply(ev.ApprovalRequested(tool_call_id="t1", tool_name="drop_table", args={"table": "sales"}))
    card = ToolCallCard(state.tool_calls["t1"], state=state)
    assert not card.collapsed
    assert card._step.status == "pending"
    assert "Needs approval" in card._header.object
    assert card._approval.visible
    assert isinstance(card._approval[0], pmui.Alert)


def test_approval_factory_and_resolution(state):
    calls = []

    def factory(request, run_state):
        calls.append((request, run_state))
        return pn.pane.Markdown("approve?")

    call = start(state, name="drop_table")
    card = ToolCallCard(call, state=state, approval_factory=factory)
    assert not card._approval.visible
    state.apply(ev.ApprovalRequested(tool_call_id="t1", tool_name="drop_table", args={"table": "sales"}))
    assert not card.collapsed
    assert card._approval.visible
    assert card._approval[0].object == "approve?"
    assert calls == [(state.approvals["t1"], state)]
    state.resolve_approval("t1", approved=True, args={"table": "sales_tmp"})
    assert not card._approval.visible
    assert card._step.status == "running"
    assert "Edited" in card._header.object


def test_approval_card_replaces_the_argument_listing(state):
    state.apply(ev.ToolCallStart(tool_call_id="t1", tool_name="drop_table"))
    state.apply(ev.ToolCallArgs(tool_call_id="t1", delta='{"table": "sales"}'))
    cards = []
    card = ToolCallCard(state.tool_calls["t1"], state=state, approval_factory=lambda *a: cards.append(approval_factory(*a)) or cards[-1])
    assert card._args_section.visible
    state.apply(ev.ApprovalRequested(tool_call_id="t1", tool_name="drop_table", args={"table": "sales"}))
    assert isinstance(cards[-1], ApprovalCard) and not cards[-1]._title.visible
    assert not card._args_section.visible
    state.resolve_approval("t1", approved=True)
    assert card._args_section.visible


def test_denied(state):
    state.apply(ev.ApprovalRequested(tool_call_id="t1", tool_name="drop_table"))
    card = ToolCallCard(state.tool_calls["t1"], state=state)
    state.resolve_approval("t1", approved=False, reason="Too risky")
    assert card._step.status == "failed"
    assert "Denied" in card._header.object
    assert card._error.visible
    assert card._error.severity == "warning"
    assert card._error.object == "Too risky"


def test_large_string_result_is_truncated(state):
    call = start(state)
    card = ToolCallCard(call, state=state, max_chars=100)
    finish(state, "x" * 1000)
    assert len(card._result[0].object) == 100
    assert "100 of 1,000 characters" in card._hint.object


def test_large_dataframe_result_is_truncated(state):
    call = start(state)
    card = ToolCallCard(call, state=state, max_items=10)
    finish(state, pd.DataFrame({"a": range(50)}))
    assert len(card._result[0].object) == 10
    assert "10 of 50 rows" in card._hint.object


def test_dataframe_result_renders_styled_table(state):
    card = ToolCallCard(start(state), state=state)
    finish(state, pd.DataFrame({"a": [1, 2]}))
    (pane,) = card._result.objects
    assert type(pane) is pn.pane.DataFrame
    assert not pane.index
    assert _TABLE_CSS in pane.stylesheets


def test_records_result_renders_styled_table(state):
    card = ToolCallCard(start(state), state=state)
    finish(state, [{"column": "order_id", "type": "BIGINT"}, {"column": "product", "type": "VARCHAR"}])
    (pane,) = card._result.objects
    assert type(pane) is pn.pane.DataFrame
    assert list(pane.object.columns) == ["column", "type"]
    assert len(pane.object) == 2


def test_nested_list_result_renders_json(state):
    card = ToolCallCard(start(state), state=state)
    finish(state, [{"a": [1, 2]}])
    assert isinstance(card._result[0], pn.pane.JSON)


def test_markdown_table_result_is_styled(state):
    card = ToolCallCard(start(state), state=state)
    finish(state, "| a | b |\n|---|---|\n| 1 | 2 |")
    (pane,) = card._result.objects
    assert isinstance(pane, pn.pane.Markdown)
    assert _TABLE_CSS in pane.stylesheets


def test_artifact_chips(state):
    call = start(state)
    card = ToolCallCard(call, state=state)
    assert not card._artifacts.visible
    state.apply(ev.ArtifactPublished(artifact_id="other", value=1, tool_call_id="t9"))
    state.apply(ev.ArtifactPublished(artifact_id="q", value="SELECT 1", kind="sql", title="Query", tool_call_id="t1"))
    assert card._artifacts.visible
    (chip,) = card._artifacts.objects
    assert chip.label == "Query"
    assert chip.icon == "code"
    state.artifacts.active = None
    chip.clicks += 1
    assert state.artifacts.active == "q"
    state.apply(ev.ArtifactPublished(artifact_id="q", value="SELECT 2", tool_call_id="t1"))
    assert card._artifacts.objects[0] is chip
    assert chip.label == "Query · v2"


def test_artifact_chips_follow_reset_store(state):
    call = start(state)
    card = ToolCallCard(call, state=state)
    state.artifacts = type(state.artifacts)()
    state.apply(ev.ArtifactPublished(artifact_id="q", value=1, tool_call_id="t1"))
    assert len(card._artifacts.objects) == 1


def test_collapsed_syncs_both_ways(state):
    card = ToolCallCard(start(state), state=state)
    card._step.collapsed = False
    assert not card.collapsed
    card.collapsed = True
    assert card._step.collapsed


def test_card_without_state(state):
    card = ToolCallCard(start(state))
    finish(state, "ok")
    assert card.published_artifacts() == []


def test_helpers():
    assert _args_preview({"a": 1, "b": "x  y\nz"}) == 'a=1, b="x y z"'
    assert _args_preview({"sql": "x" * 200}, max_length=20).endswith("…")
    assert _format_duration(None) == ""
    assert _format_duration(0.25) == "250ms"
    assert _format_duration(2.34) == "2.3s"
    assert _format_duration(125) == "2m 5s"
    assert _truncate([1, 2, 3], max_items=2) == ([1, 2], "Showing the first 2 of 3 items.")
    assert _truncate({"a": 1}) == ({"a": 1}, None)
