import json

import pytest
from pydantic import BaseModel, Field

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.audit import AuditLog
from panel_agents.components.approval import (
    ApprovalCard,
    ApprovalQueue,
    _diff_args,
    _prune_args,
    approval_factory,
)


class RunQuery(BaseModel):
    sql: str = Field(description="SQL to execute.")
    limit: int = Field(100, ge=1, le=1000)
    timeout: int | None = None


def make_state():
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ToolCallStart(tool_call_id="t1", tool_name="run_query", run_id="r"),
        ev.ApprovalRequested(
            tool_call_id="t1", tool_name="run_query", run_id="r",
            args={"sql": "SELECT * FROM sales", "limit": 10},
            schema=RunQuery.model_json_schema(), description="Run a SQL query.",
        ),
        ev.ApprovalRequested(
            tool_call_id="t2", tool_name="send_email", run_id="r",
            args={"to": "ceo@example.com", "subject": "Q3", "body": "Numbers\nattached"},
        ),
    ])
    return state


def card_for(state, tool_call_id, **params):
    return ApprovalCard(request=state.approvals[tool_call_id], state=state, **params)


def test_approve_without_edits():
    state = make_state()
    card = card_for(state, "t1", user="alice")
    assert card.approve()
    request = state.approvals["t1"]
    assert request.status == "approved"
    assert request.final_args == {"sql": "SELECT * FROM sales", "limit": 10}
    assert not request.edited
    assert request.user == "alice"
    assert state.tool_calls["t1"].status == "running"


def test_edit_is_offered_only_with_arguments():
    state = make_state()
    state.apply(ev.ApprovalRequested(tool_call_id="t3", tool_name="delete_report", run_id="r"))
    assert card_for(state, "t1")._edit_toggle.visible
    assert not card_for(state, "t3")._edit_toggle.visible


def test_deny():
    state = make_state()
    card = card_for(state, "t1")
    card._reason.value = "Too broad"
    assert card.deny()
    request = state.approvals["t1"]
    assert request.status == "denied"
    assert request.reason == "Too broad"
    assert state.tool_calls["t1"].status == "denied"


def test_edit_then_approve():
    state = make_state()
    card = card_for(state, "t1")
    assert card._approve.label == "Approve"
    card.editing = True
    # Defaults the agent did not send (timeout=None) must not count as edits.
    assert card.edited_args == {"sql": "SELECT * FROM sales", "limit": 10}
    assert not card.is_edited
    card.form.field("limit").value = 50
    assert card.is_edited
    assert card._approve.label == "Approve with edits"
    assert card._diff_section.visible
    assert "pa-old" in card._diff_pane.object
    assert card.approve()
    request = state.approvals["t1"]
    assert request.final_args == {"sql": "SELECT * FROM sales", "limit": 50}
    assert request.edited
    assert state.tool_calls["t1"].args["limit"] == 50


def test_edit_with_inferred_schema():
    state = make_state()
    card = card_for(state, "t2")
    card.editing = True
    card.form.field("subject").value = "Q3 results"
    assert card.approve()
    assert state.approvals["t2"].final_args["subject"] == "Q3 results"


def test_cancel_edit_discards_changes():
    state = make_state()
    card = card_for(state, "t1")
    card.editing = True
    card.form.field("limit").value = 50
    card.editing = False
    assert not card.is_edited
    assert card._approve.label == "Approve"
    card.approve()
    assert state.approvals["t1"].final_args["limit"] == 10


def test_invalid_edit_blocks_approval():
    state = make_state()
    card = card_for(state, "t1")
    card.editing = True
    card.form.field("limit").value = 5000
    assert not card.approve()
    assert state.approvals["t1"].status == "pending"
    assert "Fix the highlighted" in card.error
    assert card._error.visible


@pytest.mark.parametrize("mode, action, edited, required", [
    ("never", "deny", False, False),
    ("deny", "deny", False, True),
    ("deny", "approve", False, False),
    ("edit", "approve", False, False),
    ("edit", "approve", True, True),
    ("edit", "deny", False, False),
    ("always", "approve", False, True),
])
def test_require_reason(mode, action, edited, required):
    state = make_state()
    card = card_for(state, "t1", require_reason=mode)
    if edited:
        card.editing = True
        card.form.field("limit").value = 20
    resolved = getattr(card, action)()
    assert resolved is not required
    if required:
        assert card._reason.error_state
        card._reason.value = "Checked with team"
        assert getattr(card, action)()
        assert state.approvals["t1"].reason == "Checked with team"


def test_can_approve_policy():
    state = make_state()
    calls = []

    def policy(request, user):
        calls.append((request.tool_name, user))
        return user == "admin"

    card = card_for(state, "t1", can_approve=policy, user="bob", policy_message="Only admins can run queries.")
    assert card._approve.disabled and card._deny.disabled
    assert card._policy.visible
    assert card._policy.object == "Only admins can run queries."
    assert not card.approve()
    assert state.approvals["t1"].status == "pending"
    card.user = "admin"
    assert not card._approve.disabled
    assert not card._policy.visible
    assert card.approve()
    assert ("run_query", "bob") in calls


def test_resolved_card_shows_summary():
    state = make_state()
    card = card_for(state, "t1", user="alice")
    card.editing = True
    card.form.field("limit").value = 25
    card._reason.value = "Smaller sample"
    card.approve()
    column = card._layout.objects[0]
    texts = " ".join(str(getattr(o, "object", "")) for row in column.objects for o in getattr(row, "objects", [row]))
    assert "alice" in texts
    assert "Smaller sample" in texts
    assert "pa-new" in texts
    assert column.objects[0].objects[0].label == "Approved with edits"


def test_card_from_replayed_state_is_resolved():
    state = make_state()
    state.resolve_approval("t1", False, reason="no", user="carol")
    replayed = RunState.from_events(state.events)
    card = card_for(replayed, "t1")
    assert card._layout.objects[0].objects[0].objects[0].label == "Denied"
    assert not card.approve()


def test_approval_factory():
    state = make_state()
    card = approval_factory(state.approvals["t1"], state)
    assert isinstance(card, ApprovalCard)
    assert card.request is state.approvals["t1"]


def test_queue_lists_pending_and_updates():
    state = make_state()
    queue = ApprovalQueue(state)
    assert [c.request.tool_call_id for c in queue.cards] == ["t1", "t2"]
    assert queue._approve_all.visible
    queue.cards[0].approve()
    assert [c.request.tool_call_id for c in queue.cards] == ["t2"]
    assert not queue._approve_all.visible
    state.apply(ev.ApprovalRequested(tool_call_id="t3", tool_name="drop", args={"table": "x"}, run_id="r"))
    assert [c.request.tool_call_id for c in queue.cards] == ["t2", "t3"]
    queue.show_resolved = True
    assert len(queue.cards) == 3


def test_queue_approve_all():
    state = make_state()
    queue = ApprovalQueue(state, user="alice")
    assert queue.approve_all() == 2
    assert state.pending_approvals == []
    assert {r.status for r in state.approvals.values()} == {"approved"}
    assert queue._empty.visible


def test_queue_deny_all_respects_reason_requirement():
    state = make_state()
    queue = ApprovalQueue(state, require_reason="deny")
    queue.cards[0]._reason.value = "Not now"
    assert queue.deny_all() == 1
    assert state.approvals["t1"].status == "denied"
    assert state.approvals["t2"].status == "pending"
    assert "need attention" in queue.message


def test_queue_propagates_policy():
    state = make_state()
    queue = ApprovalQueue(state, can_approve=lambda request, user: False)
    assert all(c._approve.disabled for c in queue.cards)
    assert queue.approve_all() == 0
    queue.can_approve = None
    assert queue.approve_all() == 2


def test_diff_and_prune():
    schema = RunQuery.model_json_schema()
    assert _prune_args({"sql": "a"}, {"sql": "a", "limit": 100, "timeout": None}, schema) == {"sql": "a"}
    assert _prune_args({"sql": "a"}, {"sql": "a", "limit": 5}, schema) == {"sql": "a", "limit": 5}
    changes = _diff_args({"a": 1, "n": {"x": 1}, "gone": 2}, {"a": 2, "n": {"x": 1, "y": 3}})
    assert [c[0] for c in changes] == ["a", "gone", "n.y"]


def test_audit_log_callable_sink():
    state = make_state()
    records = []
    log = AuditLog(state, records.append)
    card = card_for(state, "t1", user="alice")
    card.editing = True
    card.form.field("limit").value = 30
    card._reason.value = "cap rows"
    card.approve()
    card_for(state, "t2", user="bob").deny()
    assert len(records) == 2
    first, second = records
    assert first["decision"] == "approved"
    assert first["tool_name"] == "run_query"
    assert first["run_id"] == "r"
    assert first["original_args"] == {"sql": "SELECT * FROM sales", "limit": 10}
    assert first["final_args"] == {"sql": "SELECT * FROM sales", "limit": 30}
    assert first["edited"]
    assert first["reason"] == "cap rows"
    assert first["user"] == "alice"
    assert isinstance(first["timestamp"], float) and first["time"].endswith("+00:00")
    assert second["decision"] == "denied"
    assert second["final_args"] is None
    assert second["user"] == "bob"
    assert log.records == records


def test_audit_log_jsonl(tmp_path):
    state = make_state()
    path = tmp_path / "audit" / "approvals.jsonl"
    log = AuditLog(state, path)
    state.resolve_approval("t1", True, user="alice")
    state.resolve_approval("t2", False, reason="no", user="alice")
    lines = path.read_text().splitlines()
    assert [json.loads(line)["tool_call_id"] for line in lines] == ["t1", "t2"]
    assert AuditLog.read(path)[0]["final_args"] == {"sql": "SELECT * FROM sales", "limit": 10}
    log.detach()
    state.apply(ev.ApprovalRequested(tool_call_id="t3", tool_name="x", run_id="r"))
    state.resolve_approval("t3", True)
    assert len(AuditLog.read(path)) == 2


def test_audit_log_include_existing():
    state = make_state()
    state.resolve_approval("t1", True, user="alice")
    log = AuditLog(RunState.from_events(state.events), include_existing=True)
    assert [r["tool_call_id"] for r in log.records] == ["t1"]


def test_audit_log_include_existing_warns_when_events_were_dropped():
    state = make_state()
    state.max_events = 3
    state.resolve_approval("t1", True, user="alice")
    for i in range(5):
        state.apply(ev.ArtifactPublished(artifact_id="doc", value=i))
    with pytest.warns(UserWarning, match=f"dropped its {state.dropped_events} earliest events") as record:
        log = AuditLog(state, include_existing=True)
    assert log.records == [] and record[0].filename == __file__
    with pytest.warns(UserWarning, match="decisions among the dropped events") as record:
        AuditLog().attach(state, include_existing=True)
    assert record[0].filename == __file__
