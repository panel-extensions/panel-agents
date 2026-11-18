import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect
from pydantic import BaseModel, Field

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.approval import ApprovalQueue

pytestmark = pytest.mark.ui


class RunQuery(BaseModel):
    sql: str
    limit: int = Field(100, ge=1, le=1000)


def test_edit_argument_then_approve(page):
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(
            tool_call_id="t1", tool_name="run_query", run_id="r",
            args={"sql": "SELECT 1", "limit": 10}, schema=RunQuery.model_json_schema(),
        ),
    ])
    queue = ApprovalQueue(state, user="alice", show_resolved=True)
    serve_component(page, queue)

    page.locator("button.MuiToggleButton-root").filter(has_text="Edit").click()
    limit = page.locator("input[type=text]").first
    expect(limit).to_have_value("10")
    limit.fill("50")
    limit.press("Tab")
    expect(page.get_by_role("button", name="Approve with edits")).to_be_visible()
    expect(page.locator(".pa-new")).to_have_text("50")

    page.get_by_role("button", name="Approve with edits").click()
    # The server applies the decision on its own thread, the tool call last.
    wait_until(lambda: state.tool_calls["t1"].status == "running", page)
    assert state.approvals["t1"].final_args == {"sql": "SELECT 1", "limit": 50}
    assert state.approvals["t1"].user == "alice"
    assert state.tool_calls["t1"].args["limit"] == 50
    expect(page.get_by_text("Approved with edits")).to_be_visible()


def _labelled_state():
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id="step-2", tool_name="plan_step", run_id="r", args={"question": "Revenue by region"}, labels={"approve": "Approve", "deny": "Remove"}),
        ev.ApprovalRequested(tool_call_id="call-1", tool_name="run_query", run_id="r", args={"sql": "DELETE FROM sales"}),
    ])
    return state


def test_plan_step_and_default_labels(page):
    state = _labelled_state()
    serve_component(page, ApprovalQueue(state, user="alice", show_resolved=True))

    step, call = page.locator(".MuiPaper-outlined").nth(0), page.locator(".MuiPaper-outlined").nth(1)
    expect(step.get_by_role("button", name="Remove")).to_be_visible()
    expect(step.get_by_role("button", name="Deny")).to_have_count(0)
    expect(call.get_by_role("button", name="Deny")).to_be_visible()
    expect(page.get_by_role("button", name="Approve all")).to_be_visible()
    expect(page.get_by_role("button", name="Deny all")).to_be_visible()

    step.get_by_role("button", name="Remove").click()
    wait_until(lambda: state.approvals["step-2"].status == "denied", page)
    expect(page.get_by_text("Removed", exact=True)).to_be_visible()

    page.locator(".MuiPaper-outlined").nth(1).get_by_role("button", name="Approve").click()
    wait_until(lambda: state.approvals["call-1"].status == "approved", page)
    expect(page.get_by_text("Approved", exact=True)).to_be_visible()


def test_plan_labels_on_batch_actions(page):
    state = RunState(flush_interval=0)
    plan = {"approve": "Approve", "deny": "Remove"}
    state.apply_many([
        ev.RunStarted(run_id="r"),
        *(ev.ApprovalRequested(tool_call_id=f"s{i}", tool_name="plan_step", run_id="r", args={"question": f"Step {i}"}, labels=plan) for i in (1, 2)),
    ])
    serve_component(page, ApprovalQueue(state, user="alice"))

    page.get_by_role("button", name="Remove all").click()
    wait_until(lambda: not state.pending_approvals, page)
    assert {r.status for r in state.approvals.values()} == {"denied"}
