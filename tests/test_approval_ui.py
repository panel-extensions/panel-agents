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
    wait_until(lambda: state.approvals["t1"].status == "approved", page)
    assert state.approvals["t1"].final_args == {"sql": "SELECT 1", "limit": 50}
    assert state.approvals["t1"].user == "alice"
    assert state.tool_calls["t1"].args["limit"] == 50
    expect(page.get_by_text("Approved with edits")).to_be_visible()
