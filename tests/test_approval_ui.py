import pytest

pytest.importorskip("playwright")

import panel_material_ui as pmui
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
        ev.ApprovalRequested(tool_call_id="step-2", tool_name="plan_step", run_id="r", args={"question": "Revenue by region"}, labels={"deny": "Remove", "denied": "Removed"}),
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
    plan = {"deny": "Remove", "denied": "Removed"}
    state.apply_many([
        ev.RunStarted(run_id="r"),
        *(ev.ApprovalRequested(tool_call_id=f"s{i}", tool_name="plan_step", run_id="r", args={"question": f"Step {i}"}, labels=plan) for i in (1, 2)),
    ])
    serve_component(page, ApprovalQueue(state, user="alice"))

    page.get_by_role("button", name="Remove all").click()
    wait_until(lambda: not state.pending_approvals, page)
    assert {r.status for r in state.approvals.values()} == {"denied"}


class Step(BaseModel):
    question: str
    builds_on: int | None = None
    fast: bool = False


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_form_and_reason_inputs_share_edges(page, dark):
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id="s1", tool_name="plan_step", run_id="r", args={"question": "Revenue by region?"}, schema=Step.model_json_schema()),
    ])
    queue = ApprovalQueue(state, user="alice")
    serve_component(page, pmui.Page(main=[pmui.Column(queue, width=600)], dark_theme=dark))

    toggle = page.get_by_role("button", name="Edit")
    expect(toggle).to_have_attribute("aria-pressed", "false")
    assert toggle.evaluate("e => getComputedStyle(e).textTransform") == "none"
    assert toggle.evaluate("e => getComputedStyle(e).borderTopWidth") == "0px"
    toggle.click()
    expect(toggle).to_have_attribute("aria-pressed", "true")

    expect(page.locator("fieldset")).to_have_count(3)
    edges = {(round(b["x"]), round(b["x"] + b["width"])) for b in (f.bounding_box() for f in page.locator("fieldset").all())}
    assert len(edges) == 1, edges

    approve = page.get_by_role("button", name="Approve", exact=True).bounding_box()
    edit = toggle.bounding_box()
    assert abs((edit["x"] + edit["width"]) - (approve["x"] + approve["width"])) <= 1


def test_disabled_edit_toggle_has_no_border(page):
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id="s1", tool_name="plan_step", run_id="r", args={"question": "Revenue by region?"}, schema=Step.model_json_schema()),
    ])
    serve_component(page, ApprovalQueue(state, user="bob", can_approve=lambda request, user: False))

    toggle = page.get_by_role("button", name="Edit")
    expect(toggle).to_be_disabled()
    assert toggle.evaluate("e => getComputedStyle(e).borderTopWidth") == "0px"


def test_embedded_approval_keeps_edit_right_aligned_beside_description(page):
    from panel_agents.components.tool_call import ToolCallCard

    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ToolCallStart(tool_call_id="t1", tool_name="filter_regions", run_id="r"),
        ev.ApprovalRequested(
            tool_call_id="t1", tool_name="filter_regions", run_id="r",
            args={"regions": ["North", "East"]},
            description="Show only the given regions.",
        ),
    ])
    from panel_agents.components.approval import approval_factory

    card = ToolCallCard(state.tool_calls["t1"], state=state, approval_factory=approval_factory)
    serve_component(page, pmui.Page(main=[pmui.Column(card, width=420)]))
    edit = page.get_by_role("button", name="Edit")
    approve = page.get_by_role("button", name="Approve", exact=True)
    desc = page.get_by_text("Show only the given regions.")
    expect(edit).to_be_visible()
    expect(desc).to_be_visible()
    edit_box, approve_box, desc_box = edit.bounding_box(), approve.bounding_box(), desc.bounding_box()
    assert edit_box["x"] > desc_box["x"]
    assert abs((edit_box["x"] + edit_box["width"]) - (approve_box["x"] + approve_box["width"])) <= 2
