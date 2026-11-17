import panel_material_ui as pmui
import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import AgentDrawer, RunState
from panel_agents import events as ev

pytestmark = pytest.mark.ui


def test_closed_drawer_takes_no_space(page):
    main = pmui.Paper(pmui.Typography("main"), sizing_mode="stretch_both")
    drawer = AgentDrawer(state=RunState(), main=main)
    page.set_viewport_size({"width": 1000, "height": 600})
    serve_component(page, pmui.Column(drawer, height=500, sizing_mode="stretch_width"))
    paper = page.locator(".MuiPaper-root").first
    expect(paper).to_be_visible()
    wait_until(lambda: paper.bounding_box()["width"] > 950, page)


def test_drawer_toggle_close_and_approval_badge(page):
    state = RunState(flush_interval=0)
    drawer = AgentDrawer(state=state, main=pmui.Typography("The app", variant="h4"), width=400)
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=600, sizing_mode="stretch_width"))

    expect(page.get_by_text("The app")).to_be_visible()
    expect(page.get_by_text("Assistant", exact=True).last).not_to_be_visible()

    page.get_by_role("button", name="Assistant", exact=True).click()
    wait_until(lambda: drawer.open, page)
    expect(page.get_by_text("Assistant", exact=True).last).to_be_visible()
    expect(page.get_by_role("tab", name="Chat")).to_be_visible()

    page.get_by_role("button", name="Close", exact=True).click()
    wait_until(lambda: not drawer.open, page)

    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id="t1", tool_name="save_report", run_id="r"),
    ])
    wait_until(lambda: drawer.open, page)
    toggle = page.get_by_role("button", name="Assistant, 1 pending approval", exact=True)
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(page.locator(".pa-badge")).to_have_text("1")
    page.get_by_role("tab", name="Approvals (1)").click()
    wait_until(lambda: drawer.tab == "approvals", page)
    expect(page.get_by_role("button", name="Approve")).to_be_visible()


def test_focus_moves_into_the_drawer_and_back(page):
    drawer = AgentDrawer(state=RunState(), main=pmui.Typography("The app"))
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=600, sizing_mode="stretch_width"))
    toggle = page.get_by_role("button", name="Assistant", exact=True)
    expect(toggle).to_have_attribute("aria-expanded", "false")
    toggle.click()
    wait_until(lambda: drawer.open, page)
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(page.locator("textarea:focus")).to_have_count(1)
    page.get_by_role("button", name="Close", exact=True).click()
    wait_until(lambda: not drawer.open, page)
    expect(page.locator("button.pa-toggle:focus")).to_have_count(1)
