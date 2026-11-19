import panel_material_ui as pmui
import pytest

pytest.importorskip("playwright")

from panel.io.state import state as pn_state
from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import AgentDrawer, RunState, SpanView
from panel_agents import events as ev

from .test_span import SECRET, state

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def cleanup_servers():
    try:
        yield
    finally:
        pn_state.reset()


def test_span_view_shows_the_run_and_opens_a_published_version(page):
    opened = []
    view = SpanView(state(), run_id="r1", on_artifact=lambda a, v: opened.append((a, v)))
    serve_component(page, view)

    head = page.locator(".sv-head")
    expect(head.locator(".sv-question")).to_have_text("Paid revenue by region?")
    expect(head).to_contain_text("31% of input from cache")
    expect(head).to_contain_text("1 step blocked, 1 request retried, 1 tool call failed")
    expect(page.locator(".rt-totals")).to_contain_text("Model requests4")
    expect(page.locator(".model-call-card")).to_have_count(3)
    expect(page.locator(".tool-call-card")).to_have_count(2)
    expect(page.locator(".model-call-card").nth(2).locator(".tc-head")).to_contain_text("1 retry")

    sql = page.locator(".tool-call-card").nth(1)
    sql.locator("[aria-label='show more']").first.click()
    expect(sql.locator(".tc-args")).to_contain_text("customers")

    page.get_by_text("v1 · Added revenue").click()
    wait_until(lambda: opened == [("doc", 1)], page)


def test_redacted_span_view_shows_no_content(page):
    serve_component(page, SpanView(state(), run_id="r1", redacted=True))
    expect(page.locator(".sv-question")).to_have_text("The question is hidden.")
    expect(page.locator(".tool-call-card")).to_have_count(2)
    for card in page.locator(".tool-call-card").all():
        card.locator("[aria-label='show more']").first.click()
    expect(page.get_by_text("The call failed.")).to_be_visible()
    for hidden in ("customers", "Checking.", "Added revenue", "Paid revenue"):
        expect(page.get_by_text(hidden)).to_have_count(0)
    expect(page.get_by_text("run_sql")).to_be_visible()
    expect(page.get_by_text("Dashboard").first).to_be_visible()


def test_long_run_builds_more_steps_on_request(page):
    events = [ev.RunStarted(run_id="r", timestamp=0)]
    for i in range(60):
        events += [
            ev.ToolCallStart(tool_call_id=f"t{i}", tool_name=f"step_{i}", run_id="r", timestamp=i),
            ev.ToolCallResult(tool_call_id=f"t{i}", content="ok", run_id="r", timestamp=i + 0.5),
        ]
    events.append(ev.RunFinished(run_id="r", timestamp=61))
    serve_component(page, SpanView(RunState.from_events(events), run_id="r"))
    expect(page.locator(".tool-call-card")).to_have_count(50)
    page.get_by_role("button", name="Show 10 more steps").click()
    expect(page.locator(".tool-call-card")).to_have_count(60)
    expect(page.get_by_role("button", name="Show 10 more steps")).to_have_count(0)


def test_drawer_opens_a_run_from_the_chat_and_a_version_from_the_span(page):
    s = state()
    drawer = AgentDrawer(state=s, open=True, width=480)
    serve_component(page, pmui.Column(drawer, height=700, sizing_mode="stretch_width"))

    page.locator(".run-details button").click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_text("How this was made")).to_be_visible()
    expect(dialog.locator(".sv-question")).to_have_text("Paid revenue by region?")
    assert SECRET not in dialog.locator(".sv-head").inner_text()

    dialog.get_by_text("v1 · Added revenue").click()
    expect(page.get_by_role("dialog")).to_have_count(0)
    wait_until(lambda: drawer.tab == "artifacts" and s.artifacts["doc"].current == 1, page)

    # The history's run rows open the same view.
    row = page.locator(".MuiListItemButton-root").filter(has_text="Paid revenue by region?")
    row.hover()
    row.locator('.MuiIconButton-root[title="How was this made?"]').click()
    expect(page.get_by_role("dialog").locator(".sv-question")).to_have_text("Paid revenue by region?")
