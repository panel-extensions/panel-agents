import pandas as pd
import pytest

pytest.importorskip("playwright")

from panel.io.state import state as pn_state
from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import ActionRegistry, ArtifactAction, RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def cleanup_servers():
    try:
        yield
    finally:
        pn_state.reset()


def _chat(**params):
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1", agent_name="Analyst"))
    state.add_user_message("Revenue by region?", run_id="r1")
    for name in ("paid_by_region", "paid_by_state"):
        state.publish_artifact(
            name, pd.DataFrame({"region": ["East", "West"], "total": [2, 1]}), kind="table", title=name, run_id="r1",
        )
    state.apply_many([
        ev.TextMessageStart(message_id="a", run_id="r1"),
        ev.TextMessageContent(message_id="a", delta="East leads.", run_id="r1"),
        ev.TextMessageEnd(message_id="a", run_id="r1"),
        ev.RunFinished(run_id="r1"),
    ])
    return state, AgentChat(state, layout="document", feed_params={"height": 700}, **params)


def test_cells_sit_side_by_side_and_stack_when_narrow(page):
    _, chat = _chat(readonly=True)
    page.set_viewport_size({"width": 1400, "height": 900})
    serve_component(page, chat)
    cells = page.locator(".artifact-cell")
    expect(cells).to_have_count(2)
    expect(cells.first).to_contain_text("paid_by_region")
    expect(cells.first).to_contain_text("East")
    first, second = cells.first.bounding_box(), cells.last.bounding_box()
    assert first["y"] == second["y"] and second["x"] > first["x"] + first["width"]
    answer = page.locator(".markdown", has_text="East leads.").bounding_box()
    assert first["y"] > answer["y"] + answer["height"]
    page.set_viewport_size({"width": 420, "height": 900})
    wait_until(lambda: cells.last.bounding_box()["y"] > cells.first.bounding_box()["y"] + 50, page)
    assert cells.first.bounding_box()["x"] + cells.first.bounding_box()["width"] <= 420


def test_an_action_runs_and_shows_its_done_label(page):
    actions = ActionRegistry()
    placed = []

    @actions.action
    def place(name: str):
        """Place an output."""
        placed.append(name)

    spec = ArtifactAction(
        "place", "Add to dashboard", icon="add",
        arguments=lambda a: {"name": a.title}, done=lambda a: "On dashboard" if a.title in placed else None,
    )
    _, chat = _chat(actions=actions, artifact_actions=[spec], callback=lambda text: None)
    serve_component(page, chat)
    cell = page.locator(".artifact-cell", has_text="paid_by_state")
    cell.get_by_role("button", name="Add to dashboard").click()
    wait_until(lambda: placed == ["paid_by_state"], page)
    expect(cell.locator(".artifact-cell-done")).to_contain_text("On dashboard")
    expect(page.locator(".artifact-cell", has_text="paid_by_region").get_by_role("button", name="Add to dashboard")).to_be_visible()


def test_the_idle_input_offers_send_not_stop(page):
    _, chat = _chat(callback=lambda text: None)
    serve_component(page, chat)
    expect(page.locator(".artifact-cell")).to_have_count(2)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
