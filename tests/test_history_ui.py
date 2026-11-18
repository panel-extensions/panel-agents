import panel_material_ui as pmui
import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import AgentDrawer, RunState
from panel_agents import events as ev

pytestmark = pytest.mark.ui


def history():
    state = RunState(flush_interval=0)
    value = []

    def publish(op, run_id=None, actor="agent", user=None):
        value.append(op)
        state.apply(ev.ArtifactPublished(
            artifact_id="dash", value={"edits": len(value)}, kind="json", title="Dashboard", applied=True,
            run_id=run_id, actor=actor, user=user, metadata={"label": op},
        ))

    def run(run_id, question, *ops):
        state.apply(ev.RunStarted(run_id=run_id, input=question))
        for op in ops:
            publish(op, run_id)
        state.apply(ev.RunFinished(run_id=run_id))

    run("r1", "What is paid revenue by region?", "Add node paid_by_region", "Place revenue_chart")
    publish("Move note to row 2", actor="user", user="ann")
    run("r2", "Which regions refund most?", "Add node refund_rate", "Add node refund_table", "Place refund_table")
    return state


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_timeline_in_the_drawer(page, dark):
    state = history()
    applied = []
    drawer = AgentDrawer(
        state=state, main=pmui.Typography("The app"), width=420, open=True, tab="artifacts",
        on_apply=lambda artifact, version: applied.append(version.version),
    )
    page.set_viewport_size({"width": 1000, "height": 900})
    serve_component(page, pmui.Page(main=[drawer], dark_theme=dark))

    # The latest run is open with its versions newest first, older runs are folded.
    expect(page.get_by_text("Which regions refund most?")).to_be_visible()
    expect(page.get_by_text("Place refund_table")).to_be_visible()
    expect(page.get_by_text("v6 · ", exact=False)).to_contain_text("applied")
    expect(page.get_by_text("ann edited")).to_be_visible()
    expect(page.get_by_text("Add node paid_by_region")).to_have_count(0)

    # Nothing in the timeline is wider than the drawer.
    widths = page.locator(".MuiListItemButton-root").evaluate_all("els => els.map(e => e.getBoundingClientRect().right)")
    assert widths and max(widths) <= 1000

    # Opening an older run shows its versions; viewing one does not apply it.
    page.get_by_text("What is paid revenue by region?").click()
    expect(page.get_by_text("Add node paid_by_region")).to_be_visible()
    page.get_by_text("Add node paid_by_region").click()
    expect(page.get_by_role("button", name="Version history")).to_contain_text("v1")
    expect(page.get_by_role("button", name="Revert to v1")).to_be_visible()
    assert state.artifacts["dash"].applied == 6 and applied == []

    # Folding it again removes its rows.
    page.get_by_text("What is paid revenue by region?").click()
    expect(page.get_by_text("Add node paid_by_region")).to_have_count(0)

    # Undo and redo step through applied versions with on_apply.
    undo, redo = page.get_by_role("button", name="Undo", exact=True), page.get_by_role("button", name="Redo", exact=True)
    expect(redo).to_be_disabled()
    undo.click()
    wait_until(lambda: state.artifacts["dash"].applied == 5, page)
    expect(redo).to_be_enabled()
    redo.click()
    wait_until(lambda: state.artifacts["dash"].applied == 6, page)

    # Undoing the run goes back to the version before it; a row restores one version.
    page.get_by_role("button", name="Undo run").click()
    wait_until(lambda: state.artifacts["dash"].applied == 3, page)
    page.get_by_role("button", name="Restore v5").click()
    wait_until(lambda: state.artifacts["dash"].applied == 5, page)
    assert applied == [5, 6, 3, 5]
    assert [e.action for e in state.events if isinstance(e, ev.ArtifactApplied)] == ["undo", "redo", "undo", None]
    expect(page.get_by_text("v5 · ", exact=False)).to_contain_text("applied")


def test_history_toggle_and_new_runs(page):
    state = history()
    drawer = AgentDrawer(state=state, main=pmui.Typography("The app"), width=420, open=True, tab="artifacts")
    serve_component(page, pmui.Page(main=[drawer]))

    toggle = page.get_by_role("button", name="Version history")
    expect(page.get_by_text("Which regions refund most?")).to_be_visible()
    # Without on_apply there is nothing to undo or restore.
    expect(page.get_by_role("button", name="Undo", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Restore v5")).to_have_count(0)
    toggle.click()
    expect(page.get_by_text("Which regions refund most?")).to_be_hidden()
    toggle.click()
    expect(page.get_by_text("Which regions refund most?")).to_be_visible()

    state.apply(ev.RunStarted(run_id="r3", input="Sort the table"))
    state.apply(ev.ArtifactPublished(artifact_id="dash", value=[1], applied=True, run_id="r3", actor="agent", metadata={"op": "set_config"}))
    expect(page.get_by_text("set_config")).to_be_visible()
    expect(page.get_by_text("Place refund_table")).to_have_count(0)
    expect(toggle).to_contain_text("v7")
