import re

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
    expect(page.get_by_text("v6 applied · ", exact=False)).to_be_visible()
    expect(page.get_by_text("ann edited")).to_be_visible()
    expect(page.get_by_text("Add node paid_by_region")).to_have_count(0)

    # Nothing in the timeline is wider than the drawer.
    widths = page.locator(".MuiListItemButton-root").evaluate_all("els => els.map(e => e.getBoundingClientRect().right)")
    assert widths and max(widths) <= 1000

    # Opening an older run shows its versions; viewing one does not apply it.
    page.get_by_text("What is paid revenue by region?").click()
    expect(page.get_by_text("Add node paid_by_region")).to_be_visible()
    page.get_by_text("Add node paid_by_region").click()
    expect(page.get_by_role("button", name="Version history, v1 shown")).to_have_attribute("aria-pressed", "true")
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
    # Undo's tooltip opens below it and lets clicks through, so Redo right after it is reachable.
    expect(page.get_by_role("tooltip", name="Undo")).to_be_visible()
    box = redo.bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    wait_until(lambda: state.artifacts["dash"].applied == 6, page)

    # Undoing the run goes back to the version before it; a row restores one version.
    # A row's actions show when it is hovered.
    page.locator(".MuiListItemButton-root").filter(has_text="Which regions refund most?").hover()
    page.get_by_role("button", name="Undo run").click()
    wait_until(lambda: state.artifacts["dash"].applied == 3, page)
    page.locator(".MuiListItemButton-root").filter(has_text="Add node refund_table").hover()
    page.get_by_role("button", name="Restore v5").click()
    wait_until(lambda: state.artifacts["dash"].applied == 5, page)
    assert applied == [5, 6, 3, 5]
    assert [e.action for e in state.events if isinstance(e, ev.ArtifactApplied)] == ["undo", "redo", "undo", None]
    expect(page.get_by_text("v5 applied · ", exact=False)).to_be_visible()


def test_history_toggle_and_new_runs(page):
    state = history()
    drawer = AgentDrawer(state=state, main=pmui.Typography("The app"), width=420, open=True, tab="artifacts")
    serve_component(page, pmui.Page(main=[drawer]))

    toggle = page.get_by_role("button", name=re.compile("^Version history"))
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
    expect(page.get_by_role("button", name="Version history, v7 shown")).to_be_visible()


_ROWS = """list => [...list.querySelectorAll('.MuiListItemButton-root')].map(row => {
    const icon = row.querySelector('.MuiListItemIcon-root .MuiIcon-root').getBoundingClientRect()
    return {
        label: row.querySelector('.MuiListItemText-primary').textContent,
        secondary: row.querySelector('.MuiListItemText-secondary')?.textContent ?? '',
        nested: !!row.parentElement.closest('.MuiCollapse-root'),
        caret: !!row.querySelector('.MuiIconButton-root:not([title])'),
        x: icon.x, height: row.getBoundingClientRect().height,
    }
})"""


@pytest.mark.parametrize("width", [360, 480])
def test_timeline_layout(page, width):
    from panel_agents.components._timeline import BLOCKED_NOTE

    state = history()
    state.apply(ev.RunStarted(run_id="r3", input="Sort the refund table"))
    state.apply(ev.ArtifactPublished(
        artifact_id="dash", value={"edits": 7}, kind="json", title="Dashboard", applied=True, run_id="r3",
        actor="agent", metadata={"label": "Sorted refund_table", "detail": "Sorted refund_table by refunds."},
    ))
    state.apply(ev.RunFinished(run_id="r3"))
    drawer = AgentDrawer(
        state=state, main=pmui.Typography("The app"), width=width, open=True, tab="artifacts",
        on_apply=lambda artifact, version: None,
    )
    page.set_viewport_size({"width": 1000, "height": 900})
    serve_component(page, pmui.Page(main=[drawer]))

    timeline = page.locator(".MuiList-root").filter(has_text="ann edited").first
    expect(timeline.get_by_text("Sort the refund table")).to_be_visible()
    timeline.get_by_text("What is paid revenue by region?").click()
    expect(timeline.get_by_text("Add node paid_by_region")).to_be_visible()
    rows = {row["label"]: row for row in timeline.evaluate(_ROWS)}

    # A run, or a person's edit, that made one version is one row without a caret or rows below it.
    for label in ("Sort the refund table", "ann edited"):
        assert not rows[label]["nested"] and not rows[label]["caret"]
    assert rows["Sort the refund table"]["secondary"].startswith("v7 applied · ")
    assert rows["What is paid revenue by region?"]["caret"] and rows["Place revenue_chart"]["nested"]

    # The reason a run cannot be undone is a muted indicator, not part of the meta line.
    assert all(BLOCKED_NOTE not in row["secondary"] for row in rows.values())
    ann = timeline.locator(".MuiListItemButton-root").filter(has_text="ann edited")
    expect(ann.get_by_role("button", name=BLOCKED_NOTE)).to_be_visible()

    # Every row has the same height, and every icon in a column has the same x.
    assert len({round(row["height"]) for row in rows.values()}) == 1
    for nested in (False, True):
        xs = [row["x"] for row in rows.values() if row["nested"] == nested]
        assert len(xs) >= 2 and max(xs) - min(xs) <= 1, (nested, xs)
    top = min(row["x"] for row in rows.values() if not row["nested"])
    nested_x = min(row["x"] for row in rows.values() if row["nested"])
    assert nested_x > top + 8

    # The header has each action once, all the same size and centred on one line.
    expect(page.get_by_role("button", name="Undo", exact=True)).to_have_count(1)
    expect(page.get_by_role("button", name="Redo", exact=True)).to_have_count(1)
    expect(page.get_by_role("button", name=re.compile("^Download"))).to_have_count(1)
    expect(page.get_by_role("combobox", name="Artifact")).to_have_count(0)
    controls = [
        page.get_by_role("button", name="Undo", exact=True), page.get_by_role("button", name="Redo", exact=True),
        page.get_by_role("button", name=re.compile("^Version history")), page.get_by_role("button", name=re.compile("^Download")),
    ]
    boxes = [control.bounding_box() for control in controls]
    assert max(b["height"] for b in boxes) - min(b["height"] for b in boxes) <= 1, boxes
    centres = [b["y"] + b["height"] / 2 for b in boxes]
    assert max(centres) - min(centres) <= 1, boxes
    panel = page.locator(".MuiDrawer-paper").last.bounding_box()
    assert max(b["x"] + b["width"] for b in boxes) <= panel["x"] + panel["width"]

    # The history toggle shows its tooltip on hover, like the other buttons.
    controls[2].hover()
    expect(page.get_by_role("tooltip", name="Version history, v7 shown")).to_be_visible()
    page.mouse.move(0, 0)

    # Trailing buttons sit in columns from the right edge, shared with the header's buttons.
    columns = {}
    for row in timeline.locator(".MuiListItemButton-root").all():
        row.hover()
        rights = sorted(
            (b["x"] + b["width"] for b in (button.bounding_box() for button in row.locator(".MuiIconButton-root").all()) if b),
            reverse=True,
        )
        for i, right in enumerate(rights):
            columns.setdefault(i, []).append(right)
    columns[0].append(boxes[-1]["x"] + boxes[-1]["width"])
    columns.setdefault(1, []).append(boxes[-2]["x"] + boxes[-2]["width"])
    assert len(columns[1]) >= 3
    for i, rights in columns.items():
        assert max(rights) - min(rights) <= 1, (i, rights)

    # Viewing another version offers to apply it once, in the bar above the list, not again on its row.
    timeline.get_by_text("Add node paid_by_region").click()
    expect(page.get_by_role("button", name="Revert to v1")).to_be_visible()
    expect(page.get_by_text("Viewing v1, v7 is applied")).to_be_visible()
    expect(page.get_by_role("button", name="Restore v1")).to_have_count(0)
