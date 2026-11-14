import pandas as pd
import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents.artifacts import ArtifactStore
from panel_agents.components.artifacts import ArtifactPane

pytestmark = pytest.mark.ui


@pytest.fixture
def store():
    store = ArtifactStore()
    for i, sql in enumerate(["SELECT a\nFROM t", "SELECT a, b\nFROM t", "SELECT a, b\nFROM t\nLIMIT 5"]):
        store.publish("query", sql, kind="sql", title="Query", tool_call_id=f"call_{i + 1}")
    old = pd.DataFrame({"v": [1, 2]}, index=["x", "y"])
    store.publish("result", old, title="Result")
    store.publish("result", old.assign(v=[1, 3]))
    return store


def test_artifact_pane_switch_version_and_compare(page, store):
    pane = ArtifactPane(store)
    serve_component(page, pane)

    items = page.locator(".MuiListItemButton-root")
    expect(items).to_have_count(2)
    expect(page.get_by_text("Result").first).to_be_visible()

    items.nth(0).click()
    wait_until(lambda: store.active == "query", page)
    expect(page.locator(".ace_content")).to_contain_text("LIMIT 5")

    page.locator(".select").first.click(force=True)
    page.locator(".MuiMenuItem-root").nth(0).click()
    wait_until(lambda: store["query"].current == 0, page)
    expect(page.locator(".ace_content")).not_to_contain_text("LIMIT")

    page.get_by_role("tab", name="Compare").click()
    view = pane.active_view
    wait_until(lambda: view.mode == "compare", page)
    diff = page.locator(".diff")
    expect(diff).to_be_visible()
    expect(diff.locator("tr.insert")).to_have_count(1)
    expect(diff.locator("tr.insert td.code")).to_have_text("LIMIT 5")

    page.get_by_role("button", name="Split").click()
    wait_until(lambda: view.diff_layout == "split", page)
    expect(diff.locator("td.code.insert")).to_have_text("LIMIT 5")


def test_artifact_pane_follows_published_artifact(page, store):
    pane = ArtifactPane(store)
    serve_component(page, pane)

    store.publish("notes", "Some *notes*", kind="text", title="Notes")
    expect(page.locator(".MuiListItemButton-root")).to_have_count(3)
    expect(page.get_by_text("Some notes")).to_be_visible()

    page.get_by_role("tab", name="Compare").click()
    wait_until(lambda: pane.active_view.mode == "compare", page)
    store.active = "result"
    expect(page.locator(".tabulator").first).to_be_visible()
    pane.active_view.mode = "compare"
    expect(page.get_by_text("1 cell changed")).to_be_visible()
    expect(page.get_by_role("tab", name="Compare")).to_have_attribute("aria-selected", "true")
