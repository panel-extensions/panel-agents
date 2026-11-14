import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents.components._diff import DiffViewer

pytestmark = pytest.mark.ui


def test_diff_viewer_unified_and_split(page):
    viewer = DiffViewer(old="a\nb\nc", new="a\nB\nc\nd", old_label="v1", new_label="v2")
    serve_component(page, viewer)

    diff = page.locator(".diff")
    expect(diff.locator(".diff-header")).to_contain_text("v1 → v2")
    expect(diff.locator(".stat-add")).to_have_text("+2")
    expect(diff.locator(".stat-del")).to_have_text("−1")
    expect(diff.locator("tr.insert")).to_have_count(2)
    expect(diff.locator("tr.delete td.code")).to_have_text("b")

    viewer.mode = "split"
    expect(diff.locator("td.code.insert")).to_have_count(2)
    expect(diff.locator("td.code.delete")).to_have_text("b")
    wait_until(lambda: viewer.mode == "split", page)
