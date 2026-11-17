import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.artifacts import ArtifactPane

pytestmark = pytest.mark.ui


def test_revert_to_an_earlier_version(page):
    state = RunState(flush_interval=0)
    for title in ["Sales", "Revenue"]:
        state.apply(ev.ArtifactPublished(artifact_id="doc", value={"title": title}, kind="json", title="Dashboard", applied=True))
    applied = []
    pane = ArtifactPane(state=state, on_apply=lambda artifact, version: applied.append(version.version))
    serve_component(page, pane)

    expect(page.get_by_text("json · 2 versions · v2 applied")).to_be_visible()
    state.artifacts["doc"].current = 1
    button = page.get_by_role("button", name="Revert to v1")
    expect(button).to_be_visible()
    button.click()

    wait_until(lambda: state.artifacts["doc"].applied_version.version == 1, page)
    assert applied == [1]
    expect(page.get_by_text("json · 2 versions · v1 applied")).to_be_visible()
    expect(button).to_be_hidden()
