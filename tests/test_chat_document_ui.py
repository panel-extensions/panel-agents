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
    _, chat = _chat(action_registry=actions, artifact_actions=[spec], callback=lambda text: None)
    serve_component(page, chat)
    cell = page.locator(".artifact-cell", has_text="paid_by_state")
    cell.get_by_role("button", name="Add to dashboard").click()
    wait_until(lambda: placed == ["paid_by_state"], page)
    expect(cell.locator(".artifact-cell-done")).to_contain_text("On dashboard")
    expect(page.locator(".artifact-cell", has_text="paid_by_region").get_by_role("button", name="Add to dashboard")).to_be_visible()


def _streaming_chat(tables: int, slow: float = 0):
    import asyncio as aio

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from panel_agents.adapters.pydantic_ai import AgentRunner, artifact_return

    async def stream(messages, info):
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in last.parts):
            for chunk in ["Here ", "are ", "the ", "tables."]:
                await aio.sleep(0.02)
                yield chunk
            return
        yield "Working on it. "
        for i in range(tables):
            yield {i + 1: DeltaToolCall(name="make_table", json_args='{"n": %d}' % i, tool_call_id=f"c{i}")}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain
    async def make_table(n: int):
        """Make a table."""
        await aio.sleep(slow)
        frame = pd.DataFrame({"region": ["East", "West"], "total": [n, n + 1]})
        return artifact_return("ok", f"table_{n}", frame, kind="table", title=f"table_{n}")

    runner = AgentRunner(agent, RunState(flush_interval=0))
    return runner, AgentChat(runner=runner, layout="document", feed_params={"height": 800}, max_cells=20)


def _timeline(state):
    (run,) = state.runs.values()
    start = run.started_at
    lines = [f"run {run.duration:.2f}s"]
    for item in state.items:
        begin = getattr(item, "started_at", None) or getattr(item, "created_at", None)
        end = getattr(item, "ended_at", None)
        name = getattr(item, "tool_name", None) or getattr(item, "model_name", None) or getattr(item, "kind", "")
        lines.append(
            f"{type(item).__name__} {name} {(begin or start) - start:.2f}"
            + (f" .. {end - start:.2f}" if end else "")
        )
    return "\n".join(lines)


@pytest.mark.parametrize("attempt", range(4))
def test_every_cell_of_a_streaming_run_renders(page, attempt):
    runner, chat = _streaming_chat(3)
    page.set_viewport_size({"width": 1400, "height": 1000})
    serve_component(page, chat)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 3")
    box.press("Enter")
    wait_until(lambda: bool(runner.state.runs) and not runner.state.running, page, timeout=90000)
    # Where a slow run spends its time, shown if it fails.
    print(_timeline(runner.state))
    page.get_by_text("Here are the tables.").wait_for(timeout=15000)
    cells = page.locator(".artifact-cell")
    expect(cells).to_have_count(3)
    for i in range(3):
        expect(page.locator(".artifact-cell", has_text=f"table_{i}")).to_contain_text(str(i + 1))
    (run,) = runner.state.runs.values()
    assert run.duration < 10, _timeline(runner.state)


def test_a_run_past_max_cells_shows_the_rest_on_request(page):
    runner, chat = _streaming_chat(12)
    chat.max_cells = 5
    page.set_viewport_size({"width": 1400, "height": 1000})
    serve_component(page, chat)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 12")
    box.press("Enter")
    page.get_by_text("Here are the tables.").wait_for(timeout=15000)
    expect(page.locator(".artifact-cell")).to_have_count(5)
    page.get_by_role("button", name="Show 7 more").click()
    expect(page.locator(".artifact-cell")).to_have_count(12)


def test_the_input_offers_stop_only_while_a_run_goes_on(page):
    runner, chat = _streaming_chat(1, slow=2)
    serve_component(page, chat)
    expect(page.get_by_role("button", name="Send message")).to_be_visible()
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 1")
    box.press("Enter")
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    page.get_by_text("Here are the tables.").wait_for(timeout=15000)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    expect(page.get_by_role("button", name="Send message")).to_be_visible()


def test_a_title_with_markup_is_shown_as_text(page):
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1"))
    state.add_user_message("Evil?", run_id="r1")
    state.publish_artifact(
        "evil", pd.DataFrame({"a": [1]}), kind="table", run_id="r1",
        title='<img src=x onerror="document.title=\'XSS\'"><b>bold</b>',
    )
    state.apply(ev.RunFinished(run_id="r1"))
    chat = AgentChat(state, layout="document", readonly=True, feed_params={"height": 500})
    serve_component(page, chat)
    title = page.locator(".artifact-cell-title")
    expect(title).to_contain_text("<b>bold</b>")
    assert title.locator("img, b").count() == 0
    page.wait_for_timeout(300)
    assert page.title() != "XSS"
