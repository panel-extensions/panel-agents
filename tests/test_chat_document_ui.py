import time

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
    from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart, UserPromptPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from panel_agents.adapters.pydantic_ai import AgentRunner, artifact_return

    def answering(messages):
        # Any tool result or retry since the question means the tools ran, so a
        # failing tool ends the run with an answer instead of calling tools again.
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                if any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in message.parts):
                    return True
                if any(isinstance(p, UserPromptPart) for p in message.parts):
                    return False
        return False

    async def stream(messages, info):
        if answering(messages):
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
    lines = [f"run {run.duration:.2f}s" if run.duration is not None else f"run {run.status}"]
    for item in state.items:
        begin = getattr(item, "started_at", None) or getattr(item, "created_at", None)
        end = getattr(item, "ended_at", None)
        name = getattr(item, "tool_name", None) or getattr(item, "model_name", None) or getattr(item, "kind", "")
        lines.append(
            f"{type(item).__name__} {name} {(begin or start) - start:.2f}"
            + (f" .. {end - start:.2f}" if end else "")
        )
    return "\n".join(lines)


def _wait_for_run(page, runner, stall: float = 60, limit: float = 300):
    """
    Wait for the runner's run to end. A slow machine only fails this when the
    run stops making progress for ``stall`` seconds, or runs past ``limit``,
    and the failure says where the run got to.
    """
    start = changed = time.monotonic()
    seen = -1
    while not (runner.state.runs and not runner.state.running):
        now = time.monotonic()
        count = len(runner.state.events)
        if count != seen:
            seen, changed = count, now
        if now - changed > stall or now - start > limit:
            where = _timeline(runner.state) if runner.state.runs else "no run started"
            raise AssertionError(f"The run made no progress for {now - changed:.0f}s after {now - start:.0f}s:\n{where}")
        page.wait_for_timeout(250)


def _ask_for_tables(page, chat, runner):
    serve_component(page, chat)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 3")
    box.press("Enter")
    _wait_for_run(page, runner)
    page.get_by_text("Here are the tables.").wait_for(timeout=30000)
    (run,) = runner.state.runs.values()
    return run.duration


# The runs that stream through an agent share one worker, so they do not slow each other down.
@pytest.mark.xdist_group("streaming-runs")
@pytest.mark.parametrize("attempt", range(4))
def test_every_cell_of_a_streaming_run_renders(page, attempt):
    page.set_viewport_size({"width": 1400, "height": 1000})
    # The same run in the chat layout gives the pace of this machine to compare with.
    runner, chat = _streaming_chat(3)
    chat_layout = AgentChat(runner=runner, feed_params={"height": 800})
    baseline = _ask_for_tables(page, chat_layout, runner)
    runner, chat = _streaming_chat(3)
    duration = _ask_for_tables(page, chat, runner)
    cells = page.locator(".artifact-cell")
    expect(cells).to_have_count(3)
    for i in range(3):
        expect(page.locator(".artifact-cell", has_text=f"table_{i}")).to_contain_text(str(i + 1))
    assert duration < 2 * baseline + 2, f"chat layout {baseline:.2f}s\n{_timeline(runner.state)}"


@pytest.mark.xdist_group("streaming-runs")
def test_a_run_past_max_cells_shows_the_rest_on_request(page):
    runner, chat = _streaming_chat(12)
    chat.max_cells = 5
    page.set_viewport_size({"width": 1400, "height": 1000})
    serve_component(page, chat)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 12")
    box.press("Enter")
    _wait_for_run(page, runner)
    page.get_by_text("Here are the tables.").wait_for(timeout=30000)
    expect(page.locator(".artifact-cell")).to_have_count(5)
    page.get_by_role("button", name="Show 7 more").click()
    expect(page.locator(".artifact-cell")).to_have_count(12)


@pytest.mark.xdist_group("streaming-runs")
def test_the_input_offers_stop_only_while_a_run_goes_on(page):
    runner, chat = _streaming_chat(1, slow=2)
    serve_component(page, chat)
    expect(page.get_by_role("button", name="Send message")).to_be_visible()
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 1")
    box.press("Enter")
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    _wait_for_run(page, runner)
    page.get_by_text("Here are the tables.").wait_for(timeout=30000)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    expect(page.get_by_role("button", name="Send message")).to_be_visible()


@pytest.mark.xdist_group("streaming-runs")
def test_a_feed_that_batches_a_runners_events_shows_the_answer(page):
    # As FlowDash's Explore view does: a second, document feed gets the runner's
    # events in batches a little later, in its own state.
    runner, _ = _streaming_chat(2)
    state = RunState(flush_interval=0)
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    def app():
        doc = pn_state.curdoc
        pending = []

        def flush():
            events = pending[:]
            pending.clear()
            state.apply_many(events)

        def forward(event):
            pending.append(event)
            if len(pending) == 1:
                doc.add_timeout_callback(flush, 50)

        runner.state.subscribe(forward)
        return AgentChat(state, runner=runner, layout="document", feed_params={"height": 800}, max_cells=20)

    serve_component(page, app)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("tables 2")
    box.press("Enter")
    _wait_for_run(page, runner)
    expect(page.get_by_text("Here are the tables.")).to_be_visible(timeout=30000)
    expect(page.locator(".artifact-cell")).to_have_count(2)
    page.wait_for_timeout(500)
    assert errors == []


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


def test_actions_wrap_below_the_title_on_a_narrow_card(page):
    actions = ActionRegistry()
    actions.add(lambda name: None, name="place")
    spec = ArtifactAction("place", "Add to dashboard", icon="add", arguments=lambda a: {"name": a.title})
    _, chat = _chat(readonly=True, action_registry=actions, artifact_actions=[spec])
    page.set_viewport_size({"width": 390, "height": 900})
    serve_component(page, chat)
    cell = page.locator(".artifact-cell").first
    button = cell.get_by_role("button", name="Add to dashboard")
    expect(button).to_be_visible()
    box, card = button.bounding_box(), cell.bounding_box()
    assert box["x"] + box["width"] <= card["x"] + card["width"]
