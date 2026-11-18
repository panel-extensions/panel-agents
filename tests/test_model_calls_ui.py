import panel_material_ui as pmui
import pytest

pytest.importorskip("playwright")

from panel.io.state import state as pn_state
from panel.tests.util import serve_component
from playwright.sync_api import expect

from panel_agents import RunState, RunTrace
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def cleanup_servers():
    try:
        yield
    finally:
        pn_state.reset()


def traced_run() -> RunState:
    """Two requests: the first issues two tool calls, the second answers."""
    state = RunState(flush_interval=0)
    state.add_user_message("What is paid revenue by region?", run_id="r1")
    t = 1_700_000_000.0
    events = [
        ev.RunStarted(run_id="r1", input="What is paid revenue by region?", timestamp=t),
        ev.ModelRequestStarted(request_id="q1", model="claude-haiku-4-5", run_id="r1", timestamp=t),
    ]
    for i, name in enumerate(("describe_table", "distinct_values")):
        events += [
            ev.ToolCallStart(tool_call_id=f"c{i}", tool_name=name, run_id="r1", timestamp=t + 1.2),
            ev.ToolCallArgs(tool_call_id=f"c{i}", delta='{"table": "orders"}', run_id="r1"),
            ev.ToolCallEnd(tool_call_id=f"c{i}", run_id="r1"),
        ]
    events += [
        ev.Usage(
            request_id="q1", model="claude-haiku-4-5", input_tokens=3912, output_tokens=148, cached_tokens=3000,
            cost_usd=0.0021, duration=1.42, tool_call_ids=["c0", "c1"], finish_reason="tool_call",
            provider_response_id="msg_01", run_id="r1", timestamp=t + 1.42,
        ),
        ev.ToolCallResult(tool_call_id="c0", content="ok", run_id="r1", timestamp=t + 1.6),
        ev.ToolCallResult(tool_call_id="c1", content="ok", run_id="r1", timestamp=t + 1.7),
        ev.ModelRequestStarted(request_id="q2", model="claude-haiku-4-5", run_id="r1", timestamp=t + 1.7),
        ev.TextMessageStart(message_id="a", run_id="r1", timestamp=t + 2.0),
        ev.TextMessageContent(message_id="a", delta="East leads with 31%.", run_id="r1"),
        ev.TextMessageEnd(message_id="a", run_id="r1"),
        ev.Usage(
            request_id="q2", model="claude-haiku-4-5", input_tokens=4406, output_tokens=96, cost_usd=0.0013,
            duration=1.3, finish_reason="stop", run_id="r1", timestamp=t + 3.0,
        ),
        ev.RunFinished(run_id="r1", timestamp=t + 3.2),
    ]
    state.apply_many(events)
    return state


_CENTER = """
(el) => { const r = el.getBoundingClientRect(); return r.top + r.height / 2 }
"""


def test_chat_folds_model_calls_and_shows_the_run_totals(page):
    chat = AgentChat(traced_run(), readonly=True, feed_params={"height": 600})
    serve_component(page, chat)

    expect(page.locator(".run-usage")).to_have_text("2 requests · 8.3k in · 244 out · $0.0034 · 3.2s total")
    expect(page.locator(".run-usage .usage")).to_have_attribute("title", "8,318 input tokens, 3,000 from the cache\n244 output tokens\n2.7s waiting for the model\nclaude-haiku-4-5")
    row = page.get_by_role("button", name="2 steps, 3.0s, expand")
    expect(row).to_be_visible()
    # Folded steps are only built once their group expands.
    expect(page.locator(".model-call-card")).to_have_count(0)

    row.click()
    expect(page.locator(".model-call-card")).to_have_count(2)
    card = page.locator(".model-call-card").first
    expect(card).to_be_visible()
    expect(card.locator(".tc-head")).to_contain_text("claude-haiku-4-5")
    expect(card.locator(".tc-head")).to_contain_text("3,912 in · 148 out")
    expect(card.locator(".tc-head")).to_have_attribute("title", "claude-haiku-4-5\n3,000 of 3,912 input tokens from the cache")
    # The second request answered without a tool call; it folds into the same group.
    expect(page.locator(".model-call-card").nth(1)).to_be_visible()

    card.locator("[aria-label='show more']").first.click()
    expect(card.locator("dd", has_text="q1")).to_be_visible()
    expect(card.locator("dd", has_text="describe_table · c0")).to_be_visible()


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_model_call_header_is_centered_on_its_status_icon(page, dark):
    chat = AgentChat(traced_run(), readonly=True, show_usage=False, feed_params={"height": 600})
    serve_component(page, pmui.Page(main=[chat], dark_theme=dark))
    page.get_by_role("button", name="2 steps, 3.0s, expand").click()
    card = page.locator(".model-call-card").first
    icon = card.locator("svg").first
    expect(icon).to_be_visible()
    text = card.locator(".mc-model")
    assert abs(icon.evaluate(_CENTER) - text.evaluate(_CENTER)) <= 2


@pytest.mark.parametrize("width", [400, 1000])
def test_trace_lists_the_runs_steps_on_one_time_axis(page, width):
    page.set_viewport_size({"width": width, "height": 900})
    trace = RunTrace(traced_run(), run_id="r1")
    serve_component(page, trace)

    totals = page.locator(".rt-totals")
    expect(totals).to_contain_text("Model requests2")
    expect(totals).to_contain_text("Input tokens8,3183,000 from cache")
    expect(totals).to_contain_text("Output tokens244")
    expect(totals).to_contain_text("Cost$0.0034")
    expect(totals).to_contain_text("Wall time3.2s2.7s in the model")
    expect(page.locator(".rt-question")).to_have_text("What is paid revenue by region?")
    expect(page.locator(".model-call-card")).to_have_count(2)
    expect(page.locator(".tool-call-card")).to_have_count(2)
    expect(page.locator(".rt-track")).to_have_count(4)

    model = page.locator(".model-call-card").first.bounding_box()
    tool = page.locator(".tool-call-card").first.bounding_box()
    assert tool["x"] - model["x"] == pytest.approx(24, abs=1)
    assert model["y"] < tool["y"] < page.locator(".model-call-card").nth(1).bounding_box()["y"]
    lefts = page.locator(".rt-track").evaluate_all("els => els.map(e => Math.round(e.getBoundingClientRect().left))")
    assert len(set(lefts)) == 1
    assert page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 0
    # The token counts give way before the model name is cut short.
    names = page.locator(".mc-model").evaluate_all("els => els.map(e => e.scrollWidth - e.clientWidth)")
    assert names and max(names) <= 0

    page.locator(".model-call-card").first.locator("[aria-label='show more']").first.click()
    details = page.locator(".model-call-card").first.locator("dl")
    expect(details).to_contain_text("finishtool_call")
    expect(details).to_contain_text("response idmsg_01")
    expect(details).to_contain_text("input3,912 tokens, 3,000 cached")
