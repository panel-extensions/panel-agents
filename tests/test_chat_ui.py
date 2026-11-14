import pytest

pytest.importorskip("playwright")

from panel.io.state import state as pn_state
from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat
from panel_agents.components.tool_call import ToolCallCard

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def cleanup_servers():
    try:
        yield
    finally:
        pn_state.reset()


def test_card_expands_on_failure(page):
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.ToolCallStart(tool_call_id="t1", tool_name="run_query"),
        ev.ToolCallArgs(tool_call_id="t1", delta='{"sql": "SELECT qtr FROM sales"}'),
        ev.ToolCallEnd(tool_call_id="t1"),
    ])
    card = ToolCallCard(state.tool_calls["t1"], state=state)
    serve_component(page, card)

    expect(page.locator(".tc-name")).to_have_text("run_query")
    expect(page.locator(".tc-preview")).to_contain_text("SELECT qtr FROM sales")
    expect(page.locator(".tc-args")).to_have_count(0)

    state.apply(ev.ToolCallResult(tool_call_id="t1", content="Binder Error: column qtr not found", is_error=True))

    expect(page.locator(".tc-badge")).to_have_text("Failed")
    expect(page.locator(".MuiAlert-root")).to_contain_text("Binder Error: column qtr not found")
    expect(page.locator(".tc-args pre")).to_contain_text("SELECT qtr FROM sales")
    expect(page.locator("[aria-label='show more']")).to_have_attribute("aria-expanded", "true")


def test_chat_shows_streamed_text(page):
    state = RunState(flush_interval=0)
    chat = AgentChat(state, readonly=True, feed_params={"height": 500})
    serve_component(page, chat)

    state.apply(ev.RunStarted(run_id="r1", agent_name="Analyst"))
    state.apply(ev.TextMessageStart(message_id="m1", run_id="r1"))
    state.apply(ev.TextMessageContent(message_id="m1", delta="EMEA grew ", run_id="r1"))
    expect(page.locator(".markdown", has_text="EMEA grew")).to_be_visible()
    state.apply(ev.TextMessageContent(message_id="m1", delta="by **12%**.", run_id="r1"))
    expect(page.locator(".markdown", has_text="EMEA grew by 12%.")).to_be_visible()
    expect(page.locator(".markdown strong", has_text="12%")).to_be_visible()

    state.apply(ev.ToolCallStart(tool_call_id="t1", tool_name="run_query", run_id="r1"))
    expect(page.locator(".tc-name")).to_have_text("run_query")
    expect(page.locator("textarea")).to_have_count(0)


def test_chat_submit_invokes_callback(page):
    state = RunState(flush_interval=0)
    received = []

    async def callback(text):
        received.append(text)
        state.apply(ev.RunStarted(run_id="r1"))
        state.apply(ev.TextMessageStart(message_id="m1", run_id="r1"))
        state.apply(ev.TextMessageContent(message_id="m1", delta=f"You said: {text}", run_id="r1"))
        state.apply(ev.TextMessageEnd(message_id="m1", run_id="r1"))
        state.apply(ev.RunFinished(run_id="r1"))

    chat = AgentChat(state, callback=callback, feed_params={"height": 500})
    serve_component(page, chat)

    textarea = page.locator("textarea").first
    textarea.fill("hello agent")
    textarea.press("Enter")

    wait_until(lambda: received == ["hello agent"], page)
    expect(page.locator(".markdown", has_text="You said: hello agent")).to_be_visible()
    expect(page.locator(".markdown", has_text="hello agent").first).to_be_visible()
    assert [item.content for item in state.items] == ["hello agent", "You said: hello agent"]
