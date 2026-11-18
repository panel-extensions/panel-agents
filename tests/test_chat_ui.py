import asyncio

import pandas as pd
import panel_material_ui as pmui
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
    expect(page.locator(".tc-args .codehilite pre")).to_contain_text("SELECT qtr FROM sales")
    expect(page.locator("[aria-label='show more']")).to_have_attribute("aria-expanded", "true")


@pytest.mark.parametrize("content", [
    pd.DataFrame({"column": ["order_id"], "type": ["BIGINT"]}),
    [{"column": "order_id", "type": "BIGINT"}],
    "| column | type |\n|---|---|\n| order_id | BIGINT |",
], ids=["dataframe", "records", "markdown"])
def test_card_result_table_uses_material_styling(page, content):
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.ToolCallStart(tool_call_id="t1", tool_name="describe_table"),
        ev.ToolCallEnd(tool_call_id="t1"),
        ev.ToolCallResult(tool_call_id="t1", content=content),
    ])
    card = ToolCallCard(state.tool_calls["t1"], state=state, collapsed=False)
    serve_component(page, card)

    header = page.locator("thead th", has_text="column")
    expect(header).to_have_css("text-transform", "uppercase")
    expect(header).to_have_css("border-top-width", "0px")
    expect(page.locator("table")).to_have_css("border-collapse", "separate")
    expect(page.locator("td", has_text="order_id")).to_have_css("border-left-width", "0px")


def test_chat_shows_streamed_text(page):
    state = RunState(flush_interval=0)
    chat = AgentChat(state, readonly=True, feed_params={"height": 500})
    serve_component(page, chat)

    state.apply(ev.RunStarted(run_id="r1", agent_name="Analyst"))
    state.apply(ev.TextMessageStart(message_id="m1", run_id="r1"))
    # The test changes the state from outside the server's thread: an update made while the new
    # message is still being rendered can be lost, so wait until it is. A message registers its
    # model only after its children's, so waiting for the text pane alone leaves a window.
    message = chat.view_for(state.messages["m1"])
    wait_until(lambda: bool(message._models), page)
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


def test_finished_steps_fold_into_a_row_that_expands(page):
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1"))
    for call_id, name in (("a", "read_report"), ("b", "list_regions")):
        state.apply_many([
            ev.ToolCallStart(tool_call_id=call_id, tool_name=name, run_id="r1", timestamp=100.0),
            ev.ToolCallEnd(tool_call_id=call_id, run_id="r1"),
            ev.ToolCallResult(tool_call_id=call_id, content="ok", run_id="r1", timestamp=100.5),
        ])
    chat = AgentChat(state, tool_labels={"read_report": "Read the report"})
    serve_component(page, chat)

    row = page.get_by_role("button", name="2 steps, 1.0s, expand")
    expect(row).to_be_visible()
    expect(page.get_by_text("Read the report")).to_be_hidden()
    row.click()
    expect(page.get_by_text("Read the report")).to_be_visible()
    expect(page.get_by_text("list_regions")).to_be_visible()
    expect(page.get_by_role("button", name="2 steps, 1.0s, collapse")).to_be_visible()


_INPUT_INSETS = """
(textarea) => {
  const root = textarea.closest(".MuiInputBase-root")
  const outline = root.querySelector("fieldset").getBoundingClientRect()
  const text = textarea.getBoundingClientRect().left + parseFloat(getComputedStyle(textarea).paddingLeft)
  const send = root.querySelector("button[aria-label='Send message'] svg").getBoundingClientRect()
  return {text: text - outline.left, send: outline.right - send.right}
}
"""


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_input_text_is_inset_like_the_send_button(page, theme):
    page.emulate_media(color_scheme=theme)
    chat = AgentChat(RunState(flush_interval=0), callback=lambda text: None)
    serve_component(page, chat)

    textarea = page.locator("textarea:visible").first
    expect(textarea).to_have_attribute("placeholder", "Ask the agent…")
    insets = textarea.evaluate(_INPUT_INSETS)
    # As far in as an outlined input's text (14px), and as the send icon is from the right.
    assert insets["text"] >= 14
    assert abs(insets["text"] - insets["send"]) <= 1


def _held_runs(state, received):
    """A callback whose runs last until ``finish`` is set, recording the texts sent."""
    finish = []

    async def callback(text):
        received.append(text)
        state.apply(ev.RunStarted(run_id=f"r{len(received)}"))
        while not finish:
            await asyncio.sleep(0.02)
        finish.clear()
        state.apply(ev.RunFinished(run_id=f"r{len(received)}"))

    return callback, finish


def test_input_keeps_focus_and_text_through_a_run(page):
    state = RunState(flush_interval=0)
    received = []
    callback, finish = _held_runs(state, received)
    serve_component(page, AgentChat(state, callback=callback))

    textarea = page.locator("textarea:visible").first
    textarea.click()
    textarea.press_sequentially("first")
    textarea.press("Enter")
    wait_until(lambda: received == ["first"] and state.running, page)

    # Typed during the run, without clicking back into the input; Enter waits for the run.
    expect(textarea).to_be_focused()
    page.keyboard.type("second")
    page.keyboard.press("Enter")
    page.wait_for_timeout(200)
    expect(textarea).to_have_value("second")
    expect(textarea).to_be_focused()
    assert received == ["first"]
    finish.append(True)
    wait_until(lambda: not state.running, page)

    expect(textarea).to_be_focused()
    page.keyboard.press("Enter")
    wait_until(lambda: received == ["first", "second"], page)
    finish.append(True)


def test_input_does_not_take_focus_back_after_a_run(page):
    state = RunState(flush_interval=0)
    received = []
    callback, finish = _held_runs(state, received)
    elsewhere = pmui.Button(label="Approve")
    serve_component(page, pmui.Column(AgentChat(state, callback=callback), elsewhere))

    textarea = page.locator("textarea:visible").first
    textarea.click()
    textarea.press_sequentially("first")
    textarea.press("Enter")
    wait_until(lambda: state.running, page)

    approve = page.get_by_role("button", name="Approve")
    approve.click()
    finish.append(True)
    wait_until(lambda: not state.running, page)
    page.wait_for_timeout(300)
    expect(approve).to_be_focused()
