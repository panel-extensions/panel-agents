import pytest

pytest.importorskip("playwright")
pytest.importorskip("pydantic_ai")

from panel.io.state import state as pn_state
from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from panel_agents import AgentDrawer, Mention, ModelChoice
from panel_agents.adapters.pydantic_ai import AgentRunner

pytestmark = pytest.mark.ui

# The element with the focus, looking into shadow roots.
FOCUSED = """() => {
  let a = document.activeElement
  while (a && a.shadowRoot && a.shadowRoot.activeElement) { a = a.shadowRoot.activeElement }
  return a ? (a.getAttribute("aria-label") || a.getAttribute("placeholder") || a.tagName) : null
}"""


@pytest.fixture(autouse=True)
def cleanup_servers():
    try:
        yield
    finally:
        pn_state.reset()


def echoing():
    async def stream(messages, info):
        prompt = next(
            part.content
            for message in reversed(messages) if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)
        )
        yield f"Asked: {prompt}"

    return FunctionModel(stream_function=stream, model_name="echo")


def drawer(models=()):
    runner = AgentRunner(Agent(echoing()))
    chat_params = {"models": [ModelChoice(name, model=echoing()) for name in models]} if models else {}
    return AgentDrawer(runner, open=True, chat_params=chat_params)


def test_a_mention_shows_in_the_input_and_starts_the_message(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("sales_by_region", icon="bar_chart")]
    serve_component(page, agent_drawer)

    mention = page.locator(".agent-chat-mention")
    expect(mention).to_have_count(1)
    expect(mention.locator(".pa-mention")).to_have_text("bar_chartsales_by_region")
    box = page.get_by_placeholder("Ask the agent…")
    text, chip = box.bounding_box(), mention.bounding_box()
    # In the input, under the text.
    assert chip["y"] >= text["y"] + text["height"] - 1
    assert abs(chip["x"] - text["x"]) <= 2

    box.fill("make it a bar chart")
    box.press("Enter")
    expect(page.get_by_text("Asked: @sales_by_region make it a bar chart")).to_be_visible()
    expect(page.get_by_text("@sales_by_region make it a bar chart", exact=True)).to_be_visible()
    expect(mention).to_have_count(0)


def test_a_mention_can_be_removed_and_shares_the_row_with_the_model(page):
    agent_drawer = drawer(models=["alpha", "beta"])
    agent_drawer.chat.mentions = [Mention("sales_by_region")]
    serve_component(page, agent_drawer)

    mention = page.locator(".agent-chat-mention")
    picker = page.locator(".pa-model-button")
    expect(mention).to_be_visible()
    expect(picker).to_be_visible()

    def one_row():
        a, b = mention.bounding_box(), picker.bounding_box()
        return abs((a["y"] + a["height"] / 2) - (b["y"] + b["height"] / 2)) <= 2 and b["x"] > a["x"] + a["width"]

    wait_until(one_row, page)
    page.get_by_role("button", name="Remove sales_by_region").click()
    expect(mention).to_have_count(0)
    wait_until(lambda: agent_drawer.chat.mentions == [], page)
    expect(picker).to_be_visible()


def test_enter_on_a_remove_button_removes_the_mention_and_sends_nothing(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("sales_by_region")]
    serve_component(page, agent_drawer)

    box = page.get_by_placeholder("Ask the agent…")
    box.fill("half typed")
    page.get_by_role("button", name="Remove sales_by_region").press("Enter")
    expect(page.locator(".pa-mention")).to_have_count(0)
    wait_until(lambda: agent_drawer.chat.mentions == [], page)
    page.wait_for_timeout(300)
    assert agent_drawer.chat.feed.objects == []
    expect(box).to_have_value("half typed")
    assert page.evaluate(FOCUSED) == "Ask the agent…"


def test_removing_a_mention_puts_the_focus_back_in_the_input(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("a"), Mention("b")]
    serve_component(page, agent_drawer)

    page.get_by_role("button", name="Remove a").click()
    expect(page.locator(".pa-mention")).to_have_count(1)
    assert page.evaluate(FOCUSED) == "Ask the agent…"
    page.get_by_role("button", name="Remove b").press(" ")
    expect(page.locator(".pa-mention")).to_have_count(0)
    assert page.evaluate(FOCUSED) == "Ask the agent…"


def test_only_the_remove_button_of_a_mention_takes_the_focus(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("a"), Mention("b")]
    serve_component(page, agent_drawer)

    page.get_by_placeholder("Ask the agent…").focus()
    page.keyboard.press("Tab")
    assert page.evaluate(FOCUSED) == "Remove a"
    page.keyboard.press("Tab")
    assert page.evaluate(FOCUSED) == "Remove b"


def test_backspace_in_an_empty_input_removes_the_last_mention(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("a"), Mention("b")]
    serve_component(page, agent_drawer)

    box = page.get_by_placeholder("Ask the agent…")
    box.fill("x")
    box.press("Backspace")
    expect(box).to_have_value("")
    page.wait_for_timeout(300)
    assert agent_drawer.chat.mentions == [Mention("a"), Mention("b")]
    box.press("Backspace")
    expect(page.locator(".pa-mention")).to_have_count(1)
    wait_until(lambda: agent_drawer.chat.mentions == [Mention("a")], page)


def test_a_repeated_mention_is_written_once_as_one_name(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("Sales by region"), Mention("Sales by region")]
    serve_component(page, agent_drawer)

    expect(page.locator(".pa-mention")).to_have_count(1)
    box = page.get_by_placeholder("Ask the agent…")
    box.fill("make it a bar chart")
    box.press("Enter")
    expect(page.get_by_text('@"Sales by region" make it a bar chart', exact=True)).to_be_visible()


def test_enter_on_the_single_model_sends_nothing(page):
    agent_drawer = drawer(models=["alpha"])
    serve_component(page, agent_drawer)

    box = page.get_by_placeholder("Ask the agent…")
    box.fill("half typed")
    page.locator(".pa-model-fixed").press("Enter")
    page.wait_for_timeout(500)
    assert agent_drawer.chat.feed.objects == []
    expect(box).to_have_value("half typed")


LONG = "Monthly revenue by customer segment"


@pytest.mark.parametrize("width", [390, 1280], ids=["narrow", "default-drawer"])
@pytest.mark.parametrize("labels", [[LONG], ["sales_by_region", "orders_table", "customer_segments"]], ids=["long", "three"])
def test_mentions_keep_the_model_picker_and_send_button_in_the_input(page, width, labels):
    page.set_viewport_size({"width": width, "height": 720})
    agent_drawer = drawer(models=["alpha", "beta"])
    agent_drawer.chat.mentions = [Mention(label, icon="bar_chart") for label in labels]
    serve_component(page, agent_drawer)

    removes = [page.get_by_role("button", name=f"Remove {label}") for label in labels]
    picker = page.locator(".pa-model-button")
    send = page.get_by_role("button", name="Send message")
    for control in [*removes, picker, send]:
        expect(control).to_be_visible()
    field = page.locator(".MuiOutlinedInput-root").bounding_box()

    def inside(box):
        return (
            box["x"] >= field["x"] and box["x"] + box["width"] <= field["x"] + field["width"]
            and box["y"] >= field["y"] and box["y"] + box["height"] <= field["y"] + field["height"]
        )

    def laid_out():
        boxes = [control.bounding_box() for control in [*removes, picker]]
        edge = send.bounding_box()["x"]
        return all(inside(box) and box["x"] + box["width"] <= edge for box in boxes) and inside(send.bounding_box())

    wait_until(laid_out, page)


def test_a_long_label_is_cut_short_with_its_full_name_on_hover(page):
    page.set_viewport_size({"width": 390, "height": 720})
    agent_drawer = drawer(models=["alpha", "beta"])
    agent_drawer.chat.mentions = [Mention(LONG)]
    serve_component(page, agent_drawer)

    label = page.locator(".pa-mention-label")
    expect(label).to_have_attribute("title", LONG)
    assert label.evaluate("e => e.scrollWidth > e.clientWidth")
    expect(page.get_by_role("button", name=f"Remove {LONG}")).to_be_visible()


def test_mentions_writing_the_same_name_show_as_one_chip(page):
    agent_drawer = drawer()
    agent_drawer.chat.mentions = [Mention("orders_table"), Mention("orders_table", icon="table_chart")]
    serve_component(page, agent_drawer)

    expect(page.locator(".pa-mention")).to_have_count(1)
    page.get_by_role("button", name="Remove orders_table").click()
    expect(page.locator(".pa-mention")).to_have_count(0)
    wait_until(lambda: agent_drawer.chat.mentions == [], page)
