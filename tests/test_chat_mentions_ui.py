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
    expect(mention.locator(".MuiChip-root")).to_have_text("bar_chartsales_by_region")
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
