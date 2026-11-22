import re

import pytest

pytest.importorskip("playwright")
pytest.importorskip("pydantic_ai")

from panel.io.state import state as pn_state
from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel

from panel_agents import AgentDrawer, ModelChoice
from panel_agents.adapters.pydantic_ai import AgentRunner

pytestmark = pytest.mark.ui


@pytest.fixture(autouse=True)
def cleanup_servers():
    try:
        yield
    finally:
        pn_state.reset()


def answering(name):
    async def stream(messages, info):
        yield f"Answered by {name}."

    return FunctionModel(stream_function=stream, model_name=name)


MODELS = {"alpha-model": answering("alpha-model"), "beta-model": answering("beta-model")}


def drawer(models, remember=None, layout="chat"):
    runner = AgentRunner(Agent(MODELS["alpha-model"]))
    choices = [
        ModelChoice(name, label=name.split("-")[0].title(), model=MODELS[name]) if name in MODELS else name
        for name in models
    ]
    return AgentDrawer(runner, open=True, chat_params={"models": choices, "remember_model": remember, "layout": layout})


def send(page, text):
    box = page.get_by_placeholder("Ask the agent…")
    box.fill(text)
    box.press("Enter")


def test_one_model_is_named_without_a_menu(page):
    serve_component(page, drawer(["alpha-model"]))
    label = page.locator(".pa-model-fixed")
    expect(label).to_be_visible()
    expect(label).to_have_text("Model: Alpha")
    expect(label).to_have_attribute("role", "note")
    expect(label).to_have_attribute("aria-label", "Model: Alpha, alpha-model")
    # The id is a tooltip that keyboard and screen reader users reach too.
    tip = page.locator(".pa-model-tip")
    expect(tip).to_have_attribute("role", "tooltip")
    expect(label).to_have_attribute("aria-describedby", tip.get_attribute("id"))
    expect(tip).to_be_hidden()
    expect(tip).to_have_text("alpha-model")
    label.focus()
    expect(tip).to_be_visible()
    label.press("Escape")
    expect(tip).to_be_hidden()
    label.blur()
    label.hover()
    expect(tip).to_be_visible()
    expect(page.get_by_role("button", name=re.compile("^Model"))).to_have_count(0)
    send(page, "Hi")
    expect(page.get_by_text("Answered by alpha-model.")).to_be_visible()


def test_an_unusable_single_model_looks_unavailable_and_says_why(page):
    serve_component(page, drawer(["bogus:model"]))
    label = page.locator(".pa-model-fixed")
    expect(label).to_have_text("Model: bogus:model unavailable")
    expect(label).to_have_class(re.compile(r"\bpa-model-unavailable\b"))
    label.focus()
    tip = page.locator(".pa-model-tip")
    expect(tip).to_be_visible()
    expect(tip).to_have_text("Unavailable: Unknown model: bogus:model")
    expect(label).to_have_attribute(
        "aria-label", "Model: bogus:model unavailable, Unavailable: Unknown model: bogus:model"
    )
    expect(label).to_have_attribute("aria-describedby", tip.get_attribute("id"))


def test_the_focus_stays_when_the_menu_turns_into_the_label(page):
    component = drawer(["alpha-model", "beta-model"])
    serve_component(page, component)
    button = page.get_by_role("button", name="Model: Alpha")
    button.focus()
    component.chat.models = component.chat.models[:1]
    label = page.locator(".pa-model-fixed")
    expect(label).to_be_focused()
    component.chat.models = [*component.chat.models, ModelChoice("beta-model", label="Beta", model=MODELS["beta-model"])]
    expect(page.get_by_role("button", name="Model: Alpha")).to_be_focused()


def test_the_label_turns_into_a_menu_when_models_are_added(page):
    component = drawer(["alpha-model"])
    serve_component(page, component)
    expect(page.locator(".pa-model-fixed")).to_be_visible()
    component.chat.models = [*component.chat.models, ModelChoice("beta-model", label="Beta", model=MODELS["beta-model"])]
    expect(page.get_by_role("button", name="Model: Alpha")).to_be_visible()
    expect(page.locator(".pa-model-fixed")).to_be_hidden()
    component.chat.models = []
    expect(page.get_by_role("button", name="Model: Alpha")).to_be_hidden()
    expect(page.locator(".pa-model-fixed")).to_be_hidden()


@pytest.mark.parametrize("layout", ["chat", "document"])
def test_a_picked_model_answers_the_next_request_and_shows_in_the_trace(page, layout):
    component = drawer(["alpha-model", "beta-model"], layout=layout)
    serve_component(page, component)
    picker = page.get_by_role("button", name="Model: Alpha")
    expect(picker).to_have_text("Alpha")
    send(page, "First")
    expect(page.get_by_text("Answered by alpha-model.")).to_be_visible()

    picker.click()
    page.get_by_role("listbox", name="Model").get_by_role("option", name="Beta").click()
    picker = page.get_by_role("button", name="Model: Beta")
    expect(picker).to_have_text("Beta")
    wait_until(lambda: component.runner.model is MODELS["beta-model"], page)
    send(page, "Second")
    expect(page.get_by_text("Answered by beta-model.")).to_be_visible()
    assert [run.models for run in component.state.runs.values()] == [["alpha-model"], ["beta-model"]]

    expect(page.get_by_text("Details", exact=True)).to_have_count(2, timeout=15000)
    page.get_by_text("Details", exact=True).last.click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_text("beta-model").first).to_be_visible()
    expect(dialog.get_by_text("alpha-model")).to_have_count(0)


def test_the_browser_remembers_the_picked_model(page):
    sessions = []

    def app():
        sessions.append(drawer(["alpha-model", "beta-model"], remember="pa-test-model"))
        return sessions[-1]

    serve_component(page, app)
    page.get_by_role("button", name="Model: Alpha").click()
    page.get_by_role("option", name="Beta").click()
    wait_until(lambda: page.evaluate("() => localStorage.getItem('pa-test-model')") == "beta-model", page)

    page.reload()
    expect(page.get_by_role("button", name="Model: Beta")).to_be_visible()
    send(page, "Hi")
    expect(page.get_by_text("Answered by beta-model.")).to_be_visible()
    assert len(sessions) == 2 and sessions[-1].runner.model is MODELS["beta-model"]

    page.evaluate("() => localStorage.setItem('pa-test-model', 'removed-model')")
    page.reload()
    expect(page.get_by_role("button", name="Model: Alpha")).to_be_visible()


def test_unavailable_models_show_why_and_cannot_be_picked(page):
    component = drawer(["alpha-model", "bogus:model", "beta-model"])
    serve_component(page, component)
    page.get_by_role("button", name="Model: Alpha").click()
    listbox = page.get_by_role("listbox", name="Model")
    broken = listbox.get_by_role("option", name=re.compile("^bogus:model"))
    expect(broken).to_have_attribute("aria-disabled", "true")
    expect(broken.get_by_text("Unavailable: Unknown model: bogus:model")).to_be_visible()
    broken.click(force=True)
    expect(listbox).to_be_visible()
    assert component.chat.model_picker.value == "alpha-model"

    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    expect(page.get_by_role("button", name="Model: Beta")).to_be_focused()
    expect(listbox).to_be_hidden()
    wait_until(lambda: component.runner.model is MODELS["beta-model"], page)


def test_the_keyboard_opens_and_picks_from_the_list(page):
    component = drawer(["alpha-model", "bogus:model", "beta-model"])
    serve_component(page, component)
    button = page.get_by_role("button", name="Model: Alpha")
    button.focus()
    page.keyboard.press("Enter")
    listbox = page.get_by_role("listbox", name="Model")
    expect(listbox).to_be_visible()
    page.keyboard.press("End")
    expect(listbox.get_by_role("option", name="Beta")).to_be_focused()
    page.keyboard.press("Home")
    expect(listbox.get_by_role("option", name="Alpha")).to_be_focused()
    page.keyboard.press("Escape")
    expect(listbox).to_be_hidden()
    expect(button).to_be_focused()
    page.keyboard.press(" ")
    page.keyboard.press("End")
    page.keyboard.press("Enter")
    expect(page.get_by_role("button", name="Model: Beta")).to_be_focused()
    wait_until(lambda: component.runner.model is MODELS["beta-model"], page)
    assert component.state.runs == {}


COMPOSER_GAPS = """
() => {
  const deep = (root) => [...root.querySelectorAll('*')].flatMap(e => [e, ...(e.shadowRoot ? deep(e.shadowRoot) : [])]);
  const all = deep(document);
  const box = all.find(e => e.classList.contains('MuiInputBase-root')).getBoundingClientRect();
  const content = (el) => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return {
      left: r.left + parseFloat(s.paddingLeft), top: r.top + parseFloat(s.paddingTop),
      bottom: r.bottom - parseFloat(s.paddingBottom),
    };
  };
  const text = content(all.find(e => e.tagName === 'TEXTAREA' && e.offsetHeight));
  const picker = all.find(e => (e.classList.contains('pa-model-button') || e.classList.contains('pa-model-fixed')) && e.offsetHeight);
  const last = picker ? content(picker) : text;
  return [text.top - box.top, box.bottom - last.bottom, last.top - text.bottom, last.left - text.left];
}
"""


@pytest.mark.parametrize("models", [["alpha-model"], ["alpha-model", "beta-model"]], ids=["one", "two"])
@pytest.mark.parametrize("text", ["", "First line\nSecond line"], ids=["empty", "multiline"])
def test_the_input_is_padded_evenly_above_and_below(page, models, text):
    serve_component(page, drawer(models))
    box = page.get_by_placeholder("Ask the agent…")
    box.fill(text)
    if len(models) > 1:
        expect(page.get_by_role("button", name="Model: Alpha")).to_be_visible()
    else:
        expect(page.locator(".pa-model-fixed")).to_be_visible()

    def even():
        top, bottom, *_ = page.evaluate(COMPOSER_GAPS)
        return top >= 16 and abs(top - bottom) <= 1

    wait_until(even, page)


@pytest.mark.parametrize("models", [["alpha-model"], ["alpha-model", "beta-model"]], ids=["one", "two"])
def test_the_model_sits_right_under_the_text(page, models):
    serve_component(page, drawer(models))
    expect(page.locator(".pa-model-fixed" if len(models) == 1 else ".pa-model-button")).to_be_visible()

    def tucked():
        *_, gap, indent = page.evaluate(COMPOSER_GAPS)
        return 0 <= gap <= 4 and abs(indent) <= 1

    wait_until(tucked, page)
