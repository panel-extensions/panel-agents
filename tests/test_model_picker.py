import asyncio

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from panel_agents import AgentChat, AgentDrawer, ModelChoice, ModelPicker, RunState
from panel_agents.adapters.pydantic_ai import Cassette

from .test_pydantic_ai import make_runner


def answering(name, gate=None):
    async def stream(messages, info):
        if gate is not None:
            await gate.wait()
        yield f"Answered by {name}."

    return FunctionModel(stream_function=stream, model_name=name)


def models_of(state):
    return [run.models for run in state.runs.values()]


def test_choices_are_coerced():
    assert ModelChoice.coerce("openai:a") == ModelChoice("openai:a")
    choice = ModelChoice.coerce({"id": "openai:a", "label": "A", "description": "Cheap"})
    assert (choice.name, choice.target, choice.description) == ("A", "openai:a", "Cheap")
    model = answering("b")
    assert ModelChoice("b", model=model).target is model
    with pytest.raises(TypeError):
        ModelChoice.coerce(3)


def test_the_unavailable_reason_names_what_is_missing(monkeypatch):
    runner = make_runner(Agent(answering("a")))
    assert runner.unavailable_reason("test") is None
    assert runner.unavailable_reason(answering("b")) is None
    assert runner.unavailable_reason("bogus:model") == "Unknown model: bogus:model"

    hint = (
        "Set the `OPENAI_API_KEY` environment variable or pass it via `OpenAIProvider(api_key=...)` "
        "to use the OpenAI provider. See https://ai.pydantic.dev/models/openai/ for the other options "
        "this provider takes when it is configured in code rather than through the environment."
    )

    def missing_key(model):
        raise UserError(hint + "\nmore detail")

    monkeypatch.setattr("pydantic_ai.models.infer_model", missing_key)
    assert runner.unavailable_reason("openai:x") == "Set OPENAI_API_KEY to use this model"

    def long_error(model):
        raise UserError("A long problem. " * 20 + "\nmore")

    monkeypatch.setattr("pydantic_ai.models.infer_model", long_error)
    assert runner.unavailable_reason("x:y") == ("A long problem. " * 20).strip()

    def missing_package(model):
        raise ImportError("Please install the `openai` package \u2014 `pip install x`")

    monkeypatch.setattr("pydantic_ai.models.infer_model", missing_package)
    assert runner.unavailable_reason("openai:x") == "Please install the `openai` package"


async def test_the_picked_model_applies_from_the_next_run():
    gate = asyncio.Event()
    a, b = answering("a", gate), answering("b")
    runner = make_runner(Agent(a))
    chat = AgentChat(runner=runner, models=[ModelChoice("a", model=a), ModelChoice("b", label="B", model=b)])
    assert chat.model_id == "a" and runner.model is a

    first = asyncio.ensure_future(runner.run("one"))
    await asyncio.sleep(0.05)
    chat.model_picker.value = "b"
    assert chat.model_id == "b" and runner.model is b
    gate.set()
    assert await first == "Answered by a."
    assert await runner.run("two") == "Answered by b."
    assert models_of(runner.state) == [["a"], ["b"]]


async def test_a_model_given_to_the_run_wins():
    a, b = answering("a"), answering("b")
    runner = make_runner(Agent(a), model=b)
    assert await runner.run("one") == "Answered by b."
    assert await runner.run("two", model=a) == "Answered by a."
    runner.run_kwargs = {"model": Cassette([ModelResponse([TextPart("Played.")], model_name="cassette-a")])}
    assert await runner.run("three") == "Played."
    assert models_of(runner.state) == [["b"], ["a"], ["cassette-a"]]


async def test_usage_is_counted_per_model():
    a, b = answering("a"), answering("b")
    runner = make_runner(Agent(a))
    await runner.run("one")
    runner.model = b
    await runner.run("two")
    first, second = runner.state.runs.values()
    assert (first.models, second.models) == (["a"], ["b"])
    assert all(run.requests == 1 and run.usage[0].model == run.models[0] for run in (first, second))


def test_unusable_models_are_disabled_with_the_reason():
    runner = make_runner(Agent(answering("a")))
    picker = ModelPicker(
        runner=runner, models=["bogus:x", {"id": "test", "label": "Test", "description": "Free"}, "bogus:y"],
    )
    assert picker.value == "test" and runner.model == "test"
    assert picker.unavailable == {"bogus:x": "Unknown model: bogus:x", "bogus:y": "Unknown model: bogus:y"}
    assert picker._menu.options == [
        {"id": "bogus:x", "label": "bogus:x", "reason": "Unavailable: Unknown model: bogus:x"},
        {"id": "test", "label": "Test", "description": "Free"},
        {"id": "bogus:y", "label": "bogus:y", "reason": "Unavailable: Unknown model: bogus:y"},
    ]
    assert picker._menu.visible and picker._menu.label == "Model"

    picker.value = "bogus:x"
    assert picker.value == "test"


def test_models_sharing_a_label_or_id_stay_apart():
    picker = ModelPicker(models=[
        {"id": "openai:a", "label": "Fast"}, {"id": "anthropic:b", "label": "Fast"}, "openai:a", "openai:c",
    ])
    assert [o["label"] for o in picker._menu.options] == ["Fast (openai:a)", "Fast (anthropic:b)", "openai:c"]
    assert [c.id for c in picker.choices] == ["openai:a", "anthropic:b", "openai:c"]
    assert not ModelPicker(models=["test", "test"])._menu.visible


def test_the_picker_shows_only_with_a_choice():
    runner = make_runner(Agent(answering("a")))
    chat = AgentChat(runner=runner)
    assert chat.model_picker._menu.visible is False and runner.model is None
    chat.models = ["test"]
    assert chat.model_picker._menu.visible is False and chat.model_id == "test"
    chat.models = ["test", "bogus:x"]
    assert chat.model_picker._menu.visible
    assert chat.feed._widget.footer_objects == [chat.model_picker._menu]
    assert AgentChat(state=RunState(), readonly=True, models=["test", "bogus:x"]).model_picker is None


def test_a_remembered_model_is_restored_only_if_it_is_offered_and_usable():
    a, b = answering("a"), answering("b")
    runner = make_runner(Agent(a))
    picker = ModelPicker(
        runner=runner, remember="app-model", value="a",
        models=[ModelChoice("a", model=a), ModelChoice("b", model=b), "bogus:x"],
    )
    menu = picker._menu
    assert (menu.storage_key, menu.value) == ("app-model", "a")
    menu.stored = "gone"
    menu.stored = "bogus:x"
    assert picker.value == "a"
    menu.stored = "b"
    assert picker.value == "b" and runner.model is b and menu.value == "b"
    assert ModelPicker(models=["a", "b"])._menu.storage_key == ""


def test_the_drawer_passes_models_to_its_chat():
    runner = make_runner(Agent(answering("a")))
    drawer = AgentDrawer(runner, chat_params={"models": ["test", "bogus:x"], "remember_model": "k"})
    picker = drawer.chat.model_picker
    assert picker.value == "test" and picker.remember == "k" and runner.model == "test"


def test_the_document_layout_has_the_picker_below_its_input():
    runner = make_runner(Agent(answering("a")))
    chat = AgentChat(runner=runner, layout="document", models=["test", "bogus:x"])
    assert chat.feed._widget.footer_objects == [chat.model_picker._menu]
    assert chat.model_id == "test" and runner.model == "test"
