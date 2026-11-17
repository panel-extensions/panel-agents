import asyncio
import os
import threading
from typing import Annotated, Literal

import panel as pn
import pytest
from panel.io.state import set_curdoc
from pydantic import BaseModel, Field, ValidationError

from panel_agents import Action, ActionRegistry
from panel_agents.actions import _parse_docstring

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


class Filter(BaseModel):
    column: str
    value: int


def set_filter(column: str, value: Annotated[int, Field(ge=0, description="Lowest value kept.")], mode: Literal["min", "max"] = "min") -> str:
    """
    Filter the table shown to the user.

    Args:
        column: Column to filter
            on.
        mode (str): Whether value is a minimum or a maximum.

    Returns:
        A confirmation.
    """
    return f"{column} {mode} {value}"


def test_parse_docstring():
    description, args = _parse_docstring(set_filter.__doc__)
    assert description == "Filter the table shown to the user."
    assert args == {"column": "Column to filter on.", "mode": "Whether value is a minimum or a maximum."}
    assert _parse_docstring(None) == ("", {})


def test_action_schema_from_signature_and_docstring():
    action = Action.from_function(set_filter)
    assert action.name == "set_filter"
    assert action.description == "Filter the table shown to the user."
    assert action.parameters == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "column": {"type": "string", "description": "Column to filter on."},
            "value": {"type": "integer", "minimum": 0, "description": "Lowest value kept."},
            "mode": {"type": "string", "enum": ["min", "max"], "default": "min", "description": "Whether value is a minimum or a maximum."},
        },
        "required": ["column", "value"],
    }


def test_action_validation():
    action = Action.from_function(set_filter, name="filter", description="Filter.")
    assert (action.name, action.description) == ("filter", "Filter.")
    assert action.validate({"column": "a", "value": "3"}) == {"column": "a", "value": 3, "mode": "min"}
    assert action.validate_json('{"column": "a", "value": 1, "mode": "max"}')["mode"] == "max"
    with pytest.raises(ValidationError):
        action.validate({"column": "a", "value": -1})
    with pytest.raises(ValidationError):
        action.validate({"column": "a", "value": 1, "other": 2})


def test_model_arguments_are_passed_as_models():
    def apply(filter: Filter) -> int:
        return filter.value

    action = Action.from_function(apply)
    assert isinstance(action.validate({"filter": {"column": "a", "value": 2}})["filter"], Filter)


def test_var_args_are_rejected():
    def bad(*args):
        pass

    with pytest.raises(TypeError):
        Action.from_function(bad)


def test_registry_registration():
    actions = ActionRegistry()
    seen = []
    actions.param.watch(lambda e: seen.append(list(e.new)), "actions")

    @actions.action
    def first() -> None:
        """First."""

    @actions.action(name="second_action", description="Second.")
    def second() -> None:
        pass

    assert first.__name__ == "first"
    assert list(actions.actions) == ["first", "second_action"]
    assert "first" in actions and len(actions) == 2
    assert actions["second_action"].description == "Second."
    assert [a.name for a in actions] == ["first", "second_action"]
    actions.remove("first")
    assert list(actions.actions) == ["second_action"]
    assert seen == [["first"], ["first", "second_action"], ["second_action"]]


async def test_invoke_validates_and_awaits():
    actions = ActionRegistry()
    actions.add(set_filter)

    @actions.action
    async def wait(seconds: float) -> float:
        await asyncio.sleep(seconds)
        return seconds

    assert await actions.invoke("set_filter", {"column": "a", "value": 1}) == "a min 1"
    assert await actions.invoke("wait", {"seconds": 0}) == 0
    with pytest.raises(KeyError, match="available: set_filter, wait"):
        await actions.invoke("missing")
    with pytest.raises(ValidationError):
        await actions.invoke("set_filter", {"column": "a"})


async def test_invoke_runs_with_the_session_document(document):
    with set_curdoc(document):
        actions = ActionRegistry()

    @actions.action
    def current() -> bool:
        return pn.state.curdoc is document

    assert pn.state.curdoc is not document
    assert await actions.invoke("current")


async def test_invoke_from_another_thread_runs_on_the_session_loop():
    actions = ActionRegistry()
    loop = asyncio.get_running_loop()

    @actions.action
    def where() -> bool:
        return asyncio.get_running_loop() is loop and threading.current_thread() is threading.main_thread()

    result = await asyncio.to_thread(lambda: asyncio.run(actions.invoke("where")))
    assert result is True


pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402

from panel_agents import ApprovalPolicy, RunState  # noqa: E402
from panel_agents import events as ev  # noqa: E402
from panel_agents.adapters.pydantic_ai import ActionToolset, AgentRunner  # noqa: E402

from .test_pydantic_ai import streaming, tool_returns  # noqa: E402


def calling(*calls):
    def model(messages, info):
        turn = len(tool_returns(messages))
        if turn < len(calls):
            name, args = calls[turn]
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"call-{turn}")])
        return ModelResponse(parts=[TextPart(" | ".join(str(r.content) for r in tool_returns(messages)))])
    return model


async def test_agent_calls_actions_with_the_action_schema():
    actions = ActionRegistry()
    actions.add(set_filter)
    seen = {}

    def model(messages, info):
        seen["tools"] = {t.name: (t.description, t.parameters_json_schema) for t in info.function_tools}
        return calling(("set_filter", {"column": "year", "value": 2026}))(messages, info)

    runner = AgentRunner(Agent(streaming(model), toolsets=[ActionToolset(actions)]), RunState(flush_interval=0))
    assert await runner.run("Only 2026") == "year min 2026"
    action = actions["set_filter"]
    assert seen["tools"] == {"set_filter": (action.description, action.parameters)}
    assert runner.state.tool_calls["call-0"].status == "success"


async def test_actions_registered_during_a_session_become_tools():
    actions = ActionRegistry()
    toolset = ActionToolset(actions)
    names = []

    def model(messages, info):
        names.append(sorted(t.name for t in info.function_tools))
        return ModelResponse(parts=[TextPart("ok")])

    runner = AgentRunner(Agent(streaming(model), toolsets=[toolset]), RunState(flush_interval=0))
    await runner.run("a")
    actions.add(set_filter)
    await runner.run("b")
    assert names == [[], ["set_filter"]]


async def test_failing_action_is_reported_to_the_model():
    actions = ActionRegistry()

    @actions.action
    def open_page(path: str) -> str:
        raise LookupError(f"No page {path}")

    runner = AgentRunner(Agent(streaming(calling(("open_page", {"path": "/x"}))), toolsets=[ActionToolset(actions)]), RunState(flush_interval=0))
    output = await runner.run("go")
    assert "LookupError: No page /x" in output
    call = runner.state.tool_calls["call-0"]
    assert call.status == "failed" and "No page /x" in call.error


async def test_invalid_action_args_are_retried():
    actions = ActionRegistry()
    actions.add(set_filter)
    runner = AgentRunner(
        Agent(streaming(calling(("set_filter", {"column": "a", "value": -5}), ("set_filter", {"column": "a", "value": 5}))), toolsets=[ActionToolset(actions)]),
        RunState(flush_interval=0),
    )
    output = await runner.run("go")
    assert output.endswith("a min 5")
    assert runner.state.tool_calls["call-0"].status == "failed"


async def test_actions_follow_the_approval_policy():
    actions = ActionRegistry()
    actions.add(set_filter)
    state = RunState(flush_interval=0)
    runner = AgentRunner(
        Agent(streaming(calling(("set_filter", {"column": "a", "value": 1}))), toolsets=[ActionToolset(actions)]),
        state, approval_policy=ApprovalPolicy(["set_*"]),
    )

    def approve(event):
        if isinstance(event, ev.ApprovalRequested):
            asyncio.get_running_loop().call_soon(lambda: state.resolve_approval(event.tool_call_id, approved=True))

    state.subscribe(approve)
    assert await runner.run("go") == "a min 1"
    assert state.approvals["call-0"].schema == actions["set_filter"].parameters


def test_registry_built_off_loop_hands_calls_to_the_server_loop():
    # With pn.config.nthreads, sessions are built on worker threads that have no event loop.
    import time
    import urllib.request

    holder = {}
    caller = threading.current_thread()

    def app():
        doc = pn.state.curdoc

        def build():
            try:
                with set_curdoc(doc):
                    holder["registry"] = ActionRegistry()
            except Exception as e:
                holder["error"] = e
        thread = threading.Thread(target=build)
        thread.start()
        thread.join()
        if "error" in holder:
            raise holder["error"]

        @holder["registry"].action
        def where() -> dict:
            from panel_agents.actions import _server_loop

            return {
                "server_loop": asyncio.get_running_loop() is _server_loop(doc),
                "caller_thread": threading.current_thread() is caller,
                "curdoc": pn.state.curdoc is doc,
            }

        return pn.pane.Markdown("app")

    from panel.tests.util import get_open_ports

    port = get_open_ports()[0]
    # The cleanup_panel_servers fixture stops the server; stopping it here too raises on Panel 1.9.
    pn.serve(app, port=port, threaded=True, show=False, start=True)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/").read()
            break
        except OSError:
            time.sleep(0.1)
    registry = holder["registry"]
    assert registry._loop is None
    result = asyncio.run(registry.invoke("where"))
    assert result == {"server_loop": True, "caller_thread": False, "curdoc": True}


async def test_invoke_after_the_session_ended_raises():
    actions = ActionRegistry()
    actions.add(set_filter)
    actions._on_session_destroyed(None)
    with pytest.raises(RuntimeError, match="has ended"):
        await actions.invoke("set_filter", {"column": "a", "value": 1})


def test_action_constructor_derives_the_schema():
    action = Action("filter", "Filter.", set_filter)
    assert action.parameters == Action.from_function(set_filter).parameters
    assert action.validate({"column": "a", "value": 1})["mode"] == "min"


async def test_pending_calls_fail_when_the_session_ends():
    import time

    ran = []
    actions = ActionRegistry()

    @actions.action
    def mark() -> None:
        ran.append(True)

    other = asyncio.new_event_loop()
    thread = threading.Thread(target=other.run_forever, daemon=True)
    thread.start()
    try:
        # The session's loop is busy, so the handed-over call has not run when the session ends.
        other.call_soon_threadsafe(time.sleep, 1)
        actions._loop = other
        pending = asyncio.ensure_future(actions.invoke("mark"))
        await asyncio.sleep(0.05)
        actions._on_session_destroyed(None)
        with pytest.raises(RuntimeError, match="has ended"):
            await asyncio.wait_for(pending, 0.5)
        await asyncio.sleep(1.2)
        assert ran == []
    finally:
        other.call_soon_threadsafe(other.stop)
