"""A failed run shows in the chat and in the server log, never silently."""

import logging

import panel_material_ui as pmui
import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat


def _errors(state):
    return [(i.content, i.run_id) for i in state.items if getattr(i, "kind", None) == "error"]


def test_run_error_adds_an_error_message():
    state = RunState.from_events([
        ev.RunStarted(run_id="r"),
        ev.TextMessageStart(message_id="u", role="user", run_id="r"),
        ev.TextMessageEnd(message_id="u", run_id="r"),
        ev.RunError(run_id="r", message="status_code: 401", code="ModelHTTPError"),
    ])
    assert _errors(state) == [("status_code: 401", "r")]
    assert not state.running


def test_run_error_before_the_run_started_adds_an_error_message():
    state = RunState.from_events([ev.RunError(run_id="r", message="bad arguments", code="TypeError")])
    assert _errors(state) == [("bad arguments", "r")]


def test_cancelled_and_sub_run_errors_add_no_message():
    state = RunState.from_events([
        ev.RunStarted(run_id="r"),
        ev.RunStarted(run_id="sub", parent_run_id="r"),
        ev.RunError(run_id="sub", message="sub-agent failed", code="ValueError"),
        ev.RunError(run_id="r", message="Run cancelled", code="cancelled"),
    ])
    assert _errors(state) == []


def test_chat_shows_the_error_as_plain_text():
    text = "<img src=x onerror=alert(1)> **boom**"
    state = RunState.from_events([ev.RunStarted(run_id="r", agent_name="Analyst"), ev.RunError(run_id="r", message=text)])
    chat = AgentChat(state)
    (message,) = chat.feed.objects
    assert message.user == "Analyst"
    alert = message.object
    assert isinstance(alert, pmui.Alert)
    assert alert.severity == "error"
    assert alert.object == r"\<img src\=x onerror\=alert\(1\)\> \*\*boom\*\*"


def test_chat_renders_an_error_streamed_into_a_live_state():
    state = RunState(flush_interval=0)
    chat = AgentChat(state)
    state.add_user_message("Hi", run_id="r")
    state.apply(ev.RunError(run_id="r", message="no model"))
    assert [type(m.object).__name__ for m in chat.feed.objects][-1] == "Alert"


pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from panel_agents.adapters.pydantic_ai import AgentRunner  # noqa: E402


def _failing_model():
    async def stream(messages, info):
        raise RuntimeError("provider refused the request")
        yield ""

    return FunctionModel(stream_function=stream)


async def test_failed_run_is_logged_and_shown(caplog):
    runner = AgentRunner(Agent(_failing_model()), RunState(flush_interval=0))
    chat = AgentChat(runner=runner)
    with caplog.at_level(logging.ERROR, logger="panel_agents"):
        assert await runner.run("Hi") is None
    assert _errors(runner.state) == [("provider refused the request", runner.run_id)]
    assert isinstance(chat.feed.objects[-1].object, pmui.Alert)
    (record,) = [r for r in caplog.records if r.name == "panel_agents.adapters.pydantic_ai"]
    assert runner.run_id in record.getMessage()
    assert record.exc_info[1] is runner.error


async def test_error_outside_the_event_stream_is_logged_and_shown(caplog):
    runner = AgentRunner(Agent(_failing_model()), RunState(flush_interval=0), run_kwargs={"no_such_argument": 1})
    with caplog.at_level(logging.ERROR, logger="panel_agents"):
        await runner.run("Hi")
    assert isinstance(runner.error, TypeError)
    ((content, _),) = _errors(runner.state)
    assert "no_such_argument" in content
    assert [r.exc_info[1] for r in caplog.records] == [runner.error]


async def test_cancelled_run_is_not_logged_as_an_error(caplog):
    async def stream(messages, info):
        runner.cancel()
        yield "partial"

    runner = AgentRunner(Agent(FunctionModel(stream_function=stream)), RunState(flush_interval=0))
    with caplog.at_level(logging.ERROR, logger="panel_agents"):
        await runner.run("Hi")
    assert [e.code for e in runner.state.events if isinstance(e, ev.RunError)] == ["cancelled"]
    assert _errors(runner.state) == []
    assert caplog.records == []
