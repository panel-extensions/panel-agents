import asyncio
import asyncio
import io
import json
import os
from dataclasses import dataclass

import pandas as pd
import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, ModelRetry, exceptions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturn,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from panel_agents import ApprovalPolicy
from panel_agents import events as ev
from panel_agents.adapters.model_trace import FORMAT_VERSION
from panel_agents.adapters.pydantic_ai import (
    ModelTraceRecorder,
    artifact_return,
    fingerprint,
    load_messages,
    load_model_calls,
)

from .test_pydantic_ai import make_runner, resolve_on_request, streaming

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def dumped(messages):
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prompts(messages):
    return [p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)]


def lookup_model(messages, info):
    """Looks a key up once per prompt, then answers with the result."""
    last = messages[-1]
    if any(isinstance(p, UserPromptPart) for p in last.parts):
        step = len(prompts(messages))
        return ModelResponse([ToolCallPart("lookup", {"key": f"k{step}"}, tool_call_id=f"call-{step}")])
    return ModelResponse([TextPart(f"value is {last.parts[0].content}")])


def make_agent(model=lookup_model, model_name="gpt-4o", **kwargs):
    agent = Agent(streaming(model, model_name=model_name), instructions="Look things up.", **kwargs)

    @agent.tool_plain
    def lookup(key: str) -> str:
        """Look up a key."""
        return f"secret-{key}"

    return agent


async def test_trace_records_the_message_history(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, path):
        assert await runner.run("What is k1?") == "value is secret-k1"
    header, *records = lines(path)
    assert header == {"panel_agents_model_trace": FORMAT_VERSION, "include_content": True}
    assert [r["kind"] for r in records] == ["request", "response", "request", "response"]
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_trace_is_written_as_the_run_progresses(tmp_path):
    path = tmp_path / "trace.jsonl"
    seen = []

    def peek() -> str:
        seen.append([r.get("kind") for r in lines(path)[1:]])
        return "ok"

    runner = make_runner(Agent(streaming(lambda messages, info: (
        ModelResponse([ToolCallPart("peek", {}, tool_call_id="p")]) if len(messages) == 1 else ModelResponse([TextPart("done")])
    )), tools=[peek]))
    with ModelTraceRecorder(runner, path):
        await runner.run("Go")
    assert seen == [["request", "response"]]


async def test_responses_carry_duration_and_fingerprint(tmp_path):
    runner = make_runner(make_agent())
    await runner.run("What is k1?")
    responses = [m for m in runner.messages if isinstance(m, ModelResponse)]
    stamps = [r.metadata["panel_agents"] for r in responses]
    assert all(s["duration"] >= 0 for s in stamps)
    assert len({s["fingerprint"] for s in stamps}) == 1
    assert all(m.run_id == runner.run_id for m in runner.messages)
    usage = runner.state.runs[runner.run_id].usage
    assert [u.duration for u in usage] == [s["duration"] for s in stamps]


async def test_fingerprint_changes_with_instructions_and_tools():
    captured = []

    def capture(messages, info):
        captured.append(fingerprint(info.model_request_parameters, messages))
        return ModelResponse([TextPart("ok")])

    for instructions, tool_doc in [("A", "Look up."), ("A", "Look up."), ("B", "Look up."), ("A", "Find.")]:
        agent = Agent(streaming(capture), instructions=instructions)

        def lookup(key: str) -> str:
            return key

        lookup.__doc__ = tool_doc
        agent.tool_plain(lookup)
        await make_runner(agent).run("Go")
    assert captured[0] == captured[1]
    assert len(set(captured)) == 3


async def test_load_model_calls_summarizes_each_request(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, path):
        await runner.run("What is k1?")
    first, second = load_model_calls(path)
    assert (first.model, first.run_id, first.tool_calls) == ("gpt-4o", runner.run_id, [("lookup", "call-1")])
    assert first.input_tokens > 0 and first.output_tokens > 0
    assert first.cost_usd is not None and first.cost_usd > 0
    assert first.duration is not None and first.error is None
    assert second.tool_calls == []


async def test_trace_appends_runs_and_resumes_a_conversation(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, path):
        await runner.run("What is k1?")

    resumed = make_runner(make_agent())
    resumed.messages = load_messages(path)
    with ModelTraceRecorder(resumed, path):
        assert await resumed.run("And k2?") == "value is secret-k2"
    assert prompts(resumed.messages) == ["What is k1?", "And k2?"]
    assert dumped(load_messages(path)) == dumped(resumed.messages)
    assert [c.run_id for c in load_model_calls(path)] == [runner.run_id] * 2 + [resumed.run_id] * 2
    assert sum("panel_agents_model_trace" in r for r in lines(path)) == 1
    assert not [r for r in lines(path) if "rewind" in r]


async def test_new_trace_starts_with_the_existing_history(tmp_path):
    runner = make_runner(make_agent())
    await runner.run("What is k1?")
    path = tmp_path / "trace.jsonl"
    with ModelTraceRecorder(runner, path):
        await runner.run("And k2?")
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_failed_request_is_recorded_and_left_out_of_the_history(tmp_path):
    path = tmp_path / "trace.jsonl"
    fail = [False]

    def model(messages, info):
        if fail[0]:
            raise ConnectionError("provider unreachable")
        return lookup_model(messages, info)

    runner = make_runner(make_agent(model))
    with ModelTraceRecorder(runner, path):
        await runner.run("What is k1?")
        fail[0] = True
        assert await runner.run("And k2?") is None
        failed_run = runner.run_id
        fail[0] = False
        assert await runner.run("And k3?") == "value is secret-k2"

    failed = [r for r in lines(path) if "failed_run" in r]
    assert failed == [{"failed_run": failed_run, "error": "ConnectionError: provider unreachable", "request": failed[0]["request"]}]
    assert failed[0]["request"]["model"] == "gpt-4o" and failed[0]["request"]["duration"] >= 0
    assert dumped(load_messages(path)) == dumped(runner.messages)
    assert prompts(load_messages(path, include_failed=True)) == ["What is k1?", "And k2?", "And k3?"]
    errors = [(c.run_id, c.error) for c in load_model_calls(path) if c.error]
    assert errors == [(failed_run, "ConnectionError: provider unreachable")]


@pytest.mark.skipif(not hasattr(exceptions, "RetryModelRequest"), reason="needs RetryModelRequest")
async def test_retried_attempts_are_listed_before_their_response(tmp_path):
    path = tmp_path / "trace.jsonl"
    attempts = []

    def model(messages, info):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("overloaded")
        return ModelResponse([TextPart("done")])

    @dataclass
    class RetryOnce(AbstractCapability):
        async def on_model_request_error(self, ctx, *, request_context, error):
            if request_context.attempt == 1:
                raise exceptions.RetryModelRequest()
            raise error

    runner = make_runner(Agent(streaming(model), capabilities=[RetryOnce()]))
    with ModelTraceRecorder(runner, path):
        assert await runner.run("Go") == "done"
    retried, answered = load_model_calls(path)
    assert retried.error == "ConnectionError: overloaded" and retried.duration is not None
    assert answered.error is None and retried.run_id == answered.run_id == runner.run_id


async def test_cancelled_run_keeps_its_messages(tmp_path):
    path = tmp_path / "trace.jsonl"

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse([ToolCallPart("stop", {}, tool_call_id="s")])
        return ModelResponse([TextPart("done")])

    runner = make_runner(Agent(streaming(model)))

    @runner.agent.tool_plain
    def stop() -> str:
        runner.cancel()
        return "stopped"

    with ModelTraceRecorder(runner, path):
        await runner.run("Go")
    assert runner.state.runs[runner.run_id].status == "error"
    assert runner.messages
    assert dumped(load_messages(path)) == dumped(runner.messages)
    assert not [r for r in lines(path) if "failed_run" in r]


async def test_trace_without_content(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, path, include_content=False):
        await runner.run("What is k1?")
    text = path.read_text()
    assert "secret" not in text and "What is k1" not in text and "Look things up" not in text
    assert [m.kind for m in load_messages(path)] == ["request", "response", "request", "response"]
    first, _ = load_model_calls(path)
    assert first.tool_calls == [("lookup", "call-1")] and first.input_tokens > 0 and first.duration is not None
    with pytest.raises(ValueError, match="include_content=False"):
        ModelTraceRecorder(runner, path)


async def test_trace_leaves_out_artifacts(tmp_path):
    path = tmp_path / "trace.jsonl"
    df = pd.DataFrame({"region": ["east", "west"], "amount": [65, 30]})

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse([ToolCallPart("table", {}, tool_call_id="t")])
        return ModelResponse([TextPart("done")])

    def table():
        return artifact_return("2 rows", "orders", df, kind="table")

    runner = make_runner(Agent(streaming(model), tools=[table]))
    with ModelTraceRecorder(runner, path):
        await runner.run("Go")
    assert "orders" in runner.state.artifacts.artifacts
    assert "east" not in path.read_text()
    assert [m.kind for m in load_messages(path)] == ["request", "response", "request", "response"]


async def test_trace_to_a_stream():
    buffer = io.StringIO()
    runner = make_runner(make_agent())
    recorder = ModelTraceRecorder(runner, buffer)
    await runner.run("What is k1?")
    recorder.close()
    await runner.run("And k2?")
    assert len(buffer.getvalue().splitlines()) == 5
    assert not runner._trace_listeners


def test_newer_trace_version_is_rejected(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"panel_agents_model_trace": FORMAT_VERSION + 1}) + "\n")
    with pytest.raises(ValueError, match="version"):
        load_messages(path)


async def test_usage_event_round_trips_duration():
    event = ev.Usage(model="m", duration=1.5)
    assert ev.Event.from_dict(event.to_dict()).duration == 1.5


def acting_agent(**kwargs):
    def model(messages, info):
        if any(isinstance(p, UserPromptPart) for p in messages[-1].parts):
            step = len(prompts(messages))
            return ModelResponse([ToolCallPart("act", {"x": step}, tool_call_id=f"c{step}")])
        return ModelResponse([TextPart("done")])

    agent = Agent(streaming(model), instructions="Be good.", **kwargs)

    @agent.tool_plain
    def act(x: int) -> str:
        return f"acted {x}"

    return agent


def attach_on_approval(runner, path, holder):
    def callback(event):
        if isinstance(event, ev.ApprovalRequested):
            def attach_and_approve():
                holder.append(ModelTraceRecorder(runner, path))
                runner.state.resolve_approval(event.tool_call_id, approved=True)
            asyncio.get_running_loop().call_soon(attach_and_approve)
    return runner.state.subscribe(callback)


async def test_clear_rewinds_the_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(acting_agent())
    with ModelTraceRecorder(runner, path):
        await runner.run("one")
        runner.clear()
        await runner.run("two")
    assert dumped(load_messages(path)) == dumped(runner.messages)
    assert {"rewind": 0} in lines(path)
    assert len(load_model_calls(path)) == 4


async def test_history_rewritten_by_a_capability(tmp_path):
    from pydantic_ai.capabilities import ProcessHistory

    path = tmp_path / "trace.jsonl"
    runner = make_runner(acting_agent(capabilities=[ProcessHistory(lambda messages: messages[-1:])]))
    with ModelTraceRecorder(runner, path):
        await runner.run("one")
        await runner.run("two")
    assert dumped(load_messages(path)) == dumped(runner.messages)
    assert len(load_model_calls(path)) == 4


@pytest.mark.parametrize("earlier_run", [False, True])
async def test_attach_while_a_run_waits_for_approval(tmp_path, earlier_run):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(acting_agent(), approval_policy=ApprovalPolicy(["act"]))
    if earlier_run:
        unsubscribe = resolve_on_request(runner.state, approved=True)
        await runner.run("first")
        unsubscribe()
    holder = []
    attach_on_approval(runner, path, holder)
    await runner.run("second")
    holder[0].close()
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_cancel_in_a_tool_then_continue(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(Agent(streaming(lambda messages, info: (
        ModelResponse([ToolCallPart("stop", {}, tool_call_id="s")]) if len(messages) == 1 else ModelResponse([TextPart("done")])
    ))))

    @runner.agent.tool_plain
    async def stop() -> str:
        runner.cancel()
        await asyncio.sleep(0.05)
        return "stopped"

    with ModelTraceRecorder(runner, path):
        await runner.run("go")
        assert dumped(load_messages(path)) == dumped(runner.messages)
        await runner.run("again")
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_cancel_while_waiting_for_approval_then_continue(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(acting_agent(), approval_policy=ApprovalPolicy(["act"]))

    def cancel_on_request(event):
        if isinstance(event, ev.ApprovalRequested):
            asyncio.get_running_loop().call_soon(runner.cancel)

    unsubscribe = runner.state.subscribe(cancel_on_request)
    with ModelTraceRecorder(runner, path):
        await runner.run("go")
        assert dumped(load_messages(path)) == dumped(runner.messages)
        unsubscribe()
        runner.approval_policy = None
        await runner.run("again")
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_cancel_mid_stream_marks_the_response_interrupted(tmp_path):
    path = tmp_path / "trace.jsonl"

    async def stream(messages, info):
        if len(messages) == 1:
            yield "partial "
            runner.cancel()
            await asyncio.sleep(0.2)
            yield "never"
        else:
            yield "fine"

    runner = make_runner(Agent(FunctionModel(stream_function=stream)))
    with ModelTraceRecorder(runner, path):
        await runner.run("go")
        await runner.run("again")
    assert dumped(load_messages(path)) == dumped(runner.messages)
    interrupted, answered = load_model_calls(path)
    assert interrupted.error == "Interrupted" and interrupted.duration is not None
    assert answered.error is None


async def test_task_cancel_counts_the_request_once(tmp_path):
    path = tmp_path / "trace.jsonl"

    async def stream(messages, info):
        yield "partial "
        await asyncio.sleep(10)
        yield "never"

    runner = make_runner(Agent(FunctionModel(stream_function=stream)))
    with ModelTraceRecorder(runner, path):
        task = asyncio.ensure_future(runner.run("go"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    (call,) = load_model_calls(path)
    assert call.error == "CancelledError" and call.duration is not None
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_call_deferred_and_resume(tmp_path):
    path = tmp_path / "trace.jsonl"

    def model(messages, info):
        if any(isinstance(p, UserPromptPart) for p in messages[-1].parts):
            return ModelResponse([ToolCallPart("local", {}, tool_call_id="l1"), ToolCallPart("external", {}, tool_call_id="e1")])
        return ModelResponse([TextPart("done")])

    agent = Agent(streaming(model), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def local() -> str:
        return "local ok"

    @agent.tool_plain
    def external() -> str:
        raise exceptions.CallDeferred()

    runner = make_runner(agent)
    with ModelTraceRecorder(runner, path):
        assert isinstance(await runner.run("go"), DeferredToolRequests)
        assert dumped(load_messages(path)) == dumped(runner.messages)
        await runner.run(deferred_tool_results=DeferredToolResults(calls={"e1": "external ok"}))
    assert dumped(load_messages(path)) == dumped(runner.messages)


async def test_tool_and_output_retries(tmp_path):
    path = tmp_path / "trace.jsonl"
    responses = [
        ModelResponse([ToolCallPart("act", {"x": -1}, tool_call_id="a1")]),
        ModelResponse([ToolCallPart("act", {"x": 1}, tool_call_id="a2")]),
        ModelResponse([TextPart("bad")]),
        ModelResponse([TextPart("good")]),
    ]
    agent = Agent(streaming(lambda messages, info: responses[sum(isinstance(m, ModelResponse) for m in messages)]))

    @agent.tool_plain
    def act(x: int) -> str:
        if x < 0:
            raise ModelRetry("x must be positive")
        return "ok"

    @agent.output_validator
    def check(output: str) -> str:
        if output == "bad":
            raise ModelRetry("not good")
        return output

    runner = make_runner(agent)
    with ModelTraceRecorder(runner, path):
        assert await runner.run("go") == "good"
    assert dumped(load_messages(path)) == dumped(runner.messages)
    assert len(load_model_calls(path)) == 4


def streaming_all(function):
    """Like ``streaming``, but also streams thinking."""
    async def stream(messages, info):
        for i, part in enumerate(function(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ThinkingPart):
                yield {i: DeltaThinkingPart(content=part.content, signature=part.signature)}
            else:
                yield {i: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)}
    return FunctionModel(function, stream_function=stream)


async def test_trace_without_content_keeps_every_part(tmp_path):
    path = tmp_path / "trace.jsonl"
    secrets = ["S_PROMPT", "S_INSTR", "S_SYS", "S_THINK", "S_SIG", "S_TEXT", "S_ARGS", "S_DICT", "S_RETRY", "S_OUTRETRY", "S_RICH"]
    responses = [
        ModelResponse([
            ThinkingPart("S_THINK", signature="S_SIG"),
            TextPart("S_TEXT"),
            ToolCallPart("dict_tool", {"q": "S_ARGS"}, tool_call_id="d1"),
            ToolCallPart("retry_tool", {"q": "x"}, tool_call_id="r1"),
            ToolCallPart("bin_tool", {}, tool_call_id="b1"),
            ToolCallPart("rich_tool", {}, tool_call_id="u1"),
        ]),
        ModelResponse([TextPart("bad")]),
        ModelResponse([TextPart("fine")]),
    ]
    agent = Agent(
        streaming_all(lambda messages, info: responses[sum(isinstance(m, ModelResponse) for m in messages)]),
        instructions="S_INSTR", system_prompt="S_SYS",
    )

    @agent.tool_plain
    def dict_tool(q: str) -> dict:
        return {"answer": "S_DICT"}

    @agent.tool_plain
    def retry_tool(q: str) -> str:
        raise ModelRetry("S_RETRY")

    @agent.tool_plain
    def bin_tool() -> BinaryContent:
        return BinaryContent(b"S_BINARY_DATA", media_type="image/png")

    @agent.tool_plain
    def rich_tool() -> ToolReturn:
        return ToolReturn(return_value="ok", content=["S_RICH"])

    @agent.output_validator
    def check(output: str) -> str:
        if output == "bad":
            raise ModelRetry("S_OUTRETRY")
        return output

    runner = make_runner(agent)
    with ModelTraceRecorder(runner, path, include_content=False):
        await runner.run("S_PROMPT")
    text = path.read_text()
    assert [s for s in secrets if s in text] == []
    assert "S_BINARY_DATA" not in text and "U19CSU5BUlk" not in text

    def shape(messages):
        return [(m.kind, [(p.part_kind, getattr(p, "tool_call_id", None)) for p in m.parts]) for m in messages]

    assert shape(load_messages(path)) == shape(runner.messages)


async def test_trace_without_content_keeps_only_error_types(tmp_path):
    path = tmp_path / "trace.jsonl"
    agent = Agent(streaming(lambda messages, info: ModelResponse([ToolCallPart("boom", {}, tool_call_id="x")])))

    @agent.tool_plain
    def boom() -> str:
        raise ValueError("customer 4111 S_ERROR")

    runner = make_runner(agent)
    with ModelTraceRecorder(runner, path, include_content=False):
        await runner.run("hi")
    assert "S_ERROR" not in path.read_text()
    assert [r["error"] for r in lines(path) if "failed_run" in r] == ["ValueError"]


async def test_run_id_passed_to_run_is_the_run_id():
    runner = make_runner(make_agent())
    await runner.run("What is k1?", run_id="run-mine")
    assert runner.run_id == "run-mine" and "run-mine" in runner.state.runs
    assert {m.run_id for m in runner.messages} == {"run-mine"}
    runner.run_kwargs = {"run_id": "run-fixed"}
    with pytest.raises(ValueError, match="run_kwargs cannot hold a run_id"):
        await runner.run("again")


async def test_second_recorder_on_a_stream_writes_no_second_header():
    buffer = io.StringIO()
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, buffer):
        await runner.run("What is k1?")
    with ModelTraceRecorder(runner, buffer):
        await runner.run("And k2?")
    assert sum("panel_agents_model_trace" in line for line in buffer.getvalue().splitlines()) == 1


async def test_trace_to_a_callable_and_load_the_records(tmp_path):
    records = []
    path = tmp_path / "trace.jsonl"
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, records.append), ModelTraceRecorder(runner, path):
        await runner.run("What is k1?")
        runner.clear()
        await runner.run("And k2?")
    assert records == lines(path)
    assert dumped(load_messages(records)) == dumped(load_messages(iter(records))) == dumped(runner.messages)
    assert load_model_calls(records) == load_model_calls(path)


async def test_callable_sink_gets_redacted_records():
    records = []
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, records.append, include_content=False):
        await runner.run("What is k1?")
    assert records[0] == {"panel_agents_model_trace": FORMAT_VERSION, "include_content": False}
    assert "secret" not in json.dumps(records) and "What is k1" not in json.dumps(records)


async def test_callable_sink_attached_again_does_not_repeat_the_history():
    records = []
    runner = make_runner(make_agent())
    with ModelTraceRecorder(runner, records.append):
        await runner.run("What is k1?")
    with ModelTraceRecorder(runner, records.append):
        await runner.run("And k2?")
    assert dumped(load_messages(records)) == dumped(runner.messages)
    assert len(load_model_calls(records)) == 4
