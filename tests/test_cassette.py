import asyncio
import dataclasses
import json
import os

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from panel_agents import ApprovalPolicy, Recorder, load_state
from panel_agents.adapters.pydantic_ai import (
    AgentRunner,
    Cassette,
    CassetteError,
    ModelTraceRecorder,
    RecordedFailure,
    artifact_return,
    load_messages,
    load_model_calls,
)

from .test_pydantic_ai import make_runner, resolve_on_request, streaming

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

ORDERS = {"east": [40, 25], "west": [30]}


def make_agent(model, orders=ORDERS, tool_name="total", instructions="Answer from the orders."):
    agent = Agent(model, instructions=instructions)

    def total(region: str) -> int:
        """Sum the orders of a region."""
        return sum(orders[region])

    def chart(region: str):
        """Chart the orders of a region."""
        return artifact_return("Charted.", f"chart-{region}", {"region": region, "values": orders[region]}, kind="json")

    agent.tool_plain(total, name=tool_name)
    agent.tool_plain(chart)
    return agent


def provider(messages, info):
    """Stands in for a provider: totals and charts a region, then answers with the total."""
    last = messages[-1]
    returns = [p for p in last.parts if isinstance(p, ToolReturnPart)]
    if returns:
        total = next(p.content for p in returns if p.tool_name == "total")
        return ModelResponse([TextPart(f"East sold {total}.")], model_name="gpt-4o")
    return ModelResponse([
        TextPart("Let me look."),
        ToolCallPart("total", {"region": "east"}, tool_call_id="call-total"),
        ToolCallPart("chart", {"region": "east"}, tool_call_id="call-chart"),
    ], model_name="gpt-4o")


async def record(path, prompts=("How much did east sell?",)):
    runner = make_runner(make_agent(streaming(provider, model_name="gpt-4o")), approval_policy=ApprovalPolicy(["chart"]))
    resolve_on_request(runner.state, approved=True, user="ann")
    with ModelTraceRecorder(runner, path):
        for prompt in prompts:
            await runner.run(prompt)
    return runner


def play(cassette, agent=None, **params):
    params.setdefault("approval_policy", ApprovalPolicy(["chart"]))
    runner = make_runner(agent or make_agent(cassette), **params)
    resolve_on_request(runner.state, approved=True, user="ann")
    return runner


def summary(state):
    return {
        "items": [(m.role, m.kind, m.content) for m in state.items if hasattr(m, "role")],
        "tool_calls": [(c.tool_name, c.args, c.status, c.result) for c in state.tool_calls.values()],
        "approvals": [(a.tool_name, a.status, a.user) for a in state.approvals.values()],
        "artifacts": {k: [v.value for v in a.versions] for k, a in state.artifacts.artifacts.items()},
        "usage": [[(u.model, u.input_tokens, u.output_tokens) for u in run.usage] for run in state.runs.values()],
        "runs": [(run.status, run.result) for run in state.runs.values()],
    }


async def test_playback_reproduces_the_recorded_run(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorded = await record(path, ["How much did east sell?", "And again?"])
    cassette = Cassette.from_trace(path)
    runner = play(cassette)
    for prompt in ["How much did east sell?", "And again?"]:
        await runner.run(prompt)
    assert summary(runner.state) == summary(recorded.state)
    assert summary(runner.state)["approvals"] == [("chart", "approved", "ann")]
    assert (cassette.played, cassette.remaining, cassette.divergences) == (4, 0, [])


async def test_playback_matches_a_replayed_conversation_recording(tmp_path):
    runner = make_runner(make_agent(streaming(provider, model_name="gpt-4o")))
    with Recorder(runner.state, tmp_path / "events.jsonl"):
        await runner.run("How much did east sell?")
    played = play(Cassette(runner.messages), approval_policy=None)
    await played.run("How much did east sell?")
    assert summary(played.state) == summary(load_state(tmp_path / "events.jsonl"))


async def test_playback_without_streaming(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path)
    result = await make_agent(cassette).run("How much did east sell?")
    assert result.output == "East sold 65."
    responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    assert [r.model_name for r in responses] == ["gpt-4o", "gpt-4o"]
    assert [r.usage for r in responses] == [r.usage for r in cassette.responses]
    assert cassette.remaining == 0


async def test_changed_tool_result_is_a_divergence(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path)
    runner = play(cassette, make_agent(cassette, orders={"east": [40, 25, 5], "west": [30]}))
    await runner.run("How much did east sell?")
    assert {(d.kind, d.tool_name, d.tool_call_id, d.expected, d.actual) for d in cassette.divergences} == {
        ("tool_return", "total", "call-total", 65, 70),
    }
    assert "call-total" in str(cassette.divergences[0])


async def test_strict_playback_raises_on_divergence(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path, strict=True)
    runner = play(cassette, make_agent(cassette, orders={"east": [1], "west": [30]}), raise_errors=True)
    with pytest.raises(CassetteError, match="tool_return of 'total'"):
        await runner.run("How much did east sell?")


async def test_denied_approval_is_a_divergence(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path)
    runner = make_runner(make_agent(cassette), approval_policy=ApprovalPolicy(["chart"]))
    resolve_on_request(runner.state, approved=False)
    await runner.run("How much did east sell?")
    assert [(d.kind, d.tool_name) for d in cassette.divergences] == [("tool_return", "chart")]


async def test_changed_prompt_is_a_divergence(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path)
    await play(cassette).run("How much did west sell?")
    assert [(d.kind, d.expected, d.actual) for d in cassette.divergences] == [
        ("prompt", ["How much did east sell?"], ["How much did west sell?"]),
    ]


async def test_renamed_tool_fails_loudly(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path, check_fingerprint=False)
    runner = play(cassette, make_agent(cassette, tool_name="sum_orders"))
    assert await runner.run("How much did east sell?") is None
    assert isinstance(runner.error, CassetteError)
    assert "'total', which the agent no longer offers" in str(runner.error)
    assert runner.state.runs[runner.run_id].status == "error"


async def test_running_out_of_responses_fails_loudly(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path)
    runner = play(cassette)
    await runner.run("How much did east sell?")
    await runner.run("And again?")
    assert isinstance(runner.error, CassetteError)
    assert "recorded 2 model requests and the agent made another" in str(runner.error)


async def test_changed_instructions_fail_the_fingerprint_check(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path)
    cassette = Cassette.from_trace(path)
    runner = play(cassette, make_agent(cassette, instructions="Answer briefly."))
    await runner.run("How much did east sell?")
    assert isinstance(runner.error, CassetteError) and "fingerprint" in str(runner.error)

    lenient = Cassette.from_trace(path, check_fingerprint=False)
    await play(lenient, make_agent(lenient, instructions="Answer briefly.")).run("How much did east sell?")
    assert [d.kind for d in lenient.divergences] == ["fingerprint", "fingerprint"]


async def test_cassette_from_messages_without_fingerprints():
    messages = [
        ModelRequest.user_text_prompt("Hi"),
        ModelResponse([TextPart("Hello!")], model_name="m"),
    ]
    messages = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
    runner = AgentRunner(Agent(Cassette(messages)))
    assert await runner.run("Hi") == "Hello!"


async def test_cassette_needs_content(tmp_path):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(make_agent(streaming(provider)))
    with ModelTraceRecorder(runner, path, include_content=False):
        await runner.run("How much did east sell?")
    with pytest.raises(ValueError, match="include_content=False"):
        Cassette.from_trace(path)


async def test_resumed_conversation_plays_back(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record(path, ["How much did east sell?"])
    resumed = make_runner(make_agent(streaming(provider, model_name="gpt-4o")), approval_policy=ApprovalPolicy(["chart"]))
    resolve_on_request(resumed.state, approved=True, user="ann")
    resumed.messages = load_messages(path)
    with ModelTraceRecorder(resumed, path):
        await resumed.run("And again?")
    cassette = Cassette.from_trace(path)
    runner = play(cassette)
    await runner.run("How much did east sell?")
    await runner.run("And again?")
    assert (cassette.remaining, cassette.divergences) == (0, [])
    assert runner.state.items[-1].content == resumed.state.items[-1].content == "East sold 65."


async def test_played_responses_take_the_ids_of_the_playback(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorded = await record(path)
    cassette = Cassette.from_trace(path)
    result = await make_agent(cassette).run("How much did east sell?")
    assert {m.run_id for m in result.all_messages()} == {result.run_id} != {recorded.run_id}

    cassette = Cassette.from_trace(path)
    runner = play(cassette)
    with ModelTraceRecorder(runner, tmp_path / "played.jsonl"):
        await runner.run("How much did east sell?")
    assert {m.run_id for m in load_messages(tmp_path / "played.jsonl")} == {runner.run_id}


def hand_recorded(returns):
    return [
        ModelRequest.user_text_prompt("Go"),
        ModelResponse([ToolCallPart("total", {"region": "east"}, tool_call_id="call-total")], model_name="m"),
        ModelRequest([ToolReturnPart(name, content, tool_call_id=call_id) for name, content, call_id in returns]),
        ModelResponse([TextPart("Done.")], model_name="m"),
    ]


@pytest.mark.parametrize("returns, expected, actual, call_id", [
    ([("total", 65, "call-total"), ("total", 1, "call-ghost")], 1, None, "call-ghost"),
    ([], None, 65, "call-total"),
])
async def test_missing_and_extra_tool_returns_are_divergences(returns, expected, actual, call_id):
    cassette = Cassette(hand_recorded(returns))
    result = await make_agent(cassette).run("Go")
    assert result.output == "Done."
    assert [(d.kind, d.tool_call_id, d.expected, d.actual) for d in cassette.divergences] == [
        ("tool_return", call_id, expected, actual),
    ]


async def test_changed_output_retry_is_a_divergence(tmp_path):
    from pydantic_ai import ModelRetry

    def agent_for(model, feedback):
        agent = Agent(model)

        @agent.output_validator
        def check(output: str) -> str:
            if output == "bad":
                raise ModelRetry(feedback)
            return output

        return agent

    def model(messages, info):
        return ModelResponse([TextPart("bad" if len(messages) == 1 else "good")])

    path = tmp_path / "trace.jsonl"
    with ModelTraceRecorder(recorder_runner := make_runner(agent_for(streaming(model), "Say good")), path):
        await recorder_runner.run("Go")
    cassette = Cassette.from_trace(path)
    assert await play(cassette, agent_for(cassette, "Say good"), approval_policy=None).run("Go") == "good"
    assert cassette.divergences == []

    cassette = Cassette.from_trace(path)
    await play(cassette, agent_for(cassette, "Something else"), approval_policy=None).run("Go")
    assert [d.kind for d in cassette.divergences] == ["retry"]


@pytest.mark.parametrize("request_only_context", [False, True])
async def test_streamed_and_non_streamed_playback_agree(tmp_path, request_only_context):
    from dataclasses import dataclass

    from pydantic_ai import ModelRetry
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.messages import BinaryContent, ToolReturn, UserPromptPart

    @dataclass
    class AddContext(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            last = request_context.messages[-1]
            request_context.messages = [*request_context.messages[:-1], ModelRequest([*last.parts, UserPromptPart("note")])]
            return request_context

    responses = [
        ModelResponse([
            ToolCallPart("dict_tool", {"q": "a"}, tool_call_id="d1"),
            ToolCallPart("retry_tool", {"q": "x"}, tool_call_id="r1"),
            ToolCallPart("bin_tool", {}, tool_call_id="b1"),
            ToolCallPart("rich_tool", {}, tool_call_id="u1"),
        ], model_name="m"),
        ModelResponse([TextPart("bad")], model_name="m"),
        ModelResponse([TextPart("fine")], model_name="m"),
    ]

    def agent_for(model):
        agent = Agent(model, instructions="I", capabilities=[AddContext()] if request_only_context else [])

        @agent.tool_plain
        def dict_tool(q: str) -> dict:
            return {"answer": q}

        @agent.tool_plain
        def retry_tool(q: str) -> str:
            raise ModelRetry("nope")

        @agent.tool_plain
        def bin_tool() -> BinaryContent:
            return BinaryContent(b"png", media_type="image/png")

        @agent.tool_plain
        def rich_tool() -> ToolReturn:
            return ToolReturn(return_value="ok", content=["extra"])

        @agent.output_validator
        def check(output: str) -> str:
            if output == "bad":
                raise ModelRetry("again")
            return output

        return agent

    path = tmp_path / "trace.jsonl"
    recorder_runner = make_runner(agent_for(streaming(lambda messages, info: responses[sum(isinstance(m, ModelResponse) for m in messages)])))
    with ModelTraceRecorder(recorder_runner, path):
        await recorder_runner.run("go")

    streamed = Cassette.from_trace(path, strict=True)
    assert await make_runner(agent_for(streamed), raise_errors=True).run("go") == "fine"
    non_streamed = Cassette.from_trace(path, strict=True)
    assert (await agent_for(non_streamed).run("go")).output == "fine"
    assert streamed.divergences == non_streamed.divergences == []
    assert streamed.remaining == non_streamed.remaining == 0


async def test_changed_output_retry_is_a_divergence_without_streaming(tmp_path):
    from pydantic_ai import ModelRetry

    def agent_for(model, feedback):
        agent = Agent(model)

        @agent.output_validator
        def check(output: str) -> str:
            if output == "bad":
                raise ModelRetry(feedback)
            return output

        return agent

    path = tmp_path / "trace.jsonl"
    runner = make_runner(agent_for(streaming(lambda messages, info: ModelResponse([TextPart("bad" if len(messages) == 1 else "good")])), "Say good"))
    with ModelTraceRecorder(runner, path):
        await runner.run("Go")
    same = Cassette.from_trace(path)
    assert (await agent_for(same, "Say good").run("Go")).output == "good"
    assert same.divergences == []
    changed = Cassette.from_trace(path)
    await agent_for(changed, "Something else").run("Go")
    assert [d.kind for d in changed.divergences] == ["retry"]


async def test_non_streamed_playback_without_the_run_context_lookup(tmp_path, monkeypatch):
    import importlib
    import sys

    from pydantic_ai import ModelRetry

    from panel_agents.adapters import cassette as cassette_module

    original = dict(vars(cassette_module))
    monkeypatch.setitem(sys.modules, "pydantic_ai._run_context", None)
    try:
        importlib.reload(cassette_module)
    finally:
        monkeypatch.undo()
    try:
        assert cassette_module._current_run_context() is None

        def agent_for(model):
            agent = Agent(model)

            @agent.output_validator
            def check(output: str) -> str:
                if output == "bad":
                    raise ModelRetry("Say good")
                return output

            return agent

        path = tmp_path / "trace.jsonl"
        runner = make_runner(agent_for(streaming(lambda messages, info: ModelResponse([TextPart("bad" if len(messages) == 1 else "good")]))))
        with ModelTraceRecorder(runner, path):
            await runner.run("Go")
        cassette = cassette_module.Cassette.from_trace(path)
        assert (await agent_for(cassette).run("Go")).output == "good"
        # Without the run's history, the retry is compared as the user prompt the model was shown.
        assert {d.kind for d in cassette.divergences} == {"prompt", "retry"}
        assert cassette.remaining == 0
    finally:
        # Put back the original module contents, so the classes other modules imported stay the ones it raises and returns.
        vars(cassette_module).update(original)
    assert cassette_module._current_run_context.__module__ == "pydantic_ai._run_context"
    assert cassette_module.CassetteError is CassetteError


async def test_cassette_from_the_records_of_a_callable_sink():
    records = []
    runner = make_runner(make_agent(streaming(provider, model_name="gpt-4o")), approval_policy=ApprovalPolicy(["chart"]))
    resolve_on_request(runner.state, approved=True, user="ann")
    with ModelTraceRecorder(runner, records.append):
        await runner.run("How much did east sell?")
    cassette = Cassette.from_trace(records, strict=True)
    await play(cassette).run("How much did east sell?")
    assert (cassette.played, cassette.remaining) == (2, 0)



def _stub_old_returns(messages):
    """Replace every tool result but the newest with a stub, as a history processor that shortens old results."""
    requests = [i for i, m in enumerate(messages) if isinstance(m, ModelRequest)]
    shortened = list(messages)
    for index in requests[:-1]:
        parts = [dataclasses.replace(p, content="[stub]") if isinstance(p, ToolReturnPart) else p
                 for p in messages[index].parts]
        shortened[index] = dataclasses.replace(messages[index], parts=parts)
    return shortened


def _rewrite_old_responses(messages):
    """Also blank the text and tool call arguments of every response but the newest."""
    rewritten = _stub_old_returns(messages)
    responses = [i for i, m in enumerate(rewritten) if isinstance(m, ModelResponse)]
    for index in responses[:-1]:
        parts = [
            dataclasses.replace(p, content="[old]") if isinstance(p, TextPart)
            else dataclasses.replace(p, args={}) if isinstance(p, ToolCallPart) else p
            for p in rewritten[index].parts
        ]
        rewritten[index] = dataclasses.replace(rewritten[index], parts=parts)
    return rewritten


def _keep_last(count):
    def keep(messages):
        kept = messages[-count:]
        while kept and not isinstance(kept[0], ModelRequest):
            kept = kept[1:]
        return kept

    return keep


class Regions:
    """A provider that totals east, then west, then answers with a numbered text, and can fail its first request."""

    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def __call__(self, messages, info):
        self.calls += 1
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("provider refused")
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse([ToolCallPart("total", {"region": "east"}, tool_call_id=f"c{self.calls}")], model_name="gpt-4o")
        if returns[0].content not in (30, "[stub]"):
            return ModelResponse([ToolCallPart("total", {"region": "west"}, tool_call_id=f"c{self.calls}")], model_name="gpt-4o")
        return ModelResponse([TextPart(f"Done {self.calls}.")], model_name="gpt-4o")


def regions_agent(model, process=None, orders=ORDERS):
    agent = Agent(model, instructions="Answer from the orders.",
                  capabilities=[ProcessHistory(process)] if process else [])

    def total(region: str) -> int:
        """Sum the orders of a region."""
        return sum(orders[region])

    agent.tool_plain(total)
    return agent


async def record_regions(path, prompts, process=None, between=None, fail_first=False):
    runner = make_runner(regions_agent(streaming(Regions(fail_first), model_name="gpt-4o"), process))
    outputs = []
    with ModelTraceRecorder(runner, path):
        for index, prompt in enumerate(prompts):
            outputs.append(await runner.run(prompt))
            if between:
                between(runner)
    return runner, outputs


async def play_regions(cassette, prompts, process=None, between=None, orders=ORDERS):
    runner = make_runner(regions_agent(cassette, process, orders))
    outputs = []
    for prompt in prompts:
        outputs.append(await runner.run(prompt))
        if between:
            between(runner)
    return runner, outputs


@pytest.mark.parametrize("process", [_stub_old_returns, _rewrite_old_responses, _keep_last(3)],
                         ids=["stubbed results", "rewritten responses", "truncated history"])
async def test_playback_of_an_agent_that_rewrites_its_history(tmp_path, process):
    path = tmp_path / "trace.jsonl"
    prompts = ["Totals?", "Again?", "Once more?"]
    recorded, outputs = await record_regions(path, prompts, process)
    assert any("rewind" in line for line in path.read_text().splitlines())

    cassette = Cassette.from_trace(path, strict=True)
    assert len(cassette.responses) == len(load_model_calls(path))
    replay, replayed = await play_regions(cassette, prompts, process)
    assert replay.error is None
    assert replayed == outputs
    assert (cassette.remaining, cassette.divergences) == (0, [])
    assert summary(replay.state)["tool_calls"] == summary(recorded.state)["tool_calls"]


async def test_changed_result_that_is_stubbed_later_is_reported_with_its_recorded_value(tmp_path):
    path = tmp_path / "trace.jsonl"
    await record_regions(path, ["Totals?"], _stub_old_returns)
    cassette = Cassette.from_trace(path)
    await play_regions(cassette, ["Totals?"], _stub_old_returns, orders={**ORDERS, "east": [41, 25]})
    (divergence,) = cassette.divergences
    assert (divergence.kind, divergence.tool_name, divergence.expected, divergence.actual) == ("tool_return", "total", 65, 66)


async def test_playback_after_clear(tmp_path):
    path = tmp_path / "trace.jsonl"
    _, outputs = await record_regions(path, ["Totals?", "Again?"], _stub_old_returns, between=lambda r: r.clear())
    cassette = Cassette.from_trace(path, strict=True)
    replay, replayed = await play_regions(cassette, ["Totals?", "Again?"], _stub_old_returns, between=lambda r: r.clear())
    assert replayed == outputs and cassette.remaining == 0 and replay.error is None


async def test_failed_request_fails_again_on_playback_and_the_next_run_plays(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorded, outputs = await record_regions(path, ["Totals?", "Again?"], fail_first=True)
    assert outputs[0] is None and outputs[1] is not None
    cassette = Cassette.from_trace(path, strict=True)
    runner = make_runner(regions_agent(cassette))
    assert await runner.run("Totals?") is None
    assert isinstance(runner.error, RecordedFailure) and "provider refused" in str(runner.error)
    assert await runner.run("Again?") == outputs[1]
    assert (cassette.remaining, cassette.divergences) == (0, [])


async def test_trace_records_given_as_a_generator_play_back(tmp_path):
    path = tmp_path / "trace.jsonl"
    _, outputs = await record_regions(path, ["Totals?"], _stub_old_returns)
    records = (json.loads(line) for line in path.read_text().splitlines())
    cassette = Cassette.from_trace(records, strict=True)
    _, replayed = await play_regions(cassette, ["Totals?"], _stub_old_returns)
    assert replayed == outputs and cassette.remaining == 0


async def test_trace_without_request_ids_plays_its_final_history(tmp_path):
    path = tmp_path / "trace.jsonl"
    _, outputs = await record_regions(path, ["Totals?"])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for line in lines:
        if line.get("kind") == "response":
            line["metadata"]["panel_agents"].pop("request_id")
    cassette = Cassette.from_trace(lines, strict=True)
    _, replayed = await play_regions(cassette, ["Totals?"])
    assert replayed == outputs and cassette.remaining == 0


def _rebuild_old_returns(messages):
    """Stub old tool results by building new parts and requests, which get new timestamps."""
    requests = [i for i, m in enumerate(messages) if isinstance(m, ModelRequest)]
    rebuilt = list(messages)
    for index in requests[:-1]:
        parts = [
            ToolReturnPart(tool_name=p.tool_name, content="[stub]", tool_call_id=p.tool_call_id)
            if isinstance(p, ToolReturnPart) else p
            for p in messages[index].parts
        ]
        rebuilt[index] = ModelRequest(parts=parts)
    return rebuilt


async def test_playback_of_a_processor_that_rebuilds_parts(tmp_path):
    path = tmp_path / "trace.jsonl"
    _, outputs = await record_regions(path, ["Totals?", "Again?"], _rebuild_old_returns)
    cassette = Cassette.from_trace(path, strict=True)
    replay, replayed = await play_regions(cassette, ["Totals?", "Again?"], _rebuild_old_returns)
    assert replay.error is None and replayed == outputs
    assert (cassette.remaining, cassette.divergences) == (0, [])


@pytest.mark.parametrize("process", [None, _stub_old_returns], ids=["no processor", "stubbed results"])
async def test_playback_with_a_frozen_clock(tmp_path, process):
    path = tmp_path / "trace.jsonl"
    _, outputs = await record_regions(path, ["Totals?", "Again?"], process)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        for part in record.get("parts", []):
            if "timestamp" in part:
                part["timestamp"] = "2026-01-01T00:00:00Z"
    cassette = Cassette.from_trace(records, strict=True)
    replay, replayed = await play_regions(cassette, ["Totals?", "Again?"], process)
    assert replay.error is None and replayed == outputs
    assert (cassette.remaining, cassette.divergences) == (0, [])


def _cancel_at_the_first_tool(runner):
    from panel_agents import events as ev

    gate = {"open": True}

    def cancel(event):
        if gate["open"] and isinstance(event, ev.ToolCallEnd):
            gate["open"] = False
            asyncio.get_running_loop().call_soon(runner.cancel)

    runner.state.subscribe(cancel)


@pytest.mark.parametrize("process", [None, _stub_old_returns], ids=["no processor", "stubbed results"])
async def test_playback_of_a_run_cancelled_during_a_tool_and_continued(tmp_path, process):
    path = tmp_path / "trace.jsonl"
    runner = make_runner(regions_agent(streaming(Regions(), model_name="gpt-4o"), process))
    _cancel_at_the_first_tool(runner)
    with ModelTraceRecorder(runner, path):
        outputs = [await runner.run("Totals?"), await runner.run("Again?")]
    calls = load_model_calls(path)

    cassette = Cassette.from_trace(path, strict=True)
    assert len(cassette.responses) == len(calls)
    replay = make_runner(regions_agent(cassette, process))
    _cancel_at_the_first_tool(replay)
    replayed = [await replay.run("Totals?"), await replay.run("Again?")]
    assert replayed == outputs
    assert (cassette.remaining, cassette.divergences) == (0, [])


def _summarize_earlier_turns(messages):
    """Replace the turns before the newest prompt with one request summarizing them, as a compacting processor does."""
    prompts = [
        i for i, m in enumerate(messages)
        if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) and p.content != "summary of earlier turns" for p in m.parts)
    ]
    if len(prompts) < 2:
        return messages
    return [ModelRequest(parts=[UserPromptPart("summary of earlier turns")]), *messages[prompts[-1]:]]


async def test_playback_of_a_processor_that_summarizes_earlier_turns(tmp_path):
    path = tmp_path / "trace.jsonl"
    _, outputs = await record_regions(path, ["Totals?", "Again?"], _summarize_earlier_turns)
    cassette = Cassette.from_trace(path, strict=True)
    replay, replayed = await play_regions(cassette, ["Totals?", "Again?"], _summarize_earlier_turns)
    assert replay.error is None and replayed == outputs
    assert (cassette.remaining, cassette.divergences) == (0, [])
