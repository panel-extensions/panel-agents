import asyncio
import os
import time
from dataclasses import dataclass

import pytest

from panel_agents import Recorder, RunState, load_events, load_state
from panel_agents import events as ev
from panel_agents.state import ModelCallState, ToolCallState

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def kinds(state):
    return [type(i).__name__.removesuffix("State") for i in state.items]


def model_run(run_id="r1"):
    return [
        ev.RunStarted(run_id=run_id, timestamp=100.0),
        ev.ModelRequestStarted(request_id="q1", model="gpt-4o", run_id=run_id, timestamp=100.0),
        ev.TextMessageStart(message_id="m1", run_id=run_id, timestamp=100.5),
        ev.TextMessageContent(message_id="m1", delta="Checking.", run_id=run_id),
        ev.TextMessageEnd(message_id="m1", run_id=run_id),
        ev.ToolCallStart(tool_call_id="t1", tool_name="lookup", run_id=run_id, timestamp=100.8),
        ev.ToolCallEnd(tool_call_id="t1", run_id=run_id),
        ev.Usage(
            model="gpt-4o", input_tokens=1000, output_tokens=40, cached_tokens=800, cost_usd=0.002, duration=1.0,
            request_id="q1", tool_call_ids=["t1"], finish_reason="tool_call", provider_response_id="resp-1",
            run_id=run_id, timestamp=101.0,
        ),
        ev.ToolCallResult(tool_call_id="t1", content="ok", run_id=run_id, timestamp=101.2),
        ev.ModelRequestStarted(request_id="q2", model="gpt-4o", run_id=run_id, timestamp=101.2),
        ev.TextMessageStart(message_id="m2", run_id=run_id, timestamp=101.5),
        ev.TextMessageContent(message_id="m2", delta="Done.", run_id=run_id),
        ev.TextMessageEnd(message_id="m2", run_id=run_id),
        ev.Usage(
            model="gpt-4o", input_tokens=1100, output_tokens=10, cost_usd=0.001, duration=0.6,
            request_id="q2", finish_reason="stop", run_id=run_id, timestamp=101.8,
        ),
        ev.RunFinished(run_id=run_id, timestamp=102.0),
    ]


def test_model_requests_become_timeline_items():
    state = RunState(flush_interval=0)
    state.apply_many(model_run()[:2])
    (request,) = state.model_calls.values()
    assert (request.status, request.model, request.started_at) == ("running", "gpt-4o", 100.0)

    state.apply_many(model_run()[2:])
    assert kinds(state) == ["ModelCall", "Message", "ToolCall", "ModelCall", "Message"]
    first, second = state.model_calls.values()
    assert first is state.items[0] and second is state.items[3]
    assert (first.status, first.input_tokens, first.output_tokens, first.cached_tokens) == ("success", 1000, 40, 800)
    assert (first.cost_usd, first.duration, first.ended_at) == (0.002, 1.0, 101.0)
    assert (first.tool_call_ids, first.finish_reason, first.provider_response_id) == (["t1"], "tool_call", "resp-1")
    assert second.tool_call_ids == [] and second.finish_reason == "stop"


def test_run_totals_count_every_request():
    run = RunState.from_events(model_run()).runs["r1"]
    assert (run.requests, run.input_tokens, run.output_tokens, run.cached_tokens) == (2, 2100, 50, 800)
    assert run.cost_usd == pytest.approx(0.003)
    assert run.model_duration == pytest.approx(1.6)
    assert (run.duration, run.models) == (2.0, ["gpt-4o"])


def test_a_failed_run_fails_its_request_in_flight():
    state = RunState.from_events([
        ev.RunStarted(run_id="r1", timestamp=1.0),
        ev.ModelRequestStarted(request_id="q1", model="m", run_id="r1", timestamp=1.0),
        ev.RunError(run_id="r1", message="The model provider refused the request (429)", timestamp=2.5),
    ])
    request = state.model_calls["q1"]
    assert (request.status, request.error, request.duration) == ("failed", "The model provider refused the request (429)", 1.5)


def test_usage_with_an_error_is_a_failed_request():
    state = RunState.from_events([
        ev.RunStarted(run_id="r1"),
        ev.ModelRequestStarted(request_id="q1", model="m", run_id="r1"),
        ev.Usage(request_id="q1", model="m", error="overloaded", duration=0.3, run_id="r1"),
        ev.RunError(run_id="r1", message="overloaded"),
    ])
    request = state.model_calls["q1"]
    assert (request.status, request.error, request.duration) == ("failed", "overloaded", 0.3)
    assert state.runs["r1"].requests == 1


def old_recording(events):
    """The events as recorded before model requests had ids: no start events, bare Usage."""
    old = []
    for e in events:
        if isinstance(e, ev.ModelRequestStarted) or (isinstance(e, ev.Usage) and e.error):
            continue
        if isinstance(e, ev.Usage):
            e = ev.Usage(
                model=e.model, input_tokens=e.input_tokens, output_tokens=e.output_tokens, cached_tokens=e.cached_tokens,
                cost_usd=e.cost_usd, duration=e.duration, run_id=e.run_id, timestamp=e.timestamp,
            )
        old.append(e)
    return old


def test_usage_without_request_id_is_placed_before_its_response():
    user = [
        ev.TextMessageStart(message_id="u1", role="user", run_id="r1"),
        ev.TextMessageContent(message_id="u1", delta="Hi", run_id="r1"),
        ev.TextMessageEnd(message_id="u1", run_id="r1"),
    ]
    state = RunState.from_events(user + old_recording(model_run()))
    assert kinds(state) == ["Message", "ModelCall", "Message", "ToolCall", "ModelCall", "Message"]
    first = state.items[1]
    assert (first.status, first.input_tokens, first.started_at, first.duration) == ("success", 1000, 100.0, 1.0)
    assert state.runs["r1"].requests == 2


def test_model_calls_round_trip_through_a_recording(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with Recorder(state, path):
        state.apply_many(model_run())
    replayed = load_state(path)
    assert kinds(replayed) == kinds(state)
    for request_id, request in state.model_calls.items():
        values = {k: v for k, v in request.param.values().items() if k != "name"}
        assert {k: v for k, v in replayed.model_calls[request_id].param.values().items() if k != "name"} == values


def test_reset_clears_model_calls():
    state = RunState.from_events(model_run())
    state.reset()
    assert state.model_calls == {} and state.items == []


# The pydantic-ai adapter

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, exceptions  # noqa: E402
from pydantic_ai.capabilities import AbstractCapability  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from panel_agents.adapters.pydantic_ai import (  # noqa: E402
    ModelTraceRecorder,
    load_model_calls,
    merge_model_calls,
)

from .test_model_trace import make_agent  # noqa: E402
from .test_pydantic_ai import make_runner, streaming  # noqa: E402


async def test_runner_shows_each_request_before_the_parts_it_streams(tmp_path):
    runner = make_runner(make_agent())
    seen = []

    def check(event):
        if isinstance(event, ev.ToolCallStart):
            request = runner.state.items[-2]
            seen.append((type(request).__name__, request.status))

    runner.state.subscribe(check)
    with ModelTraceRecorder(runner, tmp_path / "trace.jsonl"):
        assert await runner.run("What is k1?") == "value is secret-k1"

    state = runner.state
    assert seen == [("ModelCallState", "running")]
    assert kinds(state) == ["Message", "ModelCall", "ToolCall", "ModelCall", "Message"]
    first, second = (i for i in state.items if isinstance(i, ModelCallState))
    assert first.tool_call_ids == ["call-1"] and second.tool_call_ids == []
    assert all(r.status == "success" and r.model == "gpt-4o" and r.duration is not None for r in (first, second))
    assert all(r.input_tokens > 0 and r.cost_usd > 0 for r in (first, second))
    calls = load_model_calls(tmp_path / "trace.jsonl")
    assert [c.request_id for c in calls] == [first.request_id, second.request_id]
    assert state.runs[runner.run_id].requests == 2


async def test_runner_with_test_model_emits_model_calls():
    agent = Agent(TestModel())

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    runner = make_runner(agent)
    await runner.run("Add")
    calls = [i for i in runner.state.items if isinstance(i, ModelCallState)]
    tool_ids = [i.tool_call_id for i in runner.state.items if isinstance(i, ToolCallState)]
    assert len(calls) == 2 and calls[0].tool_call_ids == tool_ids
    assert {c.model for c in calls} == {"test"}


async def test_a_failing_request_is_shown_as_failed():
    def model(messages, info):
        raise ConnectionError("provider unreachable")

    runner = make_runner(Agent(streaming(model, model_name="gpt-4o")))
    assert await runner.run("Go") is None
    (request,) = runner.state.model_calls.values()
    assert request.status == "failed" and "provider unreachable" in request.error
    assert request.duration is not None
    usage = [e for e in runner.state.events if isinstance(e, ev.Usage)]
    assert [u.request_id for u in usage] == [request.request_id]


async def test_a_cancelled_request_is_shown_as_failed():
    started = asyncio.Event()

    async def stream(messages, info):
        started.set()
        await asyncio.sleep(10)
        yield "never"

    from pydantic_ai.models.function import FunctionModel

    runner = make_runner(Agent(FunctionModel(stream_function=stream)))
    task = asyncio.ensure_future(runner.run("Go"))
    await started.wait()
    runner.cancel()
    await task
    (request,) = runner.state.model_calls.values()
    assert request.status == "failed" and request.ended_at is not None


def flaky(failures):
    """lookup_model, failing the first ``failures`` requests."""
    from .test_model_trace import lookup_model

    count = [0]

    def model(messages, info):
        count[0] += 1
        if count[0] <= failures:
            raise ConnectionError(f"overloaded {count[0]}")
        return lookup_model(messages, info)
    return model


@dataclass
class Retry(AbstractCapability):
    async def on_model_request_error(self, ctx, *, request_context, error):
        if request_context.attempt <= 3:
            raise exceptions.RetryModelRequest()
        raise error


needs_retry = pytest.mark.skipif(not hasattr(exceptions, "RetryModelRequest"), reason="needs RetryModelRequest")


@needs_retry
async def test_retried_attempts_belong_to_their_request(tmp_path):
    runner = make_runner(make_agent(flaky(2), capabilities=[Retry()]))
    events = []
    runner.state.subscribe(lambda e: events.append(e))
    before = time.time()
    with ModelTraceRecorder(runner, tmp_path / "trace.jsonl"):
        assert await runner.run("What is k1?") == "value is secret-k1"

    state = runner.state
    # The attempts do not move the request behind the tool call it issued.
    assert kinds(state) == ["Message", "ModelCall", "ToolCall", "ModelCall", "Message"]
    first = state.items[1]
    assert first.status == "success" and first.tool_call_ids == ["call-1"]
    assert [(a.request_id, a.error) for a in first.attempts] == [
        (f"{first.request_id}-a1", "ConnectionError: overloaded 1"), (f"{first.request_id}-a2", "ConnectionError: overloaded 2"),
    ]
    assert all(before <= a.timestamp - a.duration <= a.timestamp <= first.ended_at for a in first.attempts)
    assert state.runs[runner.run_id].requests == 4
    calls = load_model_calls(tmp_path / "trace.jsonl")
    assert [(c.request_id, c.attempt_of) for c in calls[:2]] == [(a.request_id, first.request_id) for a in first.attempts]


@needs_retry
async def test_attempts_are_appended_without_reordering_the_feed():
    runner = make_runner(make_agent(flaky(2), capabilities=[Retry()]))
    snapshots = []
    runner.state.param.watch(lambda e: snapshots.append((list(e.old), list(e.new))), "items")
    await runner.run("What is k1?")
    assert all(new[: len(old)] == old for old, new in snapshots)


def test_a_repeated_usage_is_ignored():
    events = model_run()
    state = RunState.from_events([*events[:8], events[7], *events[8:]])
    assert state.runs["r1"].requests == 2 and state.runs["r1"].input_tokens == 2100


def test_an_attempt_without_its_request_is_a_request_of_its_own():
    state = RunState.from_events([
        ev.RunStarted(run_id="r1"),
        ev.Usage(request_id="x-a1", attempt_of="x", model="m", error="overloaded", run_id="r1"),
    ])
    assert kinds(state) == ["ModelCall"] and state.items[0].status == "failed"


# Model calls from a model trace


async def record_session(tmp_path, model=None):
    runner = make_runner(make_agent(model) if model else make_agent())
    with Recorder(runner.state, tmp_path / "events.jsonl"), ModelTraceRecorder(runner, tmp_path / "trace.jsonl"):
        await runner.run("What is k1?")
        await runner.run("And k2?")
    return runner


def snapshot(state):
    return [
        (type(i).__name__, getattr(i, "tool_call_ids", None), getattr(i, "input_tokens", None), getattr(i, "status", None))
        for i in state.items
    ]


async def test_merged_trace_restores_model_calls_of_an_old_recording(tmp_path):
    runner = await record_session(tmp_path)
    old = old_recording(load_events(tmp_path / "events.jsonl"))
    assert not RunState.from_events(old).items[1].tool_call_ids

    merged = RunState.from_events(merge_model_calls(old, load_model_calls(tmp_path / "trace.jsonl")))
    assert snapshot(merged) == snapshot(runner.state)
    for run_id, run in runner.state.runs.items():
        assert (merged.runs[run_id].requests, merged.runs[run_id].input_tokens) == (run.requests, run.input_tokens)
        assert merged.runs[run_id].cost_usd == pytest.approx(run.cost_usd)


async def test_merging_into_a_recording_with_model_calls_changes_nothing(tmp_path):
    await record_session(tmp_path)
    events = load_events(tmp_path / "events.jsonl")
    merged = merge_model_calls(events, load_model_calls(tmp_path / "trace.jsonl"))
    assert [type(e) for e in merged] == [type(e) for e in events]


async def test_merging_places_calls_by_tool_call_without_usage_events(tmp_path):
    runner = await record_session(tmp_path)
    bare = [e for e in old_recording(load_events(tmp_path / "events.jsonl")) if not isinstance(e, ev.Usage)]
    merged = RunState.from_events(merge_model_calls(bare, load_model_calls(tmp_path / "trace.jsonl")))
    assert snapshot(merged) == snapshot(runner.state)


async def test_merging_adds_the_request_a_failed_run_was_waiting_for(tmp_path):
    fail = [False]

    def model(messages, info):
        if fail[0]:
            raise ConnectionError("provider unreachable")
        from .test_model_trace import lookup_model
        return lookup_model(messages, info)

    runner = make_runner(make_agent(model))
    with Recorder(runner.state, tmp_path / "events.jsonl"), ModelTraceRecorder(runner, tmp_path / "trace.jsonl"):
        await runner.run("What is k1?")
        fail[0] = True
        await runner.run("And k2?")
    failed_run = runner.run_id
    old = old_recording(load_events(tmp_path / "events.jsonl"))
    merged = RunState.from_events(merge_model_calls(old, load_model_calls(tmp_path / "trace.jsonl")))
    failed = [r for r in merged.model_calls.values() if r.run_id == failed_run]
    assert [(r.status, r.error) for r in failed] == [("failed", "ConnectionError: provider unreachable")]
    assert [type(i).__name__ for i in merged.items if i.run_id == failed_run] == ["MessageState", "ModelCallState", "MessageState"]


async def test_usage_the_trace_does_not_cover_is_kept(tmp_path):
    runner = await record_session(tmp_path)
    old = old_recording(load_events(tmp_path / "events.jsonl"))
    calls = load_model_calls(tmp_path / "trace.jsonl")[:-1]
    merged = RunState.from_events(merge_model_calls(old, calls))
    assert [r.requests for r in merged.runs.values()] == [r.requests for r in runner.state.runs.values()]
    assert sum(r.input_tokens for r in merged.runs.values()) == sum(r.input_tokens for r in runner.state.runs.values())


@needs_retry
async def test_merging_keeps_attempts_with_their_request(tmp_path):
    runner = make_runner(make_agent(flaky(2), capabilities=[Retry()]))
    with Recorder(runner.state, tmp_path / "events.jsonl"), ModelTraceRecorder(runner, tmp_path / "trace.jsonl"):
        await runner.run("What is k1?")
    events = load_events(tmp_path / "events.jsonl")
    calls = load_model_calls(tmp_path / "trace.jsonl")
    assert merge_model_calls(events, calls) == events

    merged = RunState.from_events(merge_model_calls(old_recording(events), calls))
    assert snapshot(merged) == snapshot(runner.state)
    first = merged.items[1]
    assert [a.request_id for a in first.attempts] == [a.request_id for a in runner.state.items[1].attempts]
    assert merged.runs[runner.run_id].requests == 4


def test_calls_of_unknown_runs_are_left_out():
    from panel_agents.adapters.model_trace import ModelRequestSummary

    events = model_run()
    merged = merge_model_calls(events, [ModelRequestSummary(run_id="elsewhere", model="m")])
    assert merged == events
