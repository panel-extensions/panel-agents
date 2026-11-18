import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import CallDeferred, ModelRetry, ToolFailed
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from panel_agents import MessageState, Recorder, RunState, load_state
from panel_agents import events as ev
from panel_agents.adapters.pydantic_ai import AgentRunner, artifact_return, publish_artifact

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

FIXTURE = Path(__file__).parent / "fixtures" / "sql_agent_run.jsonl"


def of_type(state, *types):
    return [e for e in state.events if isinstance(e, types)]


def tool_returns(messages):
    return [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]


def resolve_on_request(state, **decision):
    def callback(event):
        if isinstance(event, ev.ApprovalRequested):
            asyncio.get_running_loop().call_soon(lambda: state.resolve_approval(event.tool_call_id, **decision))
    return state.subscribe(callback)


def streaming(function, **kwargs):
    """A FunctionModel streaming whatever ``function`` returns, since runs are always streamed."""
    async def stream(messages, info):
        for i, part in enumerate(function(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                yield {i: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)}
    return FunctionModel(function, stream_function=stream, **kwargs)


def make_runner(agent, **params):
    return AgentRunner(agent, RunState(flush_interval=0), **params)


async def test_text_streaming():
    async def stream(messages, info: AgentInfo):
        for chunk in ["Hello", ", ", "world", "!"]:
            yield chunk

    runner = make_runner(Agent(FunctionModel(stream_function=stream)))
    result = await runner.run("Hi")

    assert result == "Hello, world!"
    state = runner.state
    user, request, reply = state.items
    assert (request.status, request.model) == ("success", "function::stream")
    assert (user.role, user.content) == ("user", "Hi")
    assert (reply.role, reply.content, reply.finished) == ("assistant", "Hello, world!", True)
    assert [e.delta for e in of_type(state, ev.TextMessageContent) if e.message_id == reply.message_id] == ["Hello", ", ", "world", "!"]
    assert type(state.events[3]) is ev.RunStarted and type(state.events[-1]) is ev.RunFinished
    run = state.runs[runner.run_id]
    assert (run.status, run.result, run.input) == ("finished", "Hello, world!", "Hi")
    assert not state.running and not runner.running


async def test_thinking_is_streamed_as_its_own_message():
    async def stream(messages, info):
        yield {0: DeltaThinkingPart(content="Let me ")}
        yield {0: DeltaThinkingPart(content="think.")}
        yield "Done."

    runner = make_runner(Agent(FunctionModel(stream_function=stream)))
    await runner.run("Hi")

    thinking, reply = runner.state.items[2:]
    assert (thinking.kind, thinking.content, thinking.finished) == ("thinking", "Let me think.", True)
    assert (reply.kind, reply.content) == ("text", "Done.")
    assert len(of_type(runner.state, ev.ThinkingStart)) == 1


async def test_tool_call_args_stream_as_json_deltas():
    async def stream(messages, info):
        if not tool_returns(messages):
            yield "Querying."
            yield {0: DeltaToolCall(name="run_query", json_args='{"sql": "SELECT', tool_call_id="call-1")}
            yield {0: DeltaToolCall(json_args=" * FROM sales")}
            yield {0: DeltaToolCall(json_args='"}')}
        else:
            yield "There are 2 rows."

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain
    def run_query(sql: str) -> list[dict]:
        return [{"a": 1}, {"a": 2}]

    runner = make_runner(agent)
    await runner.run("How many rows?")

    state = runner.state
    call = state.tool_calls["call-1"]
    assert [e.delta for e in of_type(state, ev.ToolCallArgs)] == ['{"sql": "SELECT', " * FROM sales", '"}']
    assert call.args == {"sql": "SELECT * FROM sales"}
    assert (call.status, call.result) == ("success", [{"a": 1}, {"a": 2}])
    message = state.messages[call.parent_message_id]
    assert message.content == "Querying."
    assert [type(i).__name__ for i in state.items] == [
        "MessageState", "ModelCallState", "MessageState", "ToolCallState", "ModelCallState", "MessageState",
    ]


async def test_tool_call_mapping_with_test_model():
    agent = Agent(TestModel())

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    runner = make_runner(agent)
    await runner.run("Add")

    (call,) = runner.state.tool_calls.values()
    assert (call.tool_name, call.status, call.args, call.result) == ("add", "success", {"a": 0, "b": 0}, 0)
    types = [type(e) for e in runner.state.events if getattr(e, "tool_call_id", None) == call.tool_call_id]
    assert types == [ev.ToolCallStart, ev.ToolCallArgs, ev.ToolCallEnd, ev.ToolCallResult]


def calls_then_text(*calls):
    def model(messages, info):
        step = sum(isinstance(m, ModelResponse) for m in messages)
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"call-{step}")])
        return ModelResponse(parts=[TextPart("done")])
    return model


async def test_tool_failure_and_retry_are_errors():
    agent = Agent(streaming(calls_then_text(("lookup", {"key": "a"}), ("lookup", {"key": "b"}), ("lookup", {"key": "c"}))))

    @agent.tool_plain
    def lookup(key: str) -> str:
        if key == "a":
            raise ToolFailed("No such key")
        if key == "b":
            raise ModelRetry("Try another key")
        return "found"

    runner = make_runner(agent)
    assert await runner.run("Look up") == "done"

    calls = runner.state.tool_calls
    assert (calls["call-0"].status, calls["call-0"].error) == ("failed", "No such key")
    assert calls["call-1"].status == "failed"
    assert "Try another key" in calls["call-1"].error
    assert (calls["call-2"].status, calls["call-2"].result) == ("success", "found")
    assert runner.state.runs[runner.run_id].status == "finished"


@pytest.mark.parametrize("raise_errors", [False, True])
async def test_exception_produces_run_error(raise_errors):
    agent = Agent(streaming(calls_then_text(("explode", {}))))

    @agent.tool_plain
    def explode() -> str:
        raise ValueError("boom")

    runner = make_runner(agent, raise_errors=raise_errors)
    if raise_errors:
        with pytest.raises(ValueError, match="boom"):
            await runner.run("Go")
    else:
        assert await runner.run("Go") is None
    assert isinstance(runner.error, ValueError)
    run = runner.state.runs[runner.run_id]
    assert (run.status, run.error) == ("error", "boom")
    assert runner.state.tool_calls["call-0"].status == "failed"
    assert of_type(runner.state, ev.RunError)[0].code == "ValueError"
    assert not of_type(runner.state, ev.RunFinished)


async def test_usage_event_per_model_response():
    agent = Agent(streaming(calls_then_text(("noop", {})), model_name="gpt-4o"))

    @agent.tool_plain
    def noop() -> str:
        return "ok"

    runner = make_runner(agent)
    await runner.run("Go")

    usage = runner.state.runs[runner.run_id].usage
    assert len(usage) == 2
    assert all(u.model == "gpt-4o" and u.input_tokens > 0 for u in usage)
    assert usage[1].output_tokens > 0
    assert all(u.cost_usd is not None and u.cost_usd > 0 for u in usage)
    assert runner.state.runs[runner.run_id].cost_usd == pytest.approx(sum(u.cost_usd for u in usage))


async def test_usage_cost_is_none_for_unknown_models():
    runner = make_runner(Agent(TestModel()))
    await runner.run("Go")
    (usage,) = runner.state.runs[runner.run_id].usage
    assert (usage.model, usage.cost_usd) == ("test", None)


async def test_multi_turn_keeps_history():
    def model(messages, info):
        prompts = [p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)]
        return ModelResponse(parts=[TextPart(" | ".join(prompts))])

    runner = make_runner(Agent(streaming(model)))
    assert await runner.run("first") == "first"
    assert await runner.run("second") == "first | second"
    assert len(runner.state.runs) == 2
    assert [m.content for m in runner.state.items if isinstance(m, MessageState)] == ["first", "first", "second", "first | second"]

    runner.clear()
    assert await runner.run("third") == "third"


def approval_agent():
    def model(messages, info):
        returns = tool_returns(messages)
        if not returns:
            return ModelResponse(parts=[ToolCallPart("delete_rows", {"table": "sales", "limit": 10}, tool_call_id="call-1")])
        return ModelResponse(parts=[TextPart(f"Tool said: {returns[-1].content}")])

    agent = Agent(streaming(model))

    @agent.tool_plain(requires_approval=True)
    def delete_rows(table: str, limit: int = 100) -> str:
        """Delete rows from a table."""
        return f"deleted {limit} rows from {table}"

    return agent


async def test_approval_approve():
    runner = make_runner(approval_agent())
    resolve_on_request(runner.state, approved=True)
    assert await runner.run("Clean up") == "Tool said: deleted 10 rows from sales"

    state = runner.state
    request = state.approvals["call-1"]
    assert (request.tool_name, request.args, request.status) == ("delete_rows", {"table": "sales", "limit": 10}, "approved")
    assert request.description == "Delete rows from a table."
    assert request.schema["properties"]["limit"] == {"default": 100, "type": "integer"}
    assert state.tool_calls["call-1"].status == "success"
    assert len(state.runs) == 1 and state.runs[runner.run_id].status == "finished"
    assert not state.pending_approvals


async def test_approval_with_edited_args():
    runner = make_runner(approval_agent())
    resolve_on_request(runner.state, approved=True, args={"table": "sales", "limit": 3})
    assert await runner.run("Clean up") == "Tool said: deleted 3 rows from sales"
    call = runner.state.tool_calls["call-1"]
    assert call.approval.edited
    assert (call.status, call.args, call.result) == ("success", {"table": "sales", "limit": 3}, "deleted 3 rows from sales")


async def test_approval_deny():
    runner = make_runner(approval_agent())
    resolve_on_request(runner.state, approved=False, reason="Not today")
    assert await runner.run("Clean up") == "Tool said: Not today"
    call = runner.state.tool_calls["call-1"]
    assert (call.status, call.error, call.approval.status) == ("denied", "Not today", "denied")
    assert not [e for e in of_type(runner.state, ev.ToolCallResult) if e.tool_call_id == "call-1"]


async def test_approval_resolved_before_handler_waits():
    runner = make_runner(approval_agent())

    def callback(event):
        if isinstance(event, ev.ApprovalRequested):
            runner.state.resolve_approval(event.tool_call_id, approved=True)

    runner.state.subscribe(callback)
    assert await runner.run("Clean up") == "Tool said: deleted 10 rows from sales"


async def test_cancel_while_awaiting_approval():
    runner = make_runner(approval_agent())

    def callback(event):
        if isinstance(event, ev.ApprovalRequested):
            asyncio.get_running_loop().call_soon(runner.cancel)

    runner.state.subscribe(callback)
    assert await runner.run("Clean up") is None

    state = runner.state
    assert of_type(state, ev.RunError)[0].code == "cancelled"
    assert state.runs[runner.run_id].status == "error"
    assert state.approvals["call-1"].status == "denied"
    assert not state.pending_approvals and not state.running
    assert runner.messages


async def test_external_call_is_left_pending_and_can_be_resumed():
    def model(messages, info):
        returns = tool_returns(messages)
        if not returns:
            return ModelResponse(parts=[ToolCallPart("ask_backend", {"q": "x"}, tool_call_id="call-1")])
        return ModelResponse(parts=[TextPart(f"Backend said {returns[-1].content}")])

    agent = Agent(streaming(model), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def ask_backend(q: str) -> str:
        raise CallDeferred()

    runner = make_runner(agent)
    output = await runner.run("Ask")

    assert isinstance(output, DeferredToolRequests) and runner.deferred is output
    assert [c.tool_call_id for c in output.calls] == ["call-1"]
    assert runner.state.tool_calls["call-1"].status == "running"
    assert not runner.state.approvals

    result = await runner.run(deferred_tool_results=DeferredToolResults(calls={"call-1": "42"}))
    assert result == "Backend said 42"
    assert runner.state.tool_calls["call-1"].status == "success"
    assert runner.deferred is None


async def test_concurrent_run_is_rejected():
    runner = make_runner(approval_agent())
    task = asyncio.create_task(runner.run("Clean up"))
    while not runner.state.pending_approvals:
        await asyncio.sleep(0)
    with pytest.raises(RuntimeError):
        await runner.run("Again")
    runner.state.resolve_approval("call-1", approved=True)
    assert await task == "Tool said: deleted 10 rows from sales"


async def test_publish_artifact_versions():
    agent = Agent(streaming(calls_then_text(("write", {"text": "v1"}), ("write", {"text": "v2"}), ("table", {}))))

    @agent.tool
    async def write(ctx: RunContext, text: str) -> str:
        await publish_artifact(ctx, "draft", text, kind="text", title="Draft")
        return "saved"

    @agent.tool_plain
    def table() -> object:
        return artifact_return("2 rows", "table", pd.DataFrame({"a": [1, 2]}), title="Table", metadata={"sql": "SELECT a"})

    runner = make_runner(agent)
    await runner.run("Write")

    store = runner.state.artifacts
    draft = store["draft"]
    assert [v.value for v in draft.versions] == ["v1", "v2"]
    assert [v.tool_call_id for v in draft.versions] == ["call-0", "call-1"]
    assert (draft.kind, draft.title) == ("text", "Draft")
    frame = store["table"]
    assert (frame.kind, frame.latest.tool_call_id, frame.latest.metadata) == ("table", "call-2", {"sql": "SELECT a"})
    pd.testing.assert_frame_equal(frame.value, pd.DataFrame({"a": [1, 2]}))
    assert runner.state.tool_calls["call-2"].result == "2 rows"
    published = of_type(runner.state, ev.ArtifactPublished, ev.ToolCallResult)
    assert [type(e) for e in published[-2:]] == [ev.ArtifactPublished, ev.ToolCallResult]


async def test_agent_versions_record_the_run_and_actor():
    agent = Agent(streaming(calls_then_text(("write", {"text": "v1"}), ("table", {}))))

    @agent.tool
    async def write(ctx: RunContext, text: str) -> str:
        await publish_artifact(ctx, "draft", text, kind="text")
        return "saved"

    @agent.tool_plain
    def table() -> object:
        return artifact_return("1 row", "table", [1])

    runner = make_runner(agent)
    await runner.run("Write")
    versions = [runner.state.artifacts[a].latest for a in ("draft", "table")]
    assert [(v.run_id, v.tool_call_id, v.actor, v.user) for v in versions] == [
        (runner.run_id, "call-0", "agent", None), (runner.run_id, "call-1", "agent", None),
    ]


async def test_host_code_in_a_tool_sees_the_run_and_tool_call():
    from panel_agents import current_run_id, current_tool_call_id

    agent = Agent(streaming(calls_then_text(("edit", {"value": 1}), ("peek", {}))))
    state = RunState(flush_interval=0)
    seen = {}

    def mutate(value):
        # Host code with no RunContext, e.g. an editor method a tool calls.
        seen["edit"] = (current_run_id(), current_tool_call_id())
        state.publish_artifact("doc", {"value": value}, applied=True, metadata={"op": "set_value"})

    @agent.tool_plain
    async def edit(value: int) -> str:
        mutate(value)
        return "edited"

    @agent.tool
    def peek(ctx: RunContext) -> str:
        seen["peek"] = (current_run_id(), current_tool_call_id(), ctx.run_id, ctx.tool_call_id)
        return "ok"

    runner = AgentRunner(agent, state)
    await runner.run("Edit")
    assert seen == {"edit": (runner.run_id, "call-0"), "peek": (runner.run_id, "call-1") * 2}
    version = state.artifacts["doc"].latest
    assert (version.run_id, version.tool_call_id, version.actor, version.user) == (runner.run_id, "call-0", "agent", None)
    assert current_run_id() is None


async def test_edits_from_other_callbacks_during_a_run_are_the_users():
    import contextvars

    state = RunState(flush_interval=0)
    agent = Agent(streaming(calls_then_text(("delete", {}))))

    @agent.tool_plain(requires_approval=True)
    def delete() -> str:
        return "deleted"

    def on_request(event):
        if isinstance(event, ev.ApprovalRequested):
            def user_edit():
                # A widget callback runs in a context of its own, not the agent's.
                state.publish_artifact("doc", {"a": 1}, applied=True, user="ann")
                state.resolve_approval(event.tool_call_id, approved=True)
            asyncio.get_running_loop().call_soon(user_edit, context=contextvars.Context())

    state.subscribe(on_request)
    runner = AgentRunner(agent, state)
    await runner.run("Delete")
    version = state.artifacts["doc"].latest
    assert (version.run_id, version.tool_call_id, version.actor, version.user) == (None, None, "user", "ann")


# A fake SQL analyst used for the record/replay test and the shared fixture.

SALES = pd.DataFrame({
    "region": ["North", "South", "East", "West", "North", "West", "East", "South"],
    "quarter": ["Q3"] * 8,
    "amount": [120.0, 80.5, 95.25, 210.0, 60.0, 45.5, 30.0, 12.75],
})

FIRST_SQL = "SELECT region, SUM(amount) AS revenue FROM sales GROUP BY region"
SECOND_SQL = FIRST_SQL + " ORDER BY revenue DESC"


def _chunks(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


async def _stream_sql_analyst(messages, info: AgentInfo):
    step = sum(isinstance(m, ModelResponse) for m in messages)
    if step == 0:
        for chunk in _chunks("The user wants revenue by region, so I should look at the schema first.", 18):
            yield {0: DeltaThinkingPart(content=chunk)}
        yield "Let me check which tables are available."
        yield {1: DeltaToolCall(name="list_tables", json_args="{}", tool_call_id="call_list_tables")}
    elif step == 1:
        yield {0: DeltaToolCall(name="run_query", tool_call_id="call_query_1")}
        for chunk in _chunks(json.dumps({"sql": FIRST_SQL}), 12):
            yield {0: DeltaToolCall(json_args=chunk)}
    elif step == 2:
        yield "The totals look right; I'll sort them so the top region comes first."
        yield {1: DeltaToolCall(name="run_query", tool_call_id="call_query_2")}
        for chunk in _chunks(json.dumps({"sql": SECOND_SQL}), 12):
            yield {1: DeltaToolCall(json_args=chunk)}
    elif step == 3:
        yield {0: DeltaToolCall(name="save_report", tool_call_id="call_save_report")}
        for chunk in _chunks(json.dumps({"name": "revenue_by_region", "sql": SECOND_SQL}), 16):
            yield {0: DeltaToolCall(json_args=chunk)}
    else:
        saved = tool_returns(messages)[-1].content
        text = (
            "West leads Q3 with $255.50 in revenue, followed by North ($180.00), East ($125.25) "
            f"and South ($93.25). {saved}"
        )
        for chunk in _chunks(text, 9):
            yield chunk


def sql_agent():
    agent = Agent(
        FunctionModel(stream_function=_stream_sql_analyst, model_name="gpt-4o"),
        name="sql_analyst",
        instructions="You are a SQL analyst. Use the tools to answer questions about the sales database.",
    )

    @agent.tool_plain
    def list_tables() -> dict[str, list[str]]:
        """List the tables in the database with their columns."""
        return {"sales": list(SALES.columns)}

    @agent.tool
    async def run_query(ctx: RunContext, sql: str) -> object:
        """Run a SQL query against the sales database."""
        df = SALES.groupby("region", as_index=False)["amount"].sum().rename(columns={"amount": "revenue"})
        if "ORDER BY" in sql:
            df = df.sort_values("revenue", ascending=False, ignore_index=True)
        await publish_artifact(ctx, "query_sql", sql, kind="sql", title="Query")
        summary = f"{len(df)} rows: " + ", ".join(f"{r.region}={r.revenue:.2f}" for r in df.itertuples())
        return artifact_return(summary, "query_result", df, title="Revenue by region", metadata={"sql": sql})

    @agent.tool_plain(requires_approval=True)
    def save_report(name: str, sql: str) -> str:
        """Save a query as a named report that is shared with the team."""
        return f"I saved the query as the report '{name}'."

    return agent


async def run_sql_agent(state: RunState) -> AgentRunner:
    runner = AgentRunner(sql_agent(), state, thread_id="thread-sql-demo")
    resolve_on_request(state, approved=True, args={"name": "q3_revenue_by_region", "sql": SECOND_SQL}, user="analyst@example.com")
    await runner.run("Which regions had the highest revenue in Q3?")
    return runner


def snapshot(state: RunState) -> dict:
    return {
        "items": [(type(i).__name__, getattr(i, "content", None), getattr(i, "status", None)) for i in state.items],
        "tool_calls": {k: (c.tool_name, c.status, c.args, c.result, c.error) for k, c in state.tool_calls.items()},
        "approvals": {k: (a.status, a.args, a.final_args, a.user, a.schema) for k, a in state.approvals.items()},
        "runs": {k: (r.status, r.result, r.input_tokens, r.output_tokens, r.cost_usd) for k, r in state.runs.items()},
        "artifacts": {
            a.artifact_id: (a.kind, a.title, [(v.version, v.tool_call_id, v.metadata) for v in a.versions])
            for a in state.artifacts
        },
    }


async def test_record_replay_equivalence(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with Recorder(state, path):
        runner = await run_sql_agent(state)

    replayed = load_state(path)
    assert snapshot(replayed) == snapshot(state)
    for artifact in state.artifacts:
        for live, loaded in zip(artifact.versions, replayed.artifacts[artifact.artifact_id].versions, strict=True):
            if isinstance(live.value, pd.DataFrame):
                pd.testing.assert_frame_equal(loaded.value, live.value)
            else:
                assert loaded.value == live.value

    assert runner.error is None
    assert len(state.artifacts["query_result"].versions) == 2
    assert state.approvals["call_save_report"].edited
    assert state.tool_calls["call_save_report"].result == "I saved the query as the report 'q3_revenue_by_region'."
    assert len(state.runs) == 1
    assert len(state.runs[runner.run_id].usage) == 5


def test_sql_fixture_matches_agent():
    replayed = load_state(FIXTURE)
    state = RunState(flush_interval=0)
    asyncio.run(run_sql_agent(state))
    live, recorded = snapshot(state), snapshot(replayed)
    # Run ids are random, so compare runs by value only.
    assert list(live.pop("runs").values()) == list(recorded.pop("runs").values())
    assert live == recorded


if __name__ == "__main__":
    # Regenerate the shared fixture: python -m tests.test_pydantic_ai
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.unlink(missing_ok=True)

    async def main():
        state = RunState(flush_interval=0)
        with Recorder(state, FIXTURE):
            await run_sql_agent(state)

    asyncio.run(main())
