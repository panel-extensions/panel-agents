import asyncio
import os
from typing import Annotated

import pytest
from pydantic import BaseModel, StringConstraints

from panel_agents import AgentChat, ApprovalPolicy, ApprovalRule, RunState
from panel_agents import events as ev
from panel_agents.components.approval import ApprovalCard

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def request_for(state, tool_name, args, tool_call_id="t1"):
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id=tool_call_id, tool_name=tool_name, args=args, run_id="r"),
    ])
    return state.approvals[tool_call_id]


def test_string_rules_match_by_name_and_glob():
    policy = ApprovalPolicy(["save_dashboard", "delete_*"])
    assert policy.requires_approval("save_dashboard")
    assert policy.requires_approval("delete_node", {"name": "a"})
    assert not policy.requires_approval("add_node")
    assert not policy.requires_approval("save_dashboard_as_copy")


def test_first_matching_rule_wins_and_predicates_see_args():
    destructive = ApprovalRule("run_sql", when=lambda args: not args["sql"].lower().startswith("select"))
    policy = ApprovalPolicy([destructive, ApprovalRule("run_*")])
    assert policy.rule_for("run_sql", {"sql": "DROP TABLE x"}) is destructive
    assert policy.rule_for("run_sql", {"sql": "SELECT 1"}).tool == "run_*"
    assert policy.rule_for("other", {}) is None


def test_predicate_gets_a_copy_of_the_args():
    seen = []
    policy = ApprovalPolicy([ApprovalRule("t", when=lambda args: seen.append(args) or args.pop("x", False))])
    args = {"x": True}
    assert policy.requires_approval("t", args)
    assert args == {"x": True}


def test_can_approve_by_user_names_and_callable():
    state = RunState(flush_interval=0)
    request = request_for(state, "publish", {})
    by_name = ApprovalPolicy([ApprovalRule("publish", approvers={"admin"})])
    assert by_name.can_approve(request, "admin")
    assert not by_name.can_approve(request, "bob")
    assert not by_name.can_approve(request, None)

    by_func = ApprovalPolicy([ApprovalRule("publish", approvers=lambda req, user: user.endswith("@example.com"))])
    assert by_func.can_approve(request, "ann@example.com")
    assert not by_func.can_approve(request, "ann@other.org")

    assert ApprovalPolicy(["other"]).can_approve(request, "anyone")


def test_card_uses_policy_as_permission_hook():
    state = RunState(flush_interval=0)
    request = request_for(state, "publish", {})
    policy = ApprovalPolicy([ApprovalRule("publish", approvers={"admin"})])
    card = ApprovalCard(request=request, state=state, can_approve=policy.can_approve, user="bob")
    assert not card.approve()
    assert request.status == "pending"
    card.user = "admin"
    assert card.approve()
    assert request.status == "approved"


pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from panel_agents.adapters.pydantic_ai import AgentRunner  # noqa: E402

from .test_pydantic_ai import streaming  # noqa: E402


def sql_agent(*statements):
    calls = [ToolCallPart("run_sql", {"sql": sql}, tool_call_id=f"call-{i}") for i, sql in enumerate(statements)]

    def model(messages, info):
        returns = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=calls)
        return ModelResponse(parts=[TextPart(" | ".join(str(r.content) for r in returns))])

    agent = Agent(streaming(model))

    @agent.tool_plain
    def run_sql(sql: str) -> str:
        """Run SQL."""
        return f"ran {sql}"

    return agent


def calling_once(name, args):
    def model(messages, info):
        if [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]:
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id="call-0")])
    return model


def approve_when_requested(state, decisions):
    def callback(event):
        if isinstance(event, ev.ApprovalRequested):
            approved = decisions[event.tool_call_id]
            asyncio.get_running_loop().call_soon(lambda: state.resolve_approval(event.tool_call_id, approved=approved))
    state.subscribe(callback)


async def test_runner_policy_pauses_only_matching_calls():
    policy = ApprovalPolicy([ApprovalRule("run_sql", when=lambda args: args["sql"].startswith("DROP"))])
    runner = AgentRunner(sql_agent("SELECT 1", "DROP TABLE t"), RunState(flush_interval=0), approval_policy=policy)
    approve_when_requested(runner.state, {"call-1": True})

    output = await runner.run("go")

    assert output == "ran SELECT 1 | ran DROP TABLE t"
    assert list(runner.state.approvals) == ["call-1"]
    assert runner.state.approvals["call-1"].args == {"sql": "DROP TABLE t"}
    assert runner.state.approvals["call-1"].description == "Run SQL."
    assert {c.status for c in runner.state.tool_calls.values()} == {"success"}


async def test_runner_policy_denial_is_reported_to_the_model():
    runner = AgentRunner(sql_agent("DROP TABLE t"), RunState(flush_interval=0), approval_policy=ApprovalPolicy(["run_*"]))
    approve_when_requested(runner.state, {"call-0": False})

    output = await runner.run("go")

    assert output == "The user denied this tool call."
    assert runner.state.tool_calls["call-0"].status == "denied"


async def test_runner_without_policy_runs_everything():
    runner = AgentRunner(sql_agent("DROP TABLE t"), RunState(flush_interval=0))
    assert await runner.run("go") == "ran DROP TABLE t"
    assert not runner.state.approvals


def test_chat_takes_can_approve_from_the_runner_policy():
    policy = ApprovalPolicy([ApprovalRule("run_sql", approvers={"admin"})])
    runner = AgentRunner(sql_agent(), RunState(flush_interval=0), approval_policy=policy)
    chat = AgentChat(runner=runner)
    assert chat.can_approve == policy.can_approve

    request = request_for(runner.state, "run_sql", {"sql": "DROP"})
    card = chat._approval_factory()(request, runner.state)
    assert card.can_approve == policy.can_approve
    card.user = "bob"
    assert not card.permitted


def test_raising_predicates_fail_closed():
    state = RunState(flush_interval=0)
    request = request_for(state, "publish", {})
    rule = ApprovalRule("publish", when=lambda args: args["missing"], approvers=lambda req, user: user.endswith("@x"))
    assert rule.matches("publish", {})
    assert not rule.permits(request, None)
    assert not ApprovalPolicy([rule]).can_approve(request, None)


def test_requests_are_decided_once():
    state = RunState(flush_interval=0)
    state.apply(ev.ToolCallStart(tool_call_id="t1", tool_name="x", run_id="r"))
    request = request_for(state, "x", {})
    state.resolve_approval("t1", True, user="alice")
    with pytest.raises(ValueError, match="already approved"):
        state.resolve_approval("t1", False, user="mallory", reason="no")
    # A second decision that reaches the state as an event is ignored too.
    state.apply(ev.ApprovalResolved(tool_call_id="t1", approved=False, user="mallory", run_id="r"))
    assert (request.status, request.user, state.tool_calls["t1"].status) == ("approved", "alice", "running")


def test_card_rejects_edits_the_user_may_not_approve():
    state = RunState(flush_interval=0)
    request = request_for(state, "run_sql", {"sql": "SELECT 1"})
    policy = ApprovalPolicy([ApprovalRule("run_sql", when=lambda a: "delete" in a["sql"].lower(), approvers={"admin"}), "run_sql"])
    card = ApprovalCard(request=request, state=state, can_approve=policy.can_approve, user="bob")
    assert card.permitted
    card.editing = True
    card.form.field("sql").value = "DELETE FROM users"
    assert not card.approve()
    assert "not permitted" in card.error and request.status == "pending"


def pause_then(runner, decide):
    async def run():
        task = asyncio.create_task(runner.run("go"))
        for _ in range(500):
            if runner.state.pending_approvals:
                break
            await asyncio.sleep(0.01)
        decide(runner.state.pending_approvals[0])
        return await task
    return run()


async def test_runner_rejects_edits_that_need_another_approver():
    executed = []
    agent = Agent(streaming(calling_once("run_sql", {"sql": "SELECT 1"})))

    @agent.tool_plain
    def run_sql(sql: str) -> str:
        executed.append(sql)
        return f"ran {sql}"

    policy = ApprovalPolicy([ApprovalRule("run_sql", when=lambda a: "delete" in a["sql"].lower(), approvers={"admin"}), "run_sql"])
    runner = AgentRunner(agent, RunState(flush_interval=0), approval_policy=policy)

    def decide(request):
        with pytest.raises(PermissionError):
            runner.state.resolve_approval(request.tool_call_id, True, args={"sql": "DELETE FROM users"}, user="bob")
        # A decision applied straight to the state is not accepted either, and the request stays pending.
        runner.state.apply(ev.ApprovalResolved(
            tool_call_id=request.tool_call_id, approved=True, args={"sql": "DELETE FROM users"}, user="bob", run_id=request.run_id,
        ))
        assert request.status == "pending"
        runner.state.resolve_approval(request.tool_call_id, False, user="bob", reason="not mine")

    await pause_then(runner, decide)
    assert executed == []
    assert runner.state.tool_calls["call-0"].status == "denied"


async def test_runner_checks_approvers_on_validated_args():
    agent = Agent(streaming(calling_once("delete_node", {"name": "sales"})))
    deleted = []

    @agent.tool_plain
    def delete_node(name: str, cascade: bool = True) -> str:
        deleted.append((name, cascade))
        return "deleted"

    policy = ApprovalPolicy([ApprovalRule("delete_node", when=lambda a: a["cascade"], approvers={"admin"})])
    runner = AgentRunner(agent, RunState(flush_interval=0), approval_policy=policy)

    def decide(request):
        assert request.args == {"name": "sales", "cascade": True}
        assert not policy.can_approve(request, "bob")
        with pytest.raises(PermissionError):
            runner.state.resolve_approval(request.tool_call_id, True, user="bob")
        runner.state.resolve_approval(request.tool_call_id, True, user="admin")

    await pause_then(runner, decide)
    assert deleted == [("sales", True)]


def test_chat_follows_a_new_runner_policy():
    runner = AgentRunner(sql_agent(), RunState(flush_interval=0))
    chat = AgentChat(runner=runner)
    assert chat.can_approve is None
    policy = ApprovalPolicy(["run_sql"])
    runner.approval_policy = policy
    assert chat.can_approve == policy.can_approve



class Options(BaseModel):
    cascade: bool = True


async def test_edits_are_checked_after_validation_with_nested_defaults():
    ran = []
    agent = Agent(streaming(calling_once("delete_node", {"name": "sales", "options": {"cascade": False}})))

    @agent.tool_plain
    def delete_node(name: str, options: Options) -> str:
        ran.append(options)
        return "ok"

    policy = ApprovalPolicy([ApprovalRule("delete_node", when=lambda a: a["options"]["cascade"], approvers={"admin"}), "delete_node"])
    runner = AgentRunner(agent, RunState(flush_interval=0), approval_policy=policy)

    def decide(request):
        assert policy.can_approve(request, "bob")
        # Leaving out cascade means the default, True, which only admins may approve.
        with pytest.raises(PermissionError):
            runner.state.resolve_approval(request.tool_call_id, True, args={"name": "sales", "options": {}}, user="bob")
        runner.state.resolve_approval(request.tool_call_id, True, user="bob")

    await pause_then(runner, decide)
    assert ran == [Options(cascade=False)]


async def test_edits_are_checked_after_transforming_validators():
    ran = []
    agent = Agent(streaming(calling_once("purge", {"table": "t"})))
    Mode = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)]

    @agent.tool_plain
    def purge(table: str, mode: Mode = "soft") -> str:
        ran.append(mode)
        return "ok"

    policy = ApprovalPolicy([ApprovalRule("purge", when=lambda a: a["mode"] == "hard", approvers={"admin"}), "purge"])
    runner = AgentRunner(agent, RunState(flush_interval=0), approval_policy=policy)

    def decide(request):
        with pytest.raises(PermissionError):
            runner.state.resolve_approval(request.tool_call_id, True, args={"table": "t", "mode": " HARD"}, user="bob")
        with pytest.raises(ValueError, match="Invalid arguments"):
            runner.state.resolve_approval(request.tool_call_id, True, args={"table": 1}, user="bob")
        runner.state.resolve_approval(request.tool_call_id, False, user="bob")

    await pause_then(runner, decide)
    assert ran == []


async def test_denials_need_an_approver_too():
    agent = Agent(streaming(calling_once("publish", {})))

    @agent.tool_plain
    def publish() -> str:
        return "published"

    policy = ApprovalPolicy([ApprovalRule("publish", approvers={"admin"})])
    runner = AgentRunner(agent, RunState(flush_interval=0), approval_policy=policy)

    def decide(request):
        with pytest.raises(PermissionError, match="mallory may not decide"):
            runner.state.resolve_approval(request.tool_call_id, False, user="mallory", reason="nope")
        assert request.status == "pending"
        runner.state.resolve_approval(request.tool_call_id, True, user="admin")

    assert await pause_then(runner, decide) == "done"


async def test_the_runner_acts_on_the_decision_the_state_kept():
    executed = []
    agent = Agent(streaming(calling_once("run_sql", {"sql": "SELECT 1"})))

    @agent.tool_plain
    def run_sql(sql: str) -> str:
        executed.append(sql)
        return "ok"

    runner = AgentRunner(agent, RunState(flush_interval=0), approval_policy=ApprovalPolicy(["run_sql"]))
    state = runner.state

    def on_event(event):
        if isinstance(event, ev.ApprovalRequested):
            state.resolve_approval(event.tool_call_id, False, user="bob", reason="no")
            # A second decision, e.g. from a stale card or a host button, before the runner waits.
            state.apply(ev.ApprovalResolved(tool_call_id=event.tool_call_id, approved=True, user="alice", run_id=runner.run_id))

    state.subscribe(on_event)
    await runner.run("go")
    request = state.approvals["call-0"]
    assert (request.status, request.user, executed) == ("denied", "bob", [])
