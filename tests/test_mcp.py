import asyncio
import json
import os

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("fastmcp")

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError
from pydantic_ai import Agent, ApprovalRequired, RunContext
from pydantic_ai.toolsets import FunctionToolset

from panel_agents import ActionRegistry, ApprovalPolicy, ApprovalRule
from panel_agents.adapters.pydantic_ai import publish_artifact
from panel_agents.mcp import create_server

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def make_agent():
    agent = Agent(deps_type=dict, name="Reporter", output_type=list[str])

    @agent.tool
    def get_title(ctx: RunContext[dict]) -> str:
        """The report title."""
        return ctx.deps["title"]

    @agent.tool
    async def set_title(ctx: RunContext[dict], title: str) -> dict:
        """
        Rename the report.

        Args:
            title: The new title.
        """
        ctx.deps["title"] = title
        await publish_artifact(ctx, "report", dict(ctx.deps), kind="json", applied=True)
        return {"title": title}

    @agent.tool_plain
    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    @agent.tool
    def delete_report(ctx: RunContext[dict]) -> str:
        """Delete the report."""
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return "deleted"

    return agent


async def test_tools_have_the_agent_signatures():
    agent = make_agent()
    seen = {}

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    def model(messages, info):
        seen.update({t.name: (t.description, t.parameters_json_schema) for t in info.function_tools})
        return ModelResponse(parts=[TextPart('["x"]')])

    await agent.run("x", model=FunctionModel(model), deps={"title": "a"}, output_type=str)

    server = await create_server(agent, deps={"title": "Sales"})
    assert server.name == "Reporter"
    async with Client(server) as client:
        tools = {t.name: (t.description, t.inputSchema) for t in await client.list_tools()}
    assert tools == seen
    assert tools["set_title"][1]["properties"]["title"]["description"] == "The new title."


async def test_call_returns_text_and_structured_results():
    deps = {"title": "Sales"}
    server = await create_server(make_agent(), deps=deps)
    async with Client(server) as client:
        result = await client.call_tool("get_title", {})
        assert result.content[0].text == "Sales"
        result = await client.call_tool("set_title", {"title": "Revenue"})
        assert result.structured_content == {"title": "Revenue"}
        assert json.loads(result.content[0].text) == {"title": "Revenue"}
    assert deps["title"] == "Revenue"


async def test_validation_and_tool_errors_are_reported():
    server = await create_server(make_agent(), deps={})
    async with Client(server) as client:
        with pytest.raises(ToolError, match="b"):
            await client.call_tool("divide", {"a": 1})
        with pytest.raises(ToolError, match="division by zero"):
            await client.call_tool("divide", {"a": 1, "b": 0})


async def test_unknown_tool_is_rejected():
    server = await create_server(make_agent(), deps={})
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("missing", {})


def elicitation(action, asked):
    async def handler(message, response_type, params, context):
        asked.append(message)
        if action == "decline":
            return ElicitResult(action="decline")
        return response_type(value=action == "accept")
    return handler


@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize("action", ["accept", "reject", "decline"])
async def test_policy_approval_is_asked_from_the_client(action, mode):
    toolset = FunctionToolset()

    @toolset.tool_plain
    def publish(channel: str) -> str:
        """Publish the report."""
        return f"published to {channel}"

    asked = []
    server = await create_server(toolset, approval_policy=ApprovalPolicy(["publish"]))
    async with Client(server, elicitation_handler=elicitation(action, asked), mode=mode) as client:
        if action == "accept":
            result = await client.call_tool("publish", {"channel": "news"})
            assert result.content[0].text == "published to news"
        else:
            with pytest.raises(ToolError, match="declined"):
                await client.call_tool("publish", {"channel": "news"})
    assert len(asked) == 1 and "publish" in asked[0] and "news" in asked[0]


async def test_tool_requesting_approval_fails_without_elicitation_support():
    server = await create_server(make_agent(), deps={})
    async with Client(server) as client:
        # Depending on the protocol version the server or the client reports it.
        with pytest.raises(Exception, match="(?i)elicitation not supported|needs approval"):
            await client.call_tool("delete_report", {})


async def test_tool_requesting_approval_is_asked_from_the_client():
    asked = []
    server = await create_server(make_agent(), deps={})
    async with Client(server, elicitation_handler=elicitation("accept", asked)) as client:
        result = await client.call_tool("delete_report", {})
    assert result.content[0].text == "deleted"
    assert asked


async def test_action_registry_runs_actions_in_the_session():
    actions = ActionRegistry()
    loop = asyncio.get_running_loop()
    calls = []

    @actions.action
    def open_page(path: str) -> str:
        """Open a page of the app."""
        calls.append(asyncio.get_running_loop() is loop)
        return f"Opened {path}"

    server = await create_server(actions, name="app")
    async with Client(server) as client:
        tools = await client.list_tools()
        assert [(t.name, t.inputSchema) for t in tools] == [("open_page", actions["open_page"].parameters)]
        result = await client.call_tool("open_page", {"path": "/sales"})
    assert result.content[0].text == "Opened /sales"
    assert calls == [True]


def publish_toolset(calls):
    toolset = FunctionToolset()

    @toolset.tool_plain
    def drop_table(name: str) -> str:
        """Drop a table."""
        calls.append(name)
        return f"dropped {name}"

    return toolset


ADMIN_ONLY = ApprovalPolicy([ApprovalRule("drop_table", approvers={"admin"})])


async def test_restricted_approvers_need_an_authenticated_caller():
    calls, asked = [], []
    server = await create_server(publish_toolset(calls), approval_policy=ADMIN_ONLY)
    async with Client(server, elicitation_handler=elicitation("accept", asked)) as client:
        with pytest.raises(ToolError, match="not authenticated"):
            await client.call_tool("drop_table", {"name": "users"})
    assert calls == [] and asked == []


@pytest.mark.parametrize("caller", ["admin", "bob"])
async def test_restricted_approvers_are_checked_against_the_caller(caller):
    calls, asked = [], []
    server = await create_server(publish_toolset(calls), approval_policy=ADMIN_ONLY, user=lambda ctx: caller)
    async with Client(server, elicitation_handler=elicitation("accept", asked)) as client:
        if caller == "admin":
            assert (await client.call_tool("drop_table", {"name": "tmp"})).content[0].text == "dropped tmp"
        else:
            with pytest.raises(ToolError, match="bob may not approve"):
                await client.call_tool("drop_table", {"name": "tmp"})
    assert calls == (["tmp"] if caller == "admin" else [])


async def test_approvals_are_bound_to_the_call():
    from panel_agents.mcp import _Bridge

    bridge = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: None)
    state = bridge.sign("drop_table", {"name": "tmp"})
    assert not bridge.verify(None, "drop_table", {"name": "tmp"})
    assert not bridge.verify(state, "drop_table", {"name": "users"})
    assert not bridge.verify(state, "other", {"name": "tmp"})
    assert bridge.verify(state, "drop_table", {"name": "tmp"})
    assert not bridge.verify(state, "drop_table", {"name": "tmp"})


async def test_a_pre_filled_approval_is_not_honoured(monkeypatch):
    # A client sending an approval with its first call, before anything was asked.
    from types import SimpleNamespace

    import panel_agents.mcp as mcp

    calls = []
    fake = SimpleNamespace(
        input_responses={mcp._APPROVAL: SimpleNamespace(action="accept", content={"value": True})},
        request_state=None, _is_modern_protocol=lambda: True,
    )
    monkeypatch.setattr(mcp, "get_context", lambda: fake)
    server = await create_server(publish_toolset(calls), approval_policy=ApprovalPolicy(["drop_table"]))
    tool = await server.get_tool("drop_table")
    result = await tool.run({"name": "users"})
    assert calls == []
    assert result.input_required.request_state


async def test_deps_are_built_per_request():
    agent = Agent(deps_type=dict)

    @agent.tool
    def whoami(ctx: RunContext[dict]) -> str:
        """Who is calling."""
        return ctx.deps["user"]

    seen = []

    def deps_factory(context):
        seen.append(context is None)
        return {"user": "listing" if context is None else "caller"}

    server = await create_server(agent, deps_factory=deps_factory, approval_policy=ApprovalPolicy())
    async with Client(server) as client:
        assert (await client.call_tool("whoami", {})).content[0].text == "caller"
    assert seen == [True, False]


async def test_runner_source_keeps_its_policy_and_deps():
    from panel_agents.adapters.pydantic_ai import AgentRunner

    calls, asked = [], []
    runner = AgentRunner(Agent(toolsets=[publish_toolset(calls)]), deps={"x": 1}, approval_policy=ApprovalPolicy(["drop_table"]))
    server = await create_server(runner)
    async with Client(server, elicitation_handler=elicitation("decline", asked)) as client:
        with pytest.raises(ToolError, match="declined"):
            await client.call_tool("drop_table", {"name": "users"})
    assert calls == [] and asked


async def test_missing_policy_warns():
    with pytest.warns(UserWarning, match="no approval_policy"):
        await create_server(publish_toolset([]))


async def test_approvals_are_bound_to_the_caller_and_expire(monkeypatch):
    import time

    from panel_agents.mcp import _Bridge

    caller = {"user": "ann"}
    bridge = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: caller["user"], approval_ttl=60)
    state = bridge.sign("drop_table", {"name": "tmp"})
    caller["user"] = "bob"
    assert not bridge.verify(state, "drop_table", {"name": "tmp"})
    caller["user"] = "ann"
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 61)
    assert not bridge.verify(state, "drop_table", {"name": "tmp"})
    monkeypatch.setattr(time, "time", lambda: now)
    assert bridge.verify(state, "drop_table", {"name": "tmp"})


async def test_a_shared_secret_lets_workers_verify_each_others_states():
    from panel_agents.mcp import _Bridge

    first = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann", secret=b"k" * 32)
    second = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann", secret=b"k" * 32)
    other = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann")
    state = first.sign("drop_table", {"name": "tmp"})
    assert not other.verify(state, "drop_table", {"name": "tmp"})
    assert second.verify(state, "drop_table", {"name": "tmp"})


def test_client_ids_are_not_user_names(monkeypatch):
    from types import SimpleNamespace

    import panel_agents.mcp as mcp

    monkeypatch.setattr(mcp, "get_access_token", lambda: SimpleNamespace(claims={}, client_id="admin"))
    assert mcp._token_user(None) is None
    monkeypatch.setattr(mcp, "get_access_token", lambda: SimpleNamespace(claims={"sub": "ann"}, client_id="admin"))
    assert mcp._token_user(None) == "ann"


def test_a_state_from_another_key_is_refused_with_a_warning(caplog, monkeypatch):
    import logging

    from panel_agents.mcp import _Bridge

    first = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann")
    second = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann")
    state = first.sign("drop_table", {"name": "tmp"})
    with caplog.at_level(logging.WARNING, logger="panel_agents.mcp"):
        assert not second.verify(state, "drop_table", {"name": "tmp"})
    assert "approval_secret" in caplog.text and "PANEL_AGENTS_APPROVAL_SECRET" in caplog.text
    caplog.clear()
    tampered = state[:-1] + ("0" if state[-1] != "0" else "1")
    with caplog.at_level(logging.WARNING, logger="panel_agents.mcp"):
        assert not first.verify(tampered, "drop_table", {"name": "tmp"})
    assert "approval_secret" not in caplog.text


def test_the_secret_can_come_from_the_environment(monkeypatch):
    from panel_agents.mcp import _Bridge

    monkeypatch.setenv("PANEL_AGENTS_APPROVAL_SECRET", "shared-key")
    first = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann")
    second = _Bridge(None, None, None, [], ADMIN_ONLY, lambda ctx: "ann")
    assert second.verify(first.sign("drop_table", {"name": "tmp"}), "drop_table", {"name": "tmp"})
