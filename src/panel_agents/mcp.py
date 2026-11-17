"""
Serve an agent's tools over the Model Context Protocol.

``create_server`` turns a pydantic-ai agent, toolset or an
``ActionRegistry`` into a FastMCP server whose tools have the same names,
descriptions and argument schemas the agent sees. Each call runs through
pydantic-ai, so arguments are validated and the approval policy applies;
approvals are requested from the MCP client through elicitation, and
rules that restrict approvers need an authenticated caller.

Requires the ``mcp`` extra: ``pip install panel-agents[mcp]``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token, get_context
from fastmcp.tools import InputRequiredToolResult, Tool, ToolResult
from mcp_types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult
from pydantic import PrivateAttr
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from pydantic_ai.toolsets import AbstractToolset
from pydantic_core import to_jsonable_python

from .actions import ActionRegistry
from .policy import ApprovalPolicy
from .state import ApprovalRequest

__all__ = ["create_server"]

_CALL_ID = "mcp-call"


_APPROVAL = "panel_agents.approval"

_APPROVAL_TTL = 600.0

_SECRET_ENV = "PANEL_AGENTS_APPROVAL_SECRET"

_log = logging.getLogger(__name__)

_APPROVAL_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "boolean", "title": "Approve"}},
    "required": ["value"],
}


class _ApprovalNeeded(Exception):

    def __init__(self, message: str, state: str):
        super().__init__(message)
        self.message = message
        self.state = state


def _token_user(context: Context | None) -> str | None:
    """
    The authenticated MCP caller: the access token's subject. The client id
    names an application, not a user, so it is never used as one.
    """
    try:
        token = get_access_token()
    except Exception:
        return None
    if token is None:
        return None
    return (getattr(token, "claims", None) or {}).get("sub")


@dataclass
class _McpCapability(AbstractCapability[Any]):
    """Applies the approval policy and asks the MCP client to decide approvals."""

    bridge: _Bridge | None = None
    validated: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
        policy = self.bridge.policy

        def needs_approval(ctx: RunContext[Any], tool_def: ToolDefinition, args: dict[str, Any]) -> bool:
            validated = to_jsonable_python(args, fallback=str)
            if ctx.tool_call_id:
                self.validated[ctx.tool_call_id] = validated
            return policy is not None and policy.requires_approval(tool_def.name, validated)

        return toolset.approval_required(needs_approval)

    async def handle_deferred_tool_calls(self, ctx: RunContext[Any], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        if not requests.approvals:
            return None
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for call in requests.approvals:
            denial = self._check_approver(call.tool_call_id, call.tool_name, self.validated.get(call.tool_call_id, call.args_as_dict()))
            approvals[call.tool_call_id] = denial or await self.bridge.ask_client(call.tool_name, call.args_as_dict())
        return DeferredToolResults(approvals=approvals)

    def _check_approver(self, tool_call_id: str, tool_name: str, args: dict[str, Any]) -> ToolDenied | None:
        policy = self.bridge.policy
        rule = policy.rule_for(tool_name, args) if policy is not None else None
        if rule is None or rule.approvers is None:
            return None
        user = self.bridge.resolve_user()
        if user is None:
            return ToolDenied(f"{tool_name} must be approved by specific users, and this MCP connection is not authenticated.")
        request = ApprovalRequest(tool_call_id=tool_call_id, tool_name=tool_name, args=args)
        if not rule.permits(request, user):
            return ToolDenied(f"{user} may not approve this {tool_name} call.")
        return None


def _decision(approved: bool | None, action: str, tool_name: str) -> ToolApproved | ToolDenied:
    if action == "accept" and approved:
        return ToolApproved()
    return ToolDenied(f"The user {'cancelled' if action == 'cancel' else 'declined'} this tool call.")


class _AgentTool(Tool):
    """An MCP tool that calls a pydantic-ai tool through a run of the agent."""

    _server: _Bridge = PrivateAttr()

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return await self._server.call(self.name, arguments)
        except _ApprovalNeeded as e:
            request = ElicitRequest(params=ElicitRequestFormParams(message=e.message, requested_schema=_APPROVAL_SCHEMA))
            return InputRequiredToolResult(InputRequiredResult(input_requests={_APPROVAL: request}, request_state=e.state))


class _Bridge:

    def __init__(
        self,
        agent: Agent,
        deps: Any,
        deps_factory: Callable[[Context | None], Any] | None,
        toolsets: Sequence[AbstractToolset[Any]],
        policy: ApprovalPolicy | None,
        user: Callable[[Context | None], str | None],
        secret: bytes | None = None,
        approval_ttl: float = _APPROVAL_TTL,
    ):
        self.agent = agent
        self.deps = deps
        self.deps_factory = deps_factory
        self.toolsets = list(toolsets)
        self.policy = policy
        self.user = user
        if secret is None and (env := os.environ.get(_SECRET_ENV)):
            secret = env.encode()
        self._secret = secret or secrets.token_bytes(32)
        # Identifies the key without revealing it, so a state signed with another key can be told apart from tampering.
        self._key_id = hashlib.sha256(b"panel-agents-key-id" + self._secret).hexdigest()[:12]
        self._ttl = approval_ttl
        self._used: dict[str, float] = {}

    @staticmethod
    def _context() -> Context | None:
        try:
            return get_context()
        except Exception:
            return None

    def resolve_user(self) -> str | None:
        try:
            return self.user(self._context())
        except Exception:
            return None

    def _mac(self, tool_name: str, args: dict[str, Any], nonce: str, issued: str, user: str | None) -> str:
        payload = json.dumps(
            [tool_name, to_jsonable_python(args, fallback=str), nonce, issued, user], sort_keys=True, separators=(",", ":"),
        )
        return hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()

    def sign(self, tool_name: str, args: dict[str, Any]) -> str:
        nonce, issued = secrets.token_hex(16), str(int(time.time()))
        return f"{self._key_id}.{nonce}.{issued}.{self._mac(tool_name, args, nonce, issued, self.resolve_user())}"

    def verify(self, state: str | None, tool_name: str, args: dict[str, Any]) -> bool:
        """
        Whether ``state`` was issued for this exact call, to this caller,
        within the approval TTL, and not used before.
        """
        now = time.time()
        for nonce, issued_at in list(self._used.items()):
            if now - issued_at > self._ttl:
                del self._used[nonce]
        key_id, nonce, issued, mac = ((state or "").split(".") + ["", "", "", ""])[:4]
        if key_id and key_id != self._key_id:
            _log.warning(
                "An MCP approval was signed with a different approval_secret than this process uses, so it is "
                "refused and the client is asked again. In a deployment with several worker processes, give "
                "every worker the same key with create_server(approval_secret=...) or the %s environment variable.",
                _SECRET_ENV,
            )
            return False
        if not nonce or not issued.isdigit() or nonce in self._used or now - int(issued) > self._ttl:
            return False
        if not hmac.compare_digest(mac, self._mac(tool_name, args, nonce, issued, self.resolve_user())):
            return False
        self._used[nonce] = int(issued)
        return True

    async def ask_client(self, tool_name: str, args: dict[str, Any]) -> ToolApproved | ToolDenied:
        message = f"Allow {tool_name} with {to_jsonable_python(args, fallback=str)}?"
        context = self._context()
        if context is None:
            return ToolDenied(f"{tool_name} needs approval, which this MCP client could not give.")
        # On the 2026-07-28 protocol the client answers by retrying the call with input responses. They count
        # only together with the request state issued for this tool and these arguments, and only once.
        responses = context.input_responses or {}
        if (response := responses.get(_APPROVAL)) is not None and self.verify(context.request_state, tool_name, args):
            return _decision((response.content or {}).get("value"), response.action, tool_name)
        if context._is_modern_protocol():
            raise _ApprovalNeeded(message, self.sign(tool_name, args))
        try:
            result = await context.elicit(message, response_type=bool)
        except Exception as e:
            return ToolDenied(f"{tool_name} needs approval, which this MCP client could not give: {e}")
        return _decision(getattr(result, "data", None), result.action, tool_name)

    def _deps(self) -> Any:
        if self.deps_factory is not None:
            return self.deps_factory(self._context())
        return self.deps

    def _run(self, model: FunctionModel, deps: Any):
        return self.agent.iter(
            "", model=model, deps=deps, toolsets=self.toolsets or None,
            capabilities=[_McpCapability(bridge=self)],
        )

    async def tool_defs(self) -> list[ToolDefinition]:
        found: list[list[ToolDefinition]] = []

        def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            found.append(list(info.function_tools))
            return ModelResponse(parts=[TextPart("")])

        deps = self.deps
        if self.deps_factory is not None:
            try:
                deps = self.deps_factory(None)
            except Exception:
                deps = None
        async with self._run(FunctionModel(record), deps) as run:
            async for _node in run:
                if found:
                    break
        return found[0] if found else []

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(name, dict(arguments), tool_call_id=_CALL_ID)])

        async with self._run(FunctionModel(call_tool), self._deps()) as run:
            async for node in run:
                request = getattr(node, "request", None)
                if not isinstance(request, ModelRequest):
                    continue
                for part in request.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_call_id == _CALL_ID:
                        return _result(part)
                    if isinstance(part, RetryPromptPart) and part.tool_call_id == _CALL_ID:
                        raise ToolError(part.model_response())
        raise ToolError(f"{name} did not return a result.")


def _result(part: ToolReturnPart) -> ToolResult:
    if part.outcome != "success":
        raise ToolError(part.model_response_str(wrap_if_error=False))
    content = to_jsonable_python(part.content, fallback=str)
    if isinstance(content, str):
        return ToolResult(content=content)
    structured = content if isinstance(content, dict) else {"result": content}
    return ToolResult(content=part.model_response_str(), structured_content=structured)


async def create_server(
    source: Agent[Any, Any] | AbstractToolset[Any] | ActionRegistry | Any,
    *,
    deps: Any = None,
    deps_factory: Callable[[Context | None], Any] | None = None,
    name: str | None = None,
    instructions: str | None = None,
    approval_policy: ApprovalPolicy | None = None,
    user: Callable[[Context | None], str | None] | None = None,
    approval_secret: bytes | None = None,
) -> FastMCP:
    """
    Create a FastMCP server exposing the tools of ``source``.

    The tools are listed when the server is created. Run the server with
    ``server.run()`` (stdio) or ``server.run(transport="http")``, or
    mount ``server.http_app()`` in an ASGI app, behind authentication if
    it is reachable by anyone but the user.

    Parameters
    ----------
    source
        An ``AgentRunner`` (its agent, deps and approval policy), a
        pydantic-ai ``Agent`` (all of its tools), a toolset, or an
        ``ActionRegistry`` (its actions, run in the registry's session
        until that session ends).
    deps
        Dependencies passed to tools as ``ctx.deps``, as ``AgentRunner(deps=...)`` does.
    deps_factory
        Callable building the dependencies for each request from the
        FastMCP ``Context``, e.g. to act as the authenticated caller.
        Called with None while the tools are listed. Overrides ``deps``.
    name, instructions
        Server name and instructions shown to MCP clients. The name
        defaults to the agent's name.
    approval_policy
        Rules deciding which calls need approval, which is requested
        from the client through MCP elicitation. Tools that ask for
        approval themselves are covered too. A declined or impossible
        approval returns an error to the client.
    user
        Callable resolving the caller's user name from the FastMCP
        ``Context``, checked against a rule's ``approvers``. Defaults to
        the subject of the request's access token, so a server without
        authentication denies calls whose rule restricts approvers.
    approval_secret
        Key signing the request state that binds an approval to its call,
        caller and time. Defaults to the ``PANEL_AGENTS_APPROVAL_SECRET``
        environment variable, else a random key per process; give every
        worker of a server running several processes the same key. A
        state signed with another key is refused with a logged warning.
        Approvals expire after 10 minutes.
    """
    from .adapters.pydantic_ai import ActionToolset, AgentRunner

    toolsets: list[AbstractToolset[Any]] = []
    if isinstance(source, AgentRunner):
        if deps is None and deps_factory is None:
            deps = source.deps
        if approval_policy is None:
            approval_policy = source.approval_policy
        source = source.agent
    if isinstance(source, Agent):
        agent = source
    else:
        if isinstance(source, ActionRegistry):
            source = ActionToolset(source)
        agent = Agent()
        toolsets.append(source)
    if approval_policy is None:
        warnings.warn(
            "create_server got no approval_policy, so only tools that ask for approval themselves are confirmed with "
            "the MCP client. Pass the AgentRunner, or its approval_policy, to apply the same rules as in the app.",
            UserWarning, stacklevel=2,
        )
    bridge = _Bridge(agent, deps, deps_factory, toolsets, approval_policy, user or _token_user, secret=approval_secret)
    server = FastMCP(name=name or agent.name or "panel-agents", instructions=instructions)
    for tool_def in await bridge.tool_defs():
        tool = _AgentTool(name=tool_def.name, description=tool_def.description, parameters=tool_def.parameters_json_schema)
        tool._server = bridge
        server.add_tool(tool)
    return server
