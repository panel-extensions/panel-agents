# Serve an agent's tools over MCP

The tools you give an agent are often useful outside the app too: in Claude Desktop, an IDE or another agent. `panel_agents.mcp.create_server` serves them over the [Model Context Protocol](https://modelcontextprotocol.io) with the same names, descriptions and argument schemas the agent sees, so MCP clients and the in-app agent share one surface.

It needs the `mcp` extra, which installs FastMCP, the MCP library that pydantic-ai's own MCP client uses:

```bash
pip install "panel-agents[mcp]"
```

## Create a server

`create_server` accepts an `AgentRunner` (its agent, deps and approval policy), a pydantic-ai `Agent` (all of its tools), any pydantic-ai toolset, or an `ActionRegistry`. It is a coroutine because it lists the tools by asking pydantic-ai for them:

```python
import asyncio

from panel_agents.mcp import create_server

server = asyncio.run(create_server(runner, name="sql-analyst"))
server.run()                   # stdio, for desktop clients
server.run(transport="http")   # streamable HTTP on port 8000
```

Passing the runner keeps the app's approval policy. With an agent or a toolset, pass `approval_policy=` yourself; without one, `create_server` warns, because then only tools that ask for approval themselves are confirmed. The tools are listed when the server is created, so tools or actions added later need a new server.

`deps` reaches the tools as `ctx.deps`, as with `AgentRunner(deps=...)`, and is shared by every client. When tools must act as the caller, pass `deps_factory`, a callable that builds the dependencies for each request from the FastMCP `Context`, for example from the authenticated user. It is called with None while the tools are listed.

The result is a regular `fastmcp.FastMCP` server, so you can add your own tools and resources to it or mount `server.http_app()` in an ASGI application. An HTTP server is reachable by anyone who can reach the port, so put it behind FastMCP's authentication (`FastMCP(auth=...)` providers) or your app's own, unless it only listens locally for a single user.

Each call runs through pydantic-ai rather than calling the function directly. Arguments are validated exactly as they are for the agent. A validation error, a `ModelRetry` or an exception in the tool comes back to the client as a tool error on the first attempt, since the client, not a model, decides whether to retry. Tools may use `ctx.deps` and the rest of the `RunContext`, but artifacts they publish with `publish_artifact` or `artifact_return` are not recorded anywhere, including document versions published with `applied=True`. A tool that edits a document the app shows should publish the new version to the app's `RunState` itself (`state.publish_artifact(...)`), or the app's version history will not show the change.

## Approvals

Calls that the approval policy matches, and calls to tools that raise `ApprovalRequired` themselves, are confirmed with the MCP client's user through an elicitation before they run. A declined or cancelled confirmation is reported as an error, and so is a client that cannot ask its user because it does not support elicitation. Editing the arguments, which `ApprovalCard` allows, is not available over MCP.

On the 2026-07-28 protocol, where the client answers by retrying the call, the server only accepts an answer together with the request state it issued for that tool, those exact arguments and that caller, within 10 minutes, and only once. An approval sent before anything was asked, carried over to different arguments or another caller, or replayed, is ignored and the client is asked again. The state is signed with the key from `create_server(approval_secret=...)`, or the `PANEL_AGENTS_APPROVAL_SECRET` environment variable, or else a random key per process. A server running several worker processes must give every worker the same key: otherwise an approval answered by another worker is refused, the client is asked again, and the server logs a warning that names `approval_secret`.

When the rule matching a call restricts `approvers`, the server needs to know who is calling. By default it takes the subject (`sub`) of the request's access token, never the OAuth client id, which names an application rather than a user, so the server must use FastMCP authentication; without it such calls are denied. Pass `user=`, a callable of the FastMCP `Context` returning the user name, to resolve the caller another way. A caller the rule does not allow is denied without being asked.

## Front-end actions

Serving an `ActionRegistry` exposes the actions of one user session. The calls run in that session, so an MCP client can drive the page the user is looking at:

```python
server = await create_server(actions, name="my-app", approval_policy=policy)
```

The actions only work while that session is open; afterwards each call returns an error. Anyone who can reach the server drives that user's page, so only serve it locally or behind authentication. `examples/sql_analyst/mcp_server.py` serves the SQL analyst's tools, including the approval policy on `execute_statement`.
