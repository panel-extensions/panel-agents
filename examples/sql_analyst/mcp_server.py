"""
Serve the SQL analyst's tools over MCP, so any MCP client (Claude Desktop,
an IDE, another agent) can query the same database with the same tools.

    python examples/sql_analyst/mcp_server.py           # stdio
    python examples/sql_analyst/mcp_server.py --http    # http://localhost:8000/mcp

execute_statement needs approval, which the server asks the client for.
"""

import asyncio
import sys

from app import POLICY, AnalystDeps, agent, create_database

from panel_agents import RunState
from panel_agents.mcp import create_server

server = asyncio.run(create_server(
    agent, deps=AnalystDeps(create_database(), RunState()), name="sql-analyst", approval_policy=POLICY,
))

if __name__ == "__main__":
    server.run(transport="http" if "--http" in sys.argv else "stdio")
