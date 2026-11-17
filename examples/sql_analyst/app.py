"""
Reference app: a data-analysis agent over a DuckDB database.

    panel serve examples/sql_analyst/app.py

The model defaults to Claude Sonnet and can be changed with the
PANEL_AGENTS_MODEL environment variable (any pydantic-ai model string,
e.g. "openai:gpt-5"). Every session is recorded to runs/ and can be
replayed with examples/sql_analyst/replay.py.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
from pydantic_ai import Agent, RunContext

from panel_agents import (
    AgentChat,
    ApprovalPolicy,
    ArtifactPane,
    AuditLog,
    Recorder,
    RunState,
    register_renderer,
)
from panel_agents.adapters.pydantic_ai import AgentRunner, artifact_return, publish_artifact

MODEL = os.environ.get("PANEL_AGENTS_MODEL", "anthropic:claude-sonnet-5-5")
RUNS = Path(__file__).parent / "runs"
PREVIEW_ROWS = 20
READ_ONLY = re.compile(r"^\s*(select|with|describe|show|summarize|pivot)\b", re.IGNORECASE)

INSTRUCTIONS = """\
You are a data analyst working with a DuckDB database of sales data.
Explore the schema before writing queries. Use run_query for read-only
SQL; it shows the result to the user as a versioned artifact. When you
revise a query, pass the same `name` so the user can compare versions.
Use plot to visualize a result you already queried. Statements that
modify the database (CREATE VIEW, INSERT, UPDATE, ...) must go through
execute_statement. Never ask for permission in the chat: call
execute_statement directly and the user approves, edits or rejects the
call in the interface before it runs. Keep answers concise and
refer to artifacts by name instead of repeating their contents.
"""


def create_database() -> duckdb.DuckDBPyConnection:
    rng = np.random.default_rng(42)
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    segments = ["Enterprise", "Mid-Market", "SMB"]
    products = pd.DataFrame({
        "product": ["Atlas", "Beacon", "Cirrus", "Delta", "Echo", "Forge"],
        "category": ["Analytics", "Analytics", "Storage", "Storage", "Security", "Security"],
        "list_price": [1200.0, 450.0, 800.0, 300.0, 950.0, 600.0],
    })
    n_customers = 400
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "name": [f"Customer {i:03d}" for i in range(1, n_customers + 1)],
        "region": rng.choice(regions, n_customers, p=[0.4, 0.3, 0.2, 0.1]),
        "segment": rng.choice(segments, n_customers, p=[0.2, 0.3, 0.5]),
        "signup_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_customers), unit="D"),
    })
    n_orders = 6000
    product_idx = rng.integers(0, len(products), n_orders)
    quantity = rng.integers(1, 20, n_orders)
    discount = rng.choice([0, 0.05, 0.1, 0.2], n_orders, p=[0.5, 0.25, 0.15, 0.1])
    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": rng.integers(1, n_customers + 1, n_orders),
        "order_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 640, n_orders), unit="D"),
        "product": products["product"].to_numpy()[product_idx],
        "quantity": quantity,
        "discount": discount,
        "revenue": (products["list_price"].to_numpy()[product_idx] * quantity * (1 - discount)).round(2),
    })
    con = duckdb.connect()
    for name, df in [("products", products), ("customers", customers), ("orders", orders)]:
        con.register(f"_{name}", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
        con.unregister(f"_{name}")
    return con


@dataclass
class AnalystDeps:
    con: duckdb.DuckDBPyConnection
    state: RunState


agent = Agent(MODEL, deps_type=AnalystDeps, instructions=INSTRUCTIONS, name="Analyst")


def cursor(ctx: RunContext[AnalystDeps]) -> duckdb.DuckDBPyConnection:
    # pydantic-ai runs tool calls concurrently and a DuckDB connection is not
    # thread-safe: sharing it makes parallel queries silently return no rows.
    return ctx.deps.con.cursor()


@agent.tool
def list_tables(ctx: RunContext[AnalystDeps]) -> list[dict]:
    """List the tables in the database with their row counts."""
    con = cursor(ctx)
    tables = con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()
    return [{"table": t, "rows": con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]} for (t,) in tables]


@agent.tool
def describe_table(ctx: RunContext[AnalystDeps], table: str) -> list[dict]:
    """Show the columns and types of a table, with sample values from its first three rows."""
    con = cursor(ctx)
    schema = con.execute(f"DESCRIBE {table}").fetchall()
    sample = con.execute(f"SELECT * FROM {table} LIMIT 3").df()
    return [
        {"column": name, "type": dtype, "sample": ", ".join(str(v) for v in sample[name])}
        for name, dtype, *_ in schema
    ]


@agent.tool
async def run_query(ctx: RunContext[AnalystDeps], sql: str, name: str, title: str):
    """
    Run a read-only SQL query and show the result to the user.

    Args:
        sql: A SELECT (or WITH) query.
        name: Short snake_case name for the result. Reuse it when revising a query.
        title: Human readable title of the result.
    """
    if not READ_ONLY.match(sql):
        raise ValueError("run_query only accepts read-only queries; use execute_statement instead.")
    df = cursor(ctx).execute(sql).df()
    await publish_artifact(ctx, f"{name}_sql", sql, kind="sql", title=f"{title} (SQL)")
    preview = df.head(PREVIEW_ROWS).to_markdown(index=False)
    summary = f"{len(df)} rows x {len(df.columns)} columns, published as '{name}'.\n\n{preview}"
    return artifact_return(summary, name, df, kind="table", title=title, metadata={"sql": sql})


@agent.tool
async def plot(
    ctx: RunContext[AnalystDeps], name: str, x: str, y: str, kind: str = "bar", by: str | None = None, title: str = "",
):
    """
    Plot a result previously returned by run_query.

    Args:
        name: Name of the query result to plot.
        x: Column for the x-axis.
        y: Column for the y-axis.
        kind: One of 'bar', 'line', 'scatter' or 'area'.
        by: Optional column to group into separate series.
        title: Title of the chart.
    """
    store = ctx.deps.state.artifacts
    if name not in store:
        raise ValueError(f"No result named {name!r}; available: {', '.join(a.artifact_id for a in store)}")
    data = store[name].latest.value
    spec = {"data": data, "x": x, "y": y, "kind": kind, "by": by, "title": title or name}
    await publish_artifact(ctx, f"{name}_chart", spec, kind="chart", title=title or f"{name} chart")
    return f"Published chart '{name}_chart'."


@agent.tool
def execute_statement(ctx: RunContext[AnalystDeps], sql: str, reason: str) -> str:
    """
    Run a statement that modifies the database, e.g. CREATE VIEW. Requires user approval.

    Args:
        sql: The statement to execute.
        reason: Why the statement is needed, shown to the user.
    """
    cursor(ctx).execute(sql)
    return "Statement executed."


POLICY = ApprovalPolicy(["execute_statement"])


@register_renderer(kind="chart")
def render_chart(spec, **options):
    import hvplot.pandas  # noqa: F401

    data = spec["data"]
    kwargs = {k: spec[k] for k in ("x", "y", "by", "title") if spec.get(k)}
    return data.hvplot(kind=spec.get("kind", "bar"), responsive=True, height=400, **kwargs)


def usage_text(state: RunState) -> str:
    runs = [r for r in state.runs.values() if r.parent_run_id is None]
    tokens = sum(r.input_tokens + r.output_tokens for r in runs)
    costs = [r.cost_usd for r in runs if r.cost_usd is not None]
    cost = f" · ${sum(costs):.4f}" if costs else ""
    return f"{len(runs)} runs · {tokens:,} tokens{cost}"


async def _ask(chat: AgentChat, question: str, event=None):
    await chat.submit(question)


def create_app():
    state = RunState()
    runner = AgentRunner(agent, state, deps=AnalystDeps(create_database(), state), approval_policy=POLICY)
    RUNS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    recorder = Recorder(state, RUNS / f"session-{stamp}.jsonl")
    audit = AuditLog(state, RUNS / "audit.jsonl")

    def cleanup(session_context):
        recorder.close()
        runner.close()

    pn.state.on_session_destroyed(cleanup)

    chat = AgentChat(runner=runner, assistant="Analyst", placeholder="Ask about customers, orders or products…")
    artifacts = ArtifactPane(state.artifacts)
    usage = pmui.Typography(usage_text(state), variant="body2", sx={"opacity": 0.8})
    state.param.watch(lambda *_: setattr(usage, "object", usage_text(state)), ["runs", "items"])

    page = pmui.Page(
        title="SQL Analyst",
        header=[pn.HSpacer(), usage],
        main=[pmui.Row(
            pmui.Column(chat, sizing_mode="stretch_both", min_width=420),
            pmui.Column(artifacts, sizing_mode="stretch_both", min_width=480),
            sizing_mode="stretch_both",
        )],
        sidebar=[
            pmui.Typography("Examples", variant="subtitle2"),
            *[
                pmui.Button(label=q, variant="text", sizing_mode="stretch_width",
                            on_click=partial(_ask, chat, q))
                for q in [
                    "What were monthly revenues by region in 2025?",
                    "Which product categories grow fastest? Plot it.",
                    "Create a view of enterprise customers with their lifetime revenue.",
                ]
            ],
        ],
        sidebar_open=False,
    )
    page.audit = audit
    return page


create_app().servable()
