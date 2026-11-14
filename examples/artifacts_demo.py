"""
Versioned artifacts with diffs.

    panel serve examples/artifacts_demo.py
"""

import time

import pandas as pd
import panel_material_ui as pmui

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.artifacts import ArtifactPane

SQL_VERSIONS = [
    """SELECT region, SUM(revenue) AS revenue
FROM sales
GROUP BY region""",
    """SELECT region, SUM(revenue) AS revenue
FROM sales
WHERE order_date >= DATE '2026-01-01'
GROUP BY region
ORDER BY revenue DESC""",
    """SELECT
    region,
    SUM(revenue) AS revenue,
    COUNT(DISTINCT customer_id) AS customers
FROM sales
WHERE order_date >= DATE '2026-01-01'
  AND status = 'complete'
GROUP BY region
ORDER BY revenue DESC
LIMIT 10""",
]

first = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "revenue": [120_500.0, 98_200.0, 143_900.0, 87_300.0],
}).set_index("region")

second = first.copy()
second.loc["East", "revenue"] = 151_250.0
second.loc["West", "revenue"] = 90_100.0
second = second.drop("South")
second.loc["Central"] = 64_800.0
second["customers"] = [412, 538, 301, 199]

now = time.time() - 600
events = [ev.RunStarted(run_id="demo", timestamp=now)]
for i, sql in enumerate(SQL_VERSIONS):
    events.append(ev.ArtifactPublished(
        artifact_id="query", value=sql, kind="sql", title="Revenue by region (SQL)",
        tool_call_id=f"call_sql_{i + 1}", metadata={"dialect": "duckdb"}, timestamp=now + 60 * i,
    ))
for i, df in enumerate([first, second]):
    events.append(ev.ArtifactPublished(
        artifact_id="result", value=df, kind="table", title="Revenue by region",
        tool_call_id=f"call_run_{i + 1}", metadata={"rows": len(df)}, timestamp=now + 200 + 60 * i,
    ))

state = RunState.from_events(events)
pane = ArtifactPane(state.artifacts)

pmui.Page(title="Artifacts", main=[pane]).servable()
