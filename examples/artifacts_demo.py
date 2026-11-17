"""
Versioned artifacts with diffs, and a document artifact whose versions
can be applied to the app.

    panel serve examples/artifacts_demo.py
"""

import time

import pandas as pd
import panel as pn
import panel_material_ui as pmui

from panel_agents import ArtifactPane, RunState
from panel_agents import events as ev
from panel_agents.artifacts import diff_structured, register_diff

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

CHART_VERSIONS = [
    {"title": "Revenue", "chart": {"type": "bar", "x": "region", "y": "revenue"}, "filters": []},
    {"title": "Revenue by region", "chart": {"type": "bar", "x": "region", "y": "revenue"}, "filters": [{"column": "year", "value": 2026}]},
    {"title": "Revenue by region", "chart": {"type": "line", "x": "region", "y": "revenue", "color": "teal"}, "filters": [{"column": "year", "value": 2026}]},
]


@register_diff("chart-spec")
def diff_chart(old, new, **options):
    """Label changes the way a person would describe them."""
    result = diff_structured(old, new)
    for change in result.changes:
        if change.path == ("title",):
            change.label = f"Renamed to {change.new!r}"
        elif change.path[0] == "filters":
            change.label = f"Filter {'added' if change.op == 'add' else 'removed' if change.op == 'remove' else 'changed'}"
    return result


# The tool that produced these versions edited the chart in place, so each
# version is published as applied.
for i, spec in enumerate(CHART_VERSIONS):
    events.append(ev.ArtifactPublished(
        artifact_id="chart", value=spec, kind="chart-spec", title="Chart",
        tool_call_id=f"call_edit_{i + 1}", applied=True, timestamp=now + 400 + 60 * i,
    ))

state = RunState.from_events(events)
live = pn.pane.JSON(CHART_VERSIONS[-1], depth=3, sizing_mode="stretch_width")


def apply_chart(artifact, version):
    live.object = version.value


pane = ArtifactPane(state=state, on_apply=apply_chart)

pmui.Page(
    title="Artifacts",
    main=[pmui.Row(
        pmui.Column(pane, sizing_mode="stretch_both"),
        pmui.Paper(pmui.Typography("Live chart spec", variant="h6"), live, width=320, margin=10, sx={"padding": "12px"}),
        sizing_mode="stretch_both",
    )],
).servable()
