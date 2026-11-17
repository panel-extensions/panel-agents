"""
An existing app with an agent docked beside it.

    panel serve examples/drawer_app.py

A small sales report whose filters and title the agent can change: the
report settings are a document artifact the agent edits in place, the
filters are front-end actions, and exporting needs approval. The model
defaults to Claude Sonnet and can be changed with the PANEL_AGENTS_MODEL
environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import panel as pn
import panel_material_ui as pmui
from pydantic_ai import Agent, RunContext

from panel_agents import ActionRegistry, AgentDrawer, ApprovalPolicy
from panel_agents.adapters.pydantic_ai import ActionToolset, AgentRunner, publish_artifact
from panel_agents.artifacts import diff_structured, register_diff

MODEL = os.environ.get("PANEL_AGENTS_MODEL", "anthropic:claude-sonnet-5-5")
REGIONS = ["North", "South", "East", "West"]

INSTRUCTIONS = """\
You help the user with the sales report they are looking at. Use
get_report to see its settings and update_report to change its title,
regions or grouping; the report updates as you edit it. Use the
show_region action to point the user at one region. Exporting needs
the user's approval, which the interface asks for.
"""


def sales_data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    months = pd.date_range("2026-01-01", periods=10, freq="MS")
    return pd.DataFrame([
        {"month": m, "region": r, "product": p, "revenue": float(rng.integers(20, 120) * 1000)}
        for m in months for r in REGIONS for p in ["Atlas", "Beacon"]
    ])


@dataclass
class Report:
    """The host app's document: what the report shows."""

    title: str = "Monthly revenue"
    regions: list[str] = field(default_factory=lambda: list(REGIONS))
    group_by: Literal["region", "product"] = "region"

    def to_dict(self) -> dict:
        return {"title": self.title, "regions": list(self.regions), "group_by": self.group_by}


@register_diff("report")
def diff_report(old, new, **options):
    result = diff_structured(old, new)
    for change in result.changes:
        if change.path == ("title",):
            change.label = f"Title {change.old!r} → {change.new!r}"
        elif change.path[0] == "regions":
            change.label = f"Region {change.new or change.old} {'shown' if change.op == 'add' else 'hidden'}"
        elif change.path == ("group_by",):
            change.label = f"Grouped by {change.new}"
    return result


class SalesReport(pn.viewable.Viewer):
    """The existing app the agent is added to."""

    def __init__(self):
        super().__init__()
        self.data = sales_data()
        self.report = Report()
        self._title = pmui.Typography(variant="h5", margin=(10, 10, 0, 10))
        self._regions = pmui.MultiChoice(label="Regions", options=REGIONS, value=list(REGIONS), width=320)
        self._group = pmui.RadioButtonGroup(options=["region", "product"], value="region", label="Group by")
        self._table = pn.widgets.Tabulator(pd.DataFrame(), sizing_mode="stretch_width", height=420, disabled=True, show_index=False)
        self.on_user_edit = None
        self._rendering = False
        self._regions.param.watch(lambda e: self._user_edit(regions=e.new), "value")
        self._group.param.watch(lambda e: self._user_edit(group_by=e.new), "value")
        self._layout = pmui.Column(
            self._title, pmui.Row(self._regions, self._group, margin=(0, 5)), self._table,
            sizing_mode="stretch_both",
        )
        self.render()

    def __panel__(self):
        return self._layout

    def update(self, **changes):
        for key, value in changes.items():
            setattr(self.report, key, value)
        self.render()

    def _user_edit(self, **changes):
        if self._rendering:
            return
        self.update(**changes)
        if self.on_user_edit is not None:
            self.on_user_edit(self.report.to_dict())

    def render(self):
        report = self.report
        self._title.object = report.title
        self._rendering = True
        try:
            self._regions.value = list(report.regions)
            self._group.value = report.group_by
        finally:
            self._rendering = False
        data = self.data[self.data.region.isin(report.regions)]
        table = data.pivot_table(index="month", columns=report.group_by, values="revenue", aggfunc="sum").reset_index()
        table["month"] = table["month"].dt.strftime("%b %Y")
        self._table.value = table


@dataclass
class Deps:
    app: SalesReport


agent = Agent(MODEL, deps_type=Deps, instructions=INSTRUCTIONS, name="Assistant",
    # Without a limit the provider reserves the model's maximum reply against the account's credit.
    model_settings={"max_tokens": 4096},
)


@agent.tool
def get_report(ctx: RunContext[Deps]) -> dict:
    """The current report settings."""
    return ctx.deps.app.report.to_dict()


@agent.tool
async def update_report(
    ctx: RunContext[Deps], title: str | None = None, regions: list[str] | None = None,
    group_by: Literal["region", "product"] | None = None,
) -> dict:
    """
    Change the report the user is looking at; arguments left out stay as they are.

    Args:
        title: New title.
        regions: Regions to include, out of North, South, East and West.
        group_by: Whether the columns are regions or products.
    """
    changes = {k: v for k, v in {"title": title, "regions": regions, "group_by": group_by}.items() if v is not None}
    if unknown := set(changes.get("regions", [])) - set(REGIONS):
        raise ValueError(f"Unknown regions {sorted(unknown)}; use {REGIONS}.")
    ctx.deps.app.update(**changes)
    report = ctx.deps.app.report.to_dict()
    await publish_artifact(ctx, "report", report, kind="report", title="Report settings", applied=True)
    return report


@agent.tool
def export_report(ctx: RunContext[Deps], filename: str) -> str:
    """
    Export the report as CSV.

    Args:
        filename: Name of the file to write.
    """
    return f"Exported {len(ctx.deps.app._table.value)} rows to {filename}."


def create_app(model=None):
    app = SalesReport()

    actions = ActionRegistry()
    runner = AgentRunner(
        agent, deps=Deps(app), approval_policy=ApprovalPolicy(["export_*"]),
        run_kwargs={"toolsets": [ActionToolset(actions)], **({"model": model} if model else {})},
    )

    def publish(report):
        # Edits made outside update_report are versions too, so Apply and Revert act on what is on screen.
        runner.state.publish_artifact("report", report, kind="report", title="Report settings", applied=True)

    app.on_user_edit = publish

    @actions.action
    def show_region(region: Literal["North", "South", "East", "West"]) -> str:
        """Show only one region in the report."""
        app.update(regions=[region])
        publish(app.report.to_dict())
        return f"Showing {region}."
    pn.state.on_session_destroyed(lambda ctx: runner.close())

    def apply_report(artifact, version):
        app.update(**version.value)

    drawer = AgentDrawer(runner, main=app, on_apply=apply_report, open=True)
    return pmui.Page(title="Sales report", header=[pn.HSpacer(), drawer.toggle], main=[drawer])


if pn.state.served:
    create_app().servable()
