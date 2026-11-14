"""
Human-in-the-loop approvals with an audit trail.

    panel serve examples/approval_demo.py
"""

import datetime as dt
import json
from typing import Literal

import panel as pn
import panel_material_ui as pmui
from pydantic import BaseModel, Field

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.audit import AuditLog
from panel_agents.components.approval import ApprovalQueue


class RunQuery(BaseModel):
    """Run a read-only SQL query against the warehouse."""

    sql: str = Field(description="The SQL statement to execute.")
    limit: int = Field(1000, ge=1, le=10_000, description="Maximum number of rows to return.")
    warehouse: Literal["analytics", "finance", "staging"] = "analytics"
    timeout_s: int | None = Field(None, ge=1, description="Abort the query after this many seconds.")


class SendEmail(BaseModel):
    """Send an email on behalf of the user."""

    to: list[str] = Field(description="Recipient addresses.")
    subject: str
    body: str
    cc: list[str] = []


def synthetic_events() -> list[ev.Event]:
    return [
        ev.RunStarted(run_id="run-1", agent_name="analyst"),
        ev.TextMessageStart(message_id="m1", role="user", run_id="run-1"),
        ev.TextMessageContent(message_id="m1", delta="Pull Q3 revenue by region and email it to the CFO.", run_id="run-1"),
        ev.TextMessageEnd(message_id="m1", run_id="run-1"),
        ev.ToolCallStart(tool_call_id="call_query", tool_name="run_query", run_id="run-1"),
        ev.ApprovalRequested(
            tool_call_id="call_query", tool_name="run_query", run_id="run-1",
            args={
                "sql": "SELECT region, SUM(revenue) AS revenue\nFROM sales\nWHERE quarter = 'Q3'\nGROUP BY region",
                "limit": 5000,
            },
            schema=RunQuery.model_json_schema(), description=RunQuery.__doc__,
        ),
        ev.ToolCallStart(tool_call_id="call_email", tool_name="send_email", run_id="run-1"),
        ev.ApprovalRequested(
            tool_call_id="call_email", tool_name="send_email", run_id="run-1",
            args={
                "to": ["cfo@example.com"],
                "subject": "Q3 revenue by region",
                "body": "Hi,\n\nAttached is the Q3 revenue breakdown by region.\n\nBest,\nAnalyst agent",
            },
            schema=SendEmail.model_json_schema(), description=SendEmail.__doc__,
        ),
    ]


state = RunState()
records = []
audit = AuditLog(state, records.append)

event_log = pmui.Column(sizing_mode="stretch_width", height=420, scroll=True, view_latest=True, auto_scroll_limit=2000, margin=0)
audit_view = pn.pane.JSON([], depth=3, sizing_mode="stretch_width")


def _event_row(event: ev.Event):
    data = {k: v for k, v in event.to_dict().items() if k not in ("type", "timestamp", "run_id") and v not in (None, {}, [])}
    when = dt.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
    detail = json.dumps(data, default=str)
    if len(detail) > 140:
        detail = detail[:139] + "…"
    color = {"APPROVAL_REQUESTED": "warning", "APPROVAL_RESOLVED": "success"}.get(event.type, "default")
    return pmui.Row(
        pmui.Typography(when, variant="caption", color="text.secondary", margin=(2, 6), width=60),
        pmui.Chip(label=event.type, size="small", color=color, variant="outlined", margin=(0, 6)),
        pmui.Typography(f"`{detail}`", variant="caption", margin=(2, 4), sizing_mode="stretch_width"),
        sizing_mode="stretch_width", margin=(2, 0),
    )


def on_event(event):
    event_log.append(_event_row(event))
    if isinstance(event, ev.ApprovalResolved):
        audit_view.object = list(records)


state.subscribe(on_event)


def reset(event=None):
    state.reset()
    records.clear()
    audit_view.object = []
    event_log.clear()
    state.apply_many(synthetic_events())


reset_button = pmui.Button(label="Reset demo", icon="restart_alt", variant="outlined", on_click=reset)
reason_mode = pmui.RadioButtonGroup(
    label="Reason required", options=["never", "deny", "edit", "always"], value="deny", size="small",
)

queue = ApprovalQueue(state, require_reason=reason_mode.value, show_resolved=True)
reason_mode.param.watch(lambda e: setattr(queue, "require_reason", e.new), "value")

reset()

pmui.Page(
    title="panel-agents · approvals",
    sidebar=[
        pmui.Typography(
            "An agent proposed two tool calls. Review them, edit the arguments if needed, "
            "then approve or deny. Every decision is emitted as an `APPROVAL_RESOLVED` event "
            "and written to the audit log.",
            variant="body2",
        ),
        reason_mode,
        reset_button,
    ],
    main=[
        pmui.Row(
            pmui.Column(queue, sizing_mode="stretch_width", min_width=420),
            pmui.Column(
                pmui.Typography("Events", variant="h6"),
                pmui.Paper(event_log, variant="outlined", sizing_mode="stretch_width", sx={"padding": "6px"}),
                pmui.Typography("Audit log", variant="h6", margin=(16, 5, 5, 5)),
                pmui.Paper(
                    pmui.Column(audit_view, height=300, scroll=True, sizing_mode="stretch_width", margin=0),
                    variant="outlined", sizing_mode="stretch_width", sx={"padding": "6px"},
                ),
                sizing_mode="stretch_width", min_width=380,
            ),
            sizing_mode="stretch_width",
        )
    ],
).servable()
