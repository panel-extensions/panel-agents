"""
Replay of a fake SQL analyst run in AgentChat.

    panel serve examples/chat_replay.py

The run is built from synthetic events: thinking, a failing query, a
corrected query returning a DataFrame, a published artifact and a
streamed answer. "Replay" streams the same events again with delays.
"""

import asyncio
import json

import pandas as pd
import panel as pn

from panel_agents import AgentChat, RunState
from panel_agents import events as ev

RESULT = pd.DataFrame({
    "region": ["EMEA", "North America", "APAC", "LATAM"],
    "revenue_usd": [1_284_310.5, 2_045_872.0, 903_115.25, 312_400.75],
    "orders": [8_412, 12_930, 6_021, 2_377],
    "growth_pct": [12.4, 8.1, 21.7, -3.2],
})

SQL = """SELECT region,
       SUM(amount) AS revenue_usd,
       COUNT(*) AS orders,
       ROUND(100.0 * (SUM(amount) - SUM(prev_amount)) / SUM(prev_amount), 1) AS growth_pct
FROM sales
WHERE quarter = '2026-Q3'
GROUP BY region
ORDER BY revenue_usd DESC"""

ANSWER = """North America led Q3 with **$2.05M** in revenue across 12,930 orders, followed by EMEA at $1.28M.

- **APAC** is the fastest growing region at **+21.7%** quarter over quarter.
- **LATAM** is the only region that shrank (−3.2%), driven by fewer orders.

I saved the query and the result table as artifacts so you can rerun or export them."""


def chunks(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def analyst_events(start=1_762_970_000.0):
    clock = [start]

    def at(dt):
        clock[0] += dt
        return clock[0]

    run = "run-1"
    events = [
        ev.RunStarted(run_id=run, agent_name="SQL Analyst", timestamp=at(0)),
        ev.TextMessageStart(message_id="u1", role="user", run_id=run, timestamp=at(0)),
        ev.TextMessageContent(
            message_id="u1", run_id=run, timestamp=at(0),
            delta="Which regions drove revenue last quarter, and how fast are they growing?",
        ),
        ev.TextMessageEnd(message_id="u1", run_id=run, timestamp=at(0)),
        ev.ThinkingStart(message_id="th1", run_id=run, timestamp=at(0.3)),
    ]
    thinking = (
        "The user wants Q3 revenue by region with growth. I should inspect the schema first, "
        "then aggregate the sales table by region and compare against the previous quarter."
    )
    events += [ev.ThinkingContent(message_id="th1", delta=c, run_id=run, timestamp=at(0.05)) for c in chunks(thinking, 12)]
    events.append(ev.ThinkingEnd(message_id="th1", run_id=run, timestamp=at(0.1)))

    def tool_call(call_id, name, args, result=None, error=None, duration=0.4):
        args_text = json.dumps(args)
        out = [ev.ToolCallStart(tool_call_id=call_id, tool_name=name, run_id=run, timestamp=at(0.2))]
        out += [ev.ToolCallArgs(tool_call_id=call_id, delta=c, run_id=run, timestamp=at(0.03)) for c in chunks(args_text, 16)]
        out.append(ev.ToolCallEnd(tool_call_id=call_id, run_id=run, timestamp=at(0.05)))
        out.append(ev.ToolCallResult(
            tool_call_id=call_id, content=error if error else result, is_error=bool(error),
            run_id=run, timestamp=at(duration),
        ))
        return out

    events += tool_call("t1", "list_tables", {"schema": "analytics"}, result=["sales", "customers", "regions"], duration=0.12)
    events += tool_call("t2", "describe_table", {"table": "sales"}, result={
        "region": "VARCHAR", "amount": "DOUBLE", "prev_amount": "DOUBLE", "quarter": "VARCHAR", "order_id": "BIGINT",
    }, duration=0.09)
    events += tool_call(
        "t3", "run_query", {"sql": SQL.replace("quarter = '2026-Q3'", "qtr = '2026-Q3'")},
        error='Binder Error: Referenced column "qtr" not found in FROM clause!\nCandidate bindings: "sales.quarter"',
        duration=0.21,
    )
    events.append(ev.TextMessageStart(message_id="m1", run_id=run, timestamp=at(0.2)))
    events += [
        ev.TextMessageContent(message_id="m1", delta=c, run_id=run, timestamp=at(0.04))
        for c in chunks("The column is called `quarter`, not `qtr`. Retrying with the corrected filter.", 10)
    ]
    events.append(ev.TextMessageEnd(message_id="m1", run_id=run, timestamp=at(0.05)))
    events += tool_call("t4", "run_query", {"sql": SQL}, result=RESULT, duration=1.84)
    events += [
        ev.ArtifactPublished(artifact_id="q3_query", value=SQL, kind="sql", title="Q3 revenue query", tool_call_id="t4", run_id=run, timestamp=at(0.01)),
        ev.ArtifactPublished(artifact_id="q3_revenue", value=RESULT, kind="table", title="Q3 revenue by region", tool_call_id="t4", run_id=run, timestamp=at(0.01)),
        ev.TextMessageStart(message_id="m2", run_id=run, timestamp=at(0.3)),
    ]
    events += [ev.TextMessageContent(message_id="m2", delta=c, run_id=run, timestamp=at(0.03)) for c in chunks(ANSWER, 8)]
    events += [
        ev.TextMessageEnd(message_id="m2", run_id=run, timestamp=at(0.05)),
        ev.Usage(model="claude-sonnet", input_tokens=4_812, output_tokens=611, cost_usd=0.0236, run_id=run, timestamp=at(0)),
        ev.RunFinished(run_id=run, timestamp=at(0)),
    ]
    return events


EVENTS = analyst_events()

state = RunState.from_events(EVENTS)
chat = AgentChat(state, readonly=True)

active = pn.ui.Typography("No artifact selected", variant="body2", margin=(0, 10))


def show_active(*_):
    store = state.artifacts
    active.object = f"Active artifact: **{store[store.active].title}**" if store.active in store else "No artifact selected"


def watch_store(*_):
    state.artifacts.param.watch(show_active, "active")
    show_active()


state.param.watch(watch_store, "artifacts")
watch_store()


async def replay(_):
    replay_button.disabled = True
    state.reset()
    try:
        for event in EVENTS:
            state.apply(event)
            if isinstance(event, (ev.TextMessageContent, ev.ThinkingContent, ev.ToolCallArgs)):
                await asyncio.sleep(0.03)
            elif isinstance(event, ev.ToolCallResult):
                await asyncio.sleep(0.6)
            else:
                await asyncio.sleep(0.05)
        state.flush()
    finally:
        replay_button.disabled = False


replay_button = pn.ui.Button(label="Replay", icon="replay", variant="outlined", on_click=replay)

pn.ui.Page(
    title="SQL Analyst replay",
    header=[pn.ui.Row(active, replay_button, align="center")],
    main=[pn.ui.Container(chat, width_option="md", sizing_mode="stretch_both")],
).servable()
