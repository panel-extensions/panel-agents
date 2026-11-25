"""
Replay a recorded SQL Analyst session with the same components.

    panel serve examples/sql_analyst/replay.py --args runs/session-....jsonl

Without an argument the bundled test fixture is replayed.
"""

import sys
from pathlib import Path

import panel as pn

from panel_agents import AgentChat, ArtifactPane, load_state

DEFAULT = Path(__file__).parents[2] / "tests" / "fixtures" / "sql_agent_run.jsonl"

path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
state = load_state(path)

pn.ui.Page(
    title=f"Replay: {path.name}",
    main=[pn.ui.Row(
        pn.ui.Column(AgentChat(state, readonly=True), sizing_mode="stretch_both", min_width=420),
        pn.ui.Column(ArtifactPane(state.artifacts), sizing_mode="stretch_both", min_width=480),
        sizing_mode="stretch_both",
    )],
).servable()
