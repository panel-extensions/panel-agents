# panel-agents

Panel components for building user interfaces for AI agents: tool-call cards, versioned artifacts, human-in-the-loop approvals, traces and session replay.

Agent frameworks are connected through adapters that translate their output into a small set of normalized events, modelled on the [AG-UI protocol](https://docs.ag-ui.com). The components bind to a reactive `RunState` built from those events, so the same UI works live, against a recorded session, or with any framework that has an adapter.

```python
from panel_agents import RunState, Recorder, load_state

state = RunState()
with Recorder(state, "runs/session.jsonl"):
    ...  # run an agent through an adapter, which applies events to `state`

replayed = load_state("runs/session.jsonl")
```

A complete chat UI for a pydantic-ai agent, with tool-call cards, inline approvals and a versioned artifact pane:

```python
import panel_material_ui as pmui
from panel_agents import AgentChat, ArtifactPane
from panel_agents.adapters.pydantic_ai import AgentRunner

runner = AgentRunner(agent, deps=deps)
pmui.Page(main=[pmui.Row(AgentChat(runner=runner), ArtifactPane(runner.state.artifacts))]).servable()
```

See `examples/sql_analyst/` for a reference data-analysis agent over DuckDB and a replay viewer for its recordings.

This project is in early development and its API will change.
