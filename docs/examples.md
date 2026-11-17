# Examples

The [`examples/`](https://github.com/panel-extensions/panel-agents/tree/main/examples) directory of the repository has runnable apps. Install the extras they need and serve them from a checkout:

```bash
pip install "panel-agents[pydantic-ai,examples]"
panel serve examples/drawer_app.py --show
```

The apps that call a model default to Claude Sonnet and read the model from the `PANEL_AGENTS_MODEL` environment variable, which takes any pydantic-ai model string such as `openai:gpt-5`.

## Sales report with a docked agent

[`examples/drawer_app.py`](https://github.com/panel-extensions/panel-agents/blob/main/examples/drawer_app.py) is a small sales report with an agent in an `AgentDrawer`. The report's settings are a [document artifact](how-to/document-artifacts.md) the agent edits in place, with a custom semantic diff, so every change can be compared, applied and reverted. Edits the user makes in the report become versions too. Filtering to one region is a [front-end action](how-to/ui-actions.md), and exporting needs [approval](how-to/approvals.md).

## SQL analyst

[`examples/sql_analyst/`](https://github.com/panel-extensions/panel-agents/tree/main/examples/sql_analyst) is a data-analysis agent over a DuckDB database, and the most complete example:

- `app.py` is the analyst: an `AgentChat` beside an `ArtifactPane`, queries published as table artifacts, statements that change data behind an `ApprovalPolicy`, an `AuditLog` of decisions, and every session recorded to `runs/`.
- `replay.py` replays a recording with the same components, `panel serve examples/sql_analyst/replay.py --args runs/session-....jsonl`. Without an argument it replays a bundled recording, so it runs without a model.
- `mcp_server.py` serves the same tools over MCP, see [Serve an agent's tools over MCP](how-to/mcp.md). `python examples/sql_analyst/mcp_server.py` uses stdio and `--http` serves at `http://localhost:8000/mcp`.

## Components without a model

These run on synthetic events, so they need no API key:

- [`examples/approval_demo.py`](https://github.com/panel-extensions/panel-agents/blob/main/examples/approval_demo.py): approval cards for structured tool arguments, with an audit trail.
- [`examples/artifacts_demo.py`](https://github.com/panel-extensions/panel-agents/blob/main/examples/artifacts_demo.py): versioned artifacts with diffs, and a document artifact whose versions can be applied to the app.
- [`examples/chat_replay.py`](https://github.com/panel-extensions/panel-agents/blob/main/examples/chat_replay.py): a fake SQL analyst run in `AgentChat`, with thinking, a failing and a corrected query, a published artifact and a streamed answer, replayed with delays.
