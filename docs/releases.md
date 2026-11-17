# Release Notes

## 0.1.0b0

First beta of panel-agents: Panel components for building user interfaces for AI agents, connected to agent frameworks through adapters that emit a small set of normalized events modelled on the AG-UI protocol.

### Highlights

- **Chat and run state:** `AgentChat` with streaming messages, tool-call cards that fold finished steps into a summary row, readable step labels (`tool_labels`), and a reactive `RunState` that works live, from a recording, or with any framework that has an adapter.
- **pydantic-ai adapter:** `AgentRunner` runs a pydantic-ai agent against the UI, including deferred tools and approvals.
- **Approvals:** declarative `ApprovalPolicy`/`ApprovalRule` per tool (by name or glob, with optional predicates and approvers), inline approval cards with editable arguments, and an approval queue. Decisions are enforced on the validated arguments and recorded once.
- **Artifacts:** versioned artifacts with text, table and structured diffs, document artifacts that a tool edits in place with apply and revert, and a pluggable `register_diff`.
- **UI actions:** `ActionRegistry` lets an agent call functions that act on the user's own Panel session, exposed to pydantic-ai through `ActionToolset`.
- **AgentDrawer:** an agent chat that docks beside an existing app, with artifacts, a pending-approvals bar and an overlay mode for narrow windows.
- **MCP:** `panel_agents.mcp.create_server` (the `mcp` extra) serves an agent, toolset or action registry over MCP, with approvals bound to each call and to the caller.
- **Security:** model and tool output is rendered inert by default: no raw HTML, no auto-loading images, math disabled, and only http, https and mailto links, each showing its full URL. See [Security](how-to/security.md).
- **Recordings:** `Recorder` and `load_state` record and replay sessions; recordings carry a format version.

### Requirements

Python 3.11 or newer, Panel 1.9 or newer, panel-material-ui 0.16 or newer, and pydantic-ai-slim 2.60 or newer for the pydantic-ai adapter.

This is a beta: the API may still change before 0.1.0.
