# Release Notes

## 0.1.0

First release of panel-agents. Since 0.1.0b0 the components have gained model call traces, undo and redo of runs and artifact versions, a model picker, a document layout, a resizable drawer and mentions in the chat input.

### Added

- **Model calls and traces:** the chat shows each model call, `RunTrace` and `SpanView` show how a run was made (`RunState.span`), and model traces can be recorded, continued from and played back to an agent with a `Cassette`.
- **Undo and redo:** `RunState` can undo and redo single steps and whole runs (`redo_run`), and refuses to undo a run over someone else's edits. Artifact versions are shown as a history grouped by run, record who made them and can be persisted; `drop_artifact_versions` drops the values of a document's older versions.
- **Model picker:** people can pick the agent's model in the chat; a single model shows as a read-only label that keyboard and screen reader users can reach.
- **Document layout:** `AgentChat` can show a run's artifacts inline, as a document.
- **Mentions:** a host can mention items in the chat input (`Mention`). Each shows as a removable chip and is written into the message as `@name`.
- **Message placement:** `AgentChat` can place the user's messages and set parameters on every message (`user_placement`, `message_params`).
- **Approval labels:** approval cards and queues take configurable action labels.
- `SchemaForm` is public.
- `RunState(max_events=...)` caps the events a state keeps in memory.

### Changed

- **AgentDrawer is resizable by default:** a handle along the drawer's left edge resizes it by dragging, or with the arrow keys once it has focus, between `min_resize_width` and `max_resize_ratio` of the window (240 px and 60% by default). A resize sets `width`, so a host can store it. Pass `resizable=False` to keep a fixed width.
- Inside `AgentDrawer` the chat is drawn flat, and the drawer stays on screen when the window is narrower than it.
- Long chat feeds are cheaper to build, append to and publish into, and finished steps fold without blinking.
- Failed runs and provider refusals are shown and logged instead of dropped, and the chat input stays focused and typeable during a run.

### Requirements

Python 3.12 or newer, Panel 1.11 or newer, panel-material-ui 0.17 or newer, pydantic 2.12 or newer, and pydantic-ai-slim 2.60 or newer for the pydantic-ai adapter.

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
