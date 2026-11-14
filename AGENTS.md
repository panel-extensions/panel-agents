# panel-agents

Panel components for building agent user interfaces: tool-call cards, versioned artifacts, human-in-the-loop approvals, traces and session replay.

## Architecture

- `panel_agents/events.py` defines normalized run events whose taxonomy mirrors the AG-UI protocol. Kit-specific events (`ArtifactPublished`, `ApprovalRequested/Resolved`, `Usage`, `GuardrailFired`) map to AG-UI `CUSTOM` events.
- `panel_agents/state.py` holds `RunState`, a reactive store built purely by applying events. Components bind to `RunState` and its entities (`MessageState`, `ToolCallState`, `ApprovalRequest`, `RunInfo`) and never import agent frameworks.
- `panel_agents/adapters/` translates framework output (pydantic-ai, AG-UI) into events. Adapters are the only modules allowed to import a framework, and must import it lazily or from an optional extra.
- `panel_agents/recording.py` writes and reads JSON Lines recordings. Replaying a recording through `RunState.from_events` must reproduce the live state, so components must not keep state that cannot be derived from events.
- `panel_agents/components/` contains UI components built on panel-material-ui. `components/renderers.py` is the shared registry mapping values to Panel objects.

## Conventions

- Use panel-material-ui widgets and layouts (`import panel_material_ui as pmui`) over Panel's built-in equivalents, except where pmui has no equivalent (Tabulator, CodeEditor, JSON pane).
- Components subclass `panel.viewable.Viewer` or `panel.custom.PyComponent` and expose their configuration as `param` parameters.
- Only add short comments explaining why, never what. No comment is the default.
- Tests live in `tests/`, run with `pixi run test` or `PYTHONPATH=src pytest tests` in a Panel dev environment. UI tests use Playwright and are marked `@pytest.mark.ui`.
- Attribution must use the `Assisted-by: AGENT_NAME:MODEL_VERSION` commit trailer.
