# Dock an agent beside an existing app

`AgentDrawer` adds an agent to an app without giving up the app's layout. It is a collapsible drawer on the right holding an `AgentChat`, with views of the artifacts the agent publishes and the approvals it is waiting for. The drawer has no tab strip: the artifacts open from a button in its header, which counts them, and a bar above the chat says how many actions wait for the user and opens the approvals. Both views have a back button to the chat.

```python
import panel_material_ui as pmui

from panel_agents import AgentDrawer
from panel_agents.adapters.pydantic_ai import AgentRunner

runner = AgentRunner(agent, deps=deps, approval_policy=policy)
drawer = AgentDrawer(runner, main=app, title="Assistant")

pmui.Page(title="My app", header=[drawer.toggle], main=[drawer]).servable()
```

`main` is the existing content. In the default `persistent` variant the open drawer sits beside it and narrows it. With `variant="temporary"` the drawer overlays the content instead and closes when the user clicks outside it. Without `main`, the drawer renders on its own, for example as the last item of a row you already have.

`drawer.toggle` is a header button that opens and closes the drawer. Its badge counts pending approvals, the count is part of its accessible name ("Assistant, 2 pending approvals"), and `aria-expanded` tells screen readers whether the drawer is open. Opening the drawer with the toggle moves focus to the chat input, and closing it returns focus to the toggle. With `open_on_approval=True` (the default) a new approval opens a persistent drawer, so a paused run does not go unnoticed. A temporary drawer would cover the app in the middle of the user's work, so there a new approval only updates the badge. `open`, `width` and `tab` (`"chat"`, `"artifacts"` or `"approvals"`) are parameters, so the app can open the drawer on a given view, for example from a command palette.

## What the drawer passes on

- The runner's `state` is shown and its `approval_policy.can_approve` decides who may approve, in the chat and in the approvals view. Pass `can_approve` to override it.
- `on_apply` is the callback for document artifacts. The artifacts view offers to apply or revert their versions, see [document artifacts](document-artifacts.md).
- `overlay_below=1200` makes a persistent drawer overlay the app, like the temporary variant, while the window is narrower than 1200 px, so a small screen keeps its full width for the app. In that mode a new approval only updates the toggle's badge.
- Finished tool calls of a run fold into one "3 steps · 2.1s" row that expands; calls that failed, were denied, wait for approval or published an artifact always show. Turn it off with `chat_params={"group_tool_calls": False}`. `chat_params={"tool_labels": {...}}` names calls in the user's words, by tool name, with a string or a callable of the call's arguments, e.g. `{"get_report": "Read the report", "update_report": lambda args: "Change " + ", ".join(args)}`; imperative labels read right whether the call is running, done, failed or waiting for approval. The tool name stays in the card's body and its accessible name.
- `chat_params` are passed to the `AgentChat`, for example `{"placeholder": "Ask about this dashboard"}`.
- The New conversation button starts a new conversation. It clears the runner's history and the session's runs, artifacts and approvals, and is disabled while a run is in progress. When there are document artifacts, it asks first, because their versions could no longer be compared or reverted.

`drawer.chat`, `drawer.artifacts` and `drawer.approvals` give access to the `AgentChat`, `ArtifactPane` and `ApprovalQueue` inside.

## Use a page's contextbar instead

`pmui.Page` has its own right-hand contextbar. To use it, place the drawer's body there instead of the drawer itself:

```python
drawer = AgentDrawer(runner)
pmui.Page(main=[app], contextbar=[drawer.content], contextbar_width=420, contextbar_open=True)
```

`examples/drawer_app.py` docks an agent beside a small sales report. The agent edits the report settings as a document artifact, filters through a front-end action, and needs approval to export.
