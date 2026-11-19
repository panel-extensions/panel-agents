# Let the agent act on the page

Agents usually work on data, but users often want them to act on the interface they are looking at: open a page, select an item, set a filter, show a result. In panel-agents these are actions. The host app registers what the user can do as typed functions on an `ActionRegistry`, and the same actions can feed a command palette, become agent tools and be served over MCP.

## Register actions

Create the registry inside the user's session, for example while building the page, and register plain functions on it:

```python
from typing import Annotated, Literal

from pydantic import Field

from panel_agents import ActionRegistry

actions = ActionRegistry()

@actions.action
def set_filter(column: str, value: Annotated[int, Field(ge=0)], mode: Literal["min", "max"] = "min") -> str:
    """
    Filter the table shown to the user.

    Args:
        column: Column to filter on.
        mode: Whether value is a minimum or a maximum.
    """
    table.add_filter(column, value, mode)
    return f"Filtered {column}."
```

The action's description is the docstring up to its first section, and the JSON schema of its arguments (`action.parameters`) comes from the type hints, pydantic `Field` constraints and the `Args:` section. `actions.action(name=..., description=...)` overrides the name and description, and `actions.add(func)` registers a function without the decorator. `Action(name, description, function)` builds an action directly, deriving its schema the same way. Actions can be registered and removed at any time, for example when the user opens a different view, and `actions.actions` is a parameter that UIs can watch.

## Invoke actions

`await actions.invoke("set_filter", {"column": "year", "value": 2026})` validates the arguments, runs the function and returns its result, awaiting it if it is a coroutine. The action always runs in the session that created the registry: `pn.state.curdoc` is that session's document, and a call from outside the session's event loop, for example from an MCP server thread, is handed to it through the document's next-tick callback, which holds the document lock. This also works when the registry was built on a worker thread, as with `pn.config.nthreads`. A sync action therefore blocks its session while it runs, so keep slow work in async actions. Once the session is destroyed, `invoke` raises instead of acting on a page nobody sees. This is what lets a command palette, an agent tool and an MCP request share the same actions. To build a palette, iterate over the registry for each action's `name`, `description` and `parameters`.

## Give them to an agent

`ActionToolset` turns a registry into a pydantic-ai toolset. Each action becomes a tool with the same name, description and schema:

```python
from panel_agents.adapters.pydantic_ai import ActionToolset, AgentRunner

runner = AgentRunner(agent, run_kwargs={"toolsets": [ActionToolset(actions)]})
```

The agent sees actions registered later on its next model request. When an action raises, the model receives the error as a failed tool call and can try something else, so a stale selection or a missing page does not end the run. Invalid arguments are sent back to the model to retry, as with any pydantic-ai tool. Actions are covered by the runner's `approval_policy` like other tools, so an action such as `delete_selection` can still require approval.

`examples/sql_analyst/app.py` registers a `show_artifact` action that brings a result into view in the artifact pane.

## Offer actions on artifacts

In the document layout of `AgentChat`, the artifacts a run published render inline as cards, and each card can offer actions from a registry. An `ArtifactAction` names the action, its button label and icon, how to get its arguments from the artifact, and optionally the label shown once there is nothing left to do:

```python
from panel_agents import AgentChat, ArtifactAction

add = ArtifactAction(
    "add_to_dashboard", "Add to dashboard", icon="add",
    arguments=lambda artifact: {"name": artifact.title} if artifact.kind == "table" else None,
    done=lambda artifact: "On dashboard" if dashboard.shows(artifact.title) else None,
)
chat = AgentChat(runner=runner, layout="document", action_registry=actions, artifact_actions=[add])
```

An action whose `arguments` returns None is not offered on that artifact. Clicking runs it through `actions.invoke` with the arguments `arguments` gives at that moment, so the button and the agent tool are the same action, and a failure is shown on the card. Call `chat.refresh_actions()` when the host's state changes what is done, for example after the user edited the dashboard by hand.
