# Edit documents with versions you can apply and revert

Most artifacts are outputs: a query result, a chart, a report. A document artifact instead mirrors something the host app owns and the agent edits, such as a configuration, a dashboard or a form. Each edit the agent makes is published as a new version, the user can compare any two versions, and the artifact view offers to apply an earlier or later version back to the app.

## Publish versions as applied

A tool that edits the document in place publishes what the document now holds with `applied=True`:

```python
from panel_agents.adapters.pydantic_ai import publish_artifact

@agent.tool
async def set_title(ctx: RunContext[Deps], title: str) -> str:
    """Rename the dashboard."""
    ctx.deps.dashboard.title = title
    await publish_artifact(ctx, "dashboard", ctx.deps.dashboard.to_dict(), kind="dashboard", title="Dashboard", applied=True)
    return "Renamed."
```

`artifact_return(..., applied=True)` does the same from a sync tool. An artifact with an applied version is a document: `artifact.is_document` is True, `artifact.applied` is the number of the version the host holds (a read-only parameter you can watch), and `artifact.applied_version` is that `ArtifactVersion`. Version numbers count from 1 everywhere they are public, as in the `v1`, `v2` labels the user sees: `ArtifactVersion.version`, `Artifact.current`, `ArtifactView.base` and `target`, `apply_version(version)` and the `ARTIFACT_APPLIED` event all use them. A tool that only proposes a change publishes without `applied`, and the version waits for the user to apply it.

## Apply and revert from the artifact view

Give the `ArtifactPane` (or a single `ArtifactView`) the `RunState` and an `on_apply` callback that makes the host document match a version:

```python
from panel_agents import ArtifactPane

def apply_version(artifact, version):
    dashboard.load(version.value)

pane = ArtifactPane(state=runner.state, on_apply=apply_version)
```

When the user selects a version of a document, the header shows "Revert to vN" for versions before the applied one and "Apply vN" for versions after it. Clicking it calls `on_apply(artifact, version)` with the `Artifact` and the `ArtifactVersion` to apply, and the callback may be async. If the callback raises or returns False, the message is shown and nothing changes. Otherwise the pane records an `ARTIFACT_APPLIED` event through `RunState.apply_artifact_version`, so recordings and replays show which version was live. The button is disabled while the callback runs, so a double click cannot apply twice. `ArtifactView.apply_version(version)` does the same from code, taking a version number, and raises if there is no `on_apply` or no such version.

Applying does not create a version. The compare tab compares new versions of a document with the applied one, so after a revert the agent's next edit is shown against what the host actually holds. An `AgentRunner` also tells the model: the next prompt it sends starts with a short note such as "ann applied version 1 of the artifact 'Dashboard' (dashboard)", so the agent does not keep building on a version that is no longer live.

## Publish the host's own edits

When the user changes the document outside the agent, for example with the app's own widgets, publish the new state so the versions stay truthful:

```python
state.publish_artifact("dashboard", dashboard.to_dict(), applied=True)
```

`RunState.publish_artifact` records an `ARTIFACT_PUBLISHED` event like a tool would, and an `AgentRunner` mentions the new version to the model on its next prompt.

## Who made a version

Every `ArtifactVersion` records who made it: `run_id` is the run it was published in, `tool_call_id` the tool call if a tool published it, `actor` who made it (`"agent"` or `"user"`, or any string a host chooses, such as `"import"`) and `user` the person's name when a person made it. The `ARTIFACT_PUBLISHED` event carries the same fields, so a replayed recording shows the same authors.

Versions a tool publishes through `publish_artifact(ctx, ...)` or `artifact_return(...)` get the run, the tool call and the actor `"agent"`; the agent's name is on the run, `state.runs[version.run_id].agent_name`. `RunState.publish_artifact` fills them in from where it is called:

- from code an `AgentRunner` runs, such as a document method a tool calls, the current run and tool call with the actor `"agent"`;
- inside `RunState.user_run`, that run, the actor `"user"` and its user;
- anywhere else, such as a widget callback, the actor `"user"` and `pn.state.user`, with no run.

Each can be passed explicitly instead, e.g. `state.publish_artifact("dashboard", value, applied=True, actor="import")`. An actor other than the current one keeps the run but not its tool call or user, and the actor `"user"` always gets `pn.state.user` unless a user is given. A widget callback that fires while the agent is running is still the user's: the run is only visible to code the run itself calls. An `AgentRunner` tells the model about every version it did not make itself with its next prompt, including a person's edits made while it was running. Recordings made before versions recorded their actor load versions published by a tool call as the agent's.

Free-form `metadata` describes what a version did. A host editing a document one operation at a time can publish a version per operation:

```python
def place_tile(name, width):
    dashboard.place(name, width)
    state.publish_artifact(
        "dashboard", dashboard.to_dict(), applied=True,
        metadata={"op": "place_tile", "args": {"name": name, "width": width}},
    )
```

Called from an agent's tool, `place_tile` publishes an agent version stamped with that tool call; called from the app's own widgets, a user version.

## Group a person's edits into a run

Agent versions are grouped by the run that made them. To group a batch of a person's edits the same way, wrap them in `RunState.user_run`:

```python
with state.user_run("Arranged the tiles") as run_id:
    place_tile("revenue_chart", 50)
    place_tile("note", 50)
```

It emits `RUN_STARTED` with the user as `agent_name` and the given input, and `RUN_FINISHED` when the block ends. A block that raises is still recorded as a finished run, since the edits made before the error stand. A user run has no messages and no model calls. Versions published inside it default to the run, the actor `"user"` and the user, which defaults to `pn.state.user` and can be passed as `user=`. It is meant for a synchronous batch of edits: the state counts as `running` while the block is open, which disables the chat input, so do not await inside it. For edits spread over a longer time, pass the same `run_id` to each `publish_artifact` call instead. Called from code a run executes, such as a tool, `user_run` starts a child run of that run (`parent_run_id`) and the versions in it keep the enclosing attribution, so an agent's tool that groups its edits this way still publishes them as the agent's.

## Run and tool call ids

Inside a tool, pydantic-ai's `RunContext` has the ids as `ctx.run_id` and `ctx.tool_call_id`. Host code that a tool calls without passing its context reads them with `current_run_id()` and `current_tool_call_id()`:

```python
from panel_agents import current_run_id, current_tool_call_id

class Dashboard:
    def place(self, name, width):
        self.history.append({"op": "place_tile", "run_id": current_run_id(), "tool_call_id": current_tool_call_id()})
        ...
```

`AgentRunner` sets the run id while it drives the agent, including output functions, and the tool call id while each tool executes. The prompt's user message is recorded before the run id is set and the error recorded when a run is cancelled after it, so neither belongs to code that sees the id. The ids reach sync tools, which pydantic-ai runs through `anyio.to_thread.run_sync`, and actions an `ActionToolset` hands to the Panel session; do not rely on them in threads you start yourself. `RunState.user_run` sets the run id of its block. Outside them both return None.

They are backed by a context variable, so they follow awaited code and the tasks it starts, but not callbacks triggered independently, which keeps a person's concurrent edits from being attributed to the agent. Two consequences follow from that. Code that reacts to the run's own events, such as a `RunState.subscribe` callback or a param watcher that fires while the runner applies an event, runs inside the run and publishes as the agent. A task a tool starts keeps the tool's ids even after the run has ended. From outside the run, `AgentRunner.run_id` is the id of the current or last run, `AgentRunner.running` says whether it is in progress, and `RunState.runs` maps every run id to its `RunInfo`.

## Describe changes semantically

The compare tab picks a diff strategy by artifact kind. Text kinds get a line diff and tables a cell diff. For structured documents, register a strategy that returns a `StructuredDiff`, a list of `Change(op, path, old, new, label)` records. `diff_structured` computes the changes between two nested dict and list values. A change's `path` is a tuple of the keys and list positions leading to it, such as `("nodes", "sales", "config")`, so keys containing dots stay unambiguous, and `path_str` renders it for display. A strategy can relabel the changes in the document's own terms:

```python
from panel_agents.artifacts.diff import diff_structured, register_diff

@register_diff("dashboard")
def diff_dashboard(old, new, **options):
    result = diff_structured(old, new)
    for change in result.changes:
        if len(change.path) == 2 and change.path[0] == "nodes" and change.op != "change":
            name = change.path[1]
            change.label = f"Node {name!r} {'added' if change.op == 'add' else 'removed'}"
    result.old_text, result.new_text = to_yaml(old), to_yaml(new)
    return result
```

The view lists the changes, showing the label when there is one and otherwise the path with the old and new values. When `old_text` and `new_text` are set, a Text tab next to the change list shows the line diff of the serialized versions. `examples/artifacts_demo.py` shows a document artifact with a custom diff and an `on_apply` callback.
