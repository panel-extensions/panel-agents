# Show model calls and run traces

Every request an agent makes to its model is an item in the `RunState` timeline, next to the messages and tool calls. `AgentChat` shows them as steps and adds the usage of each finished run under its answer, and `RunTrace` lays out one run in full: every model request with the tool calls it issued, on a shared time axis, under the run's totals.

## Model calls in the state

`AgentRunner` emits a `ModelRequestStarted` event when a request begins and a `Usage` event when its response is in. The start event puts a `ModelCallState` in `state.items` before the text and tool calls the response streams, with `status="running"`, and the `Usage` event with the same `request_id` completes it:

```python
from panel_agents import ModelCallState

for item in state.items:
    if isinstance(item, ModelCallState):
        print(item.model, item.input_tokens, item.cached_tokens, item.output_tokens, item.cost_usd, item.duration, item.tool_call_ids)
```

`state.model_calls` maps request ids to the same objects. A request that fails, or is cut off because the run was cancelled, ends with `status="failed"` and an `error`. Attempts that pydantic-ai retried before one succeeded, for example with a fallback model, belong to the request they were made for: they are listed in its `attempts`, as `Usage` events with an `error` and `attempt_of` set to its request id, and the timeline keeps one item per request. `cached_tokens` are the input tokens the provider read from its cache; they are included in `input_tokens`.

`RunInfo` totals the requests of a run: `requests` (every `Usage` event, retried attempts included), `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `model_duration` (the seconds spent waiting for the model) and `models`, with the run's wall-clock `duration`.

Another adapter only needs to emit the same two events. A `Usage` event without a `request_id`, as recordings made before these events were written hold, still becomes a model call, placed before the messages and tool calls that followed the previous one in its run.

## In the chat

`AgentChat` shows a model request as a compact card with the model, the input and output tokens, the cost and the latency. Hovering the header shows the cached tokens and any error, and expanding it lists the finish reason, the response, request and run ids and the tool calls the response issued.

Model requests are steps like tool calls: once a request succeeded it folds into the run's "2 steps · 3.0s" row together with the tool calls around it, so a finished run shows its steps as one row, its answer and the usage line under the answer:

```
2 requests · 8.3k in · 244 out · $0.0034 · 3.2s total
```

Steps count the tool calls, and the duration is the time from the first folded step's start to the last one's end; the usage line's "total" is the wall time of the whole run, answer included. A failed request never folds. A request with no step group to fold into, such as the only request of a run that answered without tools, shows while the run is in progress and is then left to the usage line. The usage line's tooltip adds the cached tokens, the time spent in the model and the models used. A request with retried attempts shows how many in its header and lists them in its details. The usage line covers the run's own requests: those of a sub-run (a run with a `parent_run_id`) are not added to its parent's.

Both are parameters:

```python
AgentChat(state, show_model_calls=False)  # no model request cards
AgentChat(state, show_usage=False)        # no usage line; lone requests stay visible
```

In an `AgentDrawer`, pass them through `chat_params`.

## The trace of one run

`RunTrace` shows one run with every step unfolded: each model request with the tool calls it issued nested under it, a bar per step on the run's time axis, and the run's totals on top (model requests, input tokens with those read from the cache, output tokens, cost, and wall time with the time spent in the model). Expanding a model request shows its details, and expanding a tool call its arguments and result. It follows a run in progress and is readonly.

It is the building block for showing where something came from. Given a run id, for example that of the tool call that published an artifact version, a host can show the run in a dialog:

```python
import panel as pn

from panel_agents import RunTrace

run_id = state.tool_calls[version.tool_call_id].run_id
trace = RunTrace(state, run_id=run_id)
dialog = pn.ui.Dialog(trace, title="How this was made", width=900, open=True)
```

Setting `trace.run_id` shows another run. The time bars take a quarter of the width and the card headers drop the cost and the token counts as the view gets narrow, so it also fits a drawer. `show_totals=False` hides the totals.

Requests recorded with the tool call ids they issued nest those calls under them; for older recordings a tool call is nested under the request before it.

## How something was made: the span of a run

`SpanView` answers "how was this made?" for one run. It shows the question, the run's totals with the share of input tokens read from the cache, the run's steps as in `RunTrace`, the artifact versions the run published and the problems it ran into (failed requests and retries, failed tool calls, blocking guardrails). The steps of child runs, such as a sub-agent's, are part of it, under a label naming the child run, and so are their tokens and cost. A cancelled run shows as cancelled with its reason, not as failed. It is readonly and follows a run in progress.

![`SpanView` shows the run's question and totals, then its steps on a shared time axis; here a failed `run_query` call is expanded.](../assets/images/span-view.png)

```python
from panel_agents import SpanView

view = SpanView(state, run_id=version.run_id, on_artifact=lambda artifact_id, number: ...)
```

`on_artifact` is called when a published version is clicked. A view follows the state until `view.release()`, so call it when the view is closed for good. A long run builds its first 50 steps and offers the rest 50 at a time (`row_limit`). A run whose steps were not recorded, for example one restored from a stored history that keeps only versions, says so and still lists what it published.

`redacted=True` shows the shape of the run without its content, for people who may not see the data: which tools ran, the models, tokens, cost and timings, and the titles and numbers of the versions stay, while the question, the agent's text, tool arguments and results, approval descriptions, error messages and version details are hidden. `ToolCallCard`, `ModelCallCard` and `RunTrace` take the same `redacted` parameter.

The same information is available as data from `RunState.span(run_id)`, a `RunSpan` with the run and its child runs, the question, the nested `steps`, the `model_calls`, `tool_calls` and `messages`, the `published` versions, the `errors` as `SpanError(kind, message, item)` and totals such as `requests`, `input_tokens`, `cached_share`, `cost_usd` and `duration`.

`AgentDrawer` wires it up: `show_run(run_id)` opens the span in a dialog, redacted when `redact_runs` (a boolean, or a callable taking the run id, for the host's permissions) says so, each answer in its chat has a "Details" link, and the runs in the version history have a "How was this made?" action. Clicking a published version in the dialog opens it in the artifacts view. Elsewhere, pass `on_show_run` to `AgentChat`, `ArtifactView` or `ArtifactPane`.

## Sessions recorded without model calls

A conversation recording made before model requests were recorded as events can get them from the session's [model trace](model-traces.md#show-model-calls-of-an-older-recording) with `merge_model_calls`.
