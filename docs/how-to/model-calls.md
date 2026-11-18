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

`state.model_calls` maps request ids to the same objects. A request that fails, or is cut off because the run was cancelled, ends with `status="failed"` and an `error`. Attempts that pydantic-ai retried before one succeeded, for example with a fallback model, are failed requests of their own. `cached_tokens` are the input tokens the provider read from its cache; they are included in `input_tokens`.

`RunInfo` totals the requests of a run: `requests`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `model_duration` (the seconds spent waiting for the model) and `models`, with the run's wall-clock `duration`.

Another adapter only needs to emit the same two events. A `Usage` event without a `request_id`, as recordings made before these events were written hold, still becomes a model call, placed before the messages and tool calls that followed the previous one in its run.

## In the chat

`AgentChat` shows a model request as a compact card with the model, the input and output tokens, the cost and the latency. Hovering the header shows the cached tokens and any error, and expanding it lists the finish reason, the response, request and run ids and the tool calls the response issued.

Model requests are steps like tool calls: once a request succeeded it folds into the run's "2 steps · 3.0s" row together with the tool calls around it, so a finished run shows its steps as one row, its answer and the usage line under the answer:

```
2 requests · 8.3k in · 244 out · $0.0034 · 3.2s
```

Steps count the tool calls, and the duration is the time from the first folded step's start to the last one's end. A failed request never folds. A request with no step group to fold into, such as the only request of a run that answered without tools, shows while the run is in progress and is then left to the usage line. The usage line's tooltip adds the cached tokens, the time spent in the model and the models used.

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
import panel_material_ui as pmui

from panel_agents import RunTrace

run_id = state.tool_calls[version.tool_call_id].run_id
trace = RunTrace(state, run_id=run_id)
dialog = pmui.Dialog(trace, title="How this was made", width=900, open=True)
```

Setting `trace.run_id` shows another run. The time bars take a quarter of the width and the card headers drop the cost and the token counts as the view gets narrow, so it also fits a drawer. `show_totals=False` hides the totals.

Requests recorded with the tool call ids they issued nest those calls under them; for older recordings a tool call is nested under the request before it.

## Sessions recorded without model calls

A conversation recording made before model requests were recorded as events can get them from the session's [model trace](model-traces.md#show-model-calls-of-an-older-recording) with `merge_model_calls`.
