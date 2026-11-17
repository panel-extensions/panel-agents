# Trace model calls

A session has two recordings. The `Recorder` writes what the user saw: messages, tool calls, artifacts and approvals as events, which `load_state` replays into the same components. The model trace writes what the model saw: pydantic-ai's own message history, one `ModelMessage` per line, with the model, tokens, cost, duration and tool calls of every request. It is what you continue a conversation from, and the place to look when you want to know why a turn was slow or expensive.

## Record a model trace

Attach a `ModelTraceRecorder` to an `AgentRunner`:

```python
from panel_agents.adapters.pydantic_ai import AgentRunner, ModelTraceRecorder

runner = AgentRunner(agent, state)
recorder = ModelTraceRecorder(runner, "runs/model-trace.jsonl")
```

The file starts with a format line, `{"panel_agents_model_trace": 1, "include_content": true}`, followed by the messages in the format of pydantic-ai's `ModelMessagesTypeAdapter`. Messages are written while the run is in progress: a response is on disk before the tools it called run. Further runs of the conversation append to the same file, so keep one file per conversation. `close()` detaches the recorder, and it also works as a context manager.

Every response carries two values that pydantic-ai does not record, under `metadata["panel_agents"]`: the wall-clock `duration` of the request in seconds and a `fingerprint` of the instructions and tools the model was offered. `AgentRunner` also passes its run id to pydantic-ai, so the `run_id` on each message is the one on the events of the conversation recording. The `Usage` event of each response has the same `duration`.

## Failed and retried requests

pydantic-ai lists attempts that failed before a later attempt succeeded, for example with a `ModelFallback` capability, on the response under `failed_attempts`. A run that fails outright, for example because the provider refused the request, leaves no response. Its messages stay out of the conversation, as `AgentRunner` does not keep them, and the trace records them followed by a line naming the run, the error and the request that was in flight:

```json
{"failed_run": "run-3f2a9c", "error": "ModelHTTPError: status_code: 429, ...", "request": {"model": "gpt-4.1", "timestamp": "...", "duration": 0.41}}
```

Retries that the provider's HTTP client makes on its own are invisible to pydantic-ai and so to the trace.

## Summarize the model calls

`load_model_calls` returns one `ModelCall` per request, in order, including the failed ones:

```python
from panel_agents.adapters.pydantic_ai import load_model_calls

for call in load_model_calls("runs/model-trace.jsonl"):
    print(call.run_id, call.model, call.duration, call.input_tokens, call.output_tokens, call.cost_usd, call.tool_calls, call.error)
```

`tool_calls` lists the `(tool_name, tool_call_id)` of every tool the response called; the id matches the `tool_call_id` of the `TOOL_CALL_*` events in the conversation recording.

## Continue a conversation

`load_messages` returns the history as pydantic-ai messages, leaving out the messages of failed runs. Assign it to a runner and the next run continues the conversation; a recorder attached to the same file appends to it:

```python
from panel_agents.adapters.pydantic_ai import load_messages

runner = AgentRunner(agent, state)
runner.messages = load_messages("runs/model-trace.jsonl")
recorder = ModelTraceRecorder(runner, "runs/model-trace.jsonl")
await runner.run("And for last year?")
```

The chat shows only the new run unless you also replay the conversation recording into `state`. When a recorder starts a new file, it first writes the history the runner already has, so a resumed conversation can be recorded to a new file and still be complete.

## Leave out the content

The model trace is the most sensitive recording: it holds every prompt, the instructions, tool arguments and results, and anything the agent put in its context. Record without them when you only need usage and timing:

```python
ModelTraceRecorder(runner, path, include_content=False)
```

Messages then keep their parts, tool names, tool call ids, usage, timestamps and duration, but prompts, instructions, text, thinking, tool arguments and results are emptied and file parts are dropped. Such a trace still loads and summarizes, but it cannot continue a conversation. A file keeps the setting it was started with; recording to it with the other setting raises `ValueError`.

Artifacts published by tools are never written to the model trace, since the model does not see them; the conversation recording keeps them.
