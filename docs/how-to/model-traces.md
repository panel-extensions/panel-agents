# Trace and replay model calls

A session has two recordings. The `Recorder` writes what the user saw: messages, tool calls, artifacts and approvals as events, which `load_state` replays into the same components. The model trace writes what the model saw: pydantic-ai's own message history, one `ModelMessage` per line, with the model, tokens, cost, duration and tool calls of every request. It is what you continue a conversation from, and the place to look when you want to know why a turn was slow or expensive.

## Record a model trace

Attach a `ModelTraceRecorder` to an `AgentRunner`:

```python
from panel_agents.adapters.pydantic_ai import AgentRunner, ModelTraceRecorder

runner = AgentRunner(agent, state)
recorder = ModelTraceRecorder(runner, "runs/model-trace.jsonl")
```

The file starts with a format line, `{"panel_agents_model_trace": 1, "include_content": true}`, followed by the messages in the format of pydantic-ai's `ModelMessagesTypeAdapter`. The recorder brings the file up to date with the runner's history at the start of every model request, before the tools of a response run and when a run ends, so a response is on disk before the tools it called run. Further runs of the conversation append to the same file. `close()` detaches the recorder, and it also works as a context manager.

The trace records the persistent message history, the one `AgentRunner.messages` holds. Changes a capability makes only to the request it sends, without changing the history, are not in it. When the history is rewritten rather than extended, for example by a history processor that drops old messages, by `runner.clear()` or because a run failed, the recorder writes a `{"rewind": n}` record that keeps the first `n` messages, followed by the history from there on. The file therefore stays append-only, and loading it always gives the history the runner holds.

Every response carries two values that pydantic-ai does not record, under `metadata["panel_agents"]`: the wall-clock `duration` of the request in seconds and a `fingerprint` of the instructions and tools the model was offered. `AgentRunner` also passes its run id to pydantic-ai, so the `run_id` on each message is the one on the events of the conversation recording. The `Usage` event of each response has the same `duration`.

## Failed and retried requests

pydantic-ai lists attempts that failed before a later attempt succeeded, for example with a `ModelFallback` capability, on the response under `failed_attempts`. A run that fails outright, for example because the provider refused the request, leaves no response that joins the conversation. The trace records what the run sent, then a line naming the run, the error and the request that was in flight, then a rewind, since `AgentRunner` does not keep the messages of a failed run:

```json
{"failed_run": "run-3f2a9c", "error": "ModelHTTPError: status_code: 429, ...", "request": {"model": "gpt-4.1", "timestamp": "...", "duration": 0.41}}
```

Retries that the provider's HTTP client makes on its own are invisible to pydantic-ai and so to the trace.

## Summarize the model calls

`load_model_calls` returns one `ModelRequestSummary` per request, in order, including failed ones and responses a rewind later dropped from the history:

```python
from panel_agents.adapters.pydantic_ai import load_model_calls

for call in load_model_calls("runs/model-trace.jsonl"):
    print(call.run_id, call.model, call.duration, call.input_tokens, call.output_tokens, call.cost_usd, call.tool_calls, call.error)
```

`tool_calls` lists the `(tool_name, tool_call_id)` of every tool the response called; the id matches the `tool_call_id` of the `TOOL_CALL_*` events in the conversation recording. A response cut off by `runner.cancel()` has the error `"Interrupted"`.

## Continue a conversation

`load_messages` returns the history as pydantic-ai messages, as the runner held it when the trace ended. Assign it to a runner and the next run continues the conversation; a recorder attached to the same file appends to it:

```python
from panel_agents.adapters.pydantic_ai import load_messages

runner = AgentRunner(agent, state)
runner.messages = load_messages("runs/model-trace.jsonl")
recorder = ModelTraceRecorder(runner, "runs/model-trace.jsonl")
await runner.run("And for last year?")
```

The chat shows only the new run unless you also replay the conversation recording into `state`. A recorder writes the history the runner already has when it is attached: all of it to a new file, and only what differs from the conversation an existing file ends with. A resumed conversation can therefore be recorded to a new file and still be complete, and a recorder attached in the middle of a run records the whole history.

`AgentRunner` passes the id of each run to pydantic-ai. Pass `run_id` to `run()` to choose it; `run_kwargs` cannot hold one, since every run needs its own.

## Store recordings in your own database

Both recorders write to a sink. Besides a file path, a sink can be a text stream or a callable, which receives each record as the JSON-ready dict a line of the file would hold: the format record first, then one record per event or message, `rewind` and `failed_run` record. Redaction is applied before the callable sees anything. The loaders accept any iterable of those dicts in the order they were written, so a host can keep sessions next to its own data:

```python
def store(record):
    db.execute("INSERT INTO model_trace (session, record) VALUES (?, ?)", (session_id, json.dumps(record)))

ModelTraceRecorder(runner, store)
Recorder(state, lambda record: db.execute("INSERT INTO events (session, record) VALUES (?, ?)", (session_id, json.dumps(record))))

rows = db.execute("SELECT record FROM model_trace WHERE session = ? ORDER BY rowid", (session_id,))
runner.messages = load_messages(json.loads(row[0]) for row in rows)
```

A callable cannot be asked what it already holds, so a recorder writing to one starts with a format record every time it is attached, and, if the runner already has a history, with a `{"rewind": 0}` record and that history. The loaders read repeated format records and apply the rewinds, so the result is the history the runner holds.

## Leave out the content

The model trace is the most sensitive recording: it holds every prompt, the instructions, tool arguments and results, and anything the agent put in its context. Record without them when you only need usage and timing:

```python
ModelTraceRecorder(runner, path, include_content=False)
```

Messages then keep every part with its kind, tool name, tool call id and outcome, and their usage, timestamps and duration, but prompts, instructions, text, thinking, tool arguments and results are emptied, file data is removed, and errors are reduced to the exception type. Such a trace still loads and summarizes, but it cannot continue a conversation. A file keeps the setting it was started with; recording to it with the other setting raises `ValueError`.

Artifacts published by tools are never written to the model trace, since the model does not see them; the conversation recording keeps them.

## Play a trace back

A `Cassette` is a pydantic-ai model that answers each request with the next recorded response. The agent loop, the tools, the approvals and the UI run for real, but no provider is called, so a recorded session becomes a test that runs offline and costs nothing:

```python
from panel_agents.adapters.pydantic_ai import AgentRunner, Cassette

cassette = Cassette.from_trace("tests/cassettes/east-sales.jsonl")
runner = AgentRunner(agent, state, run_kwargs={"model": cassette})
await runner.run("How much did east sell?")

assert cassette.divergences == []
assert cassette.remaining == 0
```

`Cassette(messages)` takes a message history directly, for example `result.all_messages()`, and works with `agent.run`, `agent.override(model=cassette)` and streamed runs alike. The responses keep their recorded parts, model name and usage, so the chat shows the same usage as the recorded run, but take the run id of the playback, so a trace recorded during playback joins with its own events.

A cassette holds the model fixed and tests everything else. Each tool result, user prompt and output validator retry the agent sends is compared with the recording, and every difference is collected in `cassette.divergences` as a `Divergence` naming the request, the tool call and the recorded and actual values; with `strict=True` the first one raises `CassetteError` instead. Playback fails with `CassetteError` when a recorded response calls a tool the agent no longer offers and when the agent makes more requests than were recorded.

A cassette also goes stale when the instructions or tools change, even if every result still matches. `AgentRunner` stores a `fingerprint` of the instructions, system prompts and tool schemas on each response, and the cassette raises `CassetteError` when the agent it plays to offers something else. Record the cassette again after such a change. For an agent whose instructions change from run to run, for example because they include the date, pass `check_fingerprint=False`; a mismatch is then reported as a divergence of kind `"fingerprint"`.

Playback executes the agent's tools, so treat a cassette from someone else like any agent run: it runs under your approval policy and your tools' permissions. A trace recorded with `include_content=False` cannot be played back.
