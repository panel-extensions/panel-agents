# API reference

This page lists the public API of panel-agents 1.x. Every name below is covered by semantic versioning. Every public version reference is a version number counting from 1, as in the `v1`, `v2` labels users see. Anything else, including names and modules that start with an underscore, is private. Classes built on `param` expose their configuration as parameters, documented in their docstrings and listed with `Class.param`.

## `panel_agents`

### Components

| Name | Purpose |
| --- | --- |
| `AgentChat(state=None, *, runner, callback, readonly, add_user_message, approval_factory, can_approve, user, assistant, placeholder, tool_labels, group_tool_calls, show_model_calls, show_usage, feed_params)` | Chat feed of a `RunState` with streaming messages, thinking steps, model-call and tool-call cards and an error alert for a failed run; `tool_labels` names calls in the user's words and `group_tool_calls` folds runs of finished calls and model requests into one row. `show_model_calls` shows model requests as steps and `show_usage` a finished run's requests, tokens, cost and wall time under its last message. `submit(text)` sends text as the user would. |
| `AgentDrawer(runner=None, *, state, main, open, width, title, variant, overlay_below, tab, show_artifacts, show_approvals, open_on_approval, can_approve, on_apply, chat_params)` | A right-hand drawer with the chat, artifacts and approvals, docked beside `main`. `toggle` (with `aria-expanded` and the pending count in its accessible name), `content`, `chat`, `artifacts`, `approvals`, `new_conversation()`. |
| `ApprovalCard(request=None, state=None, *, can_approve, policy_message, require_reason, user, editing, labels)` | Review card for one tool call: edit arguments, approve or deny. `labels` changes the wording of the actions, e.g. `{"deny": "Remove", "denied": "Removed"}`; a changed verb needs its past tense, and a request's own labels take precedence. `approve()`, `deny()`, `form`, `permitted`, `is_edited`, `resolved_labels`. |
| `ApprovalQueue(state, *, can_approve, require_reason, user, show_resolved, title, card_factory, labels)` | The pending approvals of a state as cards. `labels` is passed to each card and names the batch actions, unless all pending requests share labels of their own. `approve_all()`, `deny_all()`, `cards`, `batch_labels`. |
| `ArtifactPane(store=None, *, state, on_apply, show_list, list_width, empty_message, view_options)` | Lists a store's artifacts and shows the active one. `views`, `active_view`. |
| `ArtifactView(artifact, *, mode, base, target, diff_layout, diff_options, show_header, on_apply, state, user)` | One artifact with versions, compare, metadata and apply/revert for documents. `base` and `target` are version numbers. `apply_version(version=None)`, `compute_diff()`. |
| `ModelCallCard(call, *, state, collapsed, show_tool_count)` | Card for one model request: model, input and output tokens, cost and latency in the header, the cached tokens, finish reason, response, request and run ids, the tool calls it issued and any error in the body. |
| `RunTrace(state, *, run_id, show_totals)` | Every step of one run, unfolded: model requests with the tool calls they issued nested under them, a bar per step on the run's time axis, and the run's totals on top. Readonly; follows a run in progress. |
| `ToolCallCard(call, *, state, approval_factory, max_chars, max_items, max_lines, collapsed, label)` | Card for one tool call with arguments, status, result and published artifacts; `label` (a string or a callable of the arguments) replaces the tool name in the header. Structured and code-like results are shown as highlighted code blocks, cut to `max_lines` until expanded. |
| `register_renderer(func=None, *, kind=None, type=None, applies=None, priority=0)` | Register how values of an artifact kind, a type or a predicate are rendered; usable as a decorator. |
| `render_output(value, max_lines=30, **options)`, `output_format(value)` | Render a tool result the way `ToolCallCard` does, and the classification behind it: `output_format` returns `(format, content)` with format `"json"`, `"yaml"`, `"python"`, `"pytb"`, `"sql"`, `"xml"`, `"text"`, `"table"` or `"object"`. Both live in `panel_agents.components.renderers`. |

### State and events

| Name | Purpose |
| --- | --- |
| `RunState` | Reactive store built from events. `apply(event)`, `apply_many(events)`, `subscribe(callback)`, `reset()`, `flush()`, `from_events(events)`, `events`, `add_user_message(text)`, `resolve_approval(tool_call_id, approved, args=None, reason=None, user=None)` (raises `ValueError` if already decided and `PermissionError` if the runner's policy does not allow `user`), `publish_artifact(artifact_id, value, *, kind, title, metadata, applied=False, run_id=None, tool_call_id=None, actor=None, user=None)` (who made the version defaults to the current run and tool call, or to the user), `user_run(input=None, *, user=None, run_id=None)` (a context manager grouping a person's edits into a run, yielding its id), `apply_artifact_version(artifact_id, version, user=None)`. |
| `current_run_id()`, `current_tool_call_id()` | The run and tool call the calling code executes in, set by `AgentRunner` and `RunState.user_run`; None outside them. Also in `panel_agents.context`. |
| `MessageState`, `ModelCallState`, `ToolCallState`, `ApprovalRequest`, `RunInfo` | The entities in a `RunState`, as parameterized objects components watch. `RunState.items` holds messages, model calls and tool calls in timeline order and `RunState.model_calls` maps request ids to `ModelCallState`. `RunInfo` totals its `Usage` events: `requests`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `model_duration`, `models` and the wall-clock `duration`. |
| `events` | The event classes, see below. |
| `Recorder(state, sink, *, on_error="log")`, `dump_events(events, path)`, `load_events(source)`, `load_state(source)` | Record runs as JSON Lines, starting with a format version line, and load them back for replay. A sink is a file path, a text stream or a callable receiving each record as a dict; a source is a file path or an iterable of those dicts. |
| `AuditLog(state=None, sink=None, include_existing=False)` | Writes one record per approval decision to a file or callable. |

### Artifacts, approvals and actions

| Name | Purpose |
| --- | --- |
| `Artifact`, `ArtifactVersion`, `ArtifactStore` | Versioned artifacts. `ArtifactVersion` records who made it as `run_id`, `tool_call_id`, `actor` (`"agent"`, `"user"` or a host's own string) and `user`. `Artifact.current` is the number of the version shown, `Artifact.is_document` and `Artifact.applied_version` describe documents; `ArtifactStore.publish(...)` and `mark_applied(artifact_id, version)`. |
| `register_diff(*kinds)` | Register a diff strategy `func(old, new, **options) -> Diff` for artifact kinds. |
| `ApprovalPolicy(rules)`, `ApprovalRule(tool, when=None, approvers=None, labels=None)` | Which tool calls need approval and who may give it. `approvers` is a collection of user names or a callable `(request, user) -> bool`. `labels` names the decision actions for the matching calls and is recorded on `ApprovalRequested.labels` and `ApprovalRequest.labels`. `requires_approval(tool, args)`, `can_approve(request, user)`, `can_decide(request, user, args=None)`, `rule_for(tool, args)`. |
| `ActionRegistry()`, `Action` | Typed front-end actions run in the user's session. `action(...)`, `add(func, *, name, description)`, `remove(name)`, `await invoke(name, args)`; `Action(name, description, function)`, `Action.from_function(function, name=None, description=None)`, `parameters`, `validate(args)`. |

## `panel_agents.events`

`Event` and its subclasses, mirroring AG-UI: `RunStarted`, `RunFinished`, `RunError`, `StepStarted`, `StepFinished`, `TextMessageStart`, `TextMessageContent`, `TextMessageEnd`, `ThinkingStart`, `ThinkingContent`, `ThinkingEnd`, `ToolCallStart`, `ToolCallArgs`, `ToolCallEnd`, `ToolCallResult`, `StateSnapshot`, `StateDelta`, `Custom`. The kit-specific events map onto AG-UI `CUSTOM` events: `ArtifactPublished` (with `actor` and `user` next to `run_id` and `tool_call_id`), `ArtifactApplied`, `ApprovalRequested`, `ApprovalResolved`, `ModelRequestStarted`, `Usage` and `GuardrailFired`. `ModelRequestStarted(request_id, model)` begins a model request and the `Usage` event with the same `request_id` completes it with its tokens, cost, duration, `tool_call_ids`, `error`, `finish_reason` and `provider_response_id`. Each has `to_dict()` and `Event.from_dict(data)`.

## `panel_agents.artifacts`

The artifact model above, plus the diff API: `diff(old, new, kind=None, **options)`, `register_diff`, the strategies `diff_values`, `diff_text`, `diff_table` and `diff_structured`, the results `Diff`, `ValueDiff`, `TextDiff`, `TableDiff` and `StructuredDiff`, `Change(op, path, old=None, new=None, label=None)` with `path` a tuple of keys and list positions and `path_str` for display, and `infer_kind(value)`.

## `panel_agents.components`

The components above, plus `render(value, kind=None, **options)`, which renders a value with the registered renderers, and `approval_factory(request, state)`, the default `AgentChat.approval_factory`. `panel_agents.components.approval.approval_labels(*layers)` returns the full action wording (`approve`, `deny`, `approved`, `denied`, `edit`, `approve_edited`, `approved_edited`, `approve_all`, `deny_all`), resolving each layer over the previous one, so a layer that changes `deny` also resets `deny_all`, and `APPROVAL_LABELS` lists the keys.

## `panel_agents.recording`

`Recorder`, `dump_events`, `load_events`, `load_state` and `FORMAT_VERSION`, plus `Codec(name, applies, encode, decode)` and `register_codec(codec, first=True)` to round-trip values that are not JSON serializable.

## `panel_agents.security`

`inert_markdown` is a markdown-it plugin that makes untrusted Markdown inert on the parsed tokens: raw HTML renders as text, images become links unless their origin is trusted, links are limited to http, https, mailto and relative ones and show their full URL. `untrusted_markdown(text, **params)` returns a Markdown pane using it with math disabled, in the page as well as on the server. Chat messages, thinking, text tool results and text or markdown artifacts are rendered with it, and titles, tool names and descriptions, reasons and error messages are shown as plain text. `set_trusted_image_origins(origins)` and `trusted_image_origins()` set and read the origins images may load from (`"*"` for all).

## `panel_agents.adapters.pydantic_ai`

Requires the `pydantic-ai` extra.

| Name | Purpose |
| --- | --- |
| `AgentRunner(agent=None, state=None, deps=None, *, approval_policy, thread_id, run_kwargs, raise_errors)` | Runs a pydantic-ai agent into a `RunState`, keeping the conversation. `await run(prompt, *, deferred_tool_results=None, **kwargs)`, `cancel()`, `clear()`, `close()`; `running`, `run_id`, `error`, `deferred`, `messages`. A failed run, unless cancelled, is logged with its traceback on the `panel_agents` logger and adds an `error` message to the timeline. |
| `publish_artifact(ctx, artifact_id, value, *, title, kind, metadata, applied=False)` | Publish an artifact version from an async tool, stamped with the run, the tool call and the actor `"agent"`. |
| `artifact_return(return_value, artifact_id, value, *, title, kind, metadata, applied=False)` | Return a value to the model and publish an artifact, from any tool. |
| `ActionToolset(registry, *, id=None, max_retries=1)` | An `ActionRegistry` as a pydantic-ai toolset. |
| `ModelTraceRecorder(runner, sink, *, include_content=True, on_error="log")` | Keeps a JSON Lines model trace in sync with a runner's pydantic-ai message history while it runs, with `rewind` records where the history was rewritten and `failed_run` records for runs that fail. `close()`. |
| `load_messages(source, *, include_failed=False)` | The history a model trace ends with, as pydantic-ai `ModelMessage`s; assign it to `AgentRunner.messages` to continue the conversation. |
| `load_model_calls(source)` | One `ModelRequestSummary(run_id, model, timestamp, duration, tool_calls, input_tokens, output_tokens, cached_tokens, cost_usd, error, provider_response_id, request_id, finish_reason)` per model request, including failed ones. |
| `merge_model_calls(events, calls)` | The events of a conversation recording with a `ModelRequestStarted` and `Usage` event added for each model call of a model trace, so a session recorded without them replays with its model calls. |
| `Cassette(messages, *, strict=False, check_fingerprint=True)`, `Cassette.from_trace(source, **kwargs)` | A pydantic-ai `FunctionModel` that plays recorded responses back in order, streamed or not. `divergences`, `played`, `remaining`, `responses`. |
| `Divergence(request, kind, expected, actual, tool_name=None, tool_call_id=None)`, `CassetteError` | A difference between playback and the recording (`kind` is `"tool_return"`, `"prompt"`, `"retry"` or `"fingerprint"`), and the error raised when playback cannot continue. |
| `fingerprint(model_request_parameters, messages=())` | A short hash of the instructions, system prompts and function tools offered to the model, stored on every recorded response. |

`AgentRunner` passes its `run_id` to pydantic-ai (`run(..., run_id=...)` chooses it) and stores the request's `duration`, `fingerprint` and `request_id` on each `ModelResponse` under `metadata["panel_agents"]`. It emits a `ModelRequestStarted` event when a request begins and a `Usage` event when it ends, failed requests included. The trace names also live in `panel_agents.adapters.model_trace`, together with its `FORMAT_VERSION`, and the cassette names in `panel_agents.adapters.cassette`.

## `panel_agents.mcp`

Requires the `mcp` extra. `await create_server(source, *, deps=None, deps_factory=None, name=None, instructions=None, approval_policy=None, user=None) -> fastmcp.FastMCP` serves an `AgentRunner`, agent, toolset or `ActionRegistry` over MCP.
