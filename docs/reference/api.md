# API reference

This page lists the public API of panel-agents 1.x. Every name below is covered by semantic versioning. Every public version reference is a version number counting from 1, as in the `v1`, `v2` labels users see. Anything else, including names and modules that start with an underscore, is private. Classes built on `param` expose their configuration as parameters, documented in their docstrings and listed with `Class.param`.

## `panel_agents`

### Components

| Name | Purpose |
| --- | --- |
| `AgentChat(state=None, *, runner, callback, readonly, add_user_message, approval_factory, can_approve, user, assistant, placeholder, tool_labels, group_tool_calls, feed_params)` | Chat feed of a `RunState` with streaming messages, thinking steps, tool-call cards and an error alert for a failed run; `tool_labels` names calls in the user's words and `group_tool_calls` folds runs of finished calls into one row. `submit(text)` sends text as the user would. |
| `AgentDrawer(runner=None, *, state, main, open, width, title, variant, overlay_below, tab, show_artifacts, show_approvals, open_on_approval, can_approve, on_apply, chat_params)` | A right-hand drawer with the chat, artifacts and approvals, docked beside `main`. `toggle` (with `aria-expanded` and the pending count in its accessible name), `content`, `chat`, `artifacts`, `approvals`, `new_conversation()`. |
| `ApprovalCard(request=None, state=None, *, can_approve, policy_message, require_reason, user, editing)` | Review card for one tool call: edit arguments, approve or deny. `approve()`, `deny()`, `form`, `permitted`, `is_edited`. |
| `ApprovalQueue(state, *, can_approve, require_reason, user, show_resolved, title, card_factory)` | The pending approvals of a state as cards. `approve_all()`, `deny_all()`, `cards`. |
| `ArtifactPane(store=None, *, state, on_apply, show_list, list_width, empty_message, view_options)` | Lists a store's artifacts and shows the active one. `views`, `active_view`. |
| `ArtifactView(artifact, *, mode, base, target, diff_layout, diff_options, show_header, on_apply, state, user)` | One artifact with versions, compare, metadata and apply/revert for documents. `base` and `target` are version numbers. `apply_version(version=None)`, `compute_diff()`. |
| `ToolCallCard(call, *, state, approval_factory, max_chars, max_items, collapsed, label)` | Card for one tool call with arguments, status, result and published artifacts; `label` (a string or a callable of the arguments) replaces the tool name in the header. |
| `register_renderer(func=None, *, kind=None, type=None, applies=None, priority=0)` | Register how values of an artifact kind, a type or a predicate are rendered; usable as a decorator. |

### State and events

| Name | Purpose |
| --- | --- |
| `RunState` | Reactive store built from events. `apply(event)`, `apply_many(events)`, `subscribe(callback)`, `reset()`, `flush()`, `from_events(events)`, `events`, `add_user_message(text)`, `resolve_approval(tool_call_id, approved, args=None, reason=None, user=None)` (raises `ValueError` if already decided and `PermissionError` if the runner's policy does not allow `user`), `publish_artifact(artifact_id, value, *, kind, title, metadata, applied=False)`, `apply_artifact_version(artifact_id, version, user=None)`. |
| `MessageState`, `ToolCallState`, `ApprovalRequest`, `RunInfo` | The entities in a `RunState`, as parameterized objects components watch. |
| `events` | The event classes, see below. |
| `Recorder(state, path)`, `dump_events(events, path)`, `load_events(path)`, `load_state(path)` | Record runs as JSON Lines, starting with a format version line, and load them back for replay. |
| `AuditLog(state=None, sink=None, include_existing=False)` | Writes one record per approval decision to a file or callable. |

### Artifacts, approvals and actions

| Name | Purpose |
| --- | --- |
| `Artifact`, `ArtifactVersion`, `ArtifactStore` | Versioned artifacts. `Artifact.current` is the number of the version shown, `Artifact.is_document` and `Artifact.applied_version` describe documents; `ArtifactStore.publish(...)` and `mark_applied(artifact_id, version)`. |
| `register_diff(*kinds)` | Register a diff strategy `func(old, new, **options) -> Diff` for artifact kinds. |
| `ApprovalPolicy(rules)`, `ApprovalRule(tool, when=None, approvers=None)` | Which tool calls need approval and who may give it. `approvers` is a collection of user names or a callable `(request, user) -> bool`. `requires_approval(tool, args)`, `can_approve(request, user)`, `can_decide(request, user, args=None)`, `rule_for(tool, args)`. |
| `ActionRegistry()`, `Action` | Typed front-end actions run in the user's session. `action(...)`, `add(func, *, name, description)`, `remove(name)`, `await invoke(name, args)`; `Action(name, description, function)`, `Action.from_function(function, name=None, description=None)`, `parameters`, `validate(args)`. |

## `panel_agents.events`

`Event` and its subclasses, mirroring AG-UI: `RunStarted`, `RunFinished`, `RunError`, `StepStarted`, `StepFinished`, `TextMessageStart`, `TextMessageContent`, `TextMessageEnd`, `ThinkingStart`, `ThinkingContent`, `ThinkingEnd`, `ToolCallStart`, `ToolCallArgs`, `ToolCallEnd`, `ToolCallResult`, `StateSnapshot`, `StateDelta`, `Custom`. The kit-specific events map onto AG-UI `CUSTOM` events: `ArtifactPublished`, `ArtifactApplied`, `ApprovalRequested`, `ApprovalResolved`, `Usage` and `GuardrailFired`. Each has `to_dict()` and `Event.from_dict(data)`.

## `panel_agents.artifacts`

The artifact model above, plus the diff API: `diff(old, new, kind=None, **options)`, `register_diff`, the strategies `diff_values`, `diff_text`, `diff_table` and `diff_structured`, the results `Diff`, `ValueDiff`, `TextDiff`, `TableDiff` and `StructuredDiff`, `Change(op, path, old=None, new=None, label=None)` with `path` a tuple of keys and list positions and `path_str` for display, and `infer_kind(value)`.

## `panel_agents.components`

The components above, plus `render(value, kind=None, **options)`, which renders a value with the registered renderers, and `approval_factory(request, state)`, the default `AgentChat.approval_factory`.

## `panel_agents.recording`

`Recorder`, `dump_events`, `load_events`, `load_state` and `FORMAT_VERSION`, plus `Codec(name, applies, encode, decode)` and `register_codec(codec, first=True)` to round-trip values that are not JSON serializable.

## `panel_agents.security`

`inert_markdown` is a markdown-it plugin that makes untrusted Markdown inert on the parsed tokens: raw HTML renders as text, images become links unless their origin is trusted, links are limited to http, https, mailto and relative ones and show their full URL. `untrusted_markdown(text, **params)` returns a Markdown pane using it with math disabled, in the page as well as on the server. Chat messages, thinking, text tool results and text or markdown artifacts are rendered with it, and titles, tool names and descriptions, reasons and error messages are shown as plain text. `set_trusted_image_origins(origins)` and `trusted_image_origins()` set and read the origins images may load from (`"*"` for all).

## `panel_agents.adapters.pydantic_ai`

Requires the `pydantic-ai` extra.

| Name | Purpose |
| --- | --- |
| `AgentRunner(agent=None, state=None, deps=None, *, approval_policy, thread_id, run_kwargs, raise_errors)` | Runs a pydantic-ai agent into a `RunState`, keeping the conversation. `await run(prompt, *, deferred_tool_results=None, **kwargs)`, `cancel()`, `clear()`, `close()`; `running`, `run_id`, `error`, `deferred`, `messages`. A failed run, unless cancelled, is logged with its traceback on the `panel_agents` logger and adds an `error` message to the timeline. |
| `publish_artifact(ctx, artifact_id, value, *, title, kind, metadata, applied=False)` | Publish an artifact version from an async tool. |
| `artifact_return(return_value, artifact_id, value, *, title, kind, metadata, applied=False)` | Return a value to the model and publish an artifact, from any tool. |
| `ActionToolset(registry, *, id=None, max_retries=1)` | An `ActionRegistry` as a pydantic-ai toolset. |

## `panel_agents.mcp`

Requires the `mcp` extra. `await create_server(source, *, deps=None, deps_factory=None, name=None, instructions=None, approval_policy=None, user=None) -> fastmcp.FastMCP` serves an `AgentRunner`, agent, toolset or `ActionRegistry` over MCP.
