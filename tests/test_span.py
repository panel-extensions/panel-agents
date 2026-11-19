import pytest

from panel_agents import AgentChat, AgentDrawer, ArtifactView, RunState, SpanView, ToolCallCard
from panel_agents import events as ev
from panel_agents.components.model_call import ModelCallCard

from .test_model_calls import model_run

SECRET = "SELECT * FROM customers WHERE name = 'Acme'"


def span_run(run_id="r1") -> list[ev.Event]:
    """A run that publishes a document twice, one of them from a child run, with a retry, a failed tool and a block."""
    events = model_run(run_id)
    finished = events.pop()
    events[0] = ev.RunStarted(run_id=run_id, input="Paid revenue by region?", timestamp=100.0)
    events += [
        ev.ArtifactPublished(
            artifact_id="doc", value={"sql": SECRET}, kind="json", title="Dashboard", applied=True,
            metadata={"label": "Added revenue"}, run_id=run_id, tool_call_id="t1", actor="agent", timestamp=101.1,
        ),
        ev.ModelRequestStarted(request_id="q3", model="gpt-4o", run_id=run_id, timestamp=101.9),
        ev.Usage(
            model="gpt-4o", request_id="q3a", attempt_of="q3", error="rate limited: " + SECRET, duration=0.1,
            run_id=run_id, timestamp=101.95,
        ),
        ev.ToolCallStart(tool_call_id="t2", tool_name="run_sql", run_id=run_id, timestamp=101.96),
        ev.ToolCallArgs(tool_call_id="t2", delta=f'{{"sql": "{SECRET}"}}', run_id=run_id),
        ev.ToolCallEnd(tool_call_id="t2", run_id=run_id),
        ev.Usage(
            model="gpt-4o", input_tokens=500, output_tokens=5, request_id="q3", tool_call_ids=["t2"],
            run_id=run_id, timestamp=101.97,
        ),
        ev.ToolCallResult(tool_call_id="t2", content="no such table: " + SECRET, is_error=True, run_id=run_id, timestamp=101.98),
        ev.GuardrailFired(name="sql", message="Blocked " + SECRET, severity="block", run_id=run_id),
        ev.RunStarted(run_id=f"{run_id}-child", parent_run_id=run_id, agent_name="ana", timestamp=101.99),
        ev.ArtifactPublished(
            artifact_id="doc", value={"sql": "SELECT 2"}, applied=True, run_id=f"{run_id}-child",
            actor="user", user="ana", timestamp=101.99,
        ),
        ev.RunFinished(run_id=f"{run_id}-child", timestamp=101.99),
        finished,
    ]
    return events


def other_run() -> list[ev.Event]:
    return [
        ev.RunStarted(run_id="r2", input="Something else", timestamp=200.0),
        ev.ArtifactPublished(artifact_id="doc", value={"sql": "SELECT 3"}, applied=True, run_id="r2", timestamp=200.5),
        ev.RunFinished(run_id="r2", timestamp=201.0),
    ]


def state() -> RunState:
    s = RunState(flush_interval=0)
    s.add_user_message("Paid revenue by region?", run_id="r1")
    s.apply_many([*span_run(), *other_run()])
    return s


def test_span_collects_what_the_run_and_its_children_did():
    span = state().span("r1")
    assert span.run_id == "r1" and [r.run_id for r in span.runs] == ["r1", "r1-child"]
    assert span.question == "Paid revenue by region?"
    assert [m.request_id for m in span.model_calls] == ["q1", "q2", "q3"]
    assert [t.tool_call_id for t in span.tool_calls] == ["t1", "t2"]
    assert [m.content for m in span.messages] == ["Checking.", "Done."]
    steps = [(getattr(i, "request_id", None) or getattr(i, "tool_call_id", None) or i.message_id, d) for i, d in span.steps]
    assert steps == [("q1", 0), ("t1", 1), ("m1", 0), ("q2", 0), ("m2", 0), ("q3", 0), ("t2", 1)]
    assert [(a.artifact_id, v.version) for a, v in span.published] == [("doc", 1), ("doc", 2)]
    assert [e.kind for e in span.errors] == ["guardrail", "attempt", "tool_call"]


def test_span_totals_cover_every_request_and_the_cached_share():
    span = state().span("r1")
    # Two answered requests, the failed attempt and the retry that answered.
    assert (span.requests, span.input_tokens, span.output_tokens, span.cached_tokens) == (4, 2600, 55, 800)
    assert span.cached_share == pytest.approx(800 / 2600)
    assert span.cost_usd == pytest.approx(0.003)
    assert span.models == ["gpt-4o"] and span.duration == 2.0


def test_span_of_a_run_without_steps_keeps_its_versions():
    s = RunState.from_events(other_run())
    span = s.span("r2")
    assert span.question == "Something else" and span.steps == () and span.errors == ()
    assert [v.version for _, v in span.published] == [1]
    assert span.cached_share is None and span.cost_usd is None


def test_span_of_an_unknown_run_raises():
    with pytest.raises(KeyError):
        state().span("nope")


def test_span_question_falls_back_to_the_user_message():
    s = RunState(flush_interval=0)
    s.add_user_message("Hello?", run_id="r1")
    s.apply_many(model_run())
    assert s.span("r1").question == "Hello?"


def test_failed_run_is_an_error_of_its_span():
    s = RunState.from_events([ev.RunStarted(run_id="r"), ev.RunError(run_id="r", message="boom")])
    assert [(e.kind, e.message) for e in s.span("r").errors] == [("run", "boom")]


def test_span_view_shows_the_question_steps_and_published_versions():
    view = SpanView(state(), run_id="r1")
    head = view._header.object
    assert "Paid revenue by region?" in head and "Finished" in head
    assert "31% of input from cache" in head
    assert "1 step blocked, 1 request retried, 1 tool call failed" in head
    assert view._trace.run_id == "r1" and view._trace.steps == 7
    assert [i["secondary"] for i in view._published.items] == ["v1 · Added revenue", "v2"]
    assert view._guardrails.visible and "Blocked SELECT" in view._guardrails.object
    assert not view._unrecorded.visible and not view._missing.visible


def test_span_view_says_when_steps_were_not_recorded_or_the_run_is_missing():
    s = RunState.from_events(other_run())
    view = SpanView(s, run_id="r2")
    assert view._unrecorded.visible and view._published.visible
    view.run_id = "gone"
    assert view._missing.visible and not view._body.visible
    s.apply(ev.RunStarted(run_id="gone", input="Back"))
    assert not view._missing.visible and "Back" in view._header.object


def test_redacted_span_view_hides_content_but_keeps_the_shape():
    view = SpanView(state(), run_id="r1", redacted=True, row_limit=None)
    trace = view._trace
    texts = [view._header.object, view._guardrails.object, str(view._published.items)]
    for item, _ in trace._shown:
        card = trace._cards[id(item)]
        if isinstance(card, ToolCallCard):
            assert not card._args_section.visible and not card._result_section.visible
            texts += [card._header.object, card._error.object, card._error_text.object]
        elif isinstance(card, ModelCallCard):
            texts += [card._header.object, card._details.object, card._error.object]
        else:
            texts.append(str(getattr(card, "object", "")))
    blob = "\n".join(str(t or "") for t in texts)
    assert SECRET not in blob and "Paid revenue" not in blob and "Checking." not in blob
    assert "run_sql" in blob and "gpt-4o" in blob and "The question is hidden." in blob
    assert "Added revenue" not in blob and "Dashboard" in blob


def test_span_view_builds_long_runs_in_pages():
    events = [ev.RunStarted(run_id="r", timestamp=0)]
    for i in range(120):
        events += [
            ev.ToolCallStart(tool_call_id=f"t{i}", tool_name="step", run_id="r", timestamp=i),
            ev.ToolCallResult(tool_call_id=f"t{i}", content="ok", run_id="r", timestamp=i + 0.5),
        ]
    view = SpanView(RunState.from_events(events), run_id="r", row_limit=50)
    trace = view._trace
    assert len(trace._rows.objects) == 50 and len(trace._cards) == 50
    assert trace._more.visible and trace._more.label == "Show 50 more steps of 70"
    trace._show_more()
    assert len(trace._rows.objects) == 100 and trace._more.label == "Show 20 more steps"
    trace._show_more()
    assert len(trace._rows.objects) == 120 and not trace._more.visible


def test_span_view_follows_a_run_in_progress():
    s = RunState(flush_interval=0)
    s.apply(ev.RunStarted(run_id="r", input="Go", timestamp=0))
    view = SpanView(s, run_id="r")
    assert "Running" in view._header.object and not view._unrecorded.visible
    s.apply(ev.ToolCallStart(tool_call_id="t", tool_name="step", run_id="r", timestamp=1))
    s.apply(ev.ArtifactPublished(artifact_id="a", value=1, run_id="r", tool_call_id="t"))
    assert view._trace.steps == 1 and len(view._published.items) == 1
    s.apply(ev.RunFinished(run_id="r", timestamp=2))
    assert "Finished" in view._header.object


def test_clicking_a_published_version_calls_on_artifact():
    s = state()
    calls = []
    view = SpanView(s, run_id="r1", on_artifact=lambda a, v: calls.append((a, v)))
    view._on_published_click(view._published.items[1])
    assert calls == [("doc", 2)]
    s.artifacts.active = None
    SpanView(s, run_id="r1")._on_published_click({"artifact_id": "doc", "version": 1})
    assert s.artifacts.active == "doc"


def test_chat_details_link_calls_on_show_run():
    s = state()
    shown = []
    chat = AgentChat(s, readonly=True, on_show_run=shown.append)
    assert chat._footers
    (message,) = chat._footers.values()
    row = message.footer_objects[0]
    usage, link = row.objects
    assert link.label == "Details"
    link.clicks += 1
    chat._show_run("r1")
    assert "r1" in shown
    chat.on_show_run = None
    assert message.footer_objects[0] is not row and "run-usage" in message.footer_objects[0].css_classes


def test_history_rows_offer_how_was_this_made():
    s = state()
    shown = []
    view = ArtifactView(s.artifacts["doc"], state=s, history=True, on_show_run=shown.append)
    items = view._timeline._list.items
    actions = [[a["action"] for a in i.get("actions", [])] for i in items]
    assert all("show_run" in a for a in actions)
    run_item = next(i for i in items if i.get("run_id") == "r1")
    view._timeline._on_show_run_action(run_item)
    assert shown == ["r1"]
    view.on_show_run = None
    assert not any("show_run" in [a["action"] for a in i.get("actions", [])] for i in view._timeline._list.items)


def test_artifact_view_show_version_opens_the_history_at_it():
    s = state()
    view = ArtifactView(s.artifacts["doc"], state=s, history=False)
    view.show_version(1)
    assert view.history and s.artifacts["doc"].current == 1
    with pytest.raises(ValueError):
        view.show_version(9)


def test_drawer_show_run_opens_a_span_dialog_and_versions_in_the_artifacts_view():
    s = state()
    drawer = AgentDrawer(state=s)
    assert drawer.chat.on_show_run == drawer.show_run
    drawer.show_run("r1")
    assert drawer._span_dialog.open and drawer.span_view.run_id == "r1"
    first = drawer.span_view
    drawer.show_run("r2")
    assert drawer.span_view is first and first.run_id == "r2"
    drawer._open_version("doc", 1)
    assert not drawer._span_dialog.open and drawer.open and drawer.tab == "artifacts"
    assert s.artifacts["doc"].current == 1


def test_span_view_includes_the_steps_of_child_runs():
    s = RunState.from_events([
        ev.RunStarted(run_id="r", timestamp=0),
        ev.RunStarted(run_id="sub", parent_run_id="r", timestamp=1),
        ev.ToolCallStart(tool_call_id="t", tool_name="step", run_id="sub", timestamp=1),
        ev.RunFinished(run_id="sub", timestamp=2),
        ev.RunFinished(run_id="r", timestamp=3),
    ])
    assert SpanView(s, run_id="r")._trace.steps == 1
    assert [t.tool_call_id for t in s.span("r").tool_calls] == ["t"]
