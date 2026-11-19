import asyncio

import pandas as pd
import panel as pn
import pytest

from panel_agents import ActionRegistry, ArtifactAction, ArtifactCell, RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat


def question(run_id, text, agent="Analyst"):
    return [
        ev.RunStarted(run_id=run_id, agent_name=agent),
        ev.TextMessageStart(message_id=f"{run_id}-q", role="user", run_id=run_id),
        ev.TextMessageContent(message_id=f"{run_id}-q", delta=text, run_id=run_id),
        ev.TextMessageEnd(message_id=f"{run_id}-q", run_id=run_id),
    ]


def answer(run_id, text):
    return [
        ev.TextMessageStart(message_id=f"{run_id}-a", run_id=run_id),
        ev.TextMessageContent(message_id=f"{run_id}-a", delta=text, run_id=run_id),
        ev.TextMessageEnd(message_id=f"{run_id}-a", run_id=run_id),
    ]


def table(run_id, artifact_id="revenue", **fields):
    return ev.ArtifactPublished(
        artifact_id=artifact_id, value=pd.DataFrame({"region": ["East"], "total": [1]}), kind="table",
        title=artifact_id, run_id=run_id, **fields,
    )


def is_cells(message):
    return "run-artifacts-message" in message.css_classes


def messages(chat):
    """The feed's messages other than rows of cells."""
    return [m for m in chat.feed.objects if not is_cells(m)]


def cells_after(chat, message):
    """The cells shown right after *message*."""
    objects = chat.feed.objects
    i = objects.index(message)
    following = objects[i + 1] if i + 1 < len(objects) else None
    return list(following.object.objects[0].objects) if following is not None and is_cells(following) else []


def test_document_layout_shows_a_runs_artifacts_after_its_messages():
    state = RunState.from_events([
        *question("r1", "Revenue by region?"), table("r1"), *answer("r1", "East leads."), ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state, layout="document", readonly=True)
    user, reply = messages(chat)
    assert user.placement == "left"
    assert cells_after(chat, user) == []
    (cell,) = chat.cells.values()
    assert isinstance(cell, ArtifactCell)
    assert cells_after(chat, reply) == [cell.__panel__()]
    row = chat.feed.objects[-1]
    assert (row.user, row.placement) == (reply.user, reply.placement)


def test_chat_layout_keeps_artifacts_out_of_the_feed():
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), *answer("r1", "Done."), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, readonly=True)
    assert chat.cells == {}
    assert chat.feed.objects[0].placement == "right"


def test_a_run_with_only_a_question_shows_its_artifacts_under_the_question():
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True)
    (user,) = messages(chat)
    assert len(cells_after(chat, user)) == 1


def test_cells_stay_where_they_were_put_while_the_run_streams():
    state = RunState(flush_interval=0)
    chat = AgentChat(state, layout="document", readonly=True)
    state.apply_many([*question("r1", "Revenue?"), table("r1", "a")])
    (user,) = messages(chat)
    row = chat.feed.objects[-1]
    first = cells_after(chat, user)
    assert len(first) == 1
    flex = row.object.objects[0]
    state.apply_many(answer("r1", "East leads."))
    state.apply(table("r1", "b"))
    state.apply(ev.RunFinished(run_id="r1"))
    user, reply = messages(chat)
    assert chat.feed.objects == [user, row, reply]
    assert row.object.objects[0] is flex
    assert flex.objects[0] is first[0] and len(flex.objects) == 2


def test_a_run_past_max_cells_shows_the_rest_on_request():
    state = RunState(flush_interval=0)
    chat = AgentChat(state, layout="document", readonly=True, max_cells=2)
    state.apply_many(question("r1", "Many?"))
    for i in range(5):
        state.apply(table("r1", f"t{i}"))
    row = chat.feed.objects[-1]
    flex, more = row.object.objects
    assert len(flex.objects) == 2 and sorted(chat.cells) == ["t0", "t1"]
    assert more.visible and more.label == "Show 3 more"
    state.apply(table("r1", "t5"))
    assert more.label == "Show 4 more" and len(flex.objects) == 2
    more.clicks += 1
    assert len(flex.objects) == 6 and not more.visible
    state.apply(table("r1", "t6"))
    assert len(flex.objects) == 7


def test_documents_and_artifacts_outside_runs_are_not_inline():
    state = RunState.from_events([
        *question("r1", "Revenue?"),
        table("r1", artifact_id="report", applied=True),
        table(None, artifact_id="loose"),
        table("r1", artifact_id="kept"),
        ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state, layout="document", readonly=True)
    assert list(chat.cells) == ["kept"]


def test_inline_artifacts_predicate_chooses_the_cells():
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1", "a"), table("r1", "b"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, inline_artifacts=lambda a: a.artifact_id == "b")
    assert list(chat.cells) == ["b"]
    chat.inline_artifacts = None
    assert sorted(chat.cells) == ["a", "b"]


def test_artifacts_of_child_runs_show_under_the_top_level_run():
    state = RunState.from_events([
        *question("r1", "Revenue?"),
        ev.RunStarted(run_id="child", parent_run_id="r1"),
        table("child"),
        ev.RunFinished(run_id="child"),
        *answer("r1", "Done."),
        ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state, layout="document", readonly=True)
    assert len(cells_after(chat, messages(chat)[-1])) == 1


def test_a_new_version_renders_the_cell_again():
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    rendered = []
    chat = AgentChat(
        state, layout="document", readonly=True,
        cell_renderer=lambda artifact: rendered.append(artifact.latest.version) or pn.pane.Str(str(len(artifact.versions))),
    )
    state.publish_artifact("revenue", pd.DataFrame({"x": [2]}), kind="table", run_id="r1")
    assert rendered == [1, 2]
    (cell,) = chat.cells.values()
    assert cell._body.objects[0].object == "2"


def test_usage_stays_under_the_last_message():
    state = RunState.from_events([
        *question("r1", "Revenue?"), table("r1"), *answer("r1", "Done."),
        ev.Usage(run_id="r1", input_tokens=10, output_tokens=5, model="test"),
        ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state, layout="document", readonly=True)
    reply = messages(chat)[-1]
    (usage,) = reply.footer_objects
    assert "run-usage" in usage.css_classes


def test_actions_offer_a_button_or_their_done_label():
    actions = ActionRegistry()
    placed = []

    @actions.action
    def add_to_dashboard(name: str):
        """Show an output on the dashboard."""
        placed.append(name)

    spec = ArtifactAction(
        "add_to_dashboard", "Add to dashboard", icon="add",
        arguments=lambda artifact: {"name": artifact.title} if artifact.kind == "table" else None,
        done=lambda artifact: "On dashboard" if artifact.title in placed else None,
    )
    state = RunState.from_events([
        *question("r1", "Revenue?"), table("r1"),
        ev.ArtifactPublished(artifact_id="note", value="hi", kind="text", run_id="r1"),
        ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=actions, artifact_actions=[spec])
    revenue, note = chat.cells["revenue"], chat.cells["note"]
    assert note._actions.objects == []
    (button,) = revenue._actions.objects
    assert button.label == "Add to dashboard"
    asyncio.run(revenue._invoke(spec, button))
    assert placed == ["revenue"]
    (chip,) = revenue._actions.objects
    assert chip.label == "On dashboard"


def test_actions_missing_from_the_registry_are_left_out():
    spec = ArtifactAction("missing", "Do it")
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=ActionRegistry(), artifact_actions=[spec])
    assert chat.cells["revenue"]._actions.objects == []


def test_a_failing_action_says_so_on_the_cell():
    actions = ActionRegistry()

    @actions.action
    def explode():
        """Fail."""
        raise ValueError("no room")

    spec = ArtifactAction("explode", "Explode")
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=actions, artifact_actions=[spec])
    cell = chat.cells["revenue"]
    (button,) = cell._actions.objects
    asyncio.run(cell._invoke(spec, button))
    assert cell._error.visible and "no room" in cell._error.object
    assert not button.loading


def test_refresh_actions_reevaluates_done_labels():
    actions = ActionRegistry()
    actions.add(lambda: None, name="noop")
    done = set()
    spec = ArtifactAction("noop", "Add", done=lambda artifact: "Added" if artifact.artifact_id in done else None)
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=actions, artifact_actions=[spec])
    done.add("revenue")
    chat.refresh_actions()
    (chip,) = chat.cells["revenue"]._actions.objects
    assert chip.label == "Added"


def test_reset_releases_the_cells():
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True)
    (cell,) = chat.cells.values()
    state.reset()
    assert chat.cells == {}
    assert cell._watchers == []
    state.apply_many([*question("r2", "Again?"), table("r2", "other"), ev.RunFinished(run_id="r2")])
    assert list(chat.cells) == ["other"]


@pytest.mark.parametrize("layout", ["chat", "document"])
def test_input_is_kept_in_both_layouts(layout):
    chat = AgentChat(RunState(), layout=layout)
    assert chat.feed._widget is not None


def test_refreshing_keeps_controls_whose_state_is_unchanged():
    actions = ActionRegistry()
    actions.add(lambda: None, name="noop")
    done = set()
    spec = ArtifactAction("noop", "Add", done=lambda artifact: "Added" if artifact.artifact_id in done else None)
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=actions, artifact_actions=[spec])
    cell = chat.cells["revenue"]
    (button,) = cell._actions.objects
    chat.refresh_actions()
    assert cell._actions.objects == [button]
    done.add("revenue")
    chat.refresh_actions()
    (chip,) = cell._actions.objects
    chat.refresh_actions()
    assert cell._actions.objects == [chip]


def test_titles_and_action_errors_are_plain_text():
    actions = ActionRegistry()

    @actions.action
    def explode():
        """Fail."""
        raise ValueError('<img src=x onerror="alert(1)">')

    spec = ArtifactAction("explode", "Explode")
    state = RunState.from_events([
        *question("r1", "Revenue?"), table("r1", artifact_id='<img src=x onerror="alert(1)">'), ev.RunFinished(run_id="r1"),
    ])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=actions, artifact_actions=[spec])
    (cell,) = chat.cells.values()
    assert cell._title.object.startswith("\\<img")
    (button,) = cell._actions.objects
    asyncio.run(cell._invoke(spec, button))
    assert "\\<img" in cell._error.object and "<img" not in cell._error.object.replace("\\<img", "")
    state.publish_artifact(cell.artifact.artifact_id, "x", kind="table", title="<b>bold</b>", run_id="r1")
    assert cell._title.object == "\\<b\\>bold\\</b\\>"


def test_a_failing_renderer_is_shown_in_its_cell_and_the_run_goes_on():
    def broken(artifact):
        raise RuntimeError("no view")

    state = RunState(flush_interval=0)
    chat = AgentChat(state, layout="document", readonly=True, cell_renderer=broken)
    state.apply_many([*question("r1", "Revenue?"), table("r1"), *answer("r1", "Done."), ev.RunFinished(run_id="r1")])
    (cell,) = chat.cells.values()
    (error,) = cell._body.objects
    assert "render-error" in " ".join(error.css_classes) and "no view" in error.object
    assert state.runs["r1"].status == "finished"


def test_a_string_from_the_renderer_is_inert_markdown():
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, cell_renderer=lambda a: "**bold** <img src=x>")
    (body,) = chat.cells["revenue"]._body.objects
    assert isinstance(body, pn.pane.Markdown) and body.renderer == "markdown-it"


def test_action_arguments_are_taken_when_clicked():
    actions = ActionRegistry()
    seen = []
    actions.add(lambda name: seen.append(name), name="place")
    current = {"name": "first"}
    spec = ArtifactAction("place", "Place", arguments=lambda artifact: dict(current))
    state = RunState.from_events([*question("r1", "Revenue?"), table("r1"), ev.RunFinished(run_id="r1")])
    chat = AgentChat(state, layout="document", readonly=True, action_registry=actions, artifact_actions=[spec])
    cell = chat.cells["revenue"]
    (button,) = cell._actions.objects
    current["name"] = "second"
    asyncio.run(cell._invoke(spec, button))
    assert seen == ["second"]


def test_a_call_whose_artifacts_are_cells_folds_and_has_no_chip():
    events = [*question("r1", "Revenue?")]
    for call in ("c1", "c2"):
        events += [
            ev.ToolCallStart(tool_call_id=call, tool_name="make_table", run_id="r1"),
            ev.ToolCallEnd(tool_call_id=call, run_id="r1"),
            ev.ToolCallResult(tool_call_id=call, content="ok", run_id="r1"),
            table("r1", artifact_id=f"t-{call}", tool_call_id=call),
        ]
    state = RunState.from_events([*events, *answer("r1", "Done."), ev.RunFinished(run_id="r1")])
    document = AgentChat(state, layout="document", readonly=True)
    assert len(document._groups) == 1
    card = document.view_for(state.tool_calls["c1"])
    assert not card._artifacts.visible
    chat = AgentChat(state, readonly=True)
    assert chat._groups == {}
    assert chat.view_for(state.tool_calls["c1"])._artifacts.visible
