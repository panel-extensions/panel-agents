import pytest

from panel_agents import HistoryRecorder, RunState, load_history, load_state
from panel_agents import events as ev
from panel_agents.components._timeline import group_versions, version_label
from panel_agents.components.artifacts import ArtifactView
from panel_agents.recording import dump_events


def agent_run(state, run_id, prompt, *ops, artifact_id="doc"):
    state.apply(ev.RunStarted(run_id=run_id, agent_name="assistant", input=prompt))
    for op in ops:
        publish(state, op, run_id=run_id, actor="agent", artifact_id=artifact_id)
    state.apply(ev.RunFinished(run_id=run_id))


def publish(state, op, run_id=None, actor="user", user=None, artifact_id="doc", applied=True):
    value = [*(state.artifacts[artifact_id].latest.value if artifact_id in state.artifacts else []), op]
    state.apply(ev.ArtifactPublished(
        artifact_id=artifact_id, value=value, kind="json", applied=applied, run_id=run_id, actor=actor, user=user,
        metadata={"op": op},
    ))


def make_state():
    state = RunState(flush_interval=0)
    agent_run(state, "r1", "Build the dashboard", "add_node", "connect", "place_tile")
    publish(state, "move_tile", user="ann")
    agent_run(state, "r2", "Add a refund table", "add_node", "place_tile")
    return state


def make_view(state, user="ann"):
    applied = []

    def on_apply(artifact, version):
        applied.append(version.version)

    view = ArtifactView(state.artifacts["doc"], state=state, on_apply=on_apply, user=user)
    return view, applied


def rows(view):
    return [row for group in view._timeline._list.items for row in group["items"] if "version" in row]


def expand_all(view):
    view._timeline._list.expanded = [(i,) for i in range(len(view._timeline._list.items))]


def test_versions_are_grouped_by_run():
    state = make_state()
    groups = group_versions(state.artifacts["doc"].versions)
    assert [(g.run_id, g.actor, g.user, [v.version for v in g.versions]) for g in groups] == [
        ("r1", "agent", None, [1, 2, 3]), (None, "user", "ann", [4]), ("r2", "agent", None, [5, 6]),
    ]


def test_interleaved_edits_split_a_run():
    state = RunState(flush_interval=0)
    publish(state, "add_node", run_id="r1", actor="agent")
    publish(state, "move_tile", user="ann")
    publish(state, "place_tile", run_id="r1", actor="agent")
    publish(state, "resize", user="bob")
    publish(state, "resize", user="bob")
    groups = group_versions(state.artifacts["doc"].versions)
    assert [(g.run_id, g.user, g.first, g.last) for g in groups] == [
        ("r1", None, 1, 1), (None, "ann", 2, 2), ("r1", None, 3, 3), (None, "bob", 4, 5),
    ]


def test_version_labels_come_from_metadata():
    state = RunState(flush_interval=0)
    for metadata in ({"label": "Placed the chart", "op": "place_tile"}, {"op": "connect"}, {}, {"label": "  "}):
        state.apply(ev.ArtifactPublished(artifact_id="doc", value=1, metadata=metadata))
    assert [version_label(v) for v in state.artifacts["doc"].versions] == [
        "Placed the chart", "connect", "Version 3", "Version 4",
    ]


def test_timeline_lists_runs_newest_first():
    state = make_state()
    view, _ = make_view(state)
    items = view._timeline._list.items
    assert [(i["label"], i["icon"]) for i in items] == [
        ("Add a refund table", "smart_toy"), ("You edited", "person"), ("Build the dashboard", "smart_toy"),
    ]
    assert [r["version"] for r in items[0]["items"]] == [6, 5]
    assert items[0]["secondary"].endswith("2 versions")
    assert view._timeline._list.expanded == [(0,)]
    assert view._history_open and view._history_panel.visible
    assert [r["label"] for r in items[2]["items"]] == ["…"]
    expand_all(view)
    items = view._timeline._list.items
    assert [r["label"] for r in items[2]["items"]] == ["place_tile", "connect", "add_node"]
    assert rows(view)[0]["secondary"].endswith("applied") and "actions" not in rows(view)[0]
    assert rows(view)[1]["actions"][0]["action"] == "restore"
    assert items[0]["actions"][0]["label"] == "Undo run" and items[0]["before"] == 4
    assert items[1]["actions"][0]["label"] == "Go back to before this run"
    assert "actions" not in items[2]


def test_other_users_and_unnamed_users_are_labelled():
    state = RunState(flush_interval=0)
    publish(state, "a", user="bob")
    publish(state, "b")
    with state.user_run("Arranged the tiles", user="ann"):
        state.publish_artifact("doc", [1], applied=True)
    view, _ = make_view(state)
    assert [i["label"] for i in view._timeline._list.items] == ["Arranged the tiles", "Edited", "bob edited"]


def test_a_new_run_expands_and_folds_the_previous_one():
    state = make_state()
    view, _ = make_view(state)
    timeline = view._timeline
    timeline._list.expanded = [(0,), (2,)]
    agent_run(state, "r3", "Sort the table", "set_config")
    assert [i["label"] for i in timeline._list.items][:2] == ["Sort the table", "Add a refund table"]
    assert timeline._list.expanded == [(0,), (3,)]


def test_clicking_a_row_views_the_version_and_compares_with_the_applied_one():
    state = make_state()
    view, applied = make_view(state)
    view._timeline._list.active = (0, 1)
    artifact = state.artifacts["doc"]
    assert artifact.current == 5 and artifact.applied == 6 and applied == []
    assert (view.base, view.target) == (6, 5)
    assert view._history_button.label == "v5"
    assert view._apply.label == "Revert to v5"


async def test_undo_and_redo_move_the_applied_version():
    state = make_state()
    view, applied = make_view(state)
    artifact = state.artifacts["doc"]
    assert view.can_undo and not view.can_redo
    assert await view.undo()
    assert artifact.applied == 5 and artifact.current == 5
    assert await view.undo()
    assert artifact.applied == 4 and view.can_redo
    assert await view.redo()
    assert artifact.applied == 5
    assert applied == [5, 4, 5]
    assert [(e.version, e.user) for e in state.events if isinstance(e, ev.ArtifactApplied)] == [(5, "ann"), (4, "ann"), (5, "ann")]
    assert len(artifact.versions) == 6


async def test_undo_run_goes_back_before_the_run_and_redo_steps_through_it():
    state = make_state()
    view, applied = make_view(state)
    artifact = state.artifacts["doc"]
    assert await view.undo_run()
    assert artifact.applied == 4 and applied == [4]
    assert artifact.redo_version == 5
    assert await view.redo() and await view.redo()
    assert artifact.applied == 6 and not view.can_redo
    with pytest.raises(ValueError, match="did not change the applied version"):
        await view.undo_run("r1")
    with pytest.raises(ValueError, match="No version"):
        await view.undo_run("missing")


async def test_undoing_a_run_also_undoes_edits_made_after_it_started():
    state = RunState(flush_interval=0)
    publish(state, "base")
    publish(state, "add_node", run_id="r1", actor="agent")
    publish(state, "move_tile", user="ann")
    publish(state, "place_tile", run_id="r1", actor="agent")
    view, applied = make_view(state)
    assert await view.undo_run("r1")
    assert state.artifacts["doc"].applied == 1
    assert [await view.redo() for _ in range(3)] == [True] * 3
    assert applied == [1, 2, 3, 4]


async def test_undo_after_a_revert_returns_to_where_the_revert_started():
    state = make_state()
    view, applied = make_view(state)
    artifact = state.artifacts["doc"]
    assert await view.apply_version(2)
    assert artifact.undo_version == 6 and artifact.redo_version is None
    assert await view.undo()
    assert artifact.applied == 6 and artifact.redo_version == 2
    assert await view.redo()
    assert artifact.applied == 2


async def test_a_new_version_ends_redo():
    state = make_state()
    view, _ = make_view(state)
    await view.undo()
    assert view.can_redo
    publish(state, "proposal", run_id="r9", actor="agent", applied=False)
    assert view.can_redo and state.artifacts["doc"].applied == 5
    publish(state, "set_config", run_id="r9", actor="agent")
    assert not view.can_redo and state.artifacts["doc"].applied == 8
    assert state.artifacts["doc"].undo_version == 5


async def test_restoring_from_the_timeline_applies_and_views_the_version():
    state = make_state()
    view, applied = make_view(state)
    expand_all(view)
    header = view._timeline._list.items[0]
    await view._timeline._on_undo_action(header)
    assert state.artifacts["doc"].redo_version == 5
    await view._timeline._on_restore_action(rows(view)[3])
    assert applied == [4, 3]
    assert state.artifacts["doc"].current == 3
    assert state.artifacts["doc"].redo_version is None and state.artifacts["doc"].undo_version == 4
    assert [e.action for e in state.events if isinstance(e, ev.ArtifactApplied)] == ["undo", None]


async def test_undo_needs_on_apply_and_a_document():
    state = make_state()
    view = ArtifactView(state.artifacts["doc"], state=state)
    assert not view.can_undo and not view._timeline_controls.visible
    assert "actions" not in rows(view)[1]
    state.apply(ev.ArtifactPublished(artifact_id="out", value=1))
    view = ArtifactView(state.artifacts["out"], state=state, on_apply=lambda *a: None)
    assert not view.can_undo and not view._history_open
    view.history = True
    assert view._history_panel.visible


async def test_a_failed_apply_changes_nothing():
    state = make_state()

    def refuse(artifact, version):
        raise RuntimeError("locked")

    view = ArtifactView(state.artifacts["doc"], state=state, on_apply=refuse)
    assert not await view.undo()
    assert state.artifacts["doc"].applied == 6 and view._apply_error.visible and view.can_undo


async def test_replay_reproduces_undo_state_and_timeline(tmp_path):
    state = make_state()
    view, _ = make_view(state)
    await view.undo_run()
    await view.redo()
    path = tmp_path / "run.jsonl"
    dump_events(state.events, path)
    replayed = load_state(path)
    live, loaded = state.artifacts["doc"], replayed.artifacts["doc"]
    assert (loaded.applied, loaded.undo_version, loaded.redo_version) == (live.applied, live.undo_version, live.redo_version)
    replayed_view, _ = make_view(replayed)
    expand_all(view)
    expand_all(replayed_view)
    strip = lambda items: [(i["label"], [r["secondary"] for r in i["items"]]) for i in items]  # noqa: E731
    assert strip(replayed_view._timeline._list.items) == strip(view._timeline._list.items)


def test_history_recorder_keeps_versions_and_their_runs(tmp_path):
    state = RunState(flush_interval=0)
    path = tmp_path / "history.jsonl"
    recorder = HistoryRecorder(state, path, artifact_ids=["doc"])
    state.add_user_message("hello")
    state.apply(ev.RunStarted(run_id="r0", input="Just chatting"))
    state.apply(ev.RunFinished(run_id="r0"))
    agent_run(state, "r1", "Build the dashboard", "add_node", "connect")
    agent_run(state, "r2", "A table", "x", artifact_id="other")
    publish(state, "move_tile", user="ann")
    state.apply_artifact_version("doc", 1, user="ann")
    recorder.close()

    stored = [e.type for e in load_state(path).events]
    assert stored == [
        "RUN_STARTED", "ARTIFACT_PUBLISHED", "ARTIFACT_PUBLISHED", "RUN_FINISHED", "ARTIFACT_PUBLISHED", "ARTIFACT_APPLIED",
    ]

    later = RunState(flush_interval=0)
    written = []
    HistoryRecorder(later, written.append)
    restored = load_history(path, later)
    assert restored is later
    doc = later.artifacts["doc"]
    assert ([v.value for v in doc.versions], doc.applied) == ([v.value for v in state.artifacts["doc"].versions], 1)
    assert later.runs["r1"].input == "Build the dashboard" and not later.messages
    assert written == [{"panel_agents_recording": 1}]
    publish(later, "resize", user="ann")
    assert [r["type"] for r in written[1:]] == ["ARTIFACT_PUBLISHED"]


async def test_a_child_run_stays_part_of_its_agent_run(tmp_path):
    state = RunState(flush_interval=0)
    path = tmp_path / "history.jsonl"
    recorder = HistoryRecorder(state, path)
    publish(state, "start", user="ann")
    state.apply(ev.RunStarted(run_id="r1", agent_name="assistant", input="Build the dashboard"))
    publish(state, "add_node", run_id="r1", actor="agent")
    state.apply(ev.RunStarted(run_id="r1-batch", parent_run_id="r1", input="Arrange"))
    publish(state, "place_tile", run_id="r1-batch", actor="agent")
    state.apply(ev.RunFinished(run_id="r1-batch"))
    publish(state, "connect", run_id="r1", actor="agent")
    state.apply(ev.RunFinished(run_id="r1"))
    recorder.close()

    groups = group_versions(state.artifacts["doc"].versions, state)
    assert [(g.run_id, g.first, g.last) for g in groups] == [(None, 1, 1), ("r1", 2, 4)]
    view, applied = make_view(state)
    assert view._timeline._list.items[0]["label"] == "Build the dashboard"
    state.artifacts["doc"].current = 3
    await view.apply_version(3)
    assert await view.undo_run()
    assert applied[-1] == 1

    restored = load_history(path)
    assert restored.runs["r1-batch"].parent_run_id == "r1"
    restored_groups = group_versions(restored.artifacts["doc"].versions, restored)
    assert [(g.run_id, g.first, g.last) for g in restored_groups] == [(None, 1, 1), ("r1", 2, 4)]


def test_load_history_finishes_runs_the_stored_session_left_open(tmp_path):
    state = RunState(flush_interval=0)
    path = tmp_path / "history.jsonl"
    with HistoryRecorder(state, path):
        state.apply(ev.RunStarted(run_id="r1", input="Interrupted"))
        publish(state, "add_node", run_id="r1", actor="agent")
    restored = load_history(path)
    assert restored.runs["r1"].status == "finished" and not restored.running
    assert restored.runs["r1"].ended_at == state.artifacts["doc"].latest.created_at
    assert len(restored.artifacts["doc"].versions) == 1


async def test_undo_run_skips_a_proposal_published_before_the_run():
    state = RunState(flush_interval=0)
    publish(state, "base")
    publish(state, "proposal", actor="agent", applied=False)
    publish(state, "a", run_id="r1", actor="agent")
    publish(state, "b", run_id="r1", actor="agent")
    view, applied = make_view(state)
    assert view._timeline._list.items[0]["before"] == 1
    assert await view.undo_run("r1")
    artifact = state.artifacts["doc"]
    assert applied == [1] and artifact.applied == 1 and artifact.redo_version == 3
    assert await view.redo() and artifact.applied == 3


async def test_a_run_that_only_proposed_cannot_be_undone():
    state = RunState(flush_interval=0)
    publish(state, "base")
    publish(state, "proposal", run_id="r1", actor="agent", applied=False)
    view, applied = make_view(state)
    assert "actions" not in view._timeline._list.items[0]
    with pytest.raises(ValueError, match="did not change"):
        await view.undo_run("r1")
    assert applied == []


async def test_undo_opens_the_group_of_the_applied_version():
    state = make_state()
    view, _ = make_view(state)
    assert view._timeline._list.expanded == [(0,)]
    await view.undo_run()
    items, expanded = view._timeline._list.items, view._timeline._list.expanded
    assert (1,) in expanded and items[1]["items"][0]["version"] == 4
    assert view._timeline._list.active == (1, 0)


def test_a_long_group_shows_its_latest_versions_first():
    from panel_agents.components._timeline import ROW_LIMIT

    state = RunState(flush_interval=0)
    for i in range(ROW_LIMIT + 7):
        publish(state, f"edit {i}", user="ann")
    view, _ = make_view(state)
    rows = view._timeline._list.items[0]["items"]
    assert len(rows) == ROW_LIMIT + 1 and rows[-1]["label"] == "Show 7 more"
    view._timeline._on_click(rows[-1])
    rows = view._timeline._list.items[0]["items"]
    assert len(rows) == ROW_LIMIT + 7 and rows[-1]["version"] == 1


async def test_apply_validates_its_action_first():
    state = make_state()
    view, applied = make_view(state)
    with pytest.raises(ValueError, match="action"):
        await view.apply_version(2, action="rewind")
    assert applied == []
    with pytest.raises(ValueError, match="needs an artifact"):
        await ArtifactView(None, on_apply=lambda *a: None).undo_run("r1")


def test_history_recorder_attached_mid_run_keeps_the_run():
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r1", input="Already running"))
    records = []
    HistoryRecorder(state, records.append)
    publish(state, "add_node", run_id="r1", actor="agent")
    assert [r["type"] for r in records[1:]] == ["RUN_STARTED", "ARTIFACT_PUBLISHED"]
    assert load_history(records).runs["r1"].input == "Already running"
