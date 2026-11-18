import io
import json

import pytest

from panel_agents import HistoryRecorder, RunState, load_history
from panel_agents import events as ev
from panel_agents.components._timeline import group_versions
from panel_agents.components.artifacts import ArtifactView

from .test_history import make_state, publish


def test_dropping_keeps_numbers_and_authors_and_frees_values():
    state = make_state()
    artifact = state.artifacts["doc"]
    assert state.drop_artifact_versions("doc", before=4) == [1, 2, 3]
    assert [v.version for v in artifact.versions] == [1, 2, 3, 4, 5, 6]
    assert [v.dropped for v in artifact.versions] == [True, True, True, False, False, False]
    assert all(v.value is None for v in artifact.versions[:3])
    assert [v.metadata["op"] for v in artifact.versions[:3]] == ["add_node", "connect", "place_tile"]
    assert artifact.versions[0].run_id == "r1" and artifact.versions[0].actor == "agent"
    assert artifact.applied == 6 and artifact.versions[5].value is not None


def test_undo_stops_at_the_oldest_version_kept():
    state = make_state()
    state.drop_artifact_versions("doc", before=5)
    assert state.undo("doc").version == 5
    assert state.undo("doc") is None
    assert state.redo_run("doc").version == 6


def test_a_run_whose_start_was_dropped_has_nothing_to_undo():
    state = make_state()
    state.drop_artifact_versions("doc", before=5)
    artifact = state.artifacts["doc"]
    assert artifact.undo_run_target("r2", state.runs) is None
    assert state.undo_run("doc", "r2") is None


def test_redo_steps_lose_the_dropped_versions():
    state = make_state()
    state.undo("doc")
    state.undo("doc")
    state.undo("doc")
    artifact = state.artifacts["doc"]
    assert artifact.applied == 3
    state.drop_artifact_versions("doc", before=5)
    assert artifact.applied == 3 and not artifact.versions[2].dropped
    assert artifact.redo_version == 5
    assert state.redo("doc").version == 5 and state.redo("doc").version == 6
    assert state.redo("doc") is None


def test_the_applied_version_is_never_dropped():
    state = make_state()
    state.undo("doc")
    assert state.drop_artifact_versions("doc", before=7) == [1, 2, 3, 4, 6]
    assert state.artifacts["doc"].applied_version.value is not None


def test_applying_a_dropped_version_is_refused():
    state = make_state()
    state.drop_artifact_versions("doc", before=3)
    with pytest.raises(ValueError, match="dropped"):
        state.apply_artifact_version("doc", 1)


def test_dropping_nothing_records_nothing():
    state = make_state()
    events = len(state.events)
    assert state.drop_artifact_versions("doc", before=1) == []
    assert len(state.events) == events
    with pytest.raises(ValueError):
        state.drop_artifact_versions("doc", before=0)


def test_a_replay_drops_the_same_versions():
    state = make_state()
    state.undo("doc")
    state.drop_artifact_versions("doc", before=4)
    publish(state, "resize")
    replay = RunState.from_events(state.events)
    live, replayed = state.artifacts["doc"], replay.artifacts["doc"]
    assert [v.dropped for v in replayed.versions] == [v.dropped for v in live.versions]
    assert [v.value for v in replayed.versions] == [v.value for v in live.versions]
    assert (replayed.applied, replayed.undo_version, replayed.redo_version) == (
        live.applied, live.undo_version, live.redo_version,
    )


def test_a_history_recorder_keeps_the_drop():
    state = RunState(flush_interval=0)
    sink = io.StringIO()
    HistoryRecorder(state, sink, artifact_ids=["doc"])
    for op in ("a", "b", "c"):
        publish(state, op)
    state.drop_artifact_versions("doc", before=3)
    records = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert records[-1]["type"] == "ARTIFACT_VERSIONS_DROPPED" and records[-1]["before"] == 3
    restored = load_history(records)
    assert [v.dropped for v in restored.artifacts["doc"].versions] == [True, True, False]


def test_views_leave_dropped_versions_out():
    state = make_state()
    view = ArtifactView(state.artifacts["doc"], state=state, on_apply=lambda a, v: None)
    state.drop_artifact_versions("doc", before=4)
    assert sorted(view._version_options().values()) == [4, 5, 6]
    groups = group_versions(state.artifacts["doc"].versions, state)
    assert [[v.version for v in g.versions] for g in groups] == [[4], [5, 6]]


async def test_a_view_refuses_to_apply_a_dropped_version():
    state = make_state()
    view = ArtifactView(state.artifacts["doc"], state=state, on_apply=lambda a, v: None)
    state.drop_artifact_versions("doc", before=4)
    with pytest.raises(ValueError, match="dropped"):
        await view.apply_version(2)


def test_tool_calls_no_longer_list_their_dropped_versions():
    state = RunState(flush_interval=0)
    state.apply(ev.ArtifactPublished(artifact_id="doc", value=[1], applied=True, tool_call_id="c1"))
    state.apply(ev.ArtifactPublished(artifact_id="doc", value=[2], applied=True, tool_call_id="c2"))
    state.drop_artifact_versions("doc", before=2)
    assert state.artifacts.published_by("c1") == []
    assert [v.version for _, v in state.artifacts.published_by("c2")] == [2]
