import gc
import io
import json
import random
import weakref

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


class Value:
    """A version value that can be watched for being freed."""


def _published_values(state, n):
    values = [Value() for _ in range(n)]
    for value in values:
        state.apply(ev.ArtifactPublished(artifact_id="doc", value=value, applied=True))
    return [weakref.ref(value) for value in values]


@pytest.mark.parametrize("max_events, freed", [(None, False), (2, True)])
def test_values_are_freed_once_the_events_let_go_too(max_events, freed):
    state = RunState(flush_interval=0, max_events=max_events)
    refs = _published_values(state, 4)
    state.drop_artifact_versions("doc", before=4)
    gc.collect()
    assert all((ref() is None) is freed for ref in refs[:3])
    assert refs[3]() is not None


def test_only_documents_drop_versions():
    state = RunState(flush_interval=0)
    for value in ([1], [2], [3]):
        state.apply(ev.ArtifactPublished(artifact_id="doc", value=value))
    with pytest.raises(ValueError, match="not a document"):
        state.drop_artifact_versions("doc", before=100)
    assert state.artifacts["doc"].latest.value == [3]


def test_a_view_comparing_dropped_versions_moves_to_kept_ones():
    state = make_state()
    view = ArtifactView(state.artifacts["doc"], state=state, on_apply=lambda a, v: None)
    view.param.update(target=2, base=1)
    state.drop_artifact_versions("doc", before=4)
    assert not view.versions[view.target - 1].dropped
    assert not view.versions[view.base - 1].dropped
    assert view.target == 6 and view.base == 5


def _stacks(artifact):
    return (
        [v.dropped for v in artifact.versions], artifact.current, artifact.applied, artifact.undo_version,
        artifact.redo_version, artifact.redo_run_version, list(artifact._undo), list(artifact._redo),
        [list(step) for step in artifact._redo_steps],
    )


def _publish(state, run_id=None, actor="user", applied=True):
    count = len(state.artifacts["doc"].versions) if "doc" in state.artifacts else 0
    state.apply(ev.ArtifactPublished(
        artifact_id="doc", value=[count], kind="json", applied=applied, run_id=run_id, actor=actor,
    ))


@pytest.mark.parametrize("seed", range(150))
def test_random_edits_undos_and_drops_keep_the_history_consistent(seed):
    rnd = random.Random(seed)
    state = RunState(flush_interval=0)
    _publish(state)
    runs = []
    for i in range(25):
        artifact = state.artifacts["doc"]
        op = rnd.choice(["run", "user", "proposal", "undo", "redo", "undo_run", "redo_run", "hand", "drop", "drop"])
        if op == "run":
            run_id = f"r{i}"
            runs.append(run_id)
            state.apply(ev.RunStarted(run_id=run_id, agent_name="a"))
            for _ in range(rnd.randint(1, 3)):
                _publish(state, run_id=run_id, actor="agent", applied=rnd.random() < 0.8)
            state.apply(ev.RunFinished(run_id=run_id))
        elif op == "user":
            _publish(state)
        elif op == "proposal":
            _publish(state, applied=False)
        elif op == "drop":
            state.drop_artifact_versions("doc", rnd.randint(1, len(artifact.versions) + 2))
        elif op == "hand":
            number = rnd.randint(1, len(artifact.versions))
            if artifact.versions[number - 1].dropped:
                with pytest.raises(ValueError):
                    state.apply_artifact_version("doc", number)
            else:
                state.apply_artifact_version("doc", number)
        elif op in ("undo", "redo", "redo_run"):
            getattr(state, op)("doc")
        elif op == "undo_run" and runs:
            try:
                state.undo_run("doc", rnd.choice(runs))
            except ValueError:
                pass
        dropped = {v.version for v in artifact.versions if v.dropped}
        assert artifact.applied not in dropped and artifact.current not in dropped
        assert not {artifact.undo_version, artifact.redo_version, artifact.redo_run_version} & dropped
        assert not (set(artifact._undo) | set(artifact._redo)) & dropped
        timeline = [*artifact._undo, artifact.applied, *artifact._redo[::-1]]
        assert len(timeline) == len(set(timeline))
    replay = RunState.from_events(state.events)
    assert _stacks(replay.artifacts["doc"]) == _stacks(state.artifacts["doc"])
