import pytest

from panel_agents import HistoryRecorder, Recorder, RunState, load_history, load_state
from panel_agents import events as ev

from .test_state import run_events


def edits(state, n):
    state.apply(ev.RunStarted(run_id="r1", agent_name="assistant", input="edit"))
    for i in range(n):
        state.apply(ev.ArtifactPublished(artifact_id="doc", value=[i], kind="json", applied=True, run_id="r1", actor="agent"))
    state.apply(ev.RunFinished(run_id="r1"))
    for i in range(2, n + 1, 3):
        state.apply(ev.ArtifactApplied(artifact_id="doc", version=i, action=None))


def summary(state):
    doc = state.artifacts["doc"]
    return (
        len(doc.versions), doc.applied, doc.undo_version, doc.redo_version,
        [(r.run_id, r.status) for r in state.runs.values()],
    )


def test_events_keep_all_by_default():
    state = RunState(flush_interval=0)
    edits(state, 10)
    assert len(state.events) == 15 and state.dropped_events == 0


def test_max_events_keeps_the_latest_and_leaves_the_state_alone():
    bounded, full = RunState(flush_interval=0, max_events=5), RunState(flush_interval=0)
    edits(bounded, 30)
    edits(full, 30)
    assert [(e.type, getattr(e, "version", None)) for e in bounded.events] == [
        (e.type, getattr(e, "version", None)) for e in full.events[-5:]
    ]
    assert bounded.dropped_events == len(full.events) - 5
    assert summary(bounded) == summary(full)


def test_max_events_also_bounds_streamed_messages():
    state = RunState(flush_interval=0, max_events=3)
    state.apply_many(run_events())
    assert len(state.events) == 3
    full = RunState.from_events(run_events())
    assert state.items[1].content == full.items[1].content
    assert state.runs["run-1"].cost_usd == full.runs["run-1"].cost_usd


def test_a_recorder_keeps_the_full_log_of_a_bounded_state(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0, max_events=4)
    with Recorder(state, path):
        edits(state, 20)
        state.apply_artifact_version("doc", 18, action="undo")
    replayed = load_state(path)
    assert replayed.dropped_events == 0 and len(replayed.events) == len(state.events) + state.dropped_events
    assert summary(replayed) == summary(state)


def test_a_history_recorder_restores_a_bounded_state(tmp_path):
    path = tmp_path / "history.jsonl"
    state = RunState(flush_interval=0, max_events=2)
    with HistoryRecorder(state, path):
        edits(state, 12)
    assert summary(load_history(path)) == summary(state)


def test_changing_max_events_trims_or_stops_dropping():
    state = RunState(flush_interval=0)
    edits(state, 10)
    events = state.events
    state.max_events = 4
    assert state.events == events[-4:] and state.dropped_events == len(events) - 4
    state.max_events = None
    edits(state, 3)
    assert len(state.events) == 4 + 6 and state.dropped_events == len(events) - 4


def test_reset_clears_the_dropped_count():
    state = RunState(flush_interval=0, max_events=2)
    edits(state, 5)
    state.reset()
    assert state.events == [] and state.dropped_events == 0
    with pytest.raises(ValueError):
        state.max_events = 0
