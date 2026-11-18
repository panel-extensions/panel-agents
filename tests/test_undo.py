import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.artifacts import root_run_id
from panel_agents.components.artifacts import ArtifactView

from .test_history import agent_run, make_state, publish


def applied_events(state):
    return [(e.version, e.action, e.user) for e in state.events if isinstance(e, ev.ArtifactApplied)]


def stacks(artifact):
    return artifact.applied, artifact.undo_version, artifact.redo_version, artifact.redo_run_version


def test_undo_and_redo_without_a_view():
    state = make_state()
    artifact = state.artifacts["doc"]
    assert state.undo("doc", user="ann").version == 5
    assert artifact.applied == 5 and artifact.current == 5 and artifact.redo_version == 6
    assert state.redo("doc", user="ann").version == 6
    assert stacks(artifact) == (6, 5, None, None)
    assert applied_events(state) == [(5, "undo", "ann"), (6, "redo", "ann")]


def test_nothing_to_undo_or_redo_returns_none():
    state = RunState(flush_interval=0)
    publish(state, "base")
    events = len(state.events)
    assert state.undo("doc") is None and state.redo("doc") is None and state.redo_run("doc") is None
    assert len(state.events) == events


async def test_model_and_view_record_the_same_events():
    by_model, by_view = make_state(), make_state()
    view = ArtifactView(by_view.artifacts["doc"], state=by_view, on_apply=lambda *a: None, user="ann")
    for name, args in [
        ("undo", ()), ("redo", ()), ("undo_run", ()), ("redo", ()), ("redo_run", ()), ("undo_run", ("r2",)),
        ("undo", ()), ("redo_run", ("r2",)),
    ]:
        getattr(by_model, name)("doc", *args, user="ann")
        await getattr(view, name)(*args)
    assert applied_events(by_model) == applied_events(by_view)
    assert stacks(by_model.artifacts["doc"]) == stacks(by_view.artifacts["doc"])
    assert by_model.artifacts["doc"].current == by_view.artifacts["doc"].current


def test_redo_run_redoes_an_undone_run_in_one_step():
    state = make_state()
    artifact = state.artifacts["doc"]
    assert state.undo_run("doc").version == 4
    assert stacks(artifact) == (4, 3, 5, 6)
    assert state.redo_run("doc").version == 6
    assert stacks(artifact) == (6, 5, None, None)
    assert [(v, a) for v, a, _ in applied_events(state)] == [(4, "undo"), (6, "redo")]
    assert state.undo("doc").version == 5


def test_redo_run_finishes_a_partly_redone_run():
    state = make_state()
    state.undo_run("doc")
    state.redo("doc")
    assert stacks(state.artifacts["doc"]) == (5, 4, 6, 6)
    assert state.redo_run("doc").version == 6
    assert state.artifacts["doc"].redo_version is None


def test_redo_run_also_redoes_edits_undone_with_the_run():
    state = RunState(flush_interval=0)
    publish(state, "base")
    publish(state, "add_node", run_id="r1", actor="agent")
    publish(state, "move_tile", user="ann")
    publish(state, "place_tile", run_id="r1", actor="agent")
    assert state.undo_run("doc", "r1").version == 1
    assert state.redo_run("doc", "r1").version == 4
    assert state.artifacts["doc"].redo_version is None


def test_redo_run_by_id_redoes_later_undos_too():
    state = make_state()
    artifact = state.artifacts["doc"]
    state.undo_run("doc", "r2")
    state.undo("doc")
    assert artifact.redo_run_version == 4
    assert artifact.redo_run_target("r2", state.runs) == 6
    assert artifact.redo_run_target("r1", state.runs) is None
    assert state.redo_run("doc", "r2").version == 6
    assert artifact.redo_version is None


def test_a_proposal_keeps_redo_and_an_applied_version_ends_it():
    state = make_state()
    artifact = state.artifacts["doc"]
    state.undo_run("doc")
    publish(state, "proposal", run_id="r3", actor="agent", applied=False)
    assert stacks(artifact) == (4, 3, 5, 6)
    publish(state, "set_config", run_id="r3", actor="agent")
    assert stacks(artifact) == (8, 4, None, None)
    assert state.redo_run("doc") is None


def test_applying_a_version_by_hand_ends_redo():
    state = make_state()
    state.undo_run("doc")
    state.apply_artifact_version("doc", 2)
    assert stacks(state.artifacts["doc"]) == (2, 4, None, None)


def test_a_refused_step_records_nothing():
    state = make_state()
    events = len(state.events)
    assert state.undo("doc", on_apply=lambda artifact, version: False) is None
    with pytest.raises(RuntimeError, match="locked"):
        state.undo_run("doc", on_apply=lambda artifact, version: (_ for _ in ()).throw(RuntimeError("locked")))
    assert len(state.events) == events and state.artifacts["doc"].applied == 6


def test_on_apply_sees_the_target_version_before_it_is_recorded():
    state = make_state()
    seen = []

    def on_apply(artifact, version):
        seen.append((version.version, artifact.applied, version.value[-1]))

    state.undo_run("doc", on_apply=on_apply)
    state.redo_run("doc", on_apply=on_apply)
    assert seen == [(4, 6, "move_tile"), (6, 4, "place_tile")]


def test_an_async_on_apply_is_refused():
    state = make_state()

    async def on_apply(artifact, version):
        pass

    with pytest.raises(TypeError, match="synchronous"):
        state.undo("doc", on_apply=on_apply)
    assert state.artifacts["doc"].applied == 6


def test_undo_run_errors():
    state = make_state()
    state.undo_run("doc")
    with pytest.raises(ValueError, match="did not change the applied version"):
        state.undo_run("doc", "r1")
    with pytest.raises(ValueError, match="No version"):
        state.undo_run("doc", "missing")
    state = RunState(flush_interval=0)
    publish(state, "base")
    with pytest.raises(ValueError, match="not published in a run"):
        state.undo_run("doc")


def test_a_child_run_is_undone_with_its_top_level_run():
    state = make_state()
    state.apply(ev.RunStarted(run_id="r3", agent_name="assistant", input="More"))
    state.apply(ev.RunStarted(run_id="r3.1", parent_run_id="r3", agent_name="assistant"))
    publish(state, "nested", run_id="r3.1", actor="agent")
    state.apply(ev.RunFinished(run_id="r3.1"))
    publish(state, "after", run_id="r3", actor="agent")
    state.apply(ev.RunFinished(run_id="r3"))
    assert root_run_id(state.runs, "r3.1") == "r3"
    assert state.artifacts["doc"].run_versions("r3.1", state.runs) == [7, 8]
    assert state.undo_run("doc", "r3.1").version == 6
    assert state.redo_run("doc", "r3.1").version == 8


def test_replay_reproduces_undo_redo_and_whole_run_steps():
    state = make_state()
    agent_run(state, "r3", "Tidy up", "rename", "resize")
    state.undo("doc")
    state.undo_run("doc", "r2")
    state.redo("doc")
    state.redo_run("doc")
    state.undo_run("doc", "r3")
    replayed = RunState.from_events(state.events)
    live, loaded = state.artifacts["doc"], replayed.artifacts["doc"]
    assert stacks(loaded) == stacks(live)
    for name in ("redo_run", "undo", "undo_run", "redo"):
        assert getattr(replayed, name)("doc").version == getattr(state, name)("doc").version
        assert stacks(loaded) == stacks(live)
    assert applied_events(replayed) == applied_events(state)
