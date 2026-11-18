import random

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


def test_redo_run_redoes_a_run_undone_as_a_whole():
    state = RunState(flush_interval=0)
    publish(state, "base")
    publish(state, "add_node", run_id="r1", actor="agent")
    publish(state, "place_tile", run_id="r1", actor="agent")
    assert state.undo_run("doc", "r1").version == 1
    assert state.redo_run("doc", "r1").version == 3
    assert state.artifacts["doc"].redo_version is None


def test_redo_run_by_id_is_not_redone_over_a_later_undone_edit():
    state = make_state()
    artifact = state.artifacts["doc"]
    state.undo_run("doc", "r2")
    state.undo("doc")
    assert artifact.redo_run_version == 4
    # Redoing r2 would redo ann's version 4 too, since redo is linear.
    assert artifact.redo_run_target("r2", state.runs) is None
    assert artifact.redo_run_target("r1", state.runs) is None
    assert state.redo_run("doc", "r2") is None
    assert state.redo("doc").version == 4 and state.redo_run("doc", "r2").version == 6
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
    assert stacks(loaded) == stacks(live) and loaded.current == live.current
    for name in ("redo_run", "undo", "undo_run", "redo"):
        assert getattr(replayed, name)("doc").version == getattr(state, name)("doc").version
        assert stacks(loaded) == stacks(live)
    assert applied_events(replayed) == applied_events(state)


def three_versions(*runs):
    state = RunState(flush_interval=0)
    publish(state, "base")
    for i, run_id in enumerate(runs):
        if run_id is None:
            publish(state, f"edit{i}")
        else:
            agent_run(state, run_id, "Edit", f"op{i}")
    return state


def test_a_version_applied_again_moves_to_the_end_of_the_undo_history():
    state = three_versions("A", "A")
    artifact = state.artifacts["doc"]
    state.apply_artifact_version("doc", 2)
    assert artifact._undo == [1, 3]
    assert state.undo("doc").version == 3 and state.undo("doc").version == 1
    assert state.undo("doc") is None


def test_redo_run_after_reapplying_versions_by_hand_redoes_everything_once():
    state = RunState(flush_interval=0)
    publish(state, "base")
    agent_run(state, "A", "Edit", "op1", "op2")
    state.apply_artifact_version("doc", 2)
    state.apply_artifact_version("doc", 3)
    artifact = state.artifacts["doc"]
    assert state.undo_run("doc").version == 1
    assert state.redo_run("doc").version == 3
    assert stacks(artifact) == (3, 2, None, None)
    events, calls = len(state.events), []
    assert state.redo_run("doc", on_apply=lambda *a: calls.append(a)) is None
    assert state.redo_run("doc", "A", on_apply=lambda *a: calls.append(a)) is None
    assert len(state.events) == events and not calls


def test_undoing_a_run_that_is_undone_already_does_nothing():
    state = three_versions(None, "A")
    state.undo("doc")
    state.undo("doc")
    artifact = state.artifacts["doc"]
    assert stacks(artifact) == (1, None, 2, 2)
    events, calls = len(state.events), []
    assert artifact.undo_run_target("A", state.runs) is None
    assert state.undo_run("doc", "A", on_apply=lambda *a: calls.append(a)) is None
    assert len(state.events) == events and not calls and artifact.redo_version == 2

    state = three_versions("A")
    assert state.undo_run("doc", "A").version == 1
    events = len(state.events)
    assert state.undo_run("doc", "A", on_apply=lambda *a: calls.append(a)) is None
    assert len(state.events) == events and not calls


async def test_a_view_does_not_undo_a_run_twice():
    state = three_versions("A")
    applied = []
    view = ArtifactView(state.artifacts["doc"], state=state, on_apply=lambda a, v: applied.append(v.version))
    assert await view.undo_run("A")
    assert not await view.undo_run("A")
    assert applied == [1]


def test_replay_shows_the_applied_version_as_current():
    state = make_state()
    state.undo_run("doc")
    state.apply_artifact_version("doc", 2)
    state.undo("doc")
    publish(state, "proposal", run_id="r3", actor="agent", applied=False)
    state.redo("doc")
    state.apply_artifact_version("doc", 6)
    state.undo_run("doc", "r2")
    replayed = RunState.from_events(state.events).artifacts["doc"]
    live = state.artifacts["doc"]
    assert (replayed.current, *stacks(replayed)) == (live.current, *stacks(live))
    assert live.current == live.applied


def full_stacks(artifact):
    return (
        artifact.current, *stacks(artifact), list(artifact._undo), list(artifact._redo),
        [list(step) for step in artifact._redo_steps],
    )


def random_history(rnd):
    state = RunState(flush_interval=0)
    publish(state, "base")
    runs = []
    for i in range(25):
        artifact = state.artifacts["doc"]
        op = rnd.choice(["run", "child", "user", "proposal", "hand", "undo", "redo", "undo_run", "redo_run", "redo_run_id"])
        if op == "run":
            run_id = f"r{len(runs)}"
            runs.append(run_id)
            state.apply(ev.RunStarted(run_id=run_id, agent_name="assistant"))
            for _ in range(rnd.randint(1, 3)):
                publish(state, f"op{i}", run_id=run_id, actor="agent", applied=rnd.random() < 0.8)
            state.apply(ev.RunFinished(run_id=run_id))
        elif op == "child" and runs:
            child = f"{rnd.choice(runs)}.{i}"
            state.apply(ev.RunStarted(run_id=child, parent_run_id=child.split(".")[0], agent_name="assistant"))
            publish(state, f"op{i}", run_id=child, actor="agent")
            state.apply(ev.RunFinished(run_id=child))
        elif op in ("user", "proposal"):
            publish(state, f"op{i}", applied=op == "user")
        elif op == "hand":
            state.apply_artifact_version("doc", rnd.randint(1, len(artifact.versions)))
        elif op in ("undo", "redo", "redo_run"):
            getattr(state, op)("doc")
        elif op == "redo_run_id" and runs:
            run_id = rnd.choice(runs)
            if state.redo_run("doc", run_id) is not None:
                assert not set(artifact.run_versions(run_id, state.runs)) & set(artifact._redo)
        elif op == "undo_run" and runs:
            applied, redo = artifact.applied, list(artifact._redo)
            try:
                version = state.undo_run("doc", rnd.choice(runs))
            except ValueError:
                continue
            assert version is None or version.version != applied
            assert artifact._redo[:len(redo)] == redo
        timeline = [*artifact._undo, artifact.applied, *reversed(artifact._redo)]
        assert len(set(timeline)) == len(timeline)
        assert [n for step in artifact._redo_steps for n in step] == artifact._redo
    return state


@pytest.mark.parametrize("seed", range(200))
def test_random_histories_replay_and_keep_each_version_once(seed):
    state = random_history(random.Random(seed))
    replayed = RunState.from_events(state.events)
    assert full_stacks(replayed.artifacts["doc"]) == full_stacks(state.artifacts["doc"])


def test_undo_does_not_step_back_to_a_version_applied_again_later():
    state = three_versions(None, None)
    state.apply_artifact_version("doc", 1)
    assert [state.undo("doc").version, state.undo("doc").version, state.undo("doc")] == [3, 2, None]
    assert [state.redo("doc").version, state.redo("doc").version] == [3, 1]


def _timeline(artifact):
    return [*artifact._undo, artifact.applied, *artifact._redo[::-1]]


@pytest.mark.parametrize("seed", range(150))
def test_whole_run_steps_never_cross_a_version_outside_the_run(seed):
    rnd = random.Random(seed)
    state = RunState(flush_interval=0)
    publish(state, "base")
    runs = []
    for i in range(30):
        artifact = state.artifacts["doc"]
        op = rnd.choice(["run", "run", "user", "undo", "redo", "undo_run", "redo_run", "drop"])
        if op == "run":
            run_id = f"r{i}"
            runs.append(run_id)
            state.apply(ev.RunStarted(run_id=run_id, agent_name="a"))
            for _ in range(rnd.randint(1, 3)):
                publish(state, "op", run_id=run_id, actor="agent")
                if rnd.random() < 0.3:
                    publish(state, "mine", user="ann")
            state.apply(ev.RunFinished(run_id=run_id))
        elif op == "user":
            publish(state, "mine", user="ann")
        elif op in ("undo", "redo"):
            getattr(state, op)("doc")
        elif op == "drop" and len(artifact.versions) > 3:
            state.drop_artifact_versions("doc", rnd.randint(1, len(artifact.versions)))
        elif op in ("undo_run", "redo_run") and runs:
            run_id = rnd.choice(runs)
            line = _timeline(artifact)
            before = artifact.applied
            try:
                version = getattr(state, op)("doc", run_id)
            except ValueError:
                continue
            if version is None:
                continue
            lo, hi = sorted((line.index(before), line.index(version.version)))
            # The versions the step undid or redid: after the lower end up to the upper one.
            crossed = line[lo + 1:hi + 1]
            assert all(artifact.versions[n - 1].run_id == run_id for n in crossed), (op, run_id, crossed)
    live = state.artifacts["doc"]
    replayed = RunState.from_events(state.events).artifacts["doc"]
    assert _timeline(replayed) == _timeline(live)
    assert len(set(_timeline(live))) == len(_timeline(live))
