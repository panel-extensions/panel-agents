import pandas as pd

from panel_agents import Recorder, RunState, load_events, load_state
from panel_agents import events as ev
from panel_agents.recording import decode_value, encode_value

from .test_state import run_events


def test_record_and_replay_reproduces_state(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with Recorder(state, path):
        state.apply_many(run_events())
    replayed = load_state(path)
    assert [type(e) for e in replayed.events] == [type(e) for e in state.events]
    assert replayed.items[0].content == state.items[0].content
    assert replayed.tool_calls["t1"].args == {"sql": "SELECT 1"}
    assert replayed.runs["run-1"].cost_usd == 0.01


def test_dataframe_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with Recorder(state, path):
        state.apply(ev.ArtifactPublished(artifact_id="t", value=df))
    (event,) = load_events(path)
    pd.testing.assert_frame_equal(event.value, df)


def test_unknown_values_fall_back_to_repr():
    class Foo:
        def __repr__(self):
            return "Foo()"
    assert decode_value(encode_value({"x": Foo()})) == {"x": "Foo()"}
