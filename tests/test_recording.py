import pandas as pd
import pytest

from panel_agents import Recorder, RunState, dump_events, load_events, load_state
from panel_agents import events as ev
from panel_agents.recording import _decode_value, _encode_value

from .test_state import run_events


def test_record_and_replay_reproduces_state(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with Recorder(state, path):
        state.apply_many(run_events())
    replayed = load_state(path)
    assert [type(e) for e in replayed.events] == [type(e) for e in state.events]
    assert replayed.items[1].content == state.items[1].content
    assert replayed.items[0].input_tokens == state.items[0].input_tokens == 10
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
    assert _decode_value(_encode_value({"x": Foo()})) == {"x": "Foo()"}


def test_recordings_start_with_the_format_version(tmp_path):
    import json

    from panel_agents.recording import FORMAT_VERSION

    state = RunState(flush_interval=0)
    path = tmp_path / "run.jsonl"
    with Recorder(state, path):
        state.add_user_message("hi")
    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"panel_agents_recording": FORMAT_VERSION}
    assert [e.type for e in load_events(path)][0] == "TEXT_MESSAGE_START"
    with Recorder(state, path):
        state.add_user_message("again")
    assert sum("panel_agents_recording" in line for line in path.read_text().splitlines()) == 1

    dumped = tmp_path / "dump.jsonl"
    dump_events(state.events, dumped)
    assert json.loads(dumped.read_text().splitlines()[0]) == {"panel_agents_recording": FORMAT_VERSION}
    assert len(load_events(dumped)) == len(state.events)


def test_newer_recordings_are_rejected(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text('{"panel_agents_recording": 99}\n')
    with pytest.raises(ValueError, match="version 99"):
        load_events(path)


def test_record_to_a_callable_and_load_the_records(tmp_path):
    import io
    import json

    records = []
    state = RunState(flush_interval=0)
    path = tmp_path / "run.jsonl"
    buffer = io.StringIO()
    df = pd.DataFrame({"a": [1, 2]})
    with Recorder(state, records.append), Recorder(state, path), Recorder(state, buffer):
        state.apply_many(run_events())
        state.apply(ev.ArtifactPublished(artifact_id="t", value=df))
    assert records == [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [json.loads(line) for line in buffer.getvalue().splitlines()]
    replayed = load_state(records)
    assert [type(e) for e in replayed.events] == [type(e) for e in state.events]
    pd.testing.assert_frame_equal(load_events(iter(records))[-1].value, df)


def test_newer_records_from_a_store_are_rejected():
    with pytest.raises(ValueError, match="The recording is a version 99"):
        load_events([{"panel_agents_recording": 99}])


def test_recorder_rejects_an_unknown_sink():
    with pytest.raises(TypeError, match="sink"):
        Recorder(RunState(), 42)


def test_path_keyword_is_a_deprecated_alias(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with pytest.warns(DeprecationWarning), Recorder(state, path=path):
        state.apply_many(run_events())
    with pytest.warns(DeprecationWarning):
        assert len(load_events(path=path)) == len(state.events)
    with pytest.warns(DeprecationWarning):
        assert load_state(path=path).runs["run-1"].cost_usd == 0.01


def test_incomplete_last_event_is_ignored(tmp_path):
    path = tmp_path / "run.jsonl"
    state = RunState(flush_interval=0)
    with Recorder(state, path):
        state.apply_many(run_events())
    path.write_text(path.read_text()[:-10])
    assert len(load_events(path)) == len(state.events) - 1
    with Recorder(state, path):
        state.add_user_message("again")
    assert len(load_events(path)) == len(state.events) - 1
