import numpy as np
import pandas as pd
import pytest

from panel_agents.artifacts.diff import (
    DIFF_STRATEGIES,
    TableDiff,
    TextDiff,
    ValueDiff,
    diff,
    diff_table,
    diff_text,
    get_diff_strategy,
    register_diff,
)

# Registry


@pytest.mark.parametrize("kind", ["text", "sql", "code", "python", "json", "markdown", "yaml"])
def test_text_kinds_registered(kind):
    assert isinstance(diff("a", "b", kind=kind), TextDiff)


def test_table_kind_registered():
    assert isinstance(diff(pd.DataFrame(), pd.DataFrame(), kind="table"), TableDiff)


def test_unknown_kind_infers_from_value():
    assert isinstance(diff("a", "b", kind="unknown"), TextDiff)
    assert isinstance(diff(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}), kind="plot"), TableDiff)


def test_unknown_kind_falls_back_to_value_diff():
    result = diff(1, 2, kind="plot")
    assert isinstance(result, ValueDiff)
    assert not result.identical
    assert result.old_repr == "1"
    assert result.new_repr == "2"
    assert result.summary == "values differ"


def test_value_diff_identical():
    result = diff(3.5, 3.5, kind="object")
    assert result.identical
    assert result.summary == "identical"


def test_value_diff_with_ambiguous_equality():
    result = diff(np.array([1, 2]), np.array([1, 3]), kind="object")
    assert isinstance(result, ValueDiff)
    assert not result.identical
    assert diff(np.array([1, 2]), np.array([1, 2]), kind="object").identical


def test_value_diff_none():
    assert diff(None, None, kind="object").identical
    assert not diff(None, 1, kind="object").identical


def test_register_custom_strategy():
    @register_diff("custom-kind")
    def custom(old, new, **options):
        return ValueDiff(identical=True, old_repr="x", new_repr="y")

    try:
        assert get_diff_strategy("custom-kind") is custom
        assert diff(1, 2, kind="custom-kind").old_repr == "x"
    finally:
        DIFF_STRATEGIES.pop("custom-kind")


def test_to_dict_is_plain_data():
    data = diff("a\nb", "a\nc", kind="sql").to_dict()
    assert data["kind"] == "sql"
    assert data["added"] == 1
    assert data["removed"] == 1
    assert data["summary"].startswith("+1")


# Text


def test_text_identical():
    result = diff_text("SELECT 1", "SELECT 1", kind="sql")
    assert result.identical
    assert result.unified == ""
    assert (result.added, result.removed, result.hunks) == (0, 0, 0)
    assert result.similarity == 1.0
    assert result.summary == "identical"


def test_text_added_and_removed_lines():
    old = "SELECT a\nFROM t\nWHERE x > 1"
    new = "SELECT a, b\nFROM t\nWHERE x > 1\nLIMIT 10"
    result = diff_text(old, new, kind="sql")
    assert not result.identical
    assert result.added == 2
    assert result.removed == 1
    assert result.hunks == 1
    assert "-SELECT a" in result.unified
    assert "+SELECT a, b" in result.unified
    assert "+LIMIT 10" in result.unified
    assert 0 < result.similarity < 1


def test_text_hunks_respect_context():
    old = "\n".join(f"line {i}" for i in range(30))
    new = old.replace("line 2\n", "line two\n").replace("line 27", "line twenty-seven")
    assert diff_text(old, new, context=3).hunks == 2
    assert diff_text(old, new, context=20).hunks == 1


def test_text_from_none():
    result = diff_text(None, "a\nb")
    assert result.old_text == ""
    assert result.added == 2
    assert result.removed == 0


def test_text_to_none():
    result = diff_text("a", None)
    assert result.removed == 1
    assert result.new_text == ""


def test_text_both_empty():
    result = diff_text("", None)
    assert result.identical
    assert result.similarity == 1.0


def test_text_non_string_values_are_stringified():
    result = diff(1, 2, kind="code")
    assert result.old_text == "1"
    assert not result.identical


def test_json_uses_stable_dumps():
    result = diff({"b": 1, "a": [1, 2]}, {"a": [1, 2], "b": 1}, kind="json")
    assert result.identical


def test_json_changes():
    result = diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": None}, kind="json")
    assert result.added == 2
    assert result.removed == 1
    assert '"c": null' in result.new_text


def test_json_parses_strings():
    assert diff('{"a": 1, "b": 2}', {"b": 2, "a": 1}, kind="json").identical


def test_json_invalid_string_is_diffed_as_text():
    result = diff("not json", "not json either", kind="json")
    assert result.old_text == "not json"
    assert not result.identical


def test_json_unserializable_values_use_repr():
    class Thing:
        def __repr__(self):
            return "Thing()"
    result = diff({"a": Thing()}, {"a": 1}, kind="json")
    assert "Thing()" in result.old_text


# Tables


@pytest.fixture
def df():
    return pd.DataFrame({"name": ["a", "b", "c"], "value": [1, 2, 3], "score": [0.5, 0.6, 0.7]})


def test_table_identical(df):
    result = diff_table(df, df.copy())
    assert result.identical
    assert result.alignment == "index"
    assert result.summary == "identical"
    assert not result.shape_changed


def test_table_modified_cells(df):
    new = df.copy()
    new.loc[1, "value"] = 20
    new.loc[2, "score"] = 0.9
    result = diff_table(df, new)
    assert not result.identical
    assert result.rows_modified == [1, 2]
    assert result.cells_modified == [(1, "value"), (2, "score")]
    assert not result.rows_added and not result.rows_removed
    assert "2 rows modified" in result.summary


def test_table_rows_added_and_removed(df):
    new = pd.concat([df.drop(0), pd.DataFrame({"name": ["d"], "value": [4], "score": [0.8]}, index=[3])])
    result = diff_table(df, new)
    assert result.rows_added == [3]
    assert result.rows_removed == [0]
    assert result.rows_modified == []
    assert result.old_shape == result.new_shape == (3, 3)


def test_table_shape_and_columns(df):
    new = df.drop(columns="score").assign(extra=1)
    result = diff_table(df, new)
    assert result.columns_added == ["extra"]
    assert result.columns_removed == ["score"]
    assert not result.shape_changed
    new = df.assign(extra=1)
    result = diff_table(df, new)
    assert result.shape_changed
    assert result.old_shape == (3, 3)
    assert result.new_shape == (3, 4)
    assert result.rows_modified == []
    assert "shape 3×3 → 3×4" in result.summary


def test_table_dtype_changes(df):
    new = df.astype({"value": float})
    result = diff_table(df, new)
    assert result.dtype_changes == {"value": ("int64", "float64")}
    assert result.rows_modified == []
    assert not result.identical


def test_table_reordered_columns(df):
    result = diff_table(df, df[["score", "name", "value"]])
    assert result.columns_reordered
    assert not result.columns_added and not result.columns_removed
    assert result.rows_modified == []
    assert not result.identical
    assert result.summary == "columns reordered"


def test_table_reordered_rows_align_on_index(df):
    result = diff_table(df, df.iloc[::-1])
    assert result.rows_modified == []
    assert result.identical


def test_table_none_values(df):
    old = df.astype({"score": object})
    old.loc[0, "score"] = None
    new = old.copy()
    assert diff_table(old, new).identical
    new.loc[1, "score"] = None
    result = diff_table(old, new)
    assert result.cells_modified == [(1, "score")]


def test_table_nan_equals_nan():
    old = pd.DataFrame({"a": [1.0, np.nan]})
    assert diff_table(old, old.copy()).identical
    result = diff_table(old, pd.DataFrame({"a": [1.0, 2.0]}))
    assert result.cells_modified == [(1, "a")]


def test_table_empty_frames():
    assert diff_table(pd.DataFrame(), pd.DataFrame()).identical
    empty = pd.DataFrame({"a": pd.Series([], dtype=int)})
    result = diff_table(empty, pd.DataFrame({"a": [1, 2]}))
    assert result.rows_added == [0, 1]
    assert result.shape_changed


def test_table_from_none(df):
    result = diff_table(None, df)
    assert result.old_shape == (0, 0)
    assert result.columns_added == ["name", "value", "score"]
    assert result.rows_added == [0, 1, 2]


def test_table_non_unique_index_same_length_aligns_by_position():
    old = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 0, 1])
    new = pd.DataFrame({"a": [1, 5, 3]}, index=[0, 0, 1])
    result = diff_table(old, new)
    assert result.alignment == "position"
    assert result.cells_modified == [(1, "a")]


def test_table_non_unique_index_different_length_not_comparable():
    old = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 0, 1])
    new = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
    result = diff_table(old, new)
    assert result.alignment is None
    assert not result.identical
    assert "rows not comparable" in result.summary
    assert result.rows_modified == []


def test_table_key_alignment(df):
    new = df.iloc[[2, 0]].reset_index(drop=True)
    new.loc[0, "value"] = 30
    result = diff_table(df, new, key="name")
    assert result.alignment == "key"
    assert result.rows_removed == ["b"]
    assert result.rows_modified == ["c"]
    assert result.cells_modified == [("c", "value")]


def test_table_multi_key_alignment():
    old = pd.DataFrame({"k1": [1, 1], "k2": ["x", "y"], "v": [1, 2]})
    new = pd.DataFrame({"k1": [1, 1], "k2": ["y", "x"], "v": [3, 1]})
    result = diff_table(old, new, key=["k1", "k2"])
    assert result.cells_modified == [((1, "y"), "v")]


def test_table_duplicate_key_not_comparable():
    old = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    new = pd.DataFrame({"k": [1], "v": [1]})
    assert diff_table(old, new, key="k").alignment is None


def test_table_missing_key_falls_back_to_index(df):
    assert diff_table(df, df, key="missing").alignment == "index"


def test_table_mixed_type_columns():
    old = pd.DataFrame({"a": [1, "x", [1, 2]]})
    new = pd.DataFrame({"a": [1, "y", [1, 2]]})
    result = diff_table(old, new)
    assert result.cells_modified == [(1, "a")]


def test_table_non_string_column_labels():
    old = pd.DataFrame({0: [1], 1: [2]})
    new = pd.DataFrame({0: [1], 1: [3], 2: [4]})
    result = diff_table(old, new)
    assert result.columns_added == ["2"]
    assert result.cells_modified == [(0, "1")]


def test_table_from_records():
    result = diff([{"a": 1}], [{"a": 2}], kind="table")
    assert result.cells_modified == [(0, "a")]
