import pandas as pd
import panel as pn
import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.artifacts import ArtifactStore
from panel_agents.components._diff import DiffViewer
from panel_agents.components.artifacts import (
    ArtifactPane,
    ArtifactView,
    table_styles,
    version_bytes,
    version_label,
)


@pytest.fixture
def store():
    return ArtifactStore()


def publish_sql(store, n=3):
    start = len(store["query"].versions) if "query" in store else 0
    for i in range(start, n):
        store.publish("query", "\n".join(f"line {j}" for j in range(i + 1)), kind="sql",
                      title="Query", tool_call_id=f"call_{i + 1}", created_at=1_700_000_000 + i)


def frames():
    old = pd.DataFrame({"name": ["a", "b", "c"], "value": [1, 2, 3]})
    new = old.drop(0)
    new.loc[1, "value"] = 20
    new.loc[3] = ["d", 4]
    return old, new


def test_pane_empty_store(store):
    pane = ArtifactPane(store)
    assert pane._list.items == []
    assert pane._container.objects == [pane._empty]
    assert pane.active_view is None


def test_pane_lists_artifacts(store):
    pane = ArtifactPane(store)
    publish_sql(store, 2)
    store.publish("df", frames()[0], title="Data")
    items = pane._list.items
    assert [i["label"] for i in items] == ["Query", "Data"]
    assert [i["icon"] for i in items] == ["storage", "table_chart"]
    assert items[0]["secondary"] == "sql · 2 versions"
    assert items[1]["secondary"] == "table · 1 version"


def test_pane_follows_store_active(store):
    pane = ArtifactPane(store)
    publish_sql(store, 1)
    assert pane._list.active == (0,)
    store.publish("df", frames()[0])
    assert pane._list.active == (1,)
    assert pane.active_view.artifact is store["df"]
    assert pane._container.objects == [pane.active_view.__panel__()]
    store.active = "query"
    assert pane._list.active == (0,)
    assert pane.active_view.artifact is store["query"]
    assert pane._container.objects == [pane.active_view.__panel__()]


def test_pane_list_click_sets_store_active(store):
    pane = ArtifactPane(store)
    publish_sql(store, 1)
    store.publish("df", frames()[0])
    pane._list.active = 0
    assert store.active == "query"


def test_pane_active_unknown_shows_empty(store):
    pane = ArtifactPane(store)
    publish_sql(store, 1)
    store.active = None
    assert pane._container.objects == [pane._empty]


def test_pane_reuses_views_on_new_version(store):
    pane = ArtifactPane(store)
    publish_sql(store, 1)
    view = pane.active_view
    layout = view._layout_obj
    first = view._preview.objects[0]
    publish_sql(store, 2)
    assert pane.active_view is view
    assert view._layout_obj is layout
    assert view._preview.objects[0] is not first
    assert view._rendered[("sql", 0)] is first


def test_pane_binds_existing_store():
    state = RunState.from_events([
        ev.ArtifactPublished(artifact_id="a", value="x", kind="text"),
        ev.ArtifactPublished(artifact_id="a", value="y", kind="text"),
    ])
    pane = ArtifactPane(state.artifacts)
    assert pane.active_view.artifact.artifact_id == "a"
    assert pane._list.items[0]["secondary"] == "text · 2 versions"


def test_pane_view_options(store):
    pane = ArtifactPane(store, view_options={"diff_layout": "split"})
    publish_sql(store, 2)
    assert pane.active_view.diff_layout == "split"


def test_pane_renders(store, document, comm):
    pane = ArtifactPane(store)
    publish_sql(store, 2)
    store.publish("df", frames()[0])
    assert pn.panel(pane).get_root(document, comm) is not None


def test_view_version_selector(store):
    publish_sql(store)
    view = ArtifactView(store["query"])
    assert list(view._version.options) == [version_label(v) for v in store["query"].versions]
    assert view._version.value == 2
    view._version.value = 0
    assert store["query"].current == 0
    assert view._preview.objects[0].value == "line 0"
    store["query"].current = 1
    assert view._version.value == 1


def test_version_label():
    store = ArtifactStore()
    store.publish("q", "x", tool_call_id="call_9", created_at=0)
    label = version_label(store["q"].latest)
    assert label.startswith("v1 · ")
    assert label.endswith(" · call_9")


def test_view_new_version_selects_latest(store):
    publish_sql(store, 2)
    view = ArtifactView(store["query"])
    view._version.value = 0
    publish_sql(store, 3)
    assert view._version.value == 2
    assert len(view._version.options) == 3
    assert view._count.object == "3 versions"


def test_view_metadata(store):
    store.publish("q", "x", kind="sql", tool_call_id="call_1", metadata={"dialect": "duckdb"})
    view = ArtifactView(store["q"])
    assert view._metadata.object["tool_call_id"] == "call_1"
    assert view._metadata.object["metadata"] == {"dialect": "duckdb"}
    assert view._metadata.object["version"] == 1


def test_view_download_text(store):
    publish_sql(store, 2)
    view = ArtifactView(store["query"])
    assert view._download.filename == "query_v2.sql"
    assert view._download_payload().read() == b"line 0\nline 1"
    store["query"].current = 0
    assert view._download.filename == "query_v1.sql"
    assert view._download_payload().read() == b"line 0"


def test_view_download_table(store):
    store.publish("df", frames()[0])
    view = ArtifactView(store["df"])
    assert view._download.filename == "df_v1.csv"
    assert view._download_payload().read().decode().splitlines()[0] == "name,value"


def test_version_bytes():
    assert version_bytes({"a": 1}, "json") == (b'{\n  "a": 1\n}', "json")
    assert version_bytes(None, "text") == (b"", "txt")
    assert version_bytes("x", "python")[1] == "py"
    data, ext = version_bytes(pd.DataFrame({"a": [1]}, index=["r"]), "table")
    assert ext == "csv"
    assert data.decode().splitlines() == [",a", "r,1"]


def test_view_compare_defaults(store):
    publish_sql(store)
    view = ArtifactView(store["query"])
    assert (view.base, view.target) == (1, 2)
    view.mode = "compare"
    assert view._tabs.active == 1
    diff_viewer = view._compare_body.objects[0]
    assert isinstance(diff_viewer, DiffViewer)
    assert diff_viewer.added == 1
    assert diff_viewer.removed == 0
    assert [c.label for c in view._summary][:2] == ["+1 lines", "−0 lines"]


def test_view_compare_selects(store):
    publish_sql(store)
    view = ArtifactView(store["query"], mode="compare")
    view._base.value = 0
    assert view.base == 0
    assert view._diff_viewer.added == 2
    view.diff_layout = "split"
    assert view._diff_viewer.mode == "split"


def test_view_compare_follows_new_versions(store):
    publish_sql(store, 2)
    view = ArtifactView(store["query"], mode="compare")
    publish_sql(store, 3)
    assert (view.base, view.target) == (1, 2)
    assert view._diff_viewer.new == "line 0\nline 1\nline 2"


def test_view_compare_pinned_target_does_not_follow(store):
    publish_sql(store, 3)
    view = ArtifactView(store["query"], mode="compare", diff_layout="split")
    view.param.update(base=0, target=1)
    store.publish("query", "new", kind="sql")
    assert (view.base, view.target) == (0, 1)


def test_view_compare_tab_click(store):
    publish_sql(store, 2)
    view = ArtifactView(store["query"])
    assert view._compare_body.objects == []
    view._tabs.active = 1
    assert view.mode == "compare"
    assert view._compare_body.objects == [view._diff_viewer]


def test_view_compare_identical(store):
    store.publish("q", "same", kind="sql")
    store.publish("q", "same", kind="sql")
    view = ArtifactView(store["q"], mode="compare")
    assert [c.label for c in view._summary] == ["Identical"]


def test_view_compare_single_version(store):
    store.publish("q", "x", kind="sql")
    view = ArtifactView(store["q"], mode="compare")
    assert (view.base, view.target) == (0, 0)
    assert [c.label for c in view._summary] == ["Identical"]


def test_view_compare_table(store):
    old, new = frames()
    store.publish("df", old)
    store.publish("df", new)
    view = ArtifactView(store["df"], mode="compare")
    labels = [c.label for c in view._summary]
    assert "1 row added" in labels
    assert "1 row removed" in labels
    assert "1 row modified" in labels
    assert "1 cell changed" in labels
    row = view._compare_body.objects[0]
    old_table, new_table = (col[1] for col in row)
    assert isinstance(new_table, pn.widgets.Tabulator)
    assert new_table.value.equals(new)
    assert not view._layout.visible


def test_view_compare_table_key(store):
    old = pd.DataFrame({"id": [1, 2], "v": [1, 2]})
    new = pd.DataFrame({"id": [2, 1], "v": [5, 1]})
    store.publish("df", old)
    store.publish("df", new)
    view = ArtifactView(store["df"], mode="compare", diff_options={"key": "id"})
    assert view.compute_diff().cells_modified == [(2, "v")]


def test_table_styles():
    old, new = frames()
    store = ArtifactStore()
    store.publish("df", old)
    store.publish("df", new)
    view = ArtifactView(store["df"])
    result = view.compute_diff()
    old_styles = table_styles(old, result, "old")
    new_styles = table_styles(new, result, "new")
    assert "211, 47, 47" in old_styles.iloc[0, 0]
    assert old_styles.iloc[1, 1].startswith("background-color: rgba(237")
    assert old_styles.iloc[1, 0] == ""
    assert new_styles.iloc[0, 1].startswith("background-color: rgba(237")
    assert "46, 125, 50" in new_styles.iloc[2, 0]


def test_table_styles_added_column():
    store = ArtifactStore()
    store.publish("df", pd.DataFrame({"a": [1]}))
    store.publish("df", pd.DataFrame({"a": [1], "b": [2]}))
    result = ArtifactView(store["df"]).compute_diff()
    styles = table_styles(store["df"].latest.value, result, "new")
    assert styles.iloc[0, 0] == ""
    assert "46, 125, 50" in styles.iloc[0, 1]


def test_view_compare_value_fallback(store):
    store.publish("obj", 1, kind="object")
    store.publish("obj", 2, kind="object")
    view = ArtifactView(store["obj"], mode="compare")
    assert [c.label for c in view._summary] == ["Values differ"]
    assert len(view._compare_body.objects[0]) == 2


def test_view_artifact_swap(store):
    publish_sql(store, 2)
    store.publish("other", "z", kind="text", title="Other")
    view = ArtifactView(store["query"])
    view.artifact = store["other"]
    assert view._title.object == "Other"
    assert len(view._version.options) == 1
    publish_sql(store, 3)
    assert len(view._version.options) == 1


def test_view_hide_header(store):
    publish_sql(store, 1)
    view = ArtifactView(store["query"], show_header=False)
    assert not view._header.visible
