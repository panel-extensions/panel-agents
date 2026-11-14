"""
Components displaying versioned artifacts and the differences between versions.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..artifacts.diff import TableDiff, TextDiff, ValueDiff, diff
from ..artifacts.model import Artifact, ArtifactStore, ArtifactVersion
from ._diff import DiffViewer
from .renderers import render

KIND_ICONS = {
    "table": "table_chart",
    "sql": "storage",
    "code": "code",
    "python": "code",
    "javascript": "javascript",
    "json": "data_object",
    "yaml": "data_object",
    "text": "article",
    "markdown": "article",
    "plot": "insert_chart",
}

DOWNLOAD_EXTENSIONS = {
    "table": "csv",
    "sql": "sql",
    "json": "json",
    "code": "py",
    "python": "py",
    "javascript": "js",
    "yaml": "yaml",
    "markdown": "md",
}

CELL_STYLES = {
    "added": "background-color: rgba(46, 125, 50, 0.18)",
    "removed": "background-color: rgba(211, 47, 47, 0.18)",
    "modified": "background-color: rgba(237, 108, 2, 0.28); font-weight: 600",
}


def kind_icon(kind: str) -> str:
    return KIND_ICONS.get(kind, "inventory_2")


def version_label(version: ArtifactVersion) -> str:
    parts = [f"v{version.version}"]
    if version.created_at:
        parts.append(datetime.fromtimestamp(version.created_at).strftime("%H:%M:%S"))
    if version.tool_call_id:
        parts.append(version.tool_call_id)
    return " · ".join(parts)


def version_bytes(value: Any, kind: str) -> tuple[bytes, str]:
    """Serialize a version for download, returning the payload and file extension."""
    if kind == "table" and hasattr(value, "to_csv"):
        return value.to_csv(index=not isinstance(value.index, pd.RangeIndex)).encode(), "csv"
    if kind == "json" and not isinstance(value, str):
        return json.dumps(value, indent=2, default=repr).encode(), "json"
    return ("" if value is None else str(value)).encode(), DOWNLOAD_EXTENSIONS.get(kind, "txt")


def _row_labels(df: pd.DataFrame, table_diff: TableDiff, key: list[str]) -> list[Any]:
    if table_diff.alignment == "position":
        return list(range(len(df)))
    if table_diff.alignment == "key":
        values = df[key].itertuples(index=False, name=None)
        return [v[0] if len(key) == 1 else v for v in values]
    return list(df.index)


def table_styles(df: pd.DataFrame, table_diff: TableDiff, side: str, key: list[str] | None = None) -> pd.DataFrame:
    """CSS for each cell of one side of a table diff, positionally aligned with ``df``."""
    styles = pd.DataFrame("", index=range(len(df)), columns=range(len(df.columns)))
    if table_diff.alignment is None:
        return styles
    columns = [str(c) for c in df.columns]
    extra_cols = set(table_diff.columns_added if side == "new" else table_diff.columns_removed)
    extra_rows = set(table_diff.rows_added if side == "new" else table_diff.rows_removed)
    status = "added" if side == "new" else "removed"
    modified = {}
    for row, col in table_diff.cells_modified:
        modified.setdefault(row, set()).add(col)
    for r, label in enumerate(_row_labels(df, table_diff, key or [])):
        if label in extra_rows:
            styles.iloc[r, :] = CELL_STYLES[status]
            continue
        for c, name in enumerate(columns):
            if name in extra_cols:
                styles.iloc[r, c] = CELL_STYLES[status]
            elif name in modified.get(label, ()):
                styles.iloc[r, c] = CELL_STYLES["modified"]
    return styles


def _apply_styles(table, styles: pd.DataFrame):
    def style(data):
        values = styles.iloc[:len(data), :data.shape[1]].to_numpy()
        return pd.DataFrame(values, index=data.index, columns=data.columns)
    table.style.apply(style, axis=None)
    return table


class ArtifactView(Viewer):
    """
    A single artifact with a version selector, download and a compare mode
    showing the difference between two versions.
    """

    artifact = param.ClassSelector(class_=Artifact, doc="The artifact to display.")

    mode = param.Selector(default="preview", objects=["preview", "compare", "metadata"], doc="""
        Which tab is shown.""")

    base = param.Integer(default=None, allow_None=True, doc="""
        Index of the version compared against; defaults to the one before the target.""")

    target = param.Integer(default=None, allow_None=True, doc="""
        Index of the version compared; defaults to the latest.""")

    diff_layout = param.Selector(default="unified", objects=["unified", "split"], doc="""
        Layout of text diffs.""")

    diff_options = param.Dict(default={}, doc="""
        Options passed to the diff strategy, e.g. ``{'key': 'id'}`` for tables.""")

    show_header = param.Boolean(default=True, doc="Whether to show the title row.")

    def __init__(self, artifact: Artifact | None = None, **params):
        super().__init__(artifact=artifact, **params)
        self._rendered: dict[tuple[str, int], pn.viewable.Viewable] = {}
        self._watchers: list = []
        self._diff_viewer = DiffViewer(context=3, max_height=520)

        self._icon = pmui.Chip(size="small", variant="outlined", margin=(0, 8, 0, 0))
        self._title = pmui.Typography(variant="h6", margin=(0, 8), styles={"font-weight": "600"})
        self._count = pmui.Typography(variant="caption", color="text.secondary", margin=(0, 8))
        self._version = pmui.Select(label="Version", options={}, width=260, size="small", margin=(0, 8))
        self._download = pmui.FileDownload(
            label="Download", icon="download", variant="outlined", size="small", color="primary",
            callback=self._download_payload, margin=(0, 0, 0, 8), embed=False,
        )
        self._header = pmui.Row(
            self._icon, self._title, self._count, pn.HSpacer(), self._version, self._download,
            align="center", sizing_mode="stretch_width", visible=self.param.show_header, margin=(12, 0, 0, 0),
            styles={"align-items": "center", "flex-wrap": "wrap"},
        )

        self._preview = pmui.Column(sizing_mode="stretch_width")

        self._base = pmui.Select(label="Base", options={}, width=220, size="small")
        self._target = pmui.Select(label="Compare to", options={}, width=220, size="small")
        self._layout = pmui.RadioButtonGroup(
            options={"Unified": "unified", "Split": "split"}, value=self.diff_layout, size="small",
            visible=False, align="center",
        )
        self._summary = pmui.Row(sizing_mode="stretch_width", styles={"flex-wrap": "wrap", "gap": "6px"}, margin=(8, 5))
        self._compare_body = pmui.Column(sizing_mode="stretch_width")
        compare = pmui.Column(
            pmui.Row(self._base, pmui.Typography("→", margin=(0, 4), align="center"), self._target, pn.HSpacer(), self._layout,
                     sizing_mode="stretch_width", styles={"align-items": "center"}),
            self._summary, self._compare_body, sizing_mode="stretch_width",
        )

        self._metadata = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width")

        self._tabs = pmui.Tabs(
            ("Preview", self._preview), ("Compare", compare), ("Metadata", self._metadata),
            active=["preview", "compare", "metadata"].index(self.mode), sizing_mode="stretch_width",
        )
        self._layout_obj = pmui.Column(self._header, self._tabs, sizing_mode="stretch_width")

        self._version.param.watch(self._on_version_select, "value")
        self._base.param.watch(lambda e: self.param.update(base=e.new), "value")
        self._target.param.watch(lambda e: self.param.update(target=e.new), "value")
        self._layout.param.watch(lambda e: self.param.update(diff_layout=e.new), "value")
        self._tabs.param.watch(lambda e: self.param.update(mode=self.param.mode.objects[e.new]), "active")
        self._bind_artifact()

    def __panel__(self):
        return self._layout_obj

    @property
    def versions(self) -> list[ArtifactVersion]:
        return self.artifact.versions if self.artifact else []

    @param.depends("artifact", watch=True)
    def _bind_artifact(self):
        for obj, watcher in self._watchers:
            obj.param.unwatch(watcher)
        self._watchers = []
        self._rendered.clear()
        self.param.update(base=None, target=None)
        if self.artifact is None:
            self._preview.objects = []
            return
        artifact = self.artifact
        self._watchers = [
            (artifact, artifact.param.watch(self._on_versions, "versions")),
            (artifact, artifact.param.watch(self._on_current, "current")),
            (artifact, artifact.param.watch(self._update_header, ["title", "kind"])),
        ]
        self._update_header()
        self._on_versions()

    def _update_header(self, *events):
        artifact = self.artifact
        self._icon.param.update(label=artifact.kind, icon=kind_icon(artifact.kind))
        self._title.object = artifact.title or artifact.artifact_id
        n = len(artifact.versions)
        self._count.object = f"{n} version{'s' if n != 1 else ''}"

    def _on_versions(self, *events):
        versions = self.versions
        previous_latest = None
        if events:
            previous_latest = len(events[0].old) - 1 if events[0].old else None
        options = {version_label(v): i for i, v in enumerate(versions)}
        self._update_header()
        self._version.options = options
        self._base.options = options
        self._target.options = options
        latest = len(versions) - 1
        if self.target is None or self.target == previous_latest or self.target > latest:
            self.param.update(
                target=latest if versions else None,
                base=max(latest - 1, 0) if versions else None,
            )
        self._on_current()

    def _on_current(self, *events):
        if not self.versions:
            self._preview.objects = []
            self._metadata.object = {}
            return
        current = min(max(self.artifact.current, 0), len(self.versions) - 1)
        self._version.value = current
        self._preview.objects = [self._render_version(current)]
        self._update_metadata(current)
        ext = DOWNLOAD_EXTENSIONS.get(self.artifact.kind, "txt")
        self._download.filename = f"{self.artifact.artifact_id}_v{self.versions[current].version}.{ext}"

    def _render_version(self, index: int) -> pn.viewable.Viewable:
        # Versions are immutable, so rendered output can be cached.
        key = (self.artifact.kind, index)
        if key not in self._rendered:
            self._rendered[key] = render(self.versions[index].value, kind=self.artifact.kind)
        return self._rendered[key]

    def _update_metadata(self, index: int):
        version = self.versions[index]
        info = {"artifact_id": self.artifact.artifact_id, "kind": self.artifact.kind, "version": version.version}
        if version.created_at:
            info["created_at"] = datetime.fromtimestamp(version.created_at).isoformat(timespec="seconds")
        if version.tool_call_id:
            info["tool_call_id"] = version.tool_call_id
        self._metadata.object = {**info, "metadata": dict(version.metadata)}

    def _on_version_select(self, event):
        if event.new is not None and self.artifact and event.new != self.artifact.current:
            self.artifact.current = event.new

    def _download_payload(self):
        version = self.versions[self.artifact.current]
        data, ext = version_bytes(version.value, self.artifact.kind)
        self._download.filename = f"{self.artifact.artifact_id}_v{version.version}.{ext}"
        return io.BytesIO(data)

    @param.depends("mode", watch=True)
    def _sync_tab(self):
        self._tabs.active = self.param.mode.objects.index(self.mode)

    @param.depends("base", "target", "diff_layout", "mode", "diff_options", watch=True)
    def _update_compare(self):
        self._base.value = self.base
        self._target.value = self.target
        self._layout.value = self.diff_layout
        if self.mode != "compare":
            return
        versions = self.versions
        if self.base is None or self.target is None or not versions:
            self._summary.objects = []
            self._compare_body.objects = [pmui.Typography("Nothing to compare yet.", color="text.secondary")]
            return
        old, new = versions[self.base], versions[self.target]
        result = self.compute_diff()
        self._layout.visible = isinstance(result, TextDiff)
        self._summary.objects = self._summary_chips(result)
        if isinstance(result, TextDiff):
            self._diff_viewer.param.update(
                old=result.old_text, new=result.new_text, mode=self.diff_layout,
                old_label=version_label(old), new_label=version_label(new),
            )
            self._compare_body.objects = [self._diff_viewer]
        elif isinstance(result, TableDiff):
            self._compare_body.objects = [self._table_comparison(old, new, result)]
        else:
            self._compare_body.objects = [self._value_comparison(old, new, result)]

    def compute_diff(self):
        """The diff between the ``base`` and ``target`` versions."""
        versions = self.versions
        if self.base is None or self.target is None or not versions:
            return None
        return diff(versions[self.base].value, versions[self.target].value, kind=self.artifact.kind, **self.diff_options)

    @staticmethod
    def _chip(label: str, color: str = "default", icon: str | None = None):
        return pmui.Chip(label=label, color=color, icon=icon, size="small", variant="outlined", margin=(0, 6, 0, 0))

    def _summary_chips(self, result) -> list:
        if result.identical:
            return [self._chip("Identical", "success", "check")]
        if isinstance(result, TextDiff):
            return [
                self._chip(f"+{result.added} lines", "success"),
                self._chip(f"−{result.removed} lines", "error"),
                self._chip(f"{result.hunks} hunk{'s' if result.hunks != 1 else ''}"),
                self._chip(f"{result.similarity:.0%} similar"),
            ]
        if isinstance(result, TableDiff):
            chips = []
            if result.shape_changed:
                (r0, c0), (r1, c1) = result.old_shape, result.new_shape
                chips.append(self._chip(f"{r0}×{c0} → {r1}×{c1}", icon="aspect_ratio"))
            for cols, sign, color in ((result.columns_added, "+", "success"), (result.columns_removed, "−", "error")):
                if cols:
                    chips.append(self._chip(f"{sign} {', '.join(cols)}", color, "view_column"))
            for col, (a, b) in result.dtype_changes.items():
                chips.append(self._chip(f"{col}: {a} → {b}", "info"))
            if result.alignment is None:
                chips.append(self._chip("Rows not comparable (non-unique index)", "warning", "warning"))
            for rows, label, color in (
                (result.rows_added, "added", "success"),
                (result.rows_removed, "removed", "error"),
                (result.rows_modified, "modified", "warning"),
            ):
                if rows:
                    chips.append(self._chip(f"{len(rows)} row{'s' if len(rows) != 1 else ''} {label}", color))
            if result.cells_modified:
                n = len(result.cells_modified)
                chips.append(self._chip(f"{n} cell{'s' if n != 1 else ''} changed", "warning"))
            if result.columns_reordered:
                chips.append(self._chip("Columns reordered"))
            return chips
        return [self._chip("Values differ", "warning")]

    def _table_comparison(self, old: ArtifactVersion, new: ArtifactVersion, result: TableDiff):
        key = self.diff_options.get("key")
        key = [key] if isinstance(key, str) else list(key or [])
        sides = []
        for side, version in (("old", old), ("new", new)):
            df = version.value if version.value is not None else pd.DataFrame()
            table = render(df, kind="table", layout="fit_data_table")
            if isinstance(table, pn.widgets.Tabulator):
                _apply_styles(table, table_styles(df, result, side, key))
            sides.append(pmui.Column(
                pmui.Typography(version_label(version), variant="subtitle2", margin=(0, 5)),
                table, sizing_mode="stretch_width",
            ))
        return pmui.Row(*sides, sizing_mode="stretch_width")

    def _value_comparison(self, old: ArtifactVersion, new: ArtifactVersion, result: ValueDiff):
        def block(version, text):
            return pmui.Column(
                pmui.Typography(version_label(version), variant="subtitle2", margin=(0, 5)),
                pn.pane.Markdown(f"```\n{text}\n```", sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )
        return pmui.Row(block(old, result.old_repr), block(new, result.new_repr), sizing_mode="stretch_width")


class ArtifactPane(Viewer):
    """
    Lists the artifacts in an `ArtifactStore` and shows the active one,
    following ``store.active`` so newly published artifacts come into focus.
    """

    store = param.ClassSelector(class_=ArtifactStore, doc="The store to display.")

    show_list = param.Boolean(default=True, doc="Whether to show the artifact list.")

    list_width = param.Integer(default=260, doc="Width of the artifact list in pixels.")

    empty_message = param.String(default="Artifacts the agent publishes will appear here.", doc="""
        Shown while the store is empty.""")

    view_options = param.Dict(default={}, doc="""
        Parameters passed to each ArtifactView, e.g. ``{'diff_layout': 'split'}``.""")

    def __init__(self, store: ArtifactStore | None = None, **params):
        if store is None:
            store = ArtifactStore()
        super().__init__(store=store, **params)
        self._views: dict[str, ArtifactView] = {}
        self._ids: list[str] = []
        self._list = pmui.MenuList(items=[], dense=False, sizing_mode="stretch_width")
        self._list.param.watch(self._on_list_select, "active")
        self._empty = pmui.Typography(
            self.param.empty_message, color="text.secondary", variant="body2",
            styles={"text-align": "center", "padding": "48px 16px"}, sizing_mode="stretch_width",
        )
        self._container = pmui.Column(self._empty, sizing_mode="stretch_both")
        self._sidebar = pmui.Paper(
            pmui.Typography("Artifacts", variant="overline", margin=(8, 16, 0, 16)),
            self._list, variant="outlined", width=self.param.list_width, visible=self.param.show_list,
            sizing_mode="stretch_height", margin=(5, 10, 5, 5),
        )
        self._layout = pmui.Row(self._sidebar, self._container, sizing_mode="stretch_both")
        self._watchers: list = []
        self._bind_store()

    def __panel__(self):
        return self._layout

    @property
    def views(self) -> dict[str, ArtifactView]:
        return dict(self._views)

    @property
    def active_view(self) -> ArtifactView | None:
        return self._views.get(self.store.active) if self.store else None

    @param.depends("store", watch=True)
    def _bind_store(self):
        for obj, watcher in self._watchers:
            obj.param.unwatch(watcher)
        self._views.clear()
        self._watchers = [
            (self.store, self.store.param.watch(self._on_artifacts, "artifacts")),
            (self.store, self.store.param.watch(self._on_active, "active")),
        ]
        self._on_artifacts()

    def _on_artifacts(self, *events):
        artifacts = self.store.artifacts
        for removed in set(self._views) - set(artifacts):
            del self._views[removed]
        self._ids = list(artifacts)
        self._list.items = [
            {
                "label": artifact.title or artifact.artifact_id,
                "icon": kind_icon(artifact.kind),
                "secondary": f"{artifact.kind} · {len(artifact.versions)} version{'s' if len(artifact.versions) != 1 else ''}",
            }
            for artifact in artifacts.values()
        ]
        self._on_active()

    def _on_active(self, *events):
        active = self.store.active
        if active not in self.store.artifacts:
            self._list.active = None
            self._container.objects = [self._empty]
            return
        if active not in self._views:
            self._views[active] = ArtifactView(self.store.artifacts[active], **self.view_options)
        index = (self._ids.index(active),)
        if self._list.active != index:
            self._list.active = index
        view = self._views[active].__panel__()
        if self._container.objects != [view]:
            self._container.objects = [view]

    def _on_list_select(self, event):
        index = event.new[0] if isinstance(event.new, tuple) else event.new
        if index is None or index >= len(self._ids):
            return
        self.store.active = self._ids[index]
