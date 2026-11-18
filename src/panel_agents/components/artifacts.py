"""
Components displaying versioned artifacts and the differences between versions.
"""

from __future__ import annotations

import html
import inspect
import io
import json
from datetime import datetime
from typing import Any

import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..artifacts.diff import (
    Change,
    StructuredDiff,
    TableDiff,
    TextDiff,
    ValueDiff,
    diff,
)
from ..artifacts.model import Artifact, ArtifactStore, ArtifactVersion
from ..security import _escape_markdown, _plain_typography
from ..state import RunState
from ._diff import DiffViewer
from ._timeline import VersionTimeline
from .renderers import render

_KIND_ICONS = {
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

_DOWNLOAD_EXTENSIONS = {
    "table": "csv",
    "sql": "sql",
    "json": "json",
    "code": "py",
    "python": "py",
    "javascript": "js",
    "yaml": "yaml",
    "markdown": "md",
}

_CELL_STYLES = {
    "added": "background-color: rgba(46, 125, 50, 0.18)",
    "removed": "background-color: rgba(211, 47, 47, 0.18)",
    "modified": "background-color: rgba(237, 108, 2, 0.28); font-weight: 600",
}


_CHANGES_CSS = """
:host { width: 100%; }
.pa-changes { display: flex; flex-direction: column; gap: 4px; font-size: 0.85rem; }
.pa-change { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px; padding: 2px 0; }
.pa-op {
  flex: none; width: 1.2em; text-align: center; font-weight: 700;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.pa-op.add { color: var(--mui-palette-success-main, #2e7d32); }
.pa-op.remove { color: var(--mui-palette-error-main, #d32f2f); }
.pa-op.change { color: var(--mui-palette-warning-main, #ed6c02); }
.pa-what { color: var(--mui-palette-text-primary, inherit); }
.pa-path {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem;
  color: var(--mui-palette-text-secondary, #6b6b6b);
}
.pa-old, .pa-new {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem;
  padding: 0 4px; border-radius: 4px;
}
.pa-old { text-decoration: line-through; color: var(--mui-palette-error-main, #c62828); background: rgba(198, 40, 40, 0.08); }
.pa-new { color: var(--mui-palette-success-dark, #1b5e20); background: rgba(46, 125, 50, 0.1); }
.pa-arrow { color: var(--mui-palette-text-disabled, #9e9e9e); }
"""

_OP_SIGNS = {"add": "+", "remove": "−", "change": "~"}


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _changes_html(changes: list[Change]) -> str:
    """HTML listing the changes of a ``StructuredDiff``."""
    rows = []
    for change in changes:
        parts = [f'<span class="pa-op {html.escape(change.op)}">{_OP_SIGNS.get(change.op, "·")}</span>']
        if change.label:
            parts.append(f'<span class="pa-what">{html.escape(change.label)}</span>')
        else:
            parts.append(f'<span class="pa-path">{html.escape(change.path_str)}</span>')
        if change.op == "change" and not change.label:
            parts.append(
                f'<span class="pa-old">{html.escape(_short(change.old))}</span><span class="pa-arrow">→</span>'
                f'<span class="pa-new">{html.escape(_short(change.new))}</span>'
            )
        elif not change.label:
            value, cls = (change.new, "pa-new") if change.op == "add" else (change.old, "pa-old")
            parts.append(f'<span class="{cls}">{html.escape(_short(value))}</span>')
        rows.append(f'<div class="pa-change">{"".join(parts)}</div>')
    return f'<div class="pa-changes">{"".join(rows)}</div>'


def _kind_icon(kind: str) -> str:
    return _KIND_ICONS.get(kind, "inventory_2")


def _version_label(version: ArtifactVersion, applied: bool = False) -> str:
    parts = [f"v{version.version}"]
    if version.created_at:
        parts.append(datetime.fromtimestamp(version.created_at).strftime("%H:%M:%S"))
    if version.tool_call_id:
        parts.append(version.tool_call_id)
    if applied:
        parts.append("applied")
    return " · ".join(parts)


def _version_bytes(value: Any, kind: str) -> tuple[bytes, str]:
    """Serialize a version for download, returning the payload and file extension."""
    if kind == "table" and hasattr(value, "to_csv"):
        return value.to_csv(index=not isinstance(value.index, pd.RangeIndex)).encode(), "csv"
    if kind == "json" and not isinstance(value, str):
        return json.dumps(value, indent=2, default=repr).encode(), "json"
    return ("" if value is None else str(value)).encode(), _DOWNLOAD_EXTENSIONS.get(kind, "txt")


def _row_labels(df: pd.DataFrame, table_diff: TableDiff, key: list[str]) -> list[Any]:
    if table_diff.alignment == "position":
        return list(range(len(df)))
    if table_diff.alignment == "key":
        values = df[key].itertuples(index=False, name=None)
        return [v[0] if len(key) == 1 else v for v in values]
    return list(df.index)


def _table_styles(df: pd.DataFrame, table_diff: TableDiff, side: str, key: list[str] | None = None) -> pd.DataFrame:
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
            styles.iloc[r, :] = _CELL_STYLES[status]
            continue
        for c, name in enumerate(columns):
            if name in extra_cols:
                styles.iloc[r, c] = _CELL_STYLES[status]
            elif name in modified.get(label, ()):
                styles.iloc[r, c] = _CELL_STYLES["modified"]
    return styles


def _apply_styles(table, styles: pd.DataFrame):
    def style(data):
        values = styles.iloc[:len(data), :data.shape[1]].to_numpy()
        return pd.DataFrame(values, index=data.index, columns=data.columns)
    table.style.apply(style, axis=None)
    return table


# A quiet toggle like the drawer's: open shows the theme's selected tint, not a filled button.
_TOGGLE_SX = {
    "border": 0, "borderRadius": "var(--mui-shape-borderRadius)", "px": 1, "py": "4px", "gap": "4px",
    "color": "var(--mui-palette-text-primary)", "textTransform": "none", "lineHeight": 1.5,
    "fontSize": "0.8125rem", "whiteSpace": "nowrap", "width": "auto",
    "& .material-icons, & .MuiIcon-root": {"fontSize": "18px", "lineHeight": 1},
    "&.Mui-selected, &.Mui-selected:hover": {
        "bgcolor": "var(--mui-palette-action-selected)", "color": "var(--mui-palette-text-primary)",
    },
}


class ArtifactView(Viewer):
    """
    A single artifact with its version history, download and a compare
    mode showing the difference between two versions.

    The history lists the versions grouped by the run that made them,
    newest first. For a document artifact with an ``on_apply`` callback
    it also offers undo and redo, which move the applied version through
    ``on_apply`` and are recorded as ``ArtifactApplied`` events; no
    version is ever changed or removed.
    """

    artifact = param.ClassSelector(class_=Artifact, doc="The artifact to display.")

    mode = param.Selector(default="preview", objects=["preview", "compare", "metadata"], doc="""
        Which tab is shown.""")

    base = param.Integer(default=None, allow_None=True, doc="""
        Number of the version compared against, counting from 1. Defaults
        to the applied version of a document, otherwise to the one before
        the target.""")

    target = param.Integer(default=None, allow_None=True, doc="""
        Number of the version compared, counting from 1; defaults to the latest.""")

    diff_layout = param.Selector(default="unified", objects=["unified", "split"], doc="""
        Layout of text diffs.""")

    diff_options = param.Dict(default={}, doc="""
        Options passed to the diff strategy, e.g. ``{'key': 'id'}`` for tables.""")

    show_header = param.Boolean(default=True, doc="Whether to show the title row.")

    on_apply = param.Callable(default=None, allow_None=True, doc="""
        Callback ``(artifact, version) -> None | bool``, possibly async,
        making the host document match a version of a document artifact.
        When set, the header offers "Apply" for versions after the
        applied one and "Revert to" for earlier ones. Raise an exception
        or return False to refuse; the message is shown.""")

    state = param.ClassSelector(class_=RunState, default=None, allow_None=True, doc="""
        RunState recording applied versions as ``ArtifactApplied`` events,
        so they show up in recordings. Without it the artifact is updated
        directly.""")

    user = param.String(default=None, allow_None=True, doc="""
        User recorded on applied versions. Defaults to pn.state.user.""")

    history = param.Boolean(default=None, allow_None=True, doc="""
        Whether the version history is open. None opens it for document
        artifacts and keeps it closed for others; the History button in
        the header toggles it.""")

    can_undo = param.Boolean(default=False, constant=True, doc="""
        Whether ``undo`` can change the applied version.""")

    can_redo = param.Boolean(default=False, constant=True, doc="""
        Whether ``redo`` can change the applied version.""")

    def __init__(self, artifact: Artifact | None = None, **params):
        super().__init__(artifact=artifact, **params)
        self._rendered: dict[tuple[str, int], pn.viewable.Viewable] = {}
        self._applying = False
        self._watchers: list = []
        self._diff_viewer = DiffViewer(context=3, max_height=520)

        self._icon = pmui.Chip(size="small", variant="outlined", margin=(0, 8, 0, 0))
        self._title = _plain_typography(variant="h6", margin=(0, 8), styles={"font-weight": "600"})
        self._count = pmui.Typography(variant="caption", color="text.secondary", margin=(0, 8))
        # A ToggleButton exposes its state as aria-pressed.
        self._history_button = pmui.Toggle(
            label="History", icon="history", icon_size="18px", description="Version history", sx=_TOGGLE_SX,
            margin=(0, 0, 0, 8),
        )
        self._history_button.param.watch(self._on_history_toggle, "value")
        self._undo = pmui.Button(
            label="Undo", icon="undo", variant="text", size="small", color="primary", disabled=True,
            margin=0, on_click=self._on_undo_click,
        )
        self._redo = pmui.Button(
            label="Redo", icon="redo", variant="text", size="small", color="primary", disabled=True,
            margin=0, on_click=self._on_redo_click,
        )
        self._timeline = VersionTimeline(on_view=self._view_version, on_restore=self._step)
        self._timeline_controls = pmui.Row(
            self._undo, self._redo, align="center", margin=(0, 0, 0, 12), styles={"align-items": "center", "gap": "4px"},
        )
        # Undo and Redo stay in view while the versions scroll.
        self._history_panel = pmui.Column(
            self._timeline_controls,
            pmui.Column(self._timeline, sizing_mode="stretch_width", max_height=360, scroll="y-auto", margin=0),
            sizing_mode="stretch_width", visible=False, margin=(8, 0, 0, 0),
            styles={"border-bottom": "1px solid var(--mui-palette-divider)"},
        )
        self._download = pmui.FileDownload(
            label="Download", icon="download", variant="outlined", size="small", color="primary",
            callback=self._download_payload, margin=(0, 0, 0, 8), embed=False,
        )
        self._apply = pmui.Button(
            label="Apply", icon="restore", variant="contained", size="small", color="primary",
            disable_elevation=True, visible=False, margin=(0, 0, 0, 8), on_click=self._on_apply_click,
        )
        self._apply_error = pmui.Alert(object="", severity="error", visible=False, closeable=True, sizing_mode="stretch_width", margin=(4, 0))
        # Wraps the controls below the title when the view is narrow, e.g. in a drawer.
        self._header = pmui.FlexBox(
            pmui.Row(self._icon, self._title, self._count, align="center", margin=0, styles={"align-items": "center"}),
            pmui.Row(self._history_button, self._apply, self._download, align="center", margin=0, styles={"align-items": "center"}),
            flex_wrap="wrap", justify_content="space-between", align_items="center", gap="8px",
            sizing_mode="stretch_width", visible=self.param.show_header, margin=(12, 0, 0, 0),
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
            pmui.FlexBox(
                self._base, pmui.Typography("→", margin=(0, 4), align="center"), self._target, pn.HSpacer(), self._layout,
                flex_wrap="wrap", align_items="center", gap="4px", sizing_mode="stretch_width",
            ),
            self._summary, self._compare_body, sizing_mode="stretch_width",
        )

        self._metadata = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width")

        self._tabs = pmui.Tabs(
            ("Preview", self._preview), ("Compare", compare), ("Metadata", self._metadata),
            active=["preview", "compare", "metadata"].index(self.mode), sizing_mode="stretch_width",
        )
        self._layout_obj = pmui.Column(self._header, self._apply_error, self._history_panel, self._tabs, sizing_mode="stretch_width")

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
            self._update_history()
            self._preview.objects = []
            return
        artifact = self.artifact
        self._watchers = [
            (artifact, artifact.param.watch(self._on_versions, "versions")),
            (artifact, artifact.param.watch(self._on_current, "current")),
            (artifact, artifact.param.watch(self._update_header, ["title", "kind"])),
            (artifact, artifact.param.watch(self._on_applied, ["applied", "undo_version", "redo_version"])),
        ]
        self._update_header()
        self._on_versions()
        self._update_history()

    def _update_header(self, *events):
        artifact = self.artifact
        self._icon.param.update(label=artifact.kind, icon=_kind_icon(artifact.kind))
        self._title.object = _escape_markdown(artifact.title or artifact.artifact_id)
        n = len(artifact.versions)
        self._count.object = f"{n} version{'s' if n != 1 else ''}"

    def _on_versions(self, *events):
        versions = self.versions
        previous_latest = None
        if events:
            previous_latest = len(events[0].old) or None
        options = self._version_options()
        self._update_header()
        self._base.options = options
        self._target.options = options
        latest = len(versions)
        if self.target is None or self.target == previous_latest or self.target > latest:
            self.param.update(target=latest if versions else None, base=self._default_base(latest) if versions else None)
        self._on_current()

    def _default_base(self, target: int) -> int:
        # A new document version is compared with what the host holds, which after a revert is not the previous one.
        applied = self.artifact.applied if self.artifact else None
        if applied is not None and applied != target:
            return applied
        return max(target - 1, 1)

    def _version_options(self) -> dict[str, int]:
        applied = self.artifact.applied if self.artifact else None
        return {_version_label(v, applied=v.version == applied): v.version for v in self.versions}

    def _on_applied(self, *events):
        options = self._version_options()
        self._base.options = options
        self._target.options = options
        # The labels changed but not the values, so the selects must be told to redraw them.
        for select in (self._base, self._target):
            select.param.trigger("value")
        self._update_apply()
        self._update_history()

    @property
    def _history_open(self) -> bool:
        if self.history is None:
            return self.artifact is not None and self.artifact.is_document
        return self.history

    @param.depends("history", "on_apply", "state", "user", watch=True)
    def _update_history(self):
        artifact = self.artifact
        is_open = self._history_open
        self._history_panel.visible = is_open
        self._history_button.value = is_open
        can_apply = artifact is not None and artifact.is_document and self.on_apply is not None
        # A closed history is not built, so artifacts nobody browses cost nothing to update.
        shown = artifact if is_open else None
        self._timeline.param.update(artifact=shown, state=self.state, user=self._user(), can_apply=can_apply)
        self._timeline_controls.visible = can_apply
        with param.edit_constant(self):
            self.param.update(
                can_undo=can_apply and artifact.undo_version is not None,
                can_redo=can_apply and artifact.redo_version is not None,
            )
        self._undo.disabled = not self.can_undo or self._applying
        self._redo.disabled = not self.can_redo or self._applying

    def _on_history_toggle(self, event):
        if event.new != self._history_open:
            self.history = event.new

    async def undo(self) -> bool:
        """
        Undo the last change to the applied version of a document through
        ``on_apply``, as ``RunState.undo`` does. Returns whether the
        version was applied.
        """
        return await self._step(self.artifact.undo_version if self.artifact else None, "undo")

    async def redo(self) -> bool:
        """
        Apply again the version the last undo moved away from through
        ``on_apply``, as ``RunState.redo`` does. Returns whether the
        version was applied.
        """
        return await self._step(self.artifact.redo_version if self.artifact else None, "redo")

    async def undo_run(self, run_id: str | None = None) -> bool:
        """
        Undo a whole run through ``on_apply``, as ``RunState.undo_run``
        does, resolving child runs through ``state``. Raises
        ``ValueError`` if the run never changed the applied version.
        Returns whether the version was applied.
        """
        artifact = self.artifact
        if artifact is None:
            raise ValueError("ArtifactView.undo_run needs an artifact.")
        return await self._step(artifact.undo_run_target(run_id, self._runs), "undo")

    async def redo_run(self, run_id: str | None = None) -> bool:
        """
        Redo a whole undo, e.g. an ``undo_run``, through ``on_apply``, as
        ``RunState.redo_run`` does. Returns whether the version was applied.
        """
        artifact = self.artifact
        return await self._step(artifact.redo_run_target(run_id, self._runs) if artifact else None, "redo")

    @property
    def _runs(self) -> dict | None:
        return self.state.runs if self.state is not None else None

    async def _step(self, number: int | None, action: str | None = None) -> bool:
        if number is None:
            return False
        if not await self.apply_version(number, action=action):
            return False
        self.artifact.current = number
        self._timeline.reveal(number)
        return True

    async def _on_undo_click(self, event):
        await self.undo()

    async def _on_redo_click(self, event):
        await self.redo()

    def _view_version(self, number: int):
        artifact = self.artifact
        if artifact is None:
            return
        artifact.current = number
        applied = artifact.applied
        if applied is not None and applied != number:
            # A past or proposed version of a document is compared with what the host holds.
            self.param.update(base=applied, target=number)

    @param.depends("on_apply", watch=True)
    def _update_apply(self):
        artifact = self.artifact
        if artifact is None or not artifact.is_document or self.on_apply is None or not self.versions:
            self._apply.visible = False
            return
        current, applied = artifact.current, artifact.applied
        self._apply.param.update(
            visible=current != applied,
            label=f"Revert to v{current}" if current < applied else f"Apply v{current}",
            icon="history" if current < applied else "done",
        )

    async def apply_version(self, version: int | None = None, *, action: str | None = None) -> bool:
        """
        Apply version number ``version`` (default: the one shown) to the
        host document through ``on_apply`` and record it. Returns whether
        it was applied; False also while another apply is in progress.
        ``action`` is ``"undo"`` or ``"redo"`` when applying the version
        undoes or redoes changes, as ``undo`` and ``redo`` do.
        """
        if self.on_apply is None:
            raise RuntimeError("ArtifactView.apply_version needs an on_apply callback that changes the host document.")
        if action not in (None, "undo", "redo"):
            raise ValueError(f"action must be 'undo', 'redo' or None, not {action!r}.")
        if self._applying:
            return False
        artifact = self.artifact
        number = artifact.current if version is None else version
        if not isinstance(number, int) or not 1 <= number <= len(self.versions):
            raise ValueError(f"{artifact.artifact_id!r} has no version {number}.")
        version = self.versions[number - 1]
        self._apply_error.visible = False
        self._applying = True
        self._apply.param.update(disabled=True, loading=True)
        self._undo.disabled = self._redo.disabled = True
        try:
            result = self.on_apply(artifact, version)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            self._apply_error.param.update(object=_escape_markdown(str(e) or type(e).__name__), visible=True, closed=False)
            return False
        finally:
            self._applying = False
            self._apply.param.update(disabled=False, loading=False)
            self._update_history()
        if result is False:
            self._apply_error.param.update(object=f"v{version.version} could not be applied.", visible=True, closed=False)
            return False
        if self.state is not None and self.state.artifacts.artifacts.get(artifact.artifact_id) is artifact:
            self.state.apply_artifact_version(artifact.artifact_id, version.version, user=self._user(), action=action)
        else:
            with param.edit_constant(artifact):
                artifact.param.update(artifact._move_applied(number, action))
        return True

    def _user(self) -> str | None:
        if self.user is not None:
            return self.user
        try:
            return pn.state.user
        except Exception:
            return None

    async def _on_apply_click(self, event):
        await self.apply_version()

    def _on_current(self, *events):
        self._update_apply()
        if not self.versions:
            self._preview.objects = []
            self._metadata.object = {}
            return
        current = min(max(self.artifact.current or 1, 1), len(self.versions))
        self._history_button.param.update(label=f"v{current}", description=f"Version history, v{current} shown")
        self._preview.objects = [self._render_version(current - 1)]
        self._update_metadata(current - 1)
        ext = _DOWNLOAD_EXTENSIONS.get(self.artifact.kind, "txt")
        self._download.filename = f"{self.artifact.artifact_id}_v{current}.{ext}"

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

    def _download_payload(self):
        version = self.versions[self.artifact.current - 1]
        data, ext = _version_bytes(version.value, self.artifact.kind)
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
        old, new = versions[self.base - 1], versions[self.target - 1]
        result = self.compute_diff()
        self._layout.visible = isinstance(result, TextDiff)
        self._summary.objects = self._summary_chips(result)
        if isinstance(result, StructuredDiff):
            self._compare_body.objects = [self._structured_comparison(old, new, result)]
        elif isinstance(result, TextDiff):
            self._diff_viewer.param.update(
                old=result.old_text, new=result.new_text, mode=self.diff_layout,
                old_label=_version_label(old), new_label=_version_label(new),
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
        return diff(versions[self.base - 1].value, versions[self.target - 1].value, kind=self.artifact.kind, **self.diff_options)

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
        if isinstance(result, StructuredDiff):
            chips = []
            for op, label, color in (("add", "added", "success"), ("remove", "removed", "error"), ("change", "changed", "warning")):
                if n := sum(c.op == op for c in result.changes):
                    chips.append(self._chip(f"{n} {label}", color))
            return chips
        return [self._chip("Values differ", "warning")]

    def _structured_comparison(self, old: ArtifactVersion, new: ArtifactVersion, result: StructuredDiff):
        changes = pn.pane.HTML(_changes_html(result.changes), stylesheets=[_CHANGES_CSS], sizing_mode="stretch_width", margin=(4, 8))
        if not (result.old_text or result.new_text):
            return changes
        text = DiffViewer(
            old=result.old_text, new=result.new_text, context=3, max_height=520,
            old_label=_version_label(old), new_label=_version_label(new), mode=self.diff_layout,
        )
        return pmui.Tabs(("Changes", changes), ("Text", text), sizing_mode="stretch_width")

    def _table_comparison(self, old: ArtifactVersion, new: ArtifactVersion, result: TableDiff):
        key = self.diff_options.get("key")
        key = [key] if isinstance(key, str) else list(key or [])
        sides = []
        for side, version in (("old", old), ("new", new)):
            df = version.value if version.value is not None else pd.DataFrame()
            table = render(df, kind="table", layout="fit_data_table")
            if isinstance(table, pn.widgets.Tabulator):
                _apply_styles(table, _table_styles(df, result, side, key))
            sides.append(pmui.Column(
                _plain_typography(_version_label(version), variant="subtitle2", margin=(0, 5)),
                table, sizing_mode="stretch_width",
            ))
        return pmui.Row(*sides, sizing_mode="stretch_width")

    def _value_comparison(self, old: ArtifactVersion, new: ArtifactVersion, result: ValueDiff):
        def block(version, text):
            return pmui.Column(
                _plain_typography(_version_label(version), variant="subtitle2", margin=(0, 5)),
                pn.pane.HTML(f"<pre>{html.escape(text)}</pre>", sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )
        return pmui.Row(block(old, result.old_repr), block(new, result.new_repr), sizing_mode="stretch_width")


def _secondary(artifact: Artifact) -> str:
    n = len(artifact.versions)
    text = f"{artifact.kind} · {n} version{'s' if n != 1 else ''}"
    if (version := artifact.applied_version) is not None:
        text += f" · v{version.version} applied"
    return text


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

    on_apply = param.Callable(default=None, allow_None=True, doc="""
        Callback applying a version of a document artifact to the host
        document; see ``ArtifactView.on_apply``.""")

    state = param.ClassSelector(class_=RunState, default=None, allow_None=True, doc="""
        RunState recording applied versions. Its artifacts are shown when
        no store is given.""")

    def __init__(self, store: ArtifactStore | None = None, **params):
        if store is None:
            state = params.get("state")
            store = state.artifacts if state is not None else ArtifactStore()
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
            self._list, variant="outlined", width=self.param.list_width,
            sizing_mode="stretch_height", margin=(5, 10, 5, 5),
        )
        self._layout = pmui.Row(sizing_mode="stretch_both")
        self._sync_list()
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
                "icon": _kind_icon(artifact.kind),
                "secondary": _secondary(artifact),
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
            self._views[active] = ArtifactView(
                self.store.artifacts[active], on_apply=self.on_apply, state=self.state, **self.view_options,
            )
        index = (self._ids.index(active),)
        if self._list.active != index:
            self._list.active = index
        view = self._views[active].__panel__()
        if self._container.objects != [view]:
            self._container.objects = [view]

    @param.depends("show_list", watch=True)
    def _sync_list(self):
        # A hidden list would still count towards the pane's minimum width.
        self._layout.objects = [self._sidebar, self._container] if self.show_list else [self._container]

    @param.depends("on_apply", "state", watch=True)
    def _propagate(self):
        for view in self._views.values():
            view.param.update(on_apply=self.on_apply, state=self.state)

    def _on_list_select(self, event):
        index = event.new[0] if isinstance(event.new, tuple) else event.new
        if index is None or index >= len(self._ids):
            return
        self.store.active = self._ids[index]


__all__ = [
    "ArtifactPane",
    "ArtifactView",
]
