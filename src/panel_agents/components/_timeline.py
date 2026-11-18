"""
The versions of an artifact as a timeline grouped by the run that made them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import panel_material_ui as pmui
import param
from panel.viewable import Viewer

from ..artifacts.model import Artifact, ArtifactVersion, root_run_id
from ..state import RunState

_ACTOR_ICONS = {"agent": "smart_toy", "user": "person"}

# Rows an open group shows before a "Show more" row, so one long group stays cheap to send and render.
ROW_LIMIT = 50

# Two-line rows as tight as a dense single-line list item, so a run's operations scan quickly. A row's
# restore button shows on hover or focus only, except on touch screens.
_ROW_ACTION = "& .MuiCollapse-root .MuiIconButton-root[title]"
_LIST_SX = {
    "& .MuiListItemButton-root": {"py": "1px", "minHeight": "40px"},
    "& .MuiListItemText-root": {"my": "2px"},
    "& .MuiListItemText-secondary": {"fontSize": "0.75rem"},
    "@media (hover: hover)": {
        _ROW_ACTION: {"opacity": 0},
        (
            "& .MuiCollapse-root .MuiListItemButton-root:hover .MuiIconButton-root[title], "
            "& .MuiCollapse-root .MuiListItemButton-root:focus-within .MuiIconButton-root[title], "
            "& .MuiCollapse-root .MuiIconButton-root[title]:focus-visible"
        ): {"opacity": 1},
    },
}


@dataclass
class VersionGroup:
    """Consecutive versions made in one run, or by one actor outside any run."""

    key: str
    run_id: str | None
    actor: str | None
    user: str | None
    versions: list[ArtifactVersion] = field(default_factory=list)

    @property
    def first(self) -> int:
        return self.versions[0].version

    @property
    def last(self) -> int:
        return self.versions[-1].version


def group_versions(versions: list[ArtifactVersion], state: RunState | None = None) -> list[VersionGroup]:
    """
    Group versions in publication order. A group is a run of consecutive
    versions made in the same top-level run of ``state``, so a child run
    a tool starts stays part of the agent's run, or without a run and by
    the same actor and user. Versions a person made while an agent ran
    split that run into two groups around them.
    """
    groups: list[VersionGroup] = []
    runs = state.runs if state is not None else None
    for version in versions:
        if version.dropped:
            continue
        run_id = root_run_id(runs, version.run_id)
        group = groups[-1] if groups else None
        if group is not None and (
            (run_id is not None and run_id == group.run_id)
            or (run_id is None and group.run_id is None and (version.actor, version.user) == (group.actor, group.user))
        ):
            group.versions.append(version)
            continue
        groups.append(VersionGroup(
            key=f"{run_id or 'edits'}@{version.version}", run_id=run_id,
            actor=version.actor, user=version.user, versions=[version],
        ))
    return groups


def version_label(version: ArtifactVersion) -> str:
    """The operation a version performed, from its ``label`` or ``op`` metadata."""
    metadata = version.metadata or {}
    for key in ("label", "op"):
        if isinstance(value := metadata.get(key), str) and value.strip():
            return _shorten(value.strip())
    return f"Version {version.version}"


def _shorten(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _time(timestamp: float | None) -> str:
    if not timestamp:
        return ""
    moment = datetime.fromtimestamp(timestamp)
    if moment.date() == datetime.now().date():
        return moment.strftime("%H:%M")
    return moment.strftime("%b %d, %H:%M")


#: Shown on a group that cannot be undone as a whole because later edits came after it.
BLOCKED_NOTE = "Later edits came after this run; undo them first"


class VersionTimeline(Viewer):
    """
    The versions of an artifact grouped by run, newest first. The latest
    group starts expanded and older ones collapsed, and an open group
    shows its latest ``ROW_LIMIT`` versions with a row showing more, so
    a long history renders only the rows a person asks for.
    """

    artifact = param.ClassSelector(class_=Artifact, allow_None=True)

    state = param.ClassSelector(class_=RunState, default=None, allow_None=True, doc="""
        RunState whose runs label the groups with their input.""")

    user = param.String(default=None, allow_None=True, doc="""
        The person viewing, whose own edits are labelled "You edited".""")

    can_apply = param.Boolean(default=False, doc="""
        Whether rows and groups offer to restore versions.""")

    on_view = param.Callable(default=None, allow_None=True, doc="Called with a version number when a row is clicked.")

    on_restore = param.Callable(default=None, allow_None=True, doc="""
        Called with the version number to apply and ``"undo"`` when it
        undoes a group, or None when it restores a single version.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._groups: list[VersionGroup] = []
        self._expanded: set[str] = set()
        self._limits: dict[str, int] = {}
        self._newest: str | None = None
        self._syncing = False
        self._list = pmui.MenuList(
            items=[], dense=True, sizing_mode="stretch_width", margin=0, level_indent=24, sx=_LIST_SX,
        )
        self._list.param.watch(self._on_active, "active")
        self._list.param.watch(self._on_expanded, "expanded")
        self._list.on_action("restore", self._on_restore_action)
        self._list.on_action("undo_group", self._on_undo_action)
        self._list.on_click(self._on_click)
        self._watchers: list = []
        self._bound = None
        self._update()

    def __panel__(self):
        return self._list

    @property
    def groups(self) -> list[VersionGroup]:
        return list(self._groups)

    @param.depends("artifact", "can_apply", "user", "state", watch=True)
    def _update(self):
        # One refresh however many of these change together.
        if self.artifact is not self._bound:
            for obj, watcher in self._watchers:
                obj.param.unwatch(watcher)
            self._watchers = []
            self._expanded.clear()
            self._limits.clear()
            self._newest = None
            self._bound = self.artifact
            if self.artifact is not None:
                self._watchers = [
                    (self.artifact, self.artifact.param.watch(self._refresh, ["versions", "applied"])),
                    (self.artifact, self.artifact.param.watch(self._on_current, "current")),
                ]
        self._refresh()

    def reveal(self, version: int):
        """Open the group holding ``version`` and show its row."""
        for group in self._groups:
            if group.first <= version <= group.last:
                self._expanded.add(group.key)
                self._limits[group.key] = max(self._limits.get(group.key, ROW_LIMIT), group.last - version + 1)
                self._refresh()
                return

    def _refresh(self, *events):
        artifact = self.artifact
        self._groups = groups = group_versions(artifact.versions, self.state) if artifact else []
        if groups and groups[-1].key != self._newest:
            # A new run starts expanded; the ones before it fold away unless a person opened them.
            if self._newest is not None:
                self._expanded.discard(self._newest)
            self._newest = groups[-1].key
            self._expanded.add(self._newest)
        items, expanded = [], []
        for gi, group in enumerate(reversed(groups)):
            if group.key in self._expanded:
                limit = self._limits.get(group.key, ROW_LIMIT)
                rows = [self._row(v) for v in reversed(group.versions[-limit:])]
                if (hidden := len(group.versions) - limit) > 0:
                    rows.append({
                        "id": f"{group.key}/more", "more": group.key, "label": f"Show {min(hidden, ROW_LIMIT)} more",
                        "secondary": f"{hidden} earlier version{'s' if hidden != 1 else ''}", "selectable": False,
                        "icon": "expand_more",
                    })
                expanded.append((gi,))
            else:
                # Rows are sent only once a group is opened, which keeps a long history cheap to update.
                rows = [{"id": f"{group.key}/rows", "label": "…", "selectable": False, "icon": None}]
            items.append(self._header(group, rows))
        self._syncing = True
        try:
            # The list remaps expanded paths itself when items change, so they are set after the items.
            self._list.items = items
            self._list.param.update(expanded=expanded, active=self._active_path(expanded))
        finally:
            self._syncing = False

    def _active_path(self, expanded: list[tuple[int]]) -> tuple[int, int] | None:
        current = self.artifact.current if self.artifact else None
        groups = self._groups
        for gi, group in enumerate(reversed(groups)):
            if group.first <= (current or 0) <= group.last:
                # A collapsed group, or a row beyond the ones shown, has nothing to highlight.
                index = group.last - current
                shown = index < self._limits.get(group.key, ROW_LIMIT)
                return (gi, index) if (gi,) in expanded and shown else None
        return None

    def _on_current(self, event):
        self._syncing = True
        try:
            self._list.active = self._active_path([tuple(p) for p in self._list.expanded])
        finally:
            self._syncing = False

    def _row(self, version: ArtifactVersion) -> dict[str, Any]:
        applied = self.artifact.applied == version.version
        secondary = " · ".join(filter(None, (f"v{version.version}", _time(version.created_at), "applied" if applied else None)))
        item = {
            "id": f"v{version.version}", "version": version.version, "label": version_label(version),
            "secondary": secondary, "icon": "radio_button_checked" if applied else "radio_button_unchecked",
            "color": "primary" if applied else "action",
        }
        if self.can_apply and not applied:
            item["actions"] = [{
                "action": "restore", "label": f"Restore v{version.version}", "icon": "settings_backup_restore", "inline": True,
            }]
        return item

    def _header(self, group: VersionGroup, rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group.versions)
        count = f"{n} version{'s' if n != 1 else ''}"
        item = {
            "id": group.key, "label": self._group_label(group), "selectable": False, "items": rows,
            "secondary": " · ".join(filter(None, (_time(group.versions[0].created_at), count))),
            "icon": _ACTOR_ICONS.get(group.actor or "", "edit_note"),
        }
        applied = self.artifact.applied
        # Only a run that changed the applied version can be undone, back to what the host held before it.
        before = self.artifact._version_before([v.version for v in group.versions])
        if self.can_apply and before is not None and applied is not None and applied >= group.first:
            runs = self.state.runs if self.state is not None else None
            mine = (
                set(self.artifact.run_versions(group.run_id, runs)) if group.run_id is not None
                else {v.version for v in group.versions}
            )
            if self.artifact._undo_crosses(before, mine):
                # Going back would undo later edits made outside the group too; they are undone first.
                item["secondary"] = " · ".join(filter(None, (item["secondary"], BLOCKED_NOTE)))
                return item
            label = "Undo run" if applied <= group.last else "Go back to before this run"
            item["actions"] = [{"action": "undo_group", "label": label, "icon": "undo", "inline": True}]
            item["before"] = before
        return item

    def _group_label(self, group: VersionGroup) -> str:
        run = self.state.runs.get(group.run_id) if self.state is not None and group.run_id else None
        if run is not None and isinstance(run.input, str) and run.input.strip():
            return _shorten(run.input)
        if group.actor == "user":
            if group.user and group.user == self.user:
                return "You edited"
            return f"{group.user} edited" if group.user else "Edited"
        if group.actor == "agent":
            return run.agent_name if run is not None and run.agent_name else "Agent"
        return "Versions"

    def _on_expanded(self, event):
        if self._syncing:
            return
        items = self._list.items
        expanded = {items[path[0]]["id"] for path in event.new if len(path) == 1 and path[0] < len(items)}
        opened = expanded - self._expanded
        self._expanded = expanded
        if opened:
            self._refresh()

    def _on_click(self, item: dict[str, Any]):
        if isinstance(item, dict) and (key := item.get("more")):
            self._limits[key] = self._limits.get(key, ROW_LIMIT) + ROW_LIMIT
            self._refresh()

    def _on_active(self, event):
        if self._syncing or event.new is None:
            return
        path = tuple(event.new) if isinstance(event.new, (list, tuple)) else (event.new,)
        if len(path) != 2:
            return
        items = self._list.items
        rows = items[path[0]]["items"] if path[0] < len(items) else []
        version = rows[path[1]].get("version") if path[1] < len(rows) else None
        if version is not None and self.on_view is not None:
            self.on_view(version)

    async def _on_restore_action(self, item: dict[str, Any]):
        await self._restore(item["version"])

    async def _on_undo_action(self, item: dict[str, Any]):
        await self._restore(item["before"], "undo")

    async def _restore(self, version: int, action: str | None = None):
        if self.on_restore is not None:
            result = self.on_restore(version, action)
            if inspect.isawaitable(result):
                await result


__all__: list[str] = []
