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

#: Shown on a group that cannot be undone as a whole because later edits came after it.
BLOCKED_NOTE = "Later edits came after this run; undo them first"

# Every row is two lines of the same height with one line of secondary text, every
# leading icon sits in one column and every trailing button is 30px wide, 8px from the
# right like the view's header buttons, so trailing buttons line up in columns too. A
# row's actions take no space until it is hovered or focused, except on touch screens;
# the muted note that a run cannot be undone stays visible and only explains.
_ROW_ACTION = f'& .MuiListItemButton-root .MuiIconButton-root[title]:not([title="{BLOCKED_NOTE}"])'
_BLOCKED = f'& .MuiIconButton-root[title="{BLOCKED_NOTE}"]'
_LIST_SX = {
    "& .MuiListItemButton-root": {"py": "2px", "pr": 1, "minHeight": "48px", "gap": "4px"},
    "& .MuiListItemIcon-root": {"minWidth": "32px", "color": "var(--mui-palette-text-secondary)"},
    "& .MuiListItemIcon-root .MuiIcon-root": {"fontSize": "20px"},
    "& .MuiListItemText-root": {"my": "2px", "minWidth": 0},
    "& .MuiListItemText-primary, & .MuiListItemText-secondary": {
        "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
    },
    "& .MuiListItemText-secondary": {"fontSize": "0.75rem"},
    # The list sets the first action's and the caret's margins inline.
    "& .MuiListItemButton-root .MuiIconButton-root": {
        "marginLeft": "0 !important", "p": "5px", "width": "30px", "height": "30px", "flex": "none",
    },
    "& .MuiListItemButton-root .MuiIconButton-root .MuiIcon-root, & .MuiListItemButton-root .MuiIconButton-root svg": {
        "fontSize": "20px",
    },
    _BLOCKED: {"color": "var(--mui-palette-text-disabled)", "pointerEvents": "none"},
    "@media (hover: hover)": {
        _ROW_ACTION: {"display": "none"},
        (
            "& .MuiListItemButton-root:hover .MuiIconButton-root[title], "
            "& .MuiListItemButton-root:focus-within .MuiIconButton-root[title]"
        ): {"display": "inline-flex"},
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


def version_detail(version: ArtifactVersion) -> str | None:
    """The full description of a version from its ``detail`` metadata, e.g. every change a run made."""
    detail = (version.metadata or {}).get("detail")
    if isinstance(detail, (list, tuple)):
        detail = " ".join(str(d) for d in detail if d)
    return " ".join(detail.split()) if isinstance(detail, str) and detail.strip() else None


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

    on_show_run = param.Callable(default=None, allow_None=True, doc="""
        Called with a run id from a group's "How was this made?" action,
        offered on groups whose run is in ``state``.""")

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
        self._list.on_action("show_run", self._on_show_run_action)
        self._list.on_click(self._on_click)
        self._watchers: list = []
        self._bound = None
        self._update()

    def __panel__(self):
        return self._list

    @property
    def groups(self) -> list[VersionGroup]:
        return list(self._groups)

    @param.depends("artifact", "can_apply", "user", "state", "on_show_run", watch=True)
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
                if len(group.versions) > 1:
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
            if len(group.versions) == 1:
                # A run that made one version is one row, without a group around it.
                items.append(self._single(group))
                continue
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

    def _active_path(self, expanded: list[tuple[int]]) -> tuple[int, ...] | None:
        current = self.artifact.current if self.artifact else None
        for gi, group in enumerate(reversed(self._groups)):
            if group.first <= (current or 0) <= group.last:
                if len(group.versions) == 1:
                    return (gi,)
                # A collapsed group, or a row beyond the ones shown, has nothing to highlight.
                index = group.last - current
                shown = index < self._limits.get(group.key, ROW_LIMIT)
                return (gi, index) if (gi,) in expanded and shown else None
        return None

    def _on_current(self, event):
        if self.can_apply:
            # The viewed row's restore action moves to the view's Apply button.
            self._refresh()
            return
        self._syncing = True
        try:
            self._list.active = self._active_path([tuple(p) for p in self._list.expanded])
        finally:
            self._syncing = False

    def _restorable(self, version: ArtifactVersion) -> bool:
        artifact = self.artifact
        return self.can_apply and version.version not in (artifact.applied, artifact.current)

    def _meta(self, version: ArtifactVersion) -> str:
        applied = self.artifact.applied == version.version
        return " · ".join(filter(None, (f"v{version.version}{' applied' if applied else ''}", _time(version.created_at))))

    def _row(self, version: ArtifactVersion) -> dict[str, Any]:
        applied = self.artifact.applied == version.version
        item = {
            "id": f"v{version.version}", "version": version.version, "label": version_label(version),
            "secondary": self._meta(version), "icon": "radio_button_checked" if applied else "radio_button_unchecked",
            "color": "primary" if applied else "action",
        }
        if detail := version_detail(version):
            item["tooltip"] = detail
        if self._restorable(version):
            item["actions"] = [self._restore_action(version)]
        return item

    @staticmethod
    def _restore_action(version: ArtifactVersion) -> dict[str, Any]:
        return {"action": "restore", "label": f"Restore v{version.version}", "icon": "settings_backup_restore", "inline": True}

    def _single(self, group: VersionGroup) -> dict[str, Any]:
        version = group.versions[0]
        applied = self.artifact.applied == version.version
        label, summary = self._group_label(group), version_label(version)
        item = {
            "id": f"v{version.version}", "version": version.version, "label": label,
            "secondary": " · ".join(filter(None, (self._meta(version), summary if summary != label else None))),
            "icon": _ACTOR_ICONS.get(group.actor or "", "edit_note"), "color": "primary" if applied else "action",
        }
        # The summary may be cut off in a narrow drawer, so the tooltip has it in full.
        shown = summary if summary != label else None
        if tooltip := ": ".join(filter(None, (shown, version_detail(version)))):
            item["tooltip"] = tooltip
        actions = self._group_actions(group, item)
        if self._restorable(version):
            actions.insert(0, self._restore_action(version))
        actions += self._run_actions(group, item)
        if actions:
            item["actions"] = actions
        return item

    def _header(self, group: VersionGroup, rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group.versions)
        item = {
            "id": group.key, "label": self._group_label(group), "selectable": False, "items": rows,
            "secondary": " · ".join(filter(None, (f"{n} versions", _time(group.versions[0].created_at)))),
            "icon": _ACTOR_ICONS.get(group.actor or "", "edit_note"), "color": "action",
        }
        if actions := self._group_actions(group, item) + self._run_actions(group, item):
            item["actions"] = actions
        return item

    def _run_actions(self, group: VersionGroup, item: dict[str, Any]) -> list[dict[str, Any]]:
        if self.on_show_run is None or self.state is None or group.run_id not in self.state.runs:
            return []
        item["run_id"] = group.run_id
        return [{"action": "show_run", "label": "How was this made?", "icon": "manage_search", "inline": True}]

    def _group_actions(self, group: VersionGroup, item: dict[str, Any]) -> list[dict[str, Any]]:
        applied = self.artifact.applied
        # Only a run that changed the applied version can be undone, back to what the host held before it.
        before = self.artifact._version_before([v.version for v in group.versions])
        if not (self.can_apply and before is not None and applied is not None and applied >= group.first):
            return []
        runs = self.state.runs if self.state is not None else None
        mine = (
            set(self.artifact.run_versions(group.run_id, runs)) if group.run_id is not None
            else {v.version for v in group.versions}
        )
        if self.artifact._undo_crosses(before, mine):
            # Going back would undo later edits made outside the group too, which must be undone first.
            item["tooltip"] = " ".join(filter(None, (f"{BLOCKED_NOTE}.", item.get("tooltip"))))
            return [{"action": "blocked", "label": BLOCKED_NOTE, "icon": "info_outline", "inline": True}]
        label = "Undo run" if applied <= group.last else "Go back to before this run"
        item["before"] = before
        return [{"action": "undo_group", "label": label, "icon": "undo", "inline": True}]

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
        items = self._list.items
        item = items[path[0]] if path and path[0] < len(items) else None
        if item is not None and len(path) == 2:
            rows = item.get("items", [])
            item = rows[path[1]] if path[1] < len(rows) else None
        version = item.get("version") if item is not None and len(path) <= 2 else None
        if version is not None and self.on_view is not None:
            self.on_view(version)

    async def _on_restore_action(self, item: dict[str, Any]):
        await self._restore(item["version"])

    async def _on_undo_action(self, item: dict[str, Any]):
        await self._restore(item["before"], "undo")

    def _on_show_run_action(self, item: dict[str, Any]):
        if self.on_show_run is not None and (run_id := item.get("run_id")):
            self.on_show_run(run_id)

    async def _restore(self, version: int, action: str | None = None):
        if self.on_restore is not None:
            result = self.on_restore(version, action)
            if inspect.isawaitable(result):
                await result


__all__: list[str] = []
