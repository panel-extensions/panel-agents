from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from typing import Any

import param


def infer_kind(value: Any) -> str:
    module = type(value).__module__.split(".")[0]
    name = type(value).__name__
    if name == "DataFrame" and module in ("pandas", "polars", "narwhals"):
        return "table"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (dict, list)):
        return "json"
    return "object"


class ArtifactVersion(param.Parameterized):

    version = param.Integer(default=1, bounds=(1, None), constant=True)

    value = param.Parameter(constant=True)

    created_at = param.Number(constant=True)

    tool_call_id = param.String(default=None, allow_None=True, constant=True, doc="""
        Id of the tool call that published the version, if a tool did.""")

    run_id = param.String(default=None, allow_None=True, constant=True, doc="""
        Id of the run the version was published in: an agent run, or a
        run grouping a person's edits (``RunState.user_run``).""")

    actor = param.String(default=None, allow_None=True, constant=True, doc="""
        Who made the version, e.g. 'agent' or 'user'. A free string, so
        hosts may add their own, such as 'import'. None in recordings
        made before versions recorded their actor.""")

    user = param.String(default=None, allow_None=True, constant=True, doc="""
        Name of the person who made the version, if a person did.""")

    metadata = param.Dict(default={}, constant=True)


class Artifact(param.Parameterized):
    """Something the agent produced, kept as a list of versions."""

    artifact_id = param.String(constant=True)

    kind = param.String(default="object", doc="""
        What the artifact holds, e.g. 'table', 'sql', 'code', 'text',
        'json' or 'plot'. Used to select renderers and diff strategies.""")

    title = param.String(default="")

    versions = param.List(default=[], item_type=ArtifactVersion)

    current = param.Integer(default=None, allow_None=True, bounds=(1, None), doc="""
        Number of the version currently selected for display, counting
        from 1 like ``ArtifactVersion.version``. None while there are no versions.""")

    applied = param.Integer(default=None, allow_None=True, constant=True, doc="""
        Number of the version the host document holds, counting from 1;
        None unless the artifact is a document. Changed by publishing an
        applied version or by ``RunState.apply_artifact_version``.""")

    undo_version = param.Integer(default=None, allow_None=True, constant=True, doc="""
        Number of the version undoing the last change to the applied
        version would apply, or None if there is nothing to undo. Every
        change to the applied version can be undone: publishing an applied
        version, and applying an earlier or later one by hand. See
        ``ArtifactView.undo``.""")

    redo_version = param.Integer(default=None, allow_None=True, constant=True, doc="""
        Number of the version redoing the last undo would apply, or None
        if there is nothing to redo. Publishing an applied version ends redo.""")

    def __init__(self, **params):
        super().__init__(**params)
        # Previously applied version numbers; derived from the order versions were published and applied.
        self._undo: list[int] = []
        self._redo: list[int] = []
        # For each version published as applied, the version the host held before it.
        self._applied_before: dict[int, int | None] = {}

    def _stacks(self) -> dict[str, int | None]:
        return {
            "undo_version": self._undo[-1] if self._undo else None,
            "redo_version": self._redo[-1] if self._redo else None,
        }

    def _move_applied(self, number: int, action: str | None = None) -> dict[str, Any]:
        """
        Update the undo and redo stacks for applying ``number`` and return
        the parameter changes. An undo or redo may skip several steps, as
        undoing a whole run does; any other apply is a change of its own,
        which can be undone and ends redo.
        """
        applied = self.applied
        if applied is None or number == applied:
            return {"applied": number}
        stacks = {"undo": (self._undo, self._redo), "redo": (self._redo, self._undo)}.get(action)
        if stacks is not None and number in stacks[0]:
            source, target = stacks
            target.append(applied)
            while (step := source.pop()) != number:
                target.append(step)
        else:
            self._undo.append(applied)
            self._redo.clear()
        return {"applied": number, **self._stacks()}

    def _version_before(self, versions: list[int]) -> int | None:
        """
        The version the host held before the first of ``versions`` that
        was published as applied, e.g. to undo a whole run; None if none
        of them changed the applied version or nothing was applied before.
        """
        changed = sorted(v for v in versions if v in self._applied_before)
        return self._applied_before[changed[0]] if changed else None

    @property
    def is_document(self) -> bool:
        """
        Whether the artifact mirrors a document the host app owns, i.e. one
        of its versions was published as applied or applied by a person.
        """
        return self.applied is not None

    @property
    def applied_version(self) -> ArtifactVersion | None:
        """The version the host document holds, or None if the artifact is not a document."""
        number = self.applied
        return self.versions[number - 1] if number is not None and number <= len(self.versions) else None

    @property
    def latest(self) -> ArtifactVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def value(self) -> Any:
        return self.versions[self.current - 1].value if self.versions and self.current else None

    def add_version(
        self,
        value: Any,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
        applied: bool = False,
        *,
        run_id: str | None = None,
        actor: str | None = None,
        user: str | None = None,
    ) -> ArtifactVersion:
        version = ArtifactVersion(
            version=len(self.versions) + 1,
            value=value,
            created_at=time.time() if created_at is None else created_at,
            tool_call_id=tool_call_id,
            run_id=run_id,
            actor=actor,
            user=user,
            metadata=metadata or {},
        )
        update = {"versions": [*self.versions, version], "current": version.version}
        if applied:
            self._applied_before[version.version] = self.applied
            if self.applied is not None:
                self._undo.append(self.applied)
            self._redo.clear()
            update["applied"] = version.version
        update.update(self._stacks())
        with param.edit_constant(self):
            self.param.update(update)
        return version


class ArtifactStore(param.Parameterized):
    """Per-session collection of artifacts."""

    artifacts = param.Dict(default={}, doc="""
        Mapping from artifact id to Artifact, in publication order.""")

    active = param.String(default=None, allow_None=True, doc="""
        Id of the artifact currently in focus.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._call_watchers: dict[str, list[weakref.WeakMethod]] = {}
        self._reindex()
        self.param.watch(self._artifacts_changed, "artifacts")

    def _reindex(self):
        # Tool call id -> artifact id -> latest version that call published.
        self._by_call: dict[str, dict[str, ArtifactVersion]] = {}
        self._publishers: dict[str, dict[str, None]] = {}
        self._order: dict[str, int] = {}
        self._indexed = self.artifacts
        for artifact in self.artifacts.values():
            for version in artifact.versions:
                self._index(artifact.artifact_id, version)

    def _artifacts_changed(self, *_):
        # publish() keeps the index up to date; artifacts replaced from outside are indexed again.
        if self.artifacts is self._indexed:
            return
        self._reindex()
        for call_id in list(self._call_watchers):
            self._notify(call_id)

    def __getitem__(self, artifact_id: str) -> Artifact:
        return self.artifacts[artifact_id]

    def __contains__(self, artifact_id: str) -> bool:
        return artifact_id in self.artifacts

    def __iter__(self):
        return iter(self.artifacts.values())

    def __len__(self) -> int:
        return len(self.artifacts)

    def publish(
        self,
        artifact_id: str,
        value: Any,
        kind: str | None = None,
        title: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
        applied: bool = False,
        *,
        run_id: str | None = None,
        actor: str | None = None,
        user: str | None = None,
    ) -> Artifact:
        """
        Create an artifact, or add a new version if the id exists.

        ``applied`` marks the version as the one the host document now
        holds, which makes the artifact a document. ``run_id``, ``actor``
        and ``user`` record who made the version, see ``ArtifactVersion``.
        """
        who = {"run_id": run_id, "actor": actor, "user": user}
        if artifact_id in self.artifacts:
            artifact = self.artifacts[artifact_id]
            if title:
                artifact.title = title
            artifact.add_version(value, tool_call_id, metadata, created_at, applied, **who)
            self._index(artifact_id, artifact.versions[-1])
            self.param.trigger("artifacts")
        else:
            artifact = Artifact(
                artifact_id=artifact_id,
                kind=kind or infer_kind(value),
                title=title or artifact_id,
            )
            artifact.add_version(value, tool_call_id, metadata, created_at, applied, **who)
            artifacts = {**self.artifacts, artifact_id: artifact}
            self._index(artifact_id, artifact.versions[-1])
            self._indexed = artifacts
            self.artifacts = artifacts
        self.active = artifact_id
        for call_id in list(self._publishers.get(artifact_id, ())):
            self._notify(call_id)
        return artifact

    def published_by(self, tool_call_id: str) -> list[tuple[Artifact, ArtifactVersion]]:
        """
        Artifacts a tool call published, in publication order, each paired
        with the latest version that call published. Looking a call up
        does not scan the versions, so it stays cheap however many there
        are. Versions are attributed to calls as ``publish`` adds them or
        when ``artifacts`` is replaced; ``Artifact.add_version`` called
        directly does not update the lookup.
        """
        found = self._by_call.get(tool_call_id)
        if not found:
            return []
        ids = sorted(found, key=lambda aid: self._order.get(aid, 0)) if len(found) > 1 else list(found)
        published = []
        for aid in ids:
            artifact, version = self.artifacts.get(aid), found[aid]
            # Artifacts changed in place outside publish() may no longer hold the version.
            if artifact is not None and version in artifact.versions[version.version - 1:version.version]:
                published.append((artifact, version))
        return published

    def _index(self, artifact_id: str, version: ArtifactVersion):
        self._order.setdefault(artifact_id, len(self._order))
        if version.tool_call_id is None:
            return
        self._by_call.setdefault(version.tool_call_id, {})[artifact_id] = version
        self._publishers.setdefault(artifact_id, {})[version.tool_call_id] = None

    def _watch_call(self, tool_call_id: str, callback: Callable[[], Any]) -> Callable[[], None]:
        """
        Call the bound method ``callback`` whenever an artifact the tool
        call published gets a new version, from any call, or the artifacts
        are replaced. Only a weak reference is kept, so the store does not
        keep a card alive; cards unwatch when released.
        """
        ref = weakref.WeakMethod(callback)
        self._call_watchers.setdefault(tool_call_id, []).append(ref)

        def unwatch():
            refs = self._call_watchers.get(tool_call_id, [])
            if ref in refs:
                refs.remove(ref)
        return unwatch

    def _notify(self, tool_call_id: str):
        refs = self._call_watchers.get(tool_call_id)
        if not refs:
            return
        for ref in list(refs):
            if (callback := ref()) is not None:
                callback()
        refs[:] = [ref for ref in refs if ref() is not None]

    def mark_applied(self, artifact_id: str, version: int, action: str | None = None):
        """
        Record that the host document now holds ``version`` (counting
        from 1) of an artifact; ``action`` is ``"undo"`` or ``"redo"`` when
        it was applied by undoing or redoing.
        """
        artifact = self.artifacts[artifact_id]
        if not 1 <= version <= len(artifact.versions):
            raise ValueError(f"{artifact_id!r} has no version {version}.")
        with param.edit_constant(artifact):
            artifact.param.update(artifact._move_applied(version, action))
        self.param.trigger("artifacts")


__all__ = [
    "Artifact",
    "ArtifactStore",
    "ArtifactVersion",
    "infer_kind",
]
