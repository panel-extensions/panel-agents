from __future__ import annotations

import time
import weakref
from collections.abc import Callable, Mapping
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


def root_run_id(runs: Mapping[str, Any] | None, run_id: str | None) -> str | None:
    """
    The top-level run that ``run_id`` belongs to, following each run's
    ``parent_run_id`` through ``runs``, e.g. ``RunState.runs``.
    """
    seen = set()
    while runs is not None and run_id is not None and run_id not in seen:
        seen.add(run_id)
        run = runs.get(run_id)
        if run is None or run.parent_run_id is None:
            break
        run_id = run.parent_run_id
    return run_id


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

    dropped = param.Boolean(default=False, constant=True, doc="""
        Whether the version's value was dropped to bound memory, see
        ``RunState.drop_artifact_versions``. It keeps its number and who
        made it, but its ``value`` is None and it cannot be applied.""")


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
        ``RunState.undo``.""")

    redo_version = param.Integer(default=None, allow_None=True, constant=True, doc="""
        Number of the version redoing the last undo would apply, or None
        if there is nothing to redo. Publishing an applied version, or
        applying a version other than by undo or redo, ends redo;
        publishing a version that is not applied, a proposal, does not.""")

    redo_run_version = param.Integer(default=None, allow_None=True, constant=True, doc="""
        Number of the version redoing the whole last undo would apply,
        e.g. the version an undone run had left the host document at, or
        None if there is nothing to redo. See ``RunState.redo_run``.""")

    def __init__(self, **params):
        super().__init__(**params)
        # Previously applied version numbers; derived from the order versions were published and applied.
        self._undo: list[int] = []
        self._redo: list[int] = []
        # ``_redo`` split into the steps each undo pushed, oldest first, so a whole undo can be redone at once.
        self._redo_steps: list[list[int]] = []
        # For each version published as applied, the version the host held before it.
        self._applied_before: dict[int, int | None] = {}

    def _stacks(self) -> dict[str, int | None]:
        return {
            "undo_version": self._undo[-1] if self._undo else None,
            "redo_version": self._redo[-1] if self._redo else None,
            "redo_run_version": self._redo_steps[-1][0] if self._redo_steps else None,
        }

    def _clear_redo(self):
        self._redo.clear()
        self._redo_steps.clear()

    def _move_applied(self, number: int, action: str | None = None) -> dict[str, Any]:
        """
        Update the undo and redo stacks for applying ``number`` and return
        the parameter changes, which also show the applied version.

        The undo stack, the applied version and the reversed redo stack
        form one timeline holding each version at most once, so undo and
        redo to a version are never ambiguous. An undo or redo moves along
        it and may skip several steps, as undoing a whole run does. Any
        other apply is a change of its own: it ends redo and moves the
        version to the end of the timeline, so undoing a revert returns to
        where the revert started.
        """
        applied = self.applied
        if applied is None or number == applied:
            return {"applied": number, "current": number}
        stacks = {"undo": (self._undo, self._redo), "redo": (self._redo, self._undo)}.get(action)
        if stacks is not None and number in stacks[0]:
            source, target = stacks
            moved = [applied]
            while (step := source.pop()) != number:
                moved.append(step)
            target.extend(moved)
            if action == "undo":
                self._redo_steps.append(moved)
            else:
                self._trim_redo_steps()
        else:
            self._undo.append(applied)
            if number in self._undo:
                self._undo.remove(number)
            self._clear_redo()
        return {"applied": number, "current": number, **self._stacks()}

    def _trim_redo_steps(self):
        excess = sum(map(len, self._redo_steps)) - len(self._redo)
        while excess > 0:
            step = self._redo_steps[-1]
            cut = min(excess, len(step))
            del step[len(step) - cut:]
            excess -= cut
            if not step:
                self._redo_steps.pop()

    def _version_before(self, versions: list[int]) -> int | None:
        """
        The version the host held before the first of ``versions`` that
        was published as applied, e.g. to undo a whole run; None if none
        of them changed the applied version or nothing was applied before.
        """
        changed = sorted(v for v in versions if v in self._applied_before)
        return self._applied_before[changed[0]] if changed else None

    def run_versions(self, run_id: str, runs: Mapping[str, Any] | None = None) -> list[int]:
        """
        Numbers of the versions published in ``run_id``. With ``runs``,
        e.g. ``RunState.runs``, both ids are resolved to their top-level
        run, so the versions of child runs count as part of it.
        """
        root = root_run_id(runs, run_id)
        return [v.version for v in self.versions if root_run_id(runs, v.run_id) == root]

    def undo_run_target(self, run_id: str | None = None, runs: Mapping[str, Any] | None = None) -> int | None:
        """
        Number of the version undoing the run ``run_id`` would apply: the
        one the host held before the run first changed the applied
        version. ``run_id`` defaults to the run of the applied version and
        ``runs`` works as for ``run_versions``. None if the run is undone
        already, i.e. that version is applied or can only be reached by
        redo. Raises ``ValueError`` if the run never changed the applied
        version, e.g. because it only proposed versions.
        """
        if run_id is None:
            applied = self.applied_version
            run_id = applied.run_id if applied is not None else None
            if run_id is None:
                raise ValueError("The applied version was not published in a run.")
        run_id = root_run_id(runs, run_id)
        numbers = self.run_versions(run_id, runs)
        if not numbers:
            raise ValueError(f"No version of {self.artifact_id!r} was published in run {run_id!r}.")
        before = self._version_before(numbers)
        if before is None:
            raise ValueError(f"Run {run_id!r} did not change the applied version of {self.artifact_id!r} from an earlier one.")
        return before if before in self._undo else None

    def redo_run_target(self, run_id: str | None = None, runs: Mapping[str, Any] | None = None) -> int | None:
        """
        Number of the version redoing a whole undo would apply, or None if
        there is nothing to redo. Without ``run_id`` this is the last undo,
        ``redo_run_version``. With it, it is the earliest undo still
        redoable that undid a version of that run (resolved through
        ``runs`` as for ``run_versions``), so that afterwards none of the
        run's versions is left to redo; undos after it are redone too,
        since redo is linear.
        """
        if run_id is None:
            return self.redo_run_version
        numbers = set(self.run_versions(run_id, runs))
        for step in self._redo_steps:
            if numbers.intersection(step):
                return step[0]
        return None

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

    def drop_versions(self, before: int) -> list[int]:
        """
        Drop the values of the versions numbered below ``before``, except
        the applied one, and return the numbers dropped. Their numbers and
        authors stay; they leave the undo and redo history, so undo stops
        at the oldest version kept. Raises ``ValueError`` for an artifact
        that is not a document, which has no applied version to keep.
        """
        if self.applied is None:
            raise ValueError(f"{self.artifact_id!r} is not a document, so it has no versions to drop.")
        numbers = [
            v.version for v in self.versions[: max(before - 1, 0)]
            if not v.dropped and v.version != self.applied
        ]
        if not numbers:
            return []
        gone = set(numbers)
        for number in numbers:
            with param.edit_constant(self.versions[number - 1]):
                self.versions[number - 1].param.update(value=None, dropped=True)
        self._undo[:] = [n for n in self._undo if n not in gone]
        self._redo[:] = [n for n in self._redo if n not in gone]
        self._redo_steps[:] = [step for step in ([n for n in s if n not in gone] for s in self._redo_steps) if step]
        update = self._stacks()
        if self.current in gone:
            update["current"] = self.applied or next(
                (v.version for v in reversed(self.versions) if not v.dropped), None
            )
        with param.edit_constant(self):
            self.param.update(update)
        self.param.trigger("versions")
        return numbers

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
            self._clear_redo()
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

    def drop_versions(self, artifact_id: str, before: int) -> list[int]:
        """Drop the values of older versions of an artifact, see ``Artifact.drop_versions``."""
        numbers = self.artifacts[artifact_id].drop_versions(before)
        if numbers:
            self._reindex()
        return numbers

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
        if version.tool_call_id is None or version.dropped:
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
            if not refs:
                self._call_watchers.pop(tool_call_id, None)
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
        from 1) of an artifact and show it as ``current``; ``action`` is
        ``"undo"`` or ``"redo"`` when it was applied by undoing or redoing.
        """
        artifact = self.artifacts[artifact_id]
        if not 1 <= version <= len(artifact.versions):
            raise ValueError(f"{artifact_id!r} has no version {version}.")
        if artifact.versions[version - 1].dropped:
            raise ValueError(f"Version {version} of {artifact_id!r} was dropped and cannot be applied.")
        with param.edit_constant(artifact):
            artifact.param.update(artifact._move_applied(version, action))
        self.param.trigger("artifacts")


__all__ = [
    "Artifact",
    "ArtifactStore",
    "ArtifactVersion",
    "infer_kind",
    "root_run_id",
]
