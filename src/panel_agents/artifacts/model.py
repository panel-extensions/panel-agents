from __future__ import annotations

import time
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

    tool_call_id = param.String(default=None, allow_None=True, constant=True)

    metadata = param.Dict(default={}, constant=True)


class Artifact(param.Parameterized):
    """Something the agent produced, kept as a list of versions."""

    artifact_id = param.String(constant=True)

    kind = param.String(default="object", doc="""
        What the artifact holds, e.g. 'table', 'sql', 'code', 'text',
        'json' or 'plot'. Used to select renderers and diff strategies.""")

    title = param.String(default="")

    versions = param.List(default=[], item_type=ArtifactVersion)

    current = param.Integer(default=0, doc="""
        Index of the version currently selected for display.""")

    @property
    def latest(self) -> ArtifactVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def value(self) -> Any:
        return self.versions[self.current].value if self.versions else None

    def add_version(
        self,
        value: Any,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> ArtifactVersion:
        version = ArtifactVersion(
            version=len(self.versions) + 1,
            value=value,
            created_at=time.time() if created_at is None else created_at,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )
        self.param.update(versions=[*self.versions, version], current=len(self.versions))
        return version


class ArtifactStore(param.Parameterized):
    """Per-session collection of artifacts."""

    artifacts = param.Dict(default={}, doc="""
        Mapping from artifact id to Artifact, in publication order.""")

    active = param.String(default=None, allow_None=True, doc="""
        Id of the artifact currently in focus.""")

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
    ) -> Artifact:
        """Create an artifact, or add a new version if the id exists."""
        if artifact_id in self.artifacts:
            artifact = self.artifacts[artifact_id]
            if title:
                artifact.title = title
            artifact.add_version(value, tool_call_id, metadata, created_at)
            self.param.trigger("artifacts")
        else:
            artifact = Artifact(
                artifact_id=artifact_id,
                kind=kind or infer_kind(value),
                title=title or artifact_id,
            )
            artifact.add_version(value, tool_call_id, metadata, created_at)
            self.artifacts = {**self.artifacts, artifact_id: artifact}
        self.active = artifact_id
        return artifact
