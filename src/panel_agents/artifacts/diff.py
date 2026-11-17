"""
Diff strategies for artifact versions, keyed by artifact kind.

Strategies are plain functions returning plain dataclasses so they can be
reused outside of the UI, e.g. by replay or evaluation tooling.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .model import infer_kind

_TEXT_KINDS = ("text", "markdown", "sql", "code", "python", "javascript", "yaml", "json")


@dataclass
class Diff:
    """Base class for diff results."""

    kind: str = "value"
    identical: bool = True

    @property
    def summary(self) -> str:
        return "identical" if self.identical else "values differ"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "summary": self.summary}


@dataclass
class ValueDiff(Diff):

    old_repr: str = ""
    new_repr: str = ""


@dataclass
class TextDiff(Diff):

    kind: str = "text"
    old_text: str = ""
    new_text: str = ""
    unified: str = ""
    added: int = 0
    removed: int = 0
    hunks: int = 0
    similarity: float = 1.0

    @property
    def summary(self) -> str:
        if self.identical:
            return "identical"
        return f"+{self.added} −{self.removed} lines in {self.hunks} hunk{'s' if self.hunks != 1 else ''}"


@dataclass
class TableDiff(Diff):

    kind: str = "table"
    old_shape: tuple[int, int] = (0, 0)
    new_shape: tuple[int, int] = (0, 0)
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    columns_reordered: bool = False
    dtype_changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    alignment: str | None = None
    rows_added: list[Any] = field(default_factory=list)
    rows_removed: list[Any] = field(default_factory=list)
    rows_modified: list[Any] = field(default_factory=list)
    cells_modified: list[tuple[Any, str]] = field(default_factory=list)

    @property
    def shape_changed(self) -> bool:
        return tuple(self.old_shape) != tuple(self.new_shape)

    @property
    def summary(self) -> str:
        if self.identical:
            return "identical"
        parts = []
        if self.shape_changed:
            parts.append(f"shape {self.old_shape[0]}×{self.old_shape[1]} → {self.new_shape[0]}×{self.new_shape[1]}")
        if self.columns_added:
            parts.append(f"+{len(self.columns_added)} col")
        if self.columns_removed:
            parts.append(f"−{len(self.columns_removed)} col")
        if self.dtype_changes:
            parts.append(f"{len(self.dtype_changes)} dtype change{'s' if len(self.dtype_changes) != 1 else ''}")
        if self.alignment is None:
            parts.append("rows not comparable")
        else:
            for label, rows in (("added", self.rows_added), ("removed", self.rows_removed), ("modified", self.rows_modified)):
                if rows:
                    parts.append(f"{len(rows)} row{'s' if len(rows) != 1 else ''} {label}")
        if self.columns_reordered and len(parts) == 0:
            parts.append("columns reordered")
        return ", ".join(parts) or "values differ"


@dataclass
class Change:
    """One semantic change between two versions of a structured value."""

    op: str
    """'add', 'remove' or 'change'."""
    path: tuple[str | int, ...]
    """Keys and list positions leading to the change, e.g. ``("nodes", "sales", "config")``."""
    old: Any = None
    new: Any = None
    label: str | None = None
    """Human readable description shown instead of the path, e.g. "Node 'sales' added"."""

    @property
    def path_str(self) -> str:
        """The path for display, e.g. ``nodes.sales.config`` or ``edges[2]``; not meant to be parsed."""
        text = ""
        for key in self.path:
            text += f"[{key}]" if isinstance(key, int) else (f".{key}" if text else str(key))
        return text


@dataclass
class StructuredDiff(Diff):
    """
    A list of semantic changes, the result to return from a custom diff
    strategy for documents such as configurations or dashboards.

    Set ``old_text`` and ``new_text`` to also offer a line diff of the
    serialized versions.
    """

    kind: str = "structured"
    changes: list[Change] = field(default_factory=list)
    old_text: str = ""
    new_text: str = ""

    @property
    def summary(self) -> str:
        if self.identical:
            return "identical"
        counts = {op: sum(c.op == op for c in self.changes) for op in ("add", "remove", "change")}
        labels = {"add": "added", "remove": "removed", "change": "changed"}
        return ", ".join(f"{n} {labels[op]}" for op, n in counts.items() if n) or "values differ"


_DIFF_STRATEGIES: dict[str, Callable[..., Diff]] = {}


def register_diff(*kinds: str):
    """Register a diff strategy ``func(old, new, **options) -> Diff`` for the given kinds."""
    def decorator(func):
        for kind in kinds:
            _DIFF_STRATEGIES[kind] = func
        return func
    return decorator


def _get_diff_strategy(kind: str | None, old: Any = None, new: Any = None) -> Callable[..., Diff]:
    if kind in _DIFF_STRATEGIES:
        return _DIFF_STRATEGIES[kind]
    for value in (new, old):
        if value is not None:
            inferred = infer_kind(value)
            if inferred in _DIFF_STRATEGIES:
                return _DIFF_STRATEGIES[inferred]
    return diff_values


def diff(old: Any, new: Any, kind: str | None = None, **options) -> Diff:
    """Diff two artifact values using the strategy registered for ``kind``."""
    return _get_diff_strategy(kind, old, new)(old, new, **options)


def diff_values(old: Any, new: Any, **options) -> ValueDiff:
    try:
        identical = bool(old == new)
    except Exception:
        identical = repr(old) == repr(new)
    return ValueDiff(identical=identical, old_repr=repr(old), new_repr=repr(new))


def _join(path: tuple, key: Any) -> tuple:
    return (*path, key)


def _changes(old: Any, new: Any, path: tuple) -> list[Change]:
    if isinstance(old, dict) and isinstance(new, dict):
        changes = []
        for key in old:
            if key not in new:
                changes.append(Change("remove", _join(path, key), old=old[key]))
            else:
                changes.extend(_changes(old[key], new[key], _join(path, key)))
        changes.extend(Change("add", _join(path, key), new=new[key]) for key in new if key not in old)
        return changes
    if isinstance(old, list) and isinstance(new, list):
        a = [json.dumps(v, sort_keys=True, default=repr) for v in old]
        b = [json.dumps(v, sort_keys=True, default=repr) for v in new]
        changes = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace" and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    changes.extend(_changes(old[i1 + k], new[j1 + k], _join(path, j1 + k)))
                continue
            changes.extend(Change("remove", _join(path, i), old=old[i]) for i in range(i1, i2))
            changes.extend(Change("add", _join(path, j), new=new[j]) for j in range(j1, j2))
        return changes
    try:
        equal = bool(old == new) and type(old) is type(new)
    except Exception:
        equal = repr(old) == repr(new)
    return [] if equal else [Change("change", path, old=old, new=new)]


def diff_structured(old: Any, new: Any, **options) -> StructuredDiff:
    """
    Diff two nested dict/list values into the changes needed to turn
    ``old`` into ``new``: keys added and removed, list items inserted and
    deleted, and leaf values changed.

    Custom strategies for structured documents can start from this and
    relabel the changes, e.g. ``("nodes", "sales")`` added as "Node 'sales' added".
    """
    changes = _changes({} if old is None else old, {} if new is None else new, ())
    return StructuredDiff(identical=not changes, changes=changes)


register_diff("structured")(diff_structured)


def _to_text(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if kind == "json":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return value
        return json.dumps(value, indent=2, sort_keys=True, default=repr, ensure_ascii=False)
    return value if isinstance(value, str) else str(value)


def diff_text(old: Any, new: Any, kind: str = "text", context: int = 3, **options) -> TextDiff:
    old_text, new_text = _to_text(old, kind), _to_text(new, kind)
    a, b = old_text.splitlines(), new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    unified = list(difflib.unified_diff(a, b, "old", "new", n=context, lineterm=""))
    return TextDiff(
        kind=kind,
        identical=old_text == new_text,
        old_text=old_text,
        new_text=new_text,
        unified="\n".join(unified),
        added=added,
        removed=removed,
        hunks=sum(line.startswith("@@") for line in unified),
        similarity=matcher.ratio() if a or b else 1.0,
    )


for _kind in _TEXT_KINDS:
    register_diff(_kind)(lambda old, new, _k=_kind, **options: diff_text(old, new, kind=_k, **options))


def _as_pandas(value: Any):
    import pandas as pd

    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _values_equal(a, b):
    import numpy as np
    import pandas as pd

    a = a.astype(object).to_numpy()
    b = b.astype(object).to_numpy()
    both_na = pd.isna(a) & pd.isna(b)
    with np.errstate(invalid="ignore"):
        try:
            equal = np.asarray(a == b, dtype=bool)
        except (TypeError, ValueError):
            equal = np.vectorize(lambda x, y: repr(x) == repr(y), otypes=[bool])(a, b)
    return equal | both_na


@register_diff("table", "dataframe")
def diff_table(old: Any, new: Any, key: str | list[str] | None = None, **options) -> TableDiff:
    """
    Diff two DataFrames.

    Rows are aligned on ``key`` columns if given, otherwise on the index
    when it is unique in both frames, otherwise by position when the row
    counts match. Values are compared on the shared columns only.
    """
    old_df, new_df = _as_pandas(old), _as_pandas(new)
    old_cols = [str(c) for c in old_df.columns]
    new_cols = [str(c) for c in new_df.columns]
    shared = [c for c in old_df.columns if c in set(new_df.columns)]
    result = TableDiff(
        old_shape=tuple(old_df.shape),
        new_shape=tuple(new_df.shape),
        columns_added=[c for c in new_cols if c not in old_cols],
        columns_removed=[c for c in old_cols if c not in new_cols],
        columns_reordered=[c for c in old_cols if c in new_cols] != [c for c in new_cols if c in old_cols],
        dtype_changes={
            str(c): (str(old_df[c].dtype), str(new_df[c].dtype))
            for c in shared if old_df[c].dtype != new_df[c].dtype
        },
    )

    keys = [key] if isinstance(key, str) else list(key or [])
    if keys and all(k in old_df.columns and k in new_df.columns for k in keys):
        old_df, new_df = old_df.set_index(keys), new_df.set_index(keys)
        shared = [c for c in shared if c not in keys]
        alignment = "key"
    else:
        alignment = "index"
    if not (old_df.index.is_unique and new_df.index.is_unique):
        if len(old_df) == len(new_df) and alignment == "index":
            old_df, new_df = old_df.reset_index(drop=True), new_df.reset_index(drop=True)
            alignment = "position"
        else:
            alignment = None

    if alignment is not None:
        result.alignment = alignment
        new_index = set(new_df.index)
        old_index = set(old_df.index)
        result.rows_added = [i for i in new_df.index if i not in old_index]
        result.rows_removed = [i for i in old_df.index if i not in new_index]
        common = [i for i in old_df.index if i in new_index]
        if common and shared:
            a = old_df.loc[common, shared]
            b = new_df.loc[common, shared]
            equal = _values_equal(a, b)
            modified_rows = []
            for r, label in enumerate(common):
                changed = [str(shared[c]) for c in range(len(shared)) if not equal[r, c]]
                if changed:
                    modified_rows.append(label)
                    result.cells_modified.extend((label, c) for c in changed)
            result.rows_modified = modified_rows

    result.identical = not (
        result.shape_changed or result.columns_added or result.columns_removed or result.columns_reordered
        or result.dtype_changes or result.rows_added or result.rows_removed or result.rows_modified
        or result.alignment is None
    )
    return result


__all__ = [
    "Change",
    "Diff",
    "StructuredDiff",
    "TableDiff",
    "TextDiff",
    "ValueDiff",
    "diff",
    "diff_structured",
    "diff_table",
    "diff_text",
    "diff_values",
    "register_diff",
]
