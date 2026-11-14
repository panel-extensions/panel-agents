"""
Registry mapping values (tool results, artifact versions) to Panel objects.

Renderers are matched by artifact ``kind`` first, then by value type or
predicate, highest priority first, falling back to ``pn.panel``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import panel as pn

from ..artifacts.model import infer_kind

CODE_KINDS = {"sql": "sql", "code": "python", "python": "python", "javascript": "javascript", "yaml": "yaml"}

# Mirrors the Material styling panel.ui applies to DataFrame panes, which
# the panel-material-ui design does not, and extends it to Markdown tables.
TABLE_CSS = """
:host {
  overflow-x: auto;
}
:host table {
  background: transparent;
  border: 1px solid var(--mui-palette-divider);
  border-collapse: separate;
  border-radius: var(--mui-shape-borderRadius);
  border-spacing: 0;
  color: var(--mui-palette-text-primary);
  font: var(--mui-font-body2);
  margin: 0;
  overflow: hidden;
  padding: 0;
  width: 100%;
}
:host th,
:host td {
  border: 0;
  border-bottom: 1px solid var(--mui-palette-divider);
  padding: 10px 16px;
}
:host thead th {
  background: var(--mui-palette-action-hover);
  color: var(--mui-palette-text-secondary);
  font: var(--mui-font-overline);
  font-weight: 500;
  letter-spacing: 0.08333em;
  text-transform: uppercase;
}
:host tbody th {
  color: var(--mui-palette-text-secondary);
  font-weight: 500;
}
:host thead:last-child th,
:host tbody tr:last-child th,
:host tbody tr:last-child td {
  border-bottom: 0;
}
:host tbody tr:hover {
  background: var(--mui-palette-action-hover);
}
"""

SCALARS = (str, int, float, bool, type(None))


@dataclass
class Renderer:
    func: Callable[..., Any]
    kind: str | None = None
    applies: Callable[[Any], bool] | None = None
    priority: int = 0


RENDERERS: list[Renderer] = []


def register_renderer(
    func: Callable[..., Any] | None = None,
    *,
    kind: str | None = None,
    type: type | tuple[type, ...] | None = None,
    applies: Callable[[Any], bool] | None = None,
    priority: int = 0,
):
    """
    Register a renderer, usable as a decorator.

    The renderer is called as ``func(value, **options)`` and must return
    something ``pn.panel`` accepts.
    """
    if type is not None:
        check_type = type
        applies = lambda value: isinstance(value, check_type)

    def decorator(f):
        RENDERERS.append(Renderer(f, kind=kind, applies=applies, priority=priority))
        RENDERERS.sort(key=lambda r: -r.priority)
        return f

    return decorator(func) if func is not None else decorator


def get_renderer(value: Any, kind: str | None = None) -> Callable[..., Any] | None:
    if kind is not None:
        for renderer in RENDERERS:
            if renderer.kind == kind:
                return renderer.func
    for renderer in RENDERERS:
        if renderer.applies is not None and renderer.applies(value):
            return renderer.func
    inferred = infer_kind(value)
    for renderer in RENDERERS:
        if renderer.kind == inferred and renderer.applies is None:
            return renderer.func
    return None


def render(value: Any, kind: str | None = None, **options) -> pn.viewable.Viewable:
    renderer = get_renderer(value, kind)
    if renderer is None:
        return pn.panel(value, sizing_mode="stretch_width")
    return pn.panel(renderer(value, **options))


@register_renderer(kind="table")
def render_table(value, max_rows: int = 1000, page_size: int = 20, **options):
    value = value.head(max_rows) if len(value) > max_rows else value
    options = {
        "disabled": True,
        "show_index": not _is_range_index(value),
        "pagination": "local" if len(value) > page_size else None,
        "page_size": page_size,
        "sizing_mode": "stretch_width",
        **options,
    }
    return pn.widgets.Tabulator(value, **options)


def _is_range_index(df) -> bool:
    index = getattr(df, "index", None)
    return index is None or type(index).__name__ == "RangeIndex"


def _with_table_css(options: dict) -> dict:
    return {**options, "stylesheets": [TABLE_CSS, *options.get("stylesheets", [])]}


@register_renderer(kind="dataframe")
def render_dataframe(value, max_height: int = 400, **options):
    """Compact read-only table, styled like ``pn.ui.DataFrame``."""
    options = {
        "index": not _is_range_index(value),
        "max_height": max_height,
        "sizing_mode": "stretch_width",
        **options,
    }
    return pn.pane.DataFrame(value, **_with_table_css(options))


def is_records(value: Any) -> bool:
    """Whether ``value`` is a non-empty list of flat dicts, i.e. rows of a table."""
    return (
        isinstance(value, list) and bool(value)
        and all(isinstance(row, dict) and all(isinstance(v, SCALARS) for v in row.values()) for row in value)
    )


@register_renderer(applies=is_records)
def render_records(value, **options):
    import pandas as pd

    return render_dataframe(pd.DataFrame.from_records(value), **options)


def _render_code(value, language: str = "python", **options):
    return pn.widgets.CodeEditor(
        value=str(value), language=language, readonly=True,
        sizing_mode="stretch_width", **options,
    )


for _kind, _language in CODE_KINDS.items():
    register_renderer(lambda value, _l=_language, **options: _render_code(value, language=_l, **options), kind=_kind)


@register_renderer(kind="json")
def render_json(value, **options):
    return pn.pane.JSON(json.loads(json.dumps(value, default=repr)), depth=2, sizing_mode="stretch_width", **options)


@register_renderer(kind="text")
def render_text(value, **options):
    return pn.pane.Markdown(str(value), sizing_mode="stretch_width", **_with_table_css(options))
