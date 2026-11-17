"""
Registry mapping values (tool results, artifact versions) to Panel objects.

Renderers are matched by artifact ``kind`` first, then by value type or
predicate, highest priority first, falling back to ``pn.panel``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import panel as pn

from ..artifacts.model import infer_kind
from ..security import untrusted_markdown

_CODE_KINDS = {"sql": "sql", "code": "python", "python": "python", "javascript": "javascript", "yaml": "yaml"}

# Mirrors the Material styling panel.ui applies to DataFrame panes, which
# the panel-material-ui design does not, and extends it to Markdown tables.
_TABLE_CSS = """
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

_SCALARS = (str, int, float, bool, type(None))


@dataclass
class _Renderer:
    func: Callable[..., Any]
    kind: str | None = None
    applies: Callable[[Any], bool] | None = None
    priority: int = 0


_RENDERERS: list[_Renderer] = []


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
        _RENDERERS.append(_Renderer(f, kind=kind, applies=applies, priority=priority))
        _RENDERERS.sort(key=lambda r: -r.priority)
        return f

    return decorator(func) if func is not None else decorator


def _get_renderer(value: Any, kind: str | None = None) -> Callable[..., Any] | None:
    if kind is not None:
        for renderer in _RENDERERS:
            if renderer.kind == kind:
                return renderer.func
    for renderer in _RENDERERS:
        if renderer.applies is not None and renderer.applies(value):
            return renderer.func
    inferred = infer_kind(value)
    for renderer in _RENDERERS:
        if renderer.kind == inferred and renderer.applies is None:
            return renderer.func
    return None


def render(value: Any, kind: str | None = None, **options) -> pn.viewable.Viewable:
    """
    Render a tool result or artifact value with the registered renderers.
    Text, including a renderer returning a string, is rendered with
    ``security.inert_markdown`` and without math, so it cannot run script
    or load remote resources by itself, and values with no renderer that
    would render as HTML are shown as plain text.
    """
    renderer = _get_renderer(value, kind)
    if renderer is None:
        if isinstance(value, str):
            return _render_text(value, **options)
        return _inert(pn.panel(value, sizing_mode="stretch_width"), value)
    result = renderer(value, **options)
    if isinstance(result, str):
        return _render_text(result)
    return pn.panel(result)


def _inert(pane: pn.viewable.Viewable, value: Any) -> pn.viewable.Viewable:
    # An object without a renderer would otherwise show its _repr_html_ or _repr_markdown_ as live HTML.
    if isinstance(pane, (pn.pane.HTML, pn.pane.Markdown)) and not isinstance(pane, pn.pane.DataFrame):
        return pn.pane.Str(str(value), sizing_mode="stretch_width")
    return pane


@register_renderer(kind="table")
def _render_table(value, max_rows: int = 1000, page_size: int = 20, **options):
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
    return {**options, "stylesheets": [_TABLE_CSS, *options.get("stylesheets", [])]}


@register_renderer(kind="dataframe")
def _render_dataframe(value, max_height: int = 400, **options):
    """Compact read-only table, styled like ``pn.ui.DataFrame``."""
    options = {
        "index": not _is_range_index(value),
        "max_height": max_height,
        "sizing_mode": "stretch_width",
        **options,
    }
    return pn.pane.DataFrame(value, **_with_table_css(options))


def _is_records(value: Any) -> bool:
    """Whether ``value`` is a non-empty list of flat dicts, i.e. rows of a table."""
    return (
        isinstance(value, list) and bool(value)
        and all(isinstance(row, dict) and all(isinstance(v, _SCALARS) for v in row.values()) for row in value)
    )


@register_renderer(applies=_is_records)
def _render_records(value, **options):
    import pandas as pd

    return _render_dataframe(pd.DataFrame.from_records(value), **options)


def _render_code(value, language: str = "python", **options):
    return pn.widgets.CodeEditor(
        value=str(value), language=language, readonly=True,
        sizing_mode="stretch_width", **options,
    )


for _kind, _language in _CODE_KINDS.items():
    register_renderer(lambda value, _l=_language, **options: _render_code(value, language=_l, **options), kind=_kind)


@register_renderer(kind="json")
def _render_json(value, **options):
    return pn.pane.JSON(json.loads(json.dumps(value, default=repr)), depth=2, sizing_mode="stretch_width", **options)


@register_renderer(kind="text")
@register_renderer(kind="markdown")
def _render_text(value, **options):
    return untrusted_markdown(str(value), sizing_mode="stretch_width", **_with_table_css(options))


_OUTPUT_TEXT_CSS = """
:host {
  font-size: 0.875rem;
  line-height: 1.5;
  color: var(--mui-palette-text-primary);
}
:host p, :host ul, :host ol, :host blockquote, :host dl, :host table { margin: 0.4em 0; }
:host ul, :host ol { padding-left: 1.4em; }
:host li { margin: 0.1em 0; }
:host h1, :host h2, :host h3, :host h4, :host h5, :host h6 {
  font-size: 0.875rem;
  font-weight: 600;
  line-height: 1.5;
  letter-spacing: normal;
  margin: 0.75em 0 0.25em;
  padding: 0;
  border: 0;
}
:host h1, :host h2 { font-size: 1rem; }
:host :is(p, ul, ol, blockquote, dl, table, pre, h1, h2, h3, h4, h5, h6):first-child { margin-top: 0; }
:host :is(p, ul, ol, blockquote, dl, table, pre, h1, h2, h3, h4, h5, h6):last-child { margin-bottom: 0; }
"""

# Panel's Markdown styles shrink code blocks to their content and wrap
# their lines; here they fill the card and long lines scroll instead,
# with the copy button pinned to the corner rather than scrolling along.
# Older Panel releases let the pane's inner container grow to its content.
_CODE_BLOCK_CSS = """
:host div:not(.codehilite) { box-sizing: border-box; width: 100%; min-width: 0; max-width: 100%; }
:host pre:has(> code > .codehilite) { margin: 0.4em 0; white-space: normal; }
:host pre:has(> code > .codehilite) > code { display: block; padding: 0; background: none; white-space: normal; }
:host pre:has(> code > .codehilite):first-child { margin-top: 0; }
:host pre:has(> code > .codehilite):last-child { margin-bottom: 0; }
:host .codehilite {
  display: block;
  position: relative;
  box-sizing: border-box;
  max-width: 100%;
  margin: 0;
  padding: 0;
  border-radius: 6px;
}
:host .codehilite .copybtn { position: absolute; top: 0.4em; right: 0.4em; z-index: 1; }
:host .codehilite pre {
  margin: 0;
  padding: 0.5em 0.75em;
  overflow-x: auto;
  white-space: pre;
}
:host .codehilite > pre > code,
:host .codehilite > code {
  font-size: 0.8rem;
  line-height: 1.45;
  white-space: pre;
}
"""

_OUTPUT_CSS = _OUTPUT_TEXT_CSS + _CODE_BLOCK_CSS

_PYTHON_LINE = re.compile(r"^(?:(?:async\s+)?def\s+\w+\s*\(|class\s+\w+[(:]|import\s+\w|from\s+[\w.]+\s+import\s|@\w)")
_SQL_START = re.compile(r"^(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b")
_SQL_CLAUSE = re.compile(r"\b(?:FROM|INTO|SET|TABLE|VALUES|AS)\b", re.IGNORECASE)
_YAML_KEY = re.compile(r"^\s*(?:-\s+)?[\w\"'][\w .\"'-]*:(?:\s|$)")
_YAML_ITEM = re.compile(r"^\s*-(?:\s|$)")
_MARKUP = re.compile(r"^<(?:!DOCTYPE|\?xml|[A-Za-z][\w:-]*)[\s\S]*>$")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return value


def _json_default(value: Any) -> Any:
    converted = _jsonable(value)
    return repr(value) if converted is value else converted


def _pretty_json(value: Any, indent: int = 0, width: int = 72) -> str:
    """JSON indented by two spaces, with containers that fit in ``width`` columns kept on one line."""
    compact = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or not value or indent + len(compact) <= width:
        return compact
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(key, ensure_ascii=False)}: {_pretty_json(item, indent + 2, width)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + _pretty_json(item, indent + 2, width) for item in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def _dump_json(value: Any) -> str:
    return _pretty_json(json.loads(json.dumps(_jsonable(value), default=_json_default, ensure_ascii=False)))


def _looks_like_yaml(text: str) -> bool:
    lines = [
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and line.strip() not in ("---", "...")
    ]
    keys = sum(bool(_YAML_KEY.match(line)) for line in lines)
    indented = [line for line in lines if line.startswith((" ", "\t"))]
    nested = bool(indented) or any(line.rstrip().endswith(":") for line in lines)
    structured = all(_YAML_KEY.match(line) or _YAML_ITEM.match(line) or line.startswith((" ", "\t")) for line in lines)
    return len(lines) >= 2 and keys >= 2 and nested and structured


def _string_format(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "text", text
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except ValueError:
            pass
        else:
            if isinstance(parsed, (dict, list)):
                return "json", _pretty_json(parsed)
    if stripped.startswith("Traceback (most recent call last):"):
        return "pytb", text.strip("\n")
    if "\n" in stripped:
        first = stripped.splitlines()[0]
        if _PYTHON_LINE.match(first):
            return "python", text.strip("\n")
        # Lowercase SQL needs a closing semicolon to tell it from prose starting with "Select ...".
        if (_SQL_START.match(first) or _SQL_START.match(first.upper()) and stripped.endswith(";")) and _SQL_CLAUSE.search(stripped):
            return "sql", text.strip("\n")
        if _MARKUP.match(stripped):
            return "xml", text.strip("\n")
        if _looks_like_yaml(stripped):
            return "yaml", text.strip("\n")
    return "text", text


def output_format(value: Any) -> tuple[str, Any]:
    """
    Classify a tool output for display, returning ``(format, content)``.

    ``format`` is ``"table"`` for DataFrames (``content`` is the value),
    ``"text"`` for prose or Markdown, ``"object"`` for values with no
    text form (``content`` is the value), or the language of a code
    block: ``"json"`` for dicts, lists, tuples, sets, pydantic models,
    dataclasses and strings holding a JSON object or array, ``"yaml"``,
    ``"python"``, ``"pytb"`` (a Python traceback), ``"sql"`` or
    ``"xml"`` for strings that look like those. ``content`` is then the
    text to show, JSON pretty-printed.
    """
    if infer_kind(value) == "table":
        return "table", value
    if isinstance(value, str):
        return _string_format(value)
    if isinstance(value, (dict, list, tuple, set, frozenset)) or _jsonable(value) is not value:
        return "json", _dump_json(value)
    return "object", value


def _code_fence(text: str, language: str = "") -> str:
    """Markdown fenced code block showing ``text`` verbatim, whatever backticks it contains."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    language = re.sub(r"[^\w+-]", "", language)
    return f"{fence}{language}\n{text}\n{fence}"


def _output_markdown(text: str = "", **params):
    """An inert Markdown pane styled for tool output: compact text, small headings and scrolling code blocks."""
    params = {"margin": 0, "sizing_mode": "stretch_width", "disable_anchors": True, **params}
    return untrusted_markdown(text, **_with_table_css({**params, "stylesheets": [_OUTPUT_CSS, *params.get("stylesheets", [])]}))


def _collapsible_code(text: str, language: str, max_lines: int, **options):
    lines = text.splitlines()
    pane = _output_markdown(_code_fence(text if len(lines) <= max_lines else "\n".join(lines[:max_lines]), language), **options)
    if len(lines) <= max_lines:
        return pane
    import panel_material_ui as pmui

    button = pmui.Button(
        label=f"Show all {len(lines):,} lines", variant="text", size="small",
        icon="unfold_more", margin=(0, 0),
    )

    def expand(_):
        pane.object = _code_fence(text, language)
        button.visible = False

    button.on_click(expand)
    return pn.Column(pane, button, sizing_mode="stretch_width", margin=0)


def render_output(value: Any, max_lines: int = 30, **options) -> pn.viewable.Viewable:
    """
    Render a tool result compactly: structured values and code as
    highlighted code blocks showing at most ``max_lines`` lines until
    expanded, text as inert Markdown with headings at body size, and
    DataFrames as tables. Renderers registered for a type or predicate
    take precedence.
    """
    if any(r.applies is not None and r.applies(value) for r in _RENDERERS):
        return render(value, **options)
    fmt, content = output_format(value)
    if fmt == "table":
        return render(content, kind="dataframe", **options)
    if fmt == "object":
        return render(content, **options)
    if fmt == "text":
        return _output_markdown(content, **options)
    return _collapsible_code(content, fmt, max_lines, **options)


__all__ = [
    "output_format",
    "register_renderer",
    "render",
    "render_output",
]
