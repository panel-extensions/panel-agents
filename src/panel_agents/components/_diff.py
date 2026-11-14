"""
A themed viewer for line-based text diffs.
"""
from __future__ import annotations

import difflib
import html
from dataclasses import dataclass

import param
from panel.custom import PyComponent
from panel.pane import HTML


@dataclass
class DiffLine:

    op: str
    old_lineno: int | None
    new_lineno: int | None
    text: str


@dataclass
class DiffRow:
    """A row of a side-by-side diff, with either side possibly empty."""

    old: DiffLine | None
    new: DiffLine | None


def _split(text: str | None) -> list[str]:
    return [] if not text else text.splitlines()


def diff_lines(old: str | None, new: str | None, context: int | None = 3) -> list[DiffLine | int]:
    """
    Compute a unified line diff.

    Returns a list of `DiffLine` objects where runs of unchanged lines
    beyond `context` are collapsed into an integer holding the number
    of skipped lines.
    """
    a, b = _split(old), _split(new)
    out: list[DiffLine | int] = []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    opcodes = matcher.get_opcodes()
    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            lines = [DiffLine("equal", i1 + k + 1, j1 + k + 1, a[i1 + k]) for k in range(i2 - i1)]
            if context is None or len(lines) <= 2 * context:
                out.extend(lines)
                continue
            head = lines[:context] if idx > 0 else []
            tail = lines[-context:] if idx < len(opcodes) - 1 else []
            out.extend(head)
            out.append(len(lines) - len(head) - len(tail))
            out.extend(tail)
            continue
        out.extend(DiffLine("delete", i + 1, None, a[i]) for i in range(i1, i2))
        out.extend(DiffLine("insert", None, j + 1, b[j]) for j in range(j1, j2))
    return out


def split_rows(lines: list[DiffLine | int]) -> list[DiffRow | int]:
    """Pair deletions with insertions so a diff can be shown side by side."""
    rows: list[DiffRow | int] = []
    deleted: list[DiffLine] = []
    inserted: list[DiffLine] = []

    def flush():
        for k in range(max(len(deleted), len(inserted))):
            rows.append(DiffRow(
                deleted[k] if k < len(deleted) else None,
                inserted[k] if k < len(inserted) else None,
            ))
        deleted.clear()
        inserted.clear()

    for line in lines:
        if isinstance(line, DiffLine) and line.op == "delete":
            deleted.append(line)
        elif isinstance(line, DiffLine) and line.op == "insert":
            inserted.append(line)
        else:
            flush()
            rows.append(line if isinstance(line, int) else DiffRow(line, line))
    flush()
    return rows


CSS = """
:host {
  --diff-add: var(--mui-palette-success-main, #2e7d32);
  --diff-del: var(--mui-palette-error-main, #d32f2f);
  --diff-border: var(--mui-palette-divider, rgba(127, 127, 127, 0.25));
  --diff-muted: var(--mui-palette-text-secondary, rgba(127, 127, 127, 0.9));
  width: 100%;
}
.diff {
  border: 1px solid var(--diff-border);
  border-radius: var(--mui-shape-borderRadius, 6px);
  overflow: hidden;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--mui-palette-text-primary, inherit);
  background: var(--mui-palette-background-paper, transparent);
}
.diff-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 12px;
  border-bottom: 1px solid var(--diff-border);
  font-family: var(--mui-font-body2, inherit);
  font-size: 12px;
  color: var(--diff-muted);
}
.diff-header .labels { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.diff-header .stat-add { color: var(--diff-add); font-weight: 600; }
.diff-header .stat-del { color: var(--diff-del); font-weight: 600; }
.diff-body { overflow: auto; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
td { padding: 0 8px; vertical-align: top; }
td.ln {
  width: 3.5em;
  text-align: right;
  color: var(--diff-muted);
  user-select: none;
  opacity: 0.8;
}
td.sign { width: 1em; padding: 0 2px; user-select: none; color: var(--diff-muted); }
td.code { white-space: pre; overflow: hidden; text-overflow: ellipsis; }
.wrap td.code { white-space: pre-wrap; word-break: break-word; }
tr.insert td, td.insert { background: color-mix(in srgb, var(--diff-add) 12%, transparent); }
tr.delete td, td.delete { background: color-mix(in srgb, var(--diff-del) 12%, transparent); }
tr.insert td.sign, td.insert.sign { color: var(--diff-add); }
tr.delete td.sign, td.delete.sign { color: var(--diff-del); }
td.empty { background: color-mix(in srgb, var(--diff-muted) 6%, transparent); }
td.split { border-left: 1px solid var(--diff-border); }
tr.skip td {
  text-align: center;
  color: var(--diff-muted);
  background: color-mix(in srgb, var(--diff-muted) 8%, transparent);
  font-size: 11.5px;
  padding: 2px 8px;
}
.diff-empty { padding: 12px; color: var(--diff-muted); font-family: var(--mui-font-body2, inherit); }
"""

_SIGNS = {"insert": "+", "delete": "−", "equal": ""}


class DiffViewer(PyComponent):
    """
    The `DiffViewer` displays the line-based difference between two texts,
    either as a unified diff or side by side.

    Colors follow the active Material UI theme.

    :Example:

    >>> DiffViewer(old="SELECT 1", new="SELECT 2", old_label="v1", new_label="v2")
    """

    old = param.String(default="", allow_None=True, doc="The original text.")

    new = param.String(default="", allow_None=True, doc="The changed text.")

    old_label = param.String(default=None, allow_None=True, doc="Label for the original text.")

    new_label = param.String(default=None, allow_None=True, doc="Label for the changed text.")

    context = param.Integer(default=3, bounds=(0, None), allow_None=True, doc="""
        Number of unchanged lines shown around each change; None shows all lines.""")

    mode = param.Selector(default="unified", objects=["unified", "split"], doc="""
        Whether to show a single unified column or the two texts side by side.""")

    max_height = param.Integer(default=None, allow_None=True, doc="""
        Maximum height of the diff body in pixels before it scrolls.""")

    show_header = param.Boolean(default=True, doc="""
        Whether to show the header with the labels and line counts.""")

    wrap = param.Boolean(default=False, doc="Whether to wrap long lines.")

    added = param.Integer(default=0, constant=True, doc="Number of inserted lines.")

    removed = param.Integer(default=0, constant=True, doc="Number of deleted lines.")

    def __init__(self, **params):
        params.setdefault("sizing_mode", "stretch_width")
        super().__init__(**params)
        self._html = HTML(sizing_mode="stretch_width", stylesheets=[CSS])
        self._update()

    def __panel__(self):
        return self._html

    @param.depends("old", "new", "old_label", "new_label", "context", "mode", "max_height", "show_header", "wrap", watch=True)
    def _update(self):
        lines = diff_lines(self.old, self.new, self.context)
        changed = [line for line in lines if isinstance(line, DiffLine)]
        with param.edit_constant(self):
            self.param.update(
                added=sum(line.op == "insert" for line in changed),
                removed=sum(line.op == "delete" for line in changed),
            )
        self._html.object = self._render(lines)

    def _render(self, lines: list[DiffLine | int]) -> str:
        parts = [f'<div class="diff{" wrap" if self.wrap else ""}">']
        if self.show_header:
            labels = " → ".join(html.escape(label) for label in (self.old_label, self.new_label) if label)
            parts.append(
                f'<div class="diff-header"><span class="labels">{labels}</span>'
                f'<span class="stat-add">+{self.added}</span><span class="stat-del">−{self.removed}</span></div>'
            )
        if not self.added and not self.removed:
            parts.append('<div class="diff-empty">No differences</div></div>')
            return "".join(parts)
        style = f' style="max-height: {self.max_height}px"' if self.max_height else ""
        parts.append(f'<div class="diff-body"{style}><table>')
        if self.mode == "split":
            parts.append('<colgroup><col style="width:3.5em"><col style="width:1em"><col>'
                         '<col style="width:3.5em"><col style="width:1em"><col></colgroup>')
            parts.extend(self._split_row(row) for row in split_rows(lines))
        else:
            parts.append('<colgroup><col style="width:3.5em"><col style="width:3.5em"><col style="width:1em"><col></colgroup>')
            parts.extend(self._unified_row(line) for line in lines)
        parts.append("</table></div></div>")
        return "".join(parts)

    @staticmethod
    def _skip(n: int, colspan: int) -> str:
        label = "line" if n == 1 else "lines"
        return f'<tr class="skip"><td colspan="{colspan}">⋯ {n} unchanged {label}</td></tr>'

    def _unified_row(self, line: DiffLine | int) -> str:
        if isinstance(line, int):
            return self._skip(line, 4)
        old = "" if line.old_lineno is None else line.old_lineno
        new = "" if line.new_lineno is None else line.new_lineno
        return (
            f'<tr class="{line.op}"><td class="ln">{old}</td><td class="ln">{new}</td>'
            f'<td class="sign">{_SIGNS[line.op]}</td><td class="code">{html.escape(line.text)}</td></tr>'
        )

    def _split_row(self, row: DiffRow | int) -> str:
        if isinstance(row, int):
            return self._skip(row, 6)
        cells = []
        for side, line in (("old", row.old), ("new", row.new)):
            border = " split" if side == "new" else ""
            if line is None:
                cells.append(f'<td class="ln empty{border}"></td><td class="sign empty"></td><td class="code empty"></td>')
                continue
            lineno = line.old_lineno if side == "old" else line.new_lineno
            op = line.op if line.op != "equal" else ""
            cells.append(
                f'<td class="ln {op}{border}">{lineno}</td><td class="sign {op}">{_SIGNS[line.op]}</td>'
                f'<td class="code {op}">{html.escape(line.text)}</td>'
            )
        return f"<tr>{''.join(cells)}</tr>"
