import dataclasses
import json

import pandas as pd
import panel as pn
import panel_material_ui as pmui
import pydantic
import pytest

from panel_agents.components.renderers import (
    _code_fence,
    _output_markdown,
    _pretty_json,
    output_format,
    render_output,
)


class Model(pydantic.BaseModel):
    name: str
    tags: list[str]


@dataclasses.dataclass
class Point:
    x: int
    y: int


@pytest.mark.parametrize("value, expected", [
    ({"a": 1}, "json"),
    ([1, 2], "json"),
    ((1, 2), "json"),
    ({3, 1}, "json"),
    (Model(name="a", tags=["x"]), "json"),
    (Point(1, 2), "json"),
    ('{"a": [1, 2]}', "json"),
    ('  [1, 2]\n', "json"),
    ("42", "text"),
    ('"quoted"', "text"),
    ("{not json}", "text"),
    ("name: app\nlayout:\n  - type: table\n    source: sales\n", "yaml"),
    ("---\nkey: value\nnested:\n  a: 1\n", "yaml"),
    ("Status: ok\nRows: 5", "text"),
    ("# Summary\n\nThe table has: 3 columns\n\n- a\n- b", "text"),
    ("Note: this is prose.\nIt continues here.", "text"),
    ("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nValueError: bad", "pytb"),
    ("import pandas as pd\ndf = pd.read_csv('x')", "python"),
    ("def f(x):\n    return x", "python"),
    ("SELECT a, b\nFROM sales\nWHERE a > 1", "sql"),
    ("Select the file you want\nfrom the list below", "text"),
    ("select a\nfrom t;", "sql"),
    ("<root>\n  <item/>\n</root>", "xml"),
    ("plain text", "text"),
    ("", "text"),
    (pd.DataFrame({"a": [1]}), "table"),
    (3.5, "object"),
    (None, "object"),
])
def test_output_format(value, expected):
    assert output_format(value)[0] == expected


def test_output_format_pretty_prints_json():
    fmt, text = output_format('{"a":1,"b":{"c":[1,2]}}')
    assert fmt == "json"
    assert text == '{"a": 1, "b": {"c": [1, 2]}}'
    assert json.loads(output_format(Model(name="a", tags=["x"]))[1]) == {"name": "a", "tags": ["x"]}
    assert json.loads(output_format({"o": object()})[1])["o"].startswith("<object object")


def test_pretty_json_breaks_only_what_does_not_fit():
    value = {"rows": [{"id": i, "name": "x" * 10} for i in range(3)], "n": 3}
    lines = _pretty_json(value, width=40).splitlines()
    assert lines[0] == "{"
    assert lines[2] == '    {"id": 0, "name": "xxxxxxxxxx"},'
    assert json.loads(_pretty_json(value, width=40)) == value
    assert _pretty_json({"a": 1}) == '{"a": 1}'
    assert _pretty_json({}) == "{}"


def test_code_fence_outlasts_backticks_in_the_text():
    fence = _code_fence("a ```` b", "json")
    assert fence.startswith("`````json\n") and fence.endswith("\n`````")
    assert _code_fence("x", "py<script>") == "```pyscript\nx\n```"


def test_render_output_paths():
    text = render_output("# Title\n\nbody")
    assert isinstance(text, pn.pane.Markdown) and text.disable_anchors
    code = render_output({"a": 1})
    assert isinstance(code, pn.pane.Markdown)
    assert code.object.startswith("```json\n")
    table = render_output(pd.DataFrame({"a": [1, 2]}))
    assert type(table) is pn.pane.DataFrame
    records = render_output([{"a": 1}, {"a": 2}])
    assert type(records) is pn.pane.DataFrame


def test_long_code_collapses_until_show_all():
    value = {f"k{i}": i for i in range(40)}
    column = render_output(value, max_lines=10)
    pane, button = column.objects
    assert isinstance(button, pmui.Button)
    assert "Show all 42 lines" in button.label
    assert pane.object.count("\n") == 11
    button.clicks += 1
    assert '"k39": 39' in pane.object
    assert not button.visible


def test_output_markdown_uses_inert_rendering():
    from panel_agents.security import inert_markdown

    pane = _output_markdown("x")
    assert pane.disable_math and inert_markdown in pane.plugins


def test_code_and_table_renderers_size_to_content():
    from panel_agents.components.renderers import render

    short_sql = render("SELECT a\nFROM t\nGROUP BY a\nORDER BY a", kind="sql")
    assert isinstance(short_sql, pn.widgets.CodeEditor)
    assert short_sql.height < 160

    explicit = render("SELECT 1", kind="sql", height=250)
    assert explicit.height == 250

    narrow = render(pd.DataFrame({"a": [1], "b": [2]}), kind="table")
    assert isinstance(narrow, pn.widgets.Tabulator) and narrow.layout == "fit_columns"

    wide = render(pd.DataFrame({f"c{i}": [i] for i in range(8)}), kind="table")
    assert isinstance(wide, pn.widgets.Tabulator) and wide.layout != "fit_columns"
