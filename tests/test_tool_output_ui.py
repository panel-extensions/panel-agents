import json
import re

import panel_material_ui as pmui
import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import AgentDrawer, RunState
from panel_agents import events as ev

pytestmark = pytest.mark.ui

MARKDOWN = "# Big title\n\n## Section\n\nSome **text**."

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/srv/app/tools/query.py", line 42, in run_query\n'
    "ValueError: " + "a very long message " * 20
)

LONG = {"rows": [{"id": i, "region": "EMEA", "note": "x" * 120} for i in range(40)]}

OUTPUTS = {
    "schema": ({"table": "sales"}, {"table": "sales", "columns": {"id": "BIGINT"}}, False),
    "query": ({"sql": "SELECT id\nFROM sales", "limit": 40}, json.dumps(LONG), False),
    "describe": ({"path": "a.md"}, MARKDOWN, False),
    "spec": ({"path": "a.yaml"}, "name: x\nlayout:\n  - type: table\n", False),
    "fails": ({"sql": "SELECT qtr"}, TRACEBACK, True),
}

_FIND = """([selector, text]) => {
  const out = [];
  const walk = (root) => root.querySelectorAll('*').forEach(e => {
    if (e.shadowRoot) walk(e.shadowRoot);
    if (e.matches(selector) && (!text || e.textContent.includes(text))) out.push(e);
  });
  walk(document);
  return out;
}"""


def find(page, selector, text=None):
    return page.evaluate_handle(_FIND, [selector, text])


def measure(page, selector, text, script):
    return page.evaluate(f"([els]) => els.map({script})", [find(page, selector, text)])


def session():
    state = RunState(flush_interval=0)
    state.apply(ev.RunStarted(run_id="r"))
    for name, (args, result, error) in OUTPUTS.items():
        state.apply_many([
            ev.ToolCallStart(tool_call_id=name, tool_name=name, run_id="r"),
            ev.ToolCallArgs(tool_call_id=name, delta=json.dumps(args), run_id="r"),
            ev.ToolCallEnd(tool_call_id=name, run_id="r"),
            ev.ToolCallResult(tool_call_id=name, content=result, is_error=error, run_id="r"),
        ])
    return state


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_tool_outputs_in_the_drawer(page, dark):
    state = session()
    drawer = AgentDrawer(state=state, main=pmui.Typography("The app"), width=420, open=True)
    drawer._chat.group_tool_calls = False
    for call in state.tool_calls.values():
        drawer._chat.view_for(call).collapsed = False
    page.set_viewport_size({"width": 1000, "height": 900})
    serve_component(page, pmui.Page(main=[drawer], dark_theme=dark))

    wait_until(lambda: measure(page, ".codehilite", '"columns"', "e => 1") == [1], page)

    # Structured results and arguments are highlighted code blocks.
    assert measure(page, ".codehilite .nt", '"table"', "e => e.textContent") != []
    assert measure(page, ".codehilite", "SELECT id", "e => e.querySelector('.k') !== null") == [True]
    assert measure(page, ".codehilite", "layout:", "e => 1") == [1]

    # Markdown headings are no bigger than a subtitle.
    sizes = measure(page, "h1, h2", None, "e => parseFloat(getComputedStyle(e).fontSize)")
    assert sizes and max(sizes) <= 16

    # Long lines scroll inside the code block instead of widening the drawer.
    (box,) = measure(page, ".codehilite pre", '"rows"', "e => [e.scrollWidth, e.clientWidth, e.getBoundingClientRect().right]")
    scroll_width, client_width, right = box
    assert scroll_width > client_width
    assert right <= 1000

    # A long result is cut until expanded.
    show_all = page.get_by_role("button", name=re.compile(r"Show all [\d,]+ lines"))
    expect(show_all).to_be_visible()
    assert measure(page, ".codehilite", '"id": 39', "e => 1") == []
    show_all.click()
    wait_until(lambda: measure(page, ".codehilite", '"id": 39', "e => 1") == [1], page)
    expect(show_all).to_have_count(0)

    # Errors are monospace text in an error alert, scrolling rather than wrapping.
    alert = page.get_by_role("alert")
    expect(alert).to_contain_text("ValueError: a very long message")
    expect(alert).to_have_class(re.compile("MuiAlert-colorError"))
    (pre,) = measure(page, "pre", "Traceback (most recent", "e => [getComputedStyle(e).fontFamily, e.scrollWidth > e.clientWidth]")
    assert "monospace" in pre[0] and pre[1]
