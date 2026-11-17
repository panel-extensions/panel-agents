"""
Rendering untrusted text safely.

Model replies, thinking, tool results, artifacts, tool descriptions and
decision reasons can contain text an attacker controls, e.g. through a
prompt injected into a document the agent read. Rendered as ordinary
Markdown, such text can load remote images (sending data out in their
URL with no click), run script through raw HTML, or abuse MathJax.

``inert_markdown`` is a markdown-it plugin that works on the parsed
tokens, the same ones that are rendered, so nothing in the source can
sneak past it: raw HTML is shown as text, images become links unless
their origin is trusted, links are limited to http, https and mailto and
show their full URL. ``untrusted_markdown`` builds a Markdown pane with
the plugin and with math disabled.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

_trusted_origins: frozenset[str] = frozenset()

_LINK_SCHEMES = ("http", "https", "mailto")

_RASTER_DATA = re.compile(r"^data:image/(png|gif|jpeg|webp);", re.IGNORECASE)

_MARKDOWN_PUNCTUATION = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$&:\"'=])")


def set_trusted_image_origins(origins: Iterable[str]) -> None:
    """
    Allow images in rendered agent output to load from ``origins``, e.g.
    ``["https://tiles.example.com"]``. ``"*"`` allows every origin. Every
    other image is shown as a link. Applies to the whole process.
    """
    global _trusted_origins
    _trusted_origins = frozenset(origin.rstrip("/").lower() for origin in origins)


def trusted_image_origins() -> frozenset[str]:
    """The origins images in agent output may load from."""
    return _trusted_origins


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc}".lower()
    return None


def _image_allowed(url: str) -> bool:
    if _RASTER_DATA.match(url):
        return True
    if "*" in _trusted_origins:
        return urlsplit(url).scheme.lower() in ("http", "https")
    return _origin(url) in _trusted_origins


def _link_allowed(url: str) -> bool:
    scheme = urlsplit(url.strip()).scheme.lower()
    if scheme:
        return scheme in _LINK_SCHEMES
    # Relative links stay within the app; protocol-relative ones leave it.
    return not url.strip().startswith("//")


def _make_token(token_cls, type_: str, tag: str, nesting: int, **attrs) -> Any:
    token = token_cls(type_, tag, nesting)
    for key, value in attrs.items():
        setattr(token, key, value)
    return token


def _text_token(token_cls, content: str) -> Any:
    return _make_token(token_cls, "text", "", 0, content=content)


def _url_suffix(token_cls, url: str) -> list[Any]:
    return [_text_token(token_cls, " ("), _make_token(token_cls, "code_inline", "code", 0, content=url, markup="`"), _text_token(token_cls, ")")]


def _inert_children(children: list[Any], token_cls) -> list[Any]:
    out: list[Any] = []
    stack: list[tuple[str, int] | None] = []
    for token in children:
        if token.type in ("html_inline", "html_block"):
            out.append(_text_token(token_cls, token.content))
        elif token.type == "image":
            src = token.attrGet("src") or ""
            if _image_allowed(src):
                out.append(token)
                continue
            alt = "".join(child.content for child in (token.children or [])) or token.content
            label = f"image: {alt}" if alt else "image"
            if _link_allowed(src):
                link = _make_token(token_cls, "link_open", "a", 1)
                link.attrSet("href", src)
                out += [link, _text_token(token_cls, label), _make_token(token_cls, "link_close", "a", -1)]
            else:
                out.append(_text_token(token_cls, label))
            out += _url_suffix(token_cls, src)
        elif token.type == "link_open":
            href = token.attrGet("href") or ""
            if not _link_allowed(href):
                stack.append(None)
                continue
            for name in list(token.attrs or {}):
                if name != "href":
                    del token.attrs[name]
            stack.append((href, len(out)))
            out.append(token)
        elif token.type == "link_close":
            opened = stack.pop() if stack else None
            if opened is None:
                continue
            href, start = opened
            text = "".join(t.content for t in out[start + 1:] if t.type in ("text", "code_inline"))
            out.append(token)
            if href not in (text, f"mailto:{text}"):
                out += _url_suffix(token_cls, href)
        else:
            out.append(token)
    return out


def _escaping_highlight(highlight):
    # markdown-it inserts what the highlighter returns as HTML, and Panel's returns the code unchanged when it fails.
    def wrapper(code: str, lang: str, attrs: str) -> str:
        try:
            html = highlight(code, lang, attrs)
        except Exception:
            return ""
        return html if isinstance(html, str) and html != code and html.lstrip().startswith("<") else ""

    wrapper._panel_agents_escaping = True
    return wrapper


def _inert_rule(state) -> None:
    from markdown_it.token import Token

    highlight = state.md.options.get("highlight")
    if highlight is not None and not getattr(highlight, "_panel_agents_escaping", False):
        state.md.options["highlight"] = _escaping_highlight(highlight)
    tokens = []
    for token in state.tokens:
        if token.type == "html_block":
            tokens.append(_make_token(Token, "paragraph_open", "p", 1, block=True))
            inline = _make_token(Token, "inline", "", 0, content=token.content, children=[_text_token(Token, token.content)])
            tokens += [inline, _make_token(Token, "paragraph_close", "p", -1, block=True)]
            continue
        if token.children:
            token.children = _inert_children(token.children, Token)
        tokens.append(token)
    state.tokens = tokens


def inert_markdown(md) -> None:
    """
    markdown-it plugin making untrusted Markdown inert: raw HTML renders
    as text, images not from a trusted origin become links labelled
    "image: ...", links other than http, https, mailto or relative ones
    are dropped, and every link shows its full URL after its text.
    Code blocks are escaped even when the highlighter fails.

    Use it with math disabled, e.g. ``pn.pane.Markdown(text,
    plugins=[inert_markdown], disable_math=True)``, or through
    ``untrusted_markdown``.
    """
    md.options["html"] = False
    md.disable(["html_block", "html_inline"], True)
    validate = md.validateLink
    md.validateLink = lambda url: validate(url) and (_link_allowed(url) or _RASTER_DATA.match(url) is not None)
    md.core.ruler.push("panel_agents_inert", _inert_rule)


def untrusted_markdown(text: str = "", **params):
    """A Markdown pane rendering ``text`` with ``inert_markdown`` and without math."""
    return _markdown_pane_class()(text, **_untrusted_params(params))


_MARKDOWN_PANE = None


def _markdown_pane_class():
    global _MARKDOWN_PANE
    if _MARKDOWN_PANE is None:
        import panel as pn

        class UntrustedMarkdown(pn.pane.Markdown):
            # Markdown does not pass disable_math to the browser, where MathJax would still typeset $$...$$.
            _rename = {**pn.pane.Markdown._rename, "disable_math": "disable_math"}

            # Never chosen by pn.panel for arbitrary strings.
            priority = 0

        _MARKDOWN_PANE = UntrustedMarkdown
    return _MARKDOWN_PANE


def _untrusted_params(params: dict[str, Any]) -> dict[str, Any]:
    return {**params, "renderer": "markdown-it", "plugins": [inert_markdown], "disable_math": True}


def _escape_markdown(text: Any) -> str:
    """Markdown source that renders ``text`` literally."""
    return _MARKDOWN_PUNCTUATION.sub(r"\\\1", "" if text is None else str(text))


def _plain_typography(text: Any = "", **params):
    """A Typography showing ``text`` as plain text, never as Markdown or HTML."""
    import panel_material_ui as pmui

    return pmui.Typography(_escape_markdown(text), **_untrusted_params(params))


__all__ = ["inert_markdown", "set_trusted_image_origins", "trusted_image_origins", "untrusted_markdown"]
