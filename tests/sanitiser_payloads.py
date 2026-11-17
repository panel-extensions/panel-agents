# ruff: noqa: E501
"""Payloads that tried to get past the untrusted-text rendering, shared by the offline check and the Playwright test.

Each payload targets https://evil.test/<name> so a request can be attributed. Raw HTML payloads use
onerror=window.__pwned.push(name) to prove script execution, not just a fetch.
"""

def img(name):
    return f'<img src="https://evil.test/{name}.png" onerror="(window.__pwned=window.__pwned||[]).push(\'{name}\')">'


PAYLOADS = {
    # Markdown image syntax the regexes miss
    "shortcut_ref": "![shortcut_ref]\n\n[shortcut_ref]: https://evil.test/shortcut_ref.png",
    "nested_brackets": "![a [b] c](https://evil.test/nested_brackets.png)",
    "escaped_paren": r"![x](https://evil.test/escaped_paren\(.png)",
    "deep_parens": "![x](https://evil.test/deep_parens((a)).png)",
    "escaped_quote_title": '![x](https://evil.test/escaped_quote_title.png "a\\"b")',
    "code_in_alt": "![`]`](https://evil.test/code_in_alt.png)",
    "code_in_dest": "![x](https://evil.test/code_in_dest`a`.png)",
    # Regions the code detector treats as code but markdown does not, so raw HTML survives
    "span_across_paragraphs": f"`a\n\n{img('span_across_paragraphs')}\n\nb`",
    "escaped_backtick": f"\\` {img('escaped_backtick')} `",
    "longer_closing_fence": f"```\nx\n````\n{img('longer_closing_fence')}\n\n```",
    "indented_fence": f"para\n\n    ```\n{img('indented_fence')}\n```",
    "fence_info_backtick": f"```a`b\n{img('fence_info_backtick')}\n```",
    # MathJax (Panel's Markdown typesets $$...$$ with all TeX packages, including html)
    "mathjax_style_bg": "$$\\style{background-image:url(https://evil.test/mathjax_style_bg.png)}{x}$$",
    "mathjax_style_cursor": "$$\\style{cursor:url(https://evil.test/mathjax_style_cursor.png),auto}{x}$$",
    "mathjax_style_mask": "$$\\style{mask-image:url(https://evil.test/mathjax_style_mask.png);-webkit-mask-image:url(https://evil.test/mathjax_style_mask.png)}{x}$$",
    "mathjax_style_filter": "$$\\style{filter:url(https://evil.test/mathjax_style_filter.svg#f)}{x}$$",
    "mathjax_href_js": "$$\\href{javascript:(window.__pwned=window.__pwned||[]).push('mathjax_href_js')}{\\style{display:block;width:3000px;height:3000px}{\\text{CLICKME}}}$$",
    # Expected to be safe
    "inline_image": "![x](https://evil.test/inline_image.png)",
    "ref_image": "![x][r]\n\n[r]: https://evil.test/ref_image.png",
    "collapsed_ref": "![collapsed_ref][]\n\n[collapsed_ref]: https://evil.test/collapsed_ref.png",
    "raw_html": img("raw_html"),
    "angle_dest": "![x](<https://evil.test/angle_dest.png>)",
    "entity_js_link": "[ENTITYJS](javascript&#58;(window.__pwned=window.__pwned||[]).push('entity_js_link'))",
    "tab_js_link": "[TABJS](java&#x09;script:(window.__pwned=window.__pwned||[]).push('tab_js_link'))",
    "data_html_link": "[DATAHTML](data:text/html,<script>alert(1)</script>)",
    "vbscript_link": "[VB](vbscript:msgbox)",
    "svg_data_image": "![x](data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnPjxpbWFnZSBocmVmPSdodHRwczovL2V2aWwudGVzdC9zdmdfZGF0YV9pbWFnZS5wbmcnLz48L3N2Zz4=)",
    "base_meta_link": '<base href="https://evil.test/base/"><meta http-equiv="refresh" content="0;url=https://evil.test/meta"><link rel="preload" as="image" href="https://evil.test/preload.png">',
    "style_attr": '<span style="background:url(https://evil.test/style_attr.png)">x</span>',
    "table_html": "| a |\n|---|\n| " + img("table_html") + " |",
    "attr_list": "![x](https://evil.test/attr_list.png){: onerror=alert(1)}",
    "linkify": "visit www.evil.test/linkify now",
    "autolink": "<https://evil.test/autolink>",
    "footnote": "x[^1]\n\n[^1]: ![x](https://evil.test/footnote.png)",
    "blockquote_html": "> " + img("blockquote_html"),
    "list_fence": f"- item\n\n  ```\n  code\n  ```\n{img('list_fence')}",
}
