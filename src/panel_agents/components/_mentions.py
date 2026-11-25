"""
The mention chips in AgentChat's input. Component class names cannot
start with an underscore, so they live in this private module instead.

They are native elements rather than panel-material-ui chips: a chip is
a focusable button of its own, and pmui's chat input takes Enter from
anything in it and sends the message, so the remove buttons handle
Enter themselves.
"""

from __future__ import annotations

import param
from panel.custom import JSComponent

_ESM = """
const CLOSE = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false">'
  + '<path fill="currentColor" d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>'

// The chat input's textarea, in the shadow root the component's host is rendered in.
const textarea = (el) => el.getRootNode().host?.getRootNode().querySelector("textarea")

export function render({ model, el }) {
  const list = document.createElement("div")
  list.className = "pa-mentions"
  list.setAttribute("role", "list")
  el.append(list)

  const remove = (index) => {
    model.send_msg({ type: "remove", index })
    textarea(el)?.focus({ preventScroll: true })
  }

  const update = () => {
    list.setAttribute("aria-label", model.label)
    list.replaceChildren(...model.mentions.map((mention, index) => {
      const chip = document.createElement("span")
      chip.className = "pa-mention"
      chip.setAttribute("role", "listitem")
      if (mention.icon) {
        const icon = document.createElement("span")
        icon.className = "pa-mention-icon material-icons"
        icon.setAttribute("aria-hidden", "true")
        icon.textContent = mention.icon
        chip.append(icon)
      }
      const label = document.createElement("span")
      label.className = "pa-mention-label"
      label.textContent = mention.label
      const button = document.createElement("button")
      button.type = "button"
      button.className = "pa-mention-remove"
      button.setAttribute("aria-label", `Remove ${mention.label}`)
      button.innerHTML = CLOSE
      button.addEventListener("click", () => remove(index))
      button.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          e.stopPropagation()
          remove(index)
        }
      })
      chip.append(label, button)
      return chip
    }))
  }
  update()
  model.on(["mentions", "label"], update)

  // Backspace in an empty input removes the last mention, as in most composers.
  let input = null
  const onKeydown = (e) => {
    const count = model.mentions.length
    if (e.key === "Backspace" && count && !input.value && input.selectionStart === 0 && input.selectionEnd === 0) {
      e.preventDefault()
      model.send_msg({ type: "remove", index: count - 1 })
    }
  }
  const attach = () => {
    input = textarea(el)
    input?.addEventListener("keydown", onKeydown)
  }
  // The host is in the page once rendering returns.
  setTimeout(attach, 0)
  model.on("remove", () => input?.removeEventListener("keydown", onKeydown))
}
"""

_CSS = """
.pa-mentions { display: flex; flex-wrap: nowrap; gap: 4px; align-items: center; }
.pa-mention {
  display: inline-flex; align-items: center; gap: 4px; height: 24px; box-sizing: border-box;
  padding: 0 2px 0 8px; border: 1px solid var(--mui-palette-Chip-defaultBorder, #bdbdbd); border-radius: 12px;
  font: var(--mui-font-body2, 400 0.875rem/1.43 Roboto, sans-serif); font-size: 0.8125rem; white-space: nowrap;
  color: var(--mui-palette-text-primary, rgba(0, 0, 0, 0.87));
}
.pa-mention-icon { font-size: 16px; margin-left: -2px; color: var(--mui-palette-action-active, rgba(0, 0, 0, 0.54)); }
.pa-mention-remove {
  display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; padding: 0;
  border: 0; border-radius: 50%; background: transparent; color: inherit; opacity: 0.6; cursor: pointer;
}
.pa-mention-remove:hover { opacity: 1; background: var(--mui-palette-action-hover, rgba(0, 0, 0, 0.04)); }
.pa-mention-remove:focus-visible { opacity: 1; outline: 2px solid currentColor; outline-offset: -1px; }
"""


class MentionChips(JSComponent):
    """
    A chip with a remove button for each mention. Removing one, or
    Backspace in the chat's empty input, sends ``{"type": "remove",
    "index": ...}`` to the ``on_msg`` callbacks and, for the button, puts
    the focus back in the input.
    """

    mentions = param.List(default=[], doc="Dicts with a ``label`` and an optional ``icon``.")

    label = param.String(default="This message is about", doc="The accessible name of the list of chips.")

    _esm = _ESM

    _stylesheets = [_CSS]
