"""
Browser-side helpers of AgentDrawer. Component class names cannot start
with an underscore, so they live in this private module instead.

The header toggle is a native button, so that it can carry
``aria-expanded`` and an accessible name that includes the number of
pending approvals, which the panel-material-ui buttons do not expose.
"""

from __future__ import annotations

import param
from panel.custom import JSComponent

_ESM = """
const PATH = (
  "M20 9V7c0-1.1-.9-2-2-2h-3c0-1.66-1.34-3-3-3S9 3.34 9 5H6c-1.1 0-2 .9-2 2v2c-1.66 0-3 1.34-3 3s1.34 3 3 3v4c0 1.1.9 2 2 2" +
  "h12c1.1 0 2-.9 2-2v-4c1.66 0 3-1.34 3-3s-1.34-3-3-3zM7.5 11.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5S9.83 13 9 13s-1.5-.67-" +
  "1.5-1.5zM16 17H8v-2h8v2zm-1-4c-.83 0-1.5-.67-1.5-1.5S14.17 10 15 10s1.5.67 1.5 1.5S15.83 13 15 13z"
)
const ICON = `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false"><path fill="currentColor" d="${PATH}"/></svg>`

export function render({ model, el }) {
  const button = document.createElement("button")
  button.type = "button"
  button.className = "pa-toggle"
  const label = document.createElement("span")
  const badge = document.createElement("span")
  badge.className = "pa-badge"
  badge.setAttribute("aria-hidden", "true")
  button.innerHTML = ICON
  button.append(label, badge)

  const update = () => {
    const count = model.count
    label.textContent = model.label
    badge.textContent = count > 99 ? "99+" : String(count)
    badge.style.display = count ? "" : "none"
    button.setAttribute("aria-expanded", String(model.expanded))
    const waiting = count ? `, ${count} pending approval${count === 1 ? "" : "s"}` : ""
    button.setAttribute("aria-label", `${model.label}${waiting}`)
  }
  update()
  model.on(["label", "count", "expanded"], update)
  button.addEventListener("click", () => { model.expanded = !model.expanded })
  model.on("msg:custom", (msg) => { if (msg.type === "focus") { button.focus() } })
  el.append(button)
}
"""

_CSS = """
.pa-toggle {
  display: inline-flex; align-items: center; gap: 6px; position: relative;
  padding: 4px 10px; border: 0; border-radius: var(--mui-shape-borderRadius, 4px);
  background: transparent; color: inherit; cursor: pointer;
  font: var(--mui-font-button, 500 0.8125rem/1.75 Roboto, sans-serif); text-transform: uppercase;
}
.pa-toggle:hover { background: rgba(127, 127, 127, 0.16); }
.pa-toggle:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
.pa-badge {
  position: absolute; top: -4px; right: -4px; min-width: 18px; height: 18px; padding: 0 5px;
  box-sizing: border-box; border-radius: 9px; font-size: 0.7rem; line-height: 18px; text-align: center;
  background: var(--mui-palette-warning-main, #ed6c02); color: var(--mui-palette-warning-contrastText, #fff);
}
"""


class DrawerToggle(JSComponent):

    label = param.String(default="Assistant")

    count = param.Integer(default=0)

    expanded = param.Boolean(default=False)

    _esm = _ESM

    _stylesheets = [_CSS]

    def focus(self):
        self._send_msg({"type": "focus"})


class DrawerResizeNotifier(JSComponent):
    """
    Tells the page its layout changed once the drawer has opened or closed.

    Plots and maps beside an inline drawer only measure themselves on window
    resizes, so without this they keep the width they had before. It also
    reports the window's width, for ``AgentDrawer.overlay_below``.
    """

    open = param.Boolean(default=False)

    window_width = param.Integer(default=None, allow_None=True)

    _esm = """
    export function render({ model }) {
      model.on("open", () => {
        // After MUI's drawer transition (225ms).
        setTimeout(() => window.dispatchEvent(new Event("resize")), 260)
      })
      let timer = null
      const report = () => {
        clearTimeout(timer)
        timer = setTimeout(() => {
          if (model.window_width !== window.innerWidth) { model.window_width = window.innerWidth }
        }, 150)
      }
      window.addEventListener("resize", report)
      model.on("remove", () => window.removeEventListener("resize", report))
      model.window_width = window.innerWidth
    }
    """
