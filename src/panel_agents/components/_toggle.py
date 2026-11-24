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
  display: inline-flex; align-items: center; gap: 8px; position: relative;
  padding: 6px 8px 6px 6px; border: 0; border-radius: var(--mui-shape-borderRadius, 4px);
  background: transparent; color: inherit; cursor: pointer;
  font: var(--mui-font-button, 500 0.875rem/1.75 Roboto, sans-serif);
  font-size: 0.8125rem; letter-spacing: 0.02857em; text-transform: none;
  transition: background-color 250ms cubic-bezier(0.4, 0, 0.2, 1);
}
.pa-toggle svg { width: 18px; height: 18px; }
.pa-toggle:hover { background: var(--mui-palette-action-hover, rgba(0, 0, 0, 0.04)); }
.pa-toggle[aria-expanded="true"] { background: var(--mui-palette-action-selected, rgba(0, 0, 0, 0.08)); }
.pa-toggle:focus-visible {
  background: var(--mui-palette-action-focus, rgba(0, 0, 0, 0.12));
  outline: 2px solid currentColor; outline-offset: -2px;
}
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


class DrawerResizeHandle(JSComponent):
    """
    A bar on the drawer's inner edge that resizes it when dragged.

    While dragging, the drawer follows the pointer in the page; ``size`` is
    set once the pointer is released. With focus, the arrow keys resize it
    by 24 pixels. It sits on the left edge of a drawer docked to the right.
    """

    size = param.Integer(default=None, allow_None=True, doc="The width the bar resized the drawer to.")

    min_size = param.Integer(default=240, doc="The narrowest the drawer gets.")

    max_ratio = param.Number(default=0.6, doc="The largest share of the window's width the drawer may take.")

    label = param.String(default="Resize the drawer", doc="The bar's accessible name.")

    _esm = """
    export function render({ model, el }) {
      const bar = document.createElement("div")
      bar.className = "pa-drawer-handle"
      bar.tabIndex = 0
      bar.setAttribute("role", "separator")
      bar.setAttribute("aria-orientation", "vertical")
      bar.setAttribute("aria-label", model.label)
      el.style.height = "100%"
      el.appendChild(bar)
      // The drawer's paper, found through the shadow roots of the layouts between.
      function paper() {
        let node = el
        while (node) {
          if (node.classList && node.classList.contains("MuiDrawer-paper")) return node
          const root = node.getRootNode && node.getRootNode()
          node = node.parentElement || (root && root !== node ? root.host : null)
        }
        return null
      }
      function clamp(size) {
        const most = Math.max(model.min_size, window.innerWidth * model.max_ratio)
        return Math.round(Math.max(model.min_size, Math.min(size, most)))
      }
      function apply(size) {
        const target = paper()
        if (!target) return
        target.style.width = `${size}px`
        // An inline drawer's own element takes its width too, so the content beside it narrows.
        const host = target.getRootNode().host
        if (host && host.style.width && host.style.width !== "0px") host.style.width = `${size}px`
        bar.setAttribute("aria-valuenow", String(size))
        bar.setAttribute("aria-valuemin", String(model.min_size))
        bar.setAttribute("aria-valuemax", String(clamp(Infinity)))
      }
      // The drag is followed on the window, so the pointer may leave the narrow bar.
      let start = null
      function move(event) {
        if (!start) return
        start.last = clamp(start.size + start.at - event.clientX)
        apply(start.last)
      }
      function end() {
        window.removeEventListener("pointermove", move)
        window.removeEventListener("pointerup", end)
        window.removeEventListener("pointercancel", end)
        if (!start) return
        bar.classList.remove("dragging")
        if (start.last != null) {
          model.size = start.last
          // Plots and maps beside the drawer measure themselves on window resizes only.
          window.dispatchEvent(new Event("resize"))
        }
        start = null
      }
      bar.addEventListener("pointerdown", (event) => {
        const target = paper()
        if (!target) return
        start = {at: event.clientX, size: target.getBoundingClientRect().width}
        bar.classList.add("dragging")
        window.addEventListener("pointermove", move)
        window.addEventListener("pointerup", end)
        window.addEventListener("pointercancel", end)
        event.preventDefault()
      })
      bar.addEventListener("keydown", (event) => {
        const step = {ArrowLeft: 24, ArrowRight: -24}[event.key]
        const target = paper()
        if (!step || !target) return
        event.preventDefault()
        const size = clamp(target.getBoundingClientRect().width + step)
        apply(size)
        model.size = size
      })
      model.on("size", () => { if (model.size != null) apply(model.size) })
    }
    """

    _stylesheets = ["""
    :host { cursor: col-resize; }
    .pa-drawer-handle { width: 8px; height: 100%; display: flex; align-items: center; justify-content: center;
      touch-action: none; outline: none; }
    .pa-drawer-handle::after { content: ""; width: 3px; height: 32px; border-radius: 2px;
      background: var(--mui-palette-divider, rgba(0, 0, 0, 0.2)); }
    .pa-drawer-handle:hover::after, .pa-drawer-handle:focus-visible::after, .pa-drawer-handle.dragging::after {
      background: var(--mui-palette-primary-main, #0072b5); }
    """]
