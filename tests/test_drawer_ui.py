import panel_material_ui as pmui
import pytest

pytest.importorskip("playwright")

from panel.tests.util import serve_component, wait_until
from playwright.sync_api import expect

from panel_agents import AgentDrawer, RunState
from panel_agents import events as ev

pytestmark = pytest.mark.ui


@pytest.mark.parametrize("resizable", [True, False])
def test_closed_drawer_takes_no_space(page, resizable):
    main = pmui.Paper(pmui.Typography("main"), sizing_mode="stretch_both")
    drawer = AgentDrawer(state=RunState(), main=main, resizable=resizable)
    page.set_viewport_size({"width": 1000, "height": 600})
    serve_component(page, pmui.Column(drawer, height=500, sizing_mode="stretch_width"))
    paper = page.locator(".MuiPaper-root").first
    expect(paper).to_be_visible()
    wait_until(lambda: abs(paper.bounding_box()["width"] - 1000) <= 1, page)


def test_drawer_toggle_close_and_approval_badge(page):
    state = RunState(flush_interval=0)
    drawer = AgentDrawer(state=state, main=pmui.Typography("The app", variant="h4"), width=400)
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=600, sizing_mode="stretch_width"))

    expect(page.get_by_text("The app")).to_be_visible()
    expect(page.get_by_text("Assistant", exact=True).last).not_to_be_visible()

    page.get_by_role("button", name="Assistant", exact=True).click()
    wait_until(lambda: drawer.open, page)
    expect(page.get_by_text("Assistant", exact=True).last).to_be_visible()
    expect(page.get_by_role("button", name="Close", exact=True)).to_be_visible()

    page.get_by_role("button", name="Close", exact=True).click()
    wait_until(lambda: not drawer.open, page)

    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id="t1", tool_name="save_report", run_id="r"),
    ])
    wait_until(lambda: drawer.open, page)
    toggle = page.get_by_role("button", name="Assistant, 1 pending approval", exact=True)
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(page.locator(".pa-badge")).to_have_text("1")
    expect(page.get_by_text("1 action waits for you")).to_be_visible()
    page.get_by_role("button", name="Review").click()
    wait_until(lambda: drawer.tab == "approvals", page)
    expect(page.get_by_role("button", name="Approve")).to_be_visible()
    page.get_by_role("button", name="Back to chat").click()
    wait_until(lambda: drawer.tab == "chat", page)


def test_focus_moves_into_the_drawer_and_back(page):
    drawer = AgentDrawer(state=RunState(), main=pmui.Typography("The app"))
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=600, sizing_mode="stretch_width"))
    toggle = page.get_by_role("button", name="Assistant", exact=True)
    expect(toggle).to_have_attribute("aria-expanded", "false")
    toggle.click()
    wait_until(lambda: drawer.open, page)
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(page.locator("textarea:focus")).to_have_count(1)
    page.get_by_role("button", name="Close", exact=True).click()
    wait_until(lambda: not drawer.open, page)
    expect(page.locator("button.pa-toggle:focus")).to_have_count(1)


def test_opening_the_drawer_tells_the_page_to_relayout(page):
    drawer = AgentDrawer(state=RunState(), main=pmui.Typography("The app"))
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=500, sizing_mode="stretch_width"))
    expect(page.get_by_text("The app")).to_be_visible()
    page.evaluate("window._resizes = 0; window.addEventListener('resize', () => window._resizes++)")
    page.get_by_role("button", name="Assistant", exact=True).click()
    wait_until(lambda: page.evaluate("window._resizes") > 0, page)


def test_a_narrow_window_overlays_the_drawer(page):
    page.set_viewport_size({"width": 900, "height": 600})
    main = pmui.Paper(pmui.Typography("main"), sizing_mode="stretch_both")
    drawer = AgentDrawer(state=RunState(), main=main, overlay_below=1200, width=360)
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=500, sizing_mode="stretch_width"))
    wait_until(lambda: drawer._drawer.variant == "temporary", page)
    page.get_by_role("button", name="Assistant", exact=True).click()
    wait_until(lambda: drawer.open, page)
    # Overlaid, the app keeps its whole width.
    paper = page.locator(".MuiPaper-root").first
    wait_until(lambda: paper.bounding_box()["width"] > 850, page)


def test_a_drawer_wider_than_a_phone_stays_on_screen(page):
    page.set_viewport_size({"width": 390, "height": 844})
    main = pmui.Paper(pmui.Typography("main"), sizing_mode="stretch_both")
    drawer = AgentDrawer(state=RunState(), main=main, overlay_below=1200, width=420)
    serve_component(page, pmui.Column(pmui.Row(drawer.toggle), drawer, height=800, sizing_mode="stretch_width"))
    wait_until(lambda: drawer._drawer.variant == "temporary", page)
    page.get_by_role("button", name="Assistant", exact=True).click()
    paper = page.locator(".MuiDrawer-paper")
    expect(paper).to_be_visible()
    wait_until(lambda: paper.bounding_box() == {**paper.bounding_box(), "x": 0, "width": 390}, page)
    assert page.evaluate("document.documentElement.scrollWidth") == 390


_LOOK = """el => {
  const style = getComputedStyle(el)
  const box = el.getBoundingClientRect()
  return {
    textTransform: style.textTransform, fontFamily: style.fontFamily, fontSize: style.fontSize,
    fontWeight: style.fontWeight, lineHeight: style.lineHeight, letterSpacing: style.letterSpacing,
    height: box.height, middle: Math.round(box.top + box.height / 2),
  }
}"""


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_toggle_looks_like_a_small_text_button(page, dark):
    drawer = AgentDrawer(state=RunState(), main=pmui.Typography("The app"))
    share = pmui.Button(label="Share", icon="share", variant="text", size="small", margin=0)
    serve_component(
        page,
        pmui.Page(main=[pmui.Row(share, drawer.toggle, styles={"align-items": "center"}), drawer], dark_theme=dark),
    )
    toggle = page.locator("button.pa-toggle")
    button = page.get_by_role("button", name="Share")
    expect(toggle).to_be_visible()
    expect(button).to_be_visible()
    assert toggle.evaluate(_LOOK) == button.evaluate(_LOOK)

    def background():
        return toggle.evaluate("el => getComputedStyle(el).backgroundColor")

    idle = background()
    toggle.focus()
    expect(toggle).to_have_css("outline-style", "solid")
    toggle.blur()
    toggle.hover()
    wait_until(lambda: background() not in (idle, "rgba(127, 127, 127, 0.16)"), page)
    page.mouse.move(0, 0)
    toggle.click()
    wait_until(lambda: drawer.open, page)
    page.get_by_role("button", name="Close", exact=True).click()
    wait_until(lambda: not drawer.open, page)
    wait_until(lambda: background() == idle, page)


def _hit(page, x, y):
    """Return the class of the element under the point, looking into shadow roots."""
    return page.evaluate(
        """([x, y]) => {
          let root = document, hit = null
          while (root) {
            const next = root.elementFromPoint(x, y)
            if (!next || next === hit) break
            hit = next
            root = hit.shadowRoot
          }
          return hit ? String(hit.className) : null
        }""",
        [x, y],
    )


def test_dragging_the_left_edge_resizes_the_drawer(page):
    main = pmui.Paper(pmui.Typography("main"), sizing_mode="stretch_both")
    drawer = AgentDrawer(state=RunState(), main=main, open=True, width=400)
    page.set_viewport_size({"width": 1200, "height": 700})
    serve_component(page, pmui.Column(drawer, height=600, sizing_mode="stretch_width"))
    paper = page.locator(".MuiDrawer-paper").first
    # Once the drawer has slid in.
    wait_until(lambda: (paper.bounding_box() or {}).get("x") == 800, page)
    handle = page.get_by_role("separator", name="Resize Assistant")
    # Described before the first resize.
    expect(handle).to_have_attribute("aria-valuenow", "400")
    expect(handle).to_have_attribute("aria-valuemin", "240")
    expect(handle).to_have_attribute("aria-valuemax", "720")
    box = handle.bounding_box()
    # The whole edge can be grabbed.
    assert box["height"] > 500
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    # Until the drawer has settled after opening, the point can be the page behind it.
    wait_until(lambda: _hit(page, x, y) == "pa-drawer-handle", page)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x - 150, y, steps=6)
    page.mouse.up()
    wait_until(lambda: abs(drawer.width - 550) <= 2, page)
    wait_until(lambda: abs(paper.bounding_box()["width"] - drawer.width) <= 2, page)
    # No wider than 60% of the window.
    handle.focus()
    for _ in range(12):
        page.keyboard.press("ArrowLeft")
    wait_until(lambda: drawer.width == 720, page)
    expect(handle).to_have_attribute("aria-valuenow", "720")
    drawer.width = 300
    wait_until(lambda: abs(paper.bounding_box()["width"] - 300) <= 2, page)
    drawer.title = "Helper"
    expect(page.get_by_role("separator", name="Resize Helper")).to_be_attached()


def test_keyboard_resizing_keeps_to_the_bounds_and_tells_the_page(page):
    drawer = AgentDrawer(state=RunState(), open=True, width=400, min_resize_width=320, max_resize_ratio=0.5)
    page.set_viewport_size({"width": 1200, "height": 700})
    serve_component(page, pmui.Column(drawer, height=600, sizing_mode="stretch_width"))
    handle = page.get_by_role("separator", name="Resize Assistant")
    expect(handle).to_have_attribute("aria-valuemax", "600")
    expect(handle).to_have_attribute("aria-valuemin", "320")
    page.evaluate("window.resizes = 0; window.addEventListener('resize', () => window.resizes++)")
    handle.focus()
    page.keyboard.press("ArrowRight")
    wait_until(lambda: drawer.width == 376, page)
    # Plots beside the drawer remeasure on window resizes, as after a drag.
    assert page.evaluate("window.resizes") >= 1
    for _ in range(4):
        page.keyboard.press("ArrowRight")
    wait_until(lambda: drawer.width == 320, page)
    for _ in range(14):
        page.keyboard.press("ArrowLeft")
    wait_until(lambda: drawer.width == 600, page)


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_toggle_in_page_header_uses_white_text(page, dark):
    drawer = AgentDrawer(state=RunState(), main=pmui.Typography("The app"))
    serve_component(page, pmui.Page(title="App", header=[drawer.toggle], main=[drawer], dark_theme=dark))
    toggle = page.locator("button.pa-toggle")
    expect(toggle).to_be_visible()
    wait_until(lambda: "pa-on-appbar" in (toggle.get_attribute("class") or ""), page)
    expect(toggle).to_have_css("color", "rgb(255, 255, 255)")
