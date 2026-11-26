"""
Playwright capture of the screenshots under docs/assets/images/, built from
the examples/ scripts and the getting-started.md snippets. Rerun it after a
UI change to the components it covers:

    pixi run -e test-ui python3 scripts/docs_screenshots.py [names...]

With no names it (re)writes every screenshot named in REGISTRY; with one or
more names it writes only those, e.g. `... docs_screenshots.py drawer-light`.

No live LLM calls are made: every agent here runs a pydantic-ai
FunctionModel or TestModel scripted in this file (see `scripted` and
`_answering` below), which is what examples/*.py also do when
PANEL_AGENTS_MODEL=test (set below, before anything imports an agent, so
`Agent(os.environ["PANEL_AGENTS_MODEL"])` resolves to a TestModel instead of
needing a provider or API key). `examples/sql_analyst/app.py`'s own agent is
driven the same way by overriding `agent.model` after import.

The test-ui pixi environment has pydantic-ai, Panel's test utilities and the
Playwright browsers this needs (`pixi run -e test-ui playwright install
chromium` installs the browser the first time). `examples/sql_analyst/app.py`
also needs the `tabulate` package for its query preview; it is not a project
dependency, so install it into the same environment if it is missing
(`pixi run -e test-ui pip install tabulate`).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PANEL_AGENTS_MODEL", "test")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "assets" / "images"
OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd
import panel as pn
import panel_material_ui as pmui
from panel.io.state import state as pn_state
from panel.tests.util import serve_and_wait, wait_until
from playwright.sync_api import sync_playwright
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from panel_agents import (
    AgentChat,
    AgentDrawer,
    ApprovalPolicy,
    ArtifactPane,
    Mention,
    ModelChoice,
    RunState,
)
from panel_agents import events as ev
from panel_agents.adapters.pydantic_ai import AgentRunner

REGISTRY: dict[str, callable] = {}


def capture(name):
    def register(fn):
        REGISTRY[name] = fn
        return fn
    return register


# --- FunctionModel helpers --------------------------------------------------

def text_response(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


def tool_response(*calls: tuple[str, dict]) -> ModelResponse:
    return ModelResponse(parts=[
        ToolCallPart(tool_name=name, args=args, tool_call_id=f"call_{name}_{i}")
        for i, (name, args) in enumerate(calls)
    ])


def scripted(*responses: ModelResponse, name: str = "scripted") -> FunctionModel:
    """
    A deterministic model that streams each response in turn (runs are
    always streamed), repeating the last once exhausted.
    """
    state = {"i": 0}

    def function(messages, info):
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[i]

    async def stream(messages, info):
        for i, part in enumerate(function(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                yield {i: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)}

    return FunctionModel(function, stream_function=stream, model_name=name)


# --- Browser plumbing --------------------------------------------------------

@dataclass
class Session:
    browser: object
    context: object
    page: object

    def close(self):
        self.context.close()
        self.browser.close()
        pn_state.reset()


def open_page(pw, width=1100, height=700):
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    return Session(browser, context, page)


def serve(page, component, **kwargs):
    msgs = []
    page.on("console", lambda msg: msgs.append(msg))
    port = serve_and_wait(component, page, **kwargs)
    page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
    wait_until(lambda: any("Websocket connection 0 is now open" in str(m) for m in msgs), page, interval=10)
    page.wait_for_function("document.readyState === 'complete'", timeout=15000)
    wait_until(lambda: any(
        "items were rendered successfully" in str(m) or "Error rendering Bokeh items" in str(m) for m in msgs
    ), page, interval=10)
    page.wait_for_timeout(300)
    return port


def send(page, text, placeholder="Ask the agent…"):
    box = page.get_by_placeholder(placeholder)
    box.fill(text)
    box.press("Enter")


def shoot(page, path, locator=None, full_page=False):
    target = locator if locator is not None else page
    kwargs = {"full_page": True} if full_page and locator is None else {}
    target.screenshot(path=str(OUT / path), **kwargs)
    print(f"wrote {path}")


# --- Getting started ---------------------------------------------------------

@capture("getting-started-chat")
def getting_started_chat(pw):
    import random

    agent = Agent("test", instructions="You are a helpful assistant.")

    @agent.tool_plain
    def roll_dice(sides: int = 6) -> int:
        """Roll a die with the given number of sides."""
        return random.randint(1, sides)

    random.seed(3)
    agent.model = scripted(
        tool_response(("roll_dice", {"sides": 6}), ("roll_dice", {"sides": 6})),
        text_response("I rolled two dice: the results are shown above."),
        name="claude-sonnet-5-5",
    )
    runner = AgentRunner(agent)
    chat = AgentChat(runner=runner, tool_labels={"roll_dice": "Roll a die"})

    s = open_page(pw, 1100, 320)
    try:
        serve(s.page, pmui.Column(chat, height=300, sizing_mode="stretch_width", css_classes=["shot-root"]))
        send(s.page, "Roll two dice for me")
        wait_until(lambda: s.page.get_by_text("I rolled two dice").count() > 0, s.page, timeout=10000)
        wait_until(lambda: s.page.get_by_text("2 steps", exact=False).count() > 0, s.page, timeout=5000)
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(400)
        shoot(s.page, "getting-started-chat.png", s.page.locator(".shot-root"))
    finally:
        s.close()


@dataclass
class _RevenueDeps:
    table: object


@capture("getting-started-approval")
def getting_started_approval(pw):
    REVENUE = pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [120_000, 95_000, 143_000, 88_000],
    })

    agent = Agent(
        "test", deps_type=_RevenueDeps,
        instructions="You help the user with the revenue table they are looking at.",
    )

    @agent.tool
    def get_table(ctx: RunContext[_RevenueDeps]) -> list[dict]:
        """The rows currently shown in the table."""
        return ctx.deps.table.value.to_dict("records")

    @agent.tool
    def filter_regions(ctx: RunContext[_RevenueDeps], regions: list[str]) -> str:
        """Show only the given regions in the table."""
        ctx.deps.table.value = REVENUE[REVENUE.region.isin(regions)]
        return f"Showing {', '.join(regions)}."

    agent.model = scripted(
        tool_response(("get_table", {})),
        tool_response(("filter_regions", {"regions": ["North", "East"]})),
    )
    table = pn.widgets.Tabulator(REVENUE, show_index=False, sizing_mode="stretch_width")
    runner = AgentRunner(agent, deps=_RevenueDeps(table), approval_policy=ApprovalPolicy(["filter_*"]))
    drawer = AgentDrawer(runner, main=table, title="Assistant", open=True)
    page_obj = pn.ui.Page(title="Revenue", header=[drawer.toggle], main=[drawer])

    s = open_page(pw, 1100, 580)
    try:
        serve(s.page, page_obj)
        send(s.page, "Show only the regions above 100k")
        wait_until(lambda: s.page.get_by_role("button", name="Approve").count() > 0, s.page, timeout=10000)
        s.page.wait_for_timeout(400)
        shoot(s.page, "getting-started-approval.png")
    finally:
        s.close()


# --- examples/drawer_app.py ---------------------------------------------------

def _drawer_app_model():
    return scripted(
        tool_response(("get_report", {})),
        tool_response(("update_report", {"regions": ["North", "East"], "group_by": "region"})),
        text_response("Updated the report to show North and East, grouped by region."),
        name="claude-sonnet-5-5",
    )


def _drawer_app_shot(pw, dark: bool, filename: str):
    import examples.drawer_app as da

    page_obj = da.create_app(model=_drawer_app_model())
    page_obj.dark_theme = dark

    s = open_page(pw, 1100, 640)
    try:
        serve(s.page, page_obj)
        send(s.page, "Show North and East grouped by region")
        wait_until(lambda: s.page.get_by_text("Updated the report").count() > 0, s.page, timeout=10000)
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(400)
        shoot(s.page, filename)
    finally:
        s.close()


@capture("drawer-light")
def drawer_light(pw):
    _drawer_app_shot(pw, dark=False, filename="drawer-light.png")


@capture("drawer-dark")
def drawer_dark(pw):
    _drawer_app_shot(pw, dark=True, filename="drawer-dark.png")


# --- Model picker & mentions ---------------------------------------------------

def _answering(name: str) -> FunctionModel:
    async def stream(messages, info):
        yield f"Answered by {name}."
    return FunctionModel(stream_function=stream, model_name=name)


@capture("model-picker-menu")
def model_picker_menu(pw):
    gpt5 = _answering("gpt-5-mini")
    claude = _answering("claude-sonnet-5-5")
    choices = [
        ModelChoice("gpt-5-mini", label="GPT-5 mini", description="Fast and cheap", model=gpt5),
        ModelChoice("claude-sonnet-5-5", label="Claude Sonnet", model=claude),
    ]
    runner = AgentRunner(Agent(gpt5))
    drawer = AgentDrawer(runner, open=True, chat_params={"models": choices}, title="Assistant")

    s = open_page(pw, 420, 340)
    try:
        serve(s.page, drawer)
        s.page.get_by_role("button", name="Model: GPT-5 mini").click()
        s.page.get_by_role("listbox", name="Model").wait_for()
        s.page.wait_for_timeout(300)
        shoot(s.page, "model-picker-menu.png")
    finally:
        s.close()


@capture("model-picker-single")
def model_picker_single(pw):
    claude = _answering("claude-sonnet-5-5")
    choices = [ModelChoice("claude-sonnet-5-5", label="Claude Sonnet", model=claude)]
    runner = AgentRunner(Agent(claude))
    drawer = AgentDrawer(runner, open=True, chat_params={"models": choices}, title="Assistant")

    s = open_page(pw, 420, 220)
    try:
        serve(s.page, drawer)
        label = s.page.locator(".pa-model-fixed")
        label.wait_for()
        s.page.wait_for_timeout(300)
        shoot(s.page, "model-picker-single.png")
    finally:
        s.close()


@capture("mentions-chip")
def mentions_chip(pw):
    agent = Agent(_answering("claude-sonnet-5-5"))
    runner = AgentRunner(agent)
    drawer = AgentDrawer(runner, open=True, title="Assistant")
    drawer.chat.mentions = [
        Mention("orders_table", icon="table_chart"),
        Mention("Monthly revenue by customer segment", icon="bar_chart"),
    ]

    s = open_page(pw, 420, 260)
    try:
        serve(s.page, drawer)
        s.page.locator(".pa-mention").first.wait_for()
        s.page.wait_for_timeout(300)
        shoot(s.page, "mentions-chip.png")
    finally:
        s.close()


# --- examples/approval_demo.py --------------------------------------------------

@capture("approval-card-edit")
def approval_card_edit(pw):
    import examples.approval_demo as ad

    s = open_page(pw, 820, 440)
    try:
        serve(s.page, pmui.Column(ad.queue, width=780, css_classes=["shot-root"]))
        s.page.locator("button.MuiToggleButton-root").filter(has_text="Edit").first.click()
        limit = s.page.locator("input[type=text]").first
        limit.wait_for()
        limit.fill("5000")
        limit.press("Tab")
        s.page.get_by_role("button", name="Approve with edits").wait_for()
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(300)
        shoot(s.page, "approval-card-edit.png", s.page.locator(".shot-root"))
    finally:
        s.close()


# --- examples/sql_analyst/app.py: document layout -------------------------------

@capture("chat-document-layout")
def chat_document_layout(pw):
    import examples.sql_analyst.app as sql_app

    model = scripted(
        tool_response(("list_tables", {})),
        tool_response(("run_query", {
            "sql": "SELECT region, SUM(revenue) AS revenue\nFROM orders o JOIN customers c USING (customer_id)\nGROUP BY region\nORDER BY revenue DESC",
            "name": "revenue_by_region", "title": "Revenue by region",
        })),
        text_response("North America leads, driven by Enterprise customers. I published the results as revenue_by_region."),
        name="claude-sonnet-5-5",
    )
    state = RunState()
    con = sql_app.create_database()
    runner = AgentRunner(
        sql_app.agent, state, deps=sql_app.AnalystDeps(con, state),
        approval_policy=sql_app.POLICY, run_kwargs={"model": model},
    )
    chat = AgentChat(
        runner=runner, layout="document", assistant="Analyst",
        feed_params={"height": 560},
    )

    s = open_page(pw, 1100, 660)
    try:
        serve(s.page, pmui.Container(chat, width_option="md", css_classes=["shot-root"]))
        send(s.page, "Which regions drove the most revenue?")
        wait_until(lambda: s.page.get_by_text("published the results").count() > 0, s.page, timeout=15000)
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(500)
        shoot(s.page, "chat-document-layout.png")
    finally:
        s.close()


# --- examples/artifacts_demo.py -------------------------------------------------

@capture("artifact-history")
def artifact_history(pw):
    import time

    import examples.artifacts_demo as adm

    # examples/artifacts_demo.py's own run has no input, so its group heading
    # falls back to the agent name ("Agent"); build the same SQL versions
    # under a run that has one, to show the heading the docs describe.
    start = time.time() - 600
    events = [ev.RunStarted(run_id="r1", input="Show revenue by region for 2026", timestamp=start)]
    for i, sql in enumerate(adm.SQL_VERSIONS):
        events.append(ev.ArtifactPublished(
            artifact_id="query", value=sql, kind="sql", title="Revenue by region (SQL)",
            tool_call_id=f"call_sql_{i + 1}", run_id="r1", metadata={"dialect": "duckdb"},
            timestamp=start + 60 * i,
        ))
    state = RunState.from_events(events)
    pane = ArtifactPane(state=state, show_list=True)
    pane.store.active = "query"

    s = open_page(pw, 900, 620)
    try:
        serve(s.page, pmui.Column(pane, width=860, height=580, css_classes=["shot-root"]))
        s.page.get_by_role("button", name="Version history").click()
        s.page.get_by_text("Version 1", exact=True).wait_for()
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(300)
        shoot(s.page, "artifact-history.png", s.page.locator(".shot-root"))
    finally:
        s.close()


@capture("artifact-undo-redo")
def artifact_undo_redo(pw):
    import examples.artifacts_demo as adm

    pane = ArtifactPane(state=adm.state, on_apply=adm.apply_chart, show_list=True)
    pane.store.active = "chart"

    s = open_page(pw, 900, 620)
    try:
        serve(s.page, pmui.Column(pane, width=860, height=580, css_classes=["shot-root"]))
        s.page.get_by_role("button", name="Undo").wait_for()
        s.page.wait_for_timeout(300)
        shoot(s.page, "artifact-undo-redo.png", s.page.locator(".shot-root"))
    finally:
        s.close()


@capture("artifact-diff")
def artifact_diff(pw):
    import examples.artifacts_demo as adm

    pane = ArtifactPane(state=adm.state, on_apply=adm.apply_chart, show_list=True)
    pane.store.active = "chart"

    s = open_page(pw, 900, 620)
    try:
        serve(s.page, pmui.Column(pane, width=860, height=580, css_classes=["shot-root"]))
        s.page.get_by_role("tab", name="Compare").click()
        s.page.wait_for_timeout(300)
        shoot(s.page, "artifact-diff.png", s.page.locator(".shot-root"))
    finally:
        s.close()


# --- examples/chat_replay.py -----------------------------------------------------

@capture("chat-replay")
def chat_replay_shot(pw):
    import examples.chat_replay as cr

    s = open_page(pw, 1100, 900)
    try:
        serve(s.page, pmui.Column(cr.chat, width=1000, css_classes=["shot-root"]))
        s.page.get_by_text("North America led Q3").wait_for()
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(300)
        shoot(s.page, "chat-replay.png", full_page=True)
    finally:
        s.close()


@capture("span-view")
def span_view_shot(pw):
    import examples.chat_replay as cr

    drawer = AgentDrawer(state=cr.state, open=True)
    drawer.show_run("run-1")

    s = open_page(pw, 920, 780)
    try:
        serve(s.page, pn.ui.Page(main=[drawer]))
        s.page.get_by_role("dialog").get_by_text("How this was made").wait_for()
        s.page.mouse.move(0, 0)
        s.page.wait_for_timeout(300)
        shoot(s.page, "span-view.png")
    finally:
        s.close()


# --- Entry point --------------------------------------------------------------

def main():
    names = sys.argv[1:] or list(REGISTRY)
    with sync_playwright() as pw:
        for name in names:
            print(f"--- {name} ---")
            REGISTRY[name](pw)


if __name__ == "__main__":
    main()
