import json

import pytest

pytest.importorskip("playwright")

import panel_material_ui as pmui
from panel.tests.util import serve_component, wait_until

from panel_agents import AgentChat, ApprovalCard, ArtifactPane, RunState
from panel_agents import events as ev

from .sanitiser_payloads import PAYLOADS, img, tool_output_variants

pytestmark = pytest.mark.ui

_LINKS = """() => {
  const out = [];
  const walk = (root) => root.querySelectorAll('*').forEach(e => {
    if (e.shadowRoot) walk(e.shadowRoot);
    const href = e.getAttribute('href') || e.getAttribute('xlink:href');
    if (href && /^\\s*(javascript|vbscript|data):/i.test(href)) out.push(href);
  });
  walk(document);
  return out;
}"""


def watch(page):
    requests = []
    page.on("request", lambda request: requests.append(request.url) if "evil.test" in request.url else None)
    page.route("**/*evil.test*/**", lambda route: route.abort())
    page.route("**/*evil.test*", lambda route: route.abort())
    return requests


def assert_nothing_happened(page, requests, marker):
    wait_until(lambda: page.get_by_text(marker).count() > 0, page)
    page.wait_for_timeout(1000)
    # Links may only fetch when clicked, so nothing is clicked here; script-carrying hrefs are checked directly.
    assert requests == []
    assert page.evaluate("window.__pwned || []") == []
    assert page.evaluate(_LINKS) == []


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_untrusted_output_loads_and_runs_nothing(page, name):
    payload = PAYLOADS[name]
    requests = watch(page)
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.TextMessageStart(message_id="m", run_id="r"),
        ev.TextMessageContent(message_id="m", delta=f"MARKER {name}\n\n{payload}", run_id="r"),
        ev.TextMessageEnd(message_id="m", run_id="r"),
        ev.ThinkingStart(message_id="t", run_id="r"),
        ev.ThinkingContent(message_id="t", delta=payload, run_id="r"),
        ev.ThinkingEnd(message_id="t", run_id="r"),
        ev.ToolCallStart(tool_call_id="c1", tool_name="fetch_page", run_id="r"),
        ev.ToolCallEnd(tool_call_id="c1", run_id="r"),
        ev.ToolCallResult(tool_call_id="c1", content=payload, run_id="r"),
        ev.ArtifactPublished(artifact_id="notes", value=payload, kind="markdown", title="Notes", run_id="r"),
    ])
    for variant, value in tool_output_variants(payload).items():
        state.apply_many([
            ev.ToolCallStart(tool_call_id=variant, tool_name=variant, run_id="r"),
            ev.ToolCallArgs(tool_call_id=variant, delta=json.dumps({"body": payload, "code": f"x = 1\n{payload}"}), run_id="r"),
            ev.ToolCallEnd(tool_call_id=variant, run_id="r"),
            ev.ToolCallResult(tool_call_id=variant, content=value, run_id="r"),
        ])
    state.apply_many([
        ev.ToolCallStart(tool_call_id="err", tool_name="fails", run_id="r"),
        ev.ToolCallEnd(tool_call_id="err", run_id="r"),
        ev.ToolCallResult(tool_call_id="err", content=payload, is_error=True, run_id="r"),
    ])
    chat = AgentChat(state, readonly=True, group_tool_calls=False)
    for call in state.tool_calls.values():
        chat.view_for(call).collapsed = False
    chat.view_for(state.messages["t"]).collapsed = False
    serve_component(page, pmui.Row(chat, ArtifactPane(state=state), height=900, sizing_mode="stretch_width"))
    assert_nothing_happened(page, requests, "MARKER")


def test_titles_descriptions_reasons_and_errors_are_plain_text(page):
    requests = watch(page)
    state = RunState(flush_interval=0)
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ArtifactPublished(artifact_id="notes", value="MARKER", kind="text", title=img("artifact_title") + " Notes", run_id="r"),
        ev.ToolCallStart(tool_call_id="c1", tool_name="publish", run_id="r"),
        ev.ApprovalRequested(
            tool_call_id="c1", tool_name="publish", run_id="r", args={"body": img("approval_arg")},
            description="Publish. " + img("approval_description"),
        ),
        ev.ToolCallStart(tool_call_id="c2", tool_name="fetch", run_id="r"),
        ev.ToolCallEnd(tool_call_id="c2", run_id="r"),
        ev.ToolCallResult(tool_call_id="c2", content="Failed: " + img("tool_error"), is_error=True, run_id="r"),
    ])
    card = ApprovalCard(request=state.approvals["c1"], state=state, user="alice")
    chat = AgentChat(state, readonly=True)
    chat.view_for(state.tool_calls["c2"]).collapsed = False
    serve_component(page, pmui.Row(pmui.Column(card, chat), ArtifactPane(state=state), height=900, sizing_mode="stretch_width"))
    assert_nothing_happened(page, requests, "Publish.")
    state.resolve_approval("c1", False, user=img("user_name"), reason=img("approval_reason"))
    wait_until(lambda: page.get_by_text("Denied").count() > 0, page)
    assert_nothing_happened(page, requests, "Denied")
