import panel_material_ui as pmui
import pytest

from panel_agents import AgentDrawer, ApprovalPolicy, ApprovalRule, RunState
from panel_agents import events as ev


class Runner:
    def __init__(self, approval_policy=None):
        self.state = RunState(flush_interval=0)
        self.approval_policy = approval_policy
        self.prompts = []
        self.cleared = 0

    async def run(self, text):
        self.prompts.append(text)
        self.state.add_user_message(text)

    def clear(self):
        self.cleared += 1


def request(state, tool_call_id="t1", tool_name="save"):
    state.apply_many([
        ev.RunStarted(run_id="r"),
        ev.ApprovalRequested(tool_call_id=tool_call_id, tool_name=tool_name, run_id="r"),
    ])


def shown(drawer):
    return next(name for name, view in drawer._views.items() if drawer._body.objects == [view])


def test_drawer_beside_main_content():
    main = pmui.Typography("app")
    drawer = AgentDrawer(Runner(), main=main, width=380)
    wrapper, drawer_view, resize = drawer.__panel__().objects
    assert wrapper.objects == [main] and drawer_view is drawer._drawer and resize is drawer._resize
    assert (drawer._drawer.anchor, drawer._drawer.size, drawer._drawer.inline) == ("right", 380, True)
    assert shown(drawer) == "chat" and not drawer._back.visible
    assert drawer._heading.object == "Assistant"
    drawer.main = None
    assert drawer.__panel__().objects == [drawer._drawer, drawer._resize]
    drawer.variant = "temporary"
    assert drawer._drawer.variant == "temporary" and not drawer._drawer.inline


def test_open_is_synced_with_drawer_and_toggle():
    drawer = AgentDrawer(Runner())
    drawer.open = True
    assert drawer._drawer.open
    drawer._drawer.open = False
    assert not drawer.open
    drawer.toggle.expanded = True
    assert drawer.open and drawer._drawer.open
    assert drawer.toggle.label == "Assistant"
    drawer.title = "Helper"
    assert drawer.toggle.label == "Helper"


def test_pending_approval_opens_and_badges():
    runner = Runner()
    drawer = AgentDrawer(runner)
    request(runner.state)
    assert drawer.open
    assert drawer.toggle.count == 1
    assert drawer._pending_bar.visible and drawer._pending_text.object == "1 action waits for you"
    assert [c.request.tool_call_id for c in drawer.approvals.cards] == ["t1"]
    drawer.tab = "approvals"
    assert shown(drawer) == "approvals" and drawer._back.visible and not drawer._pending_bar.visible
    assert drawer._heading.object == "Approvals"
    drawer.tab = "chat"
    runner.state.resolve_approval("t1", approved=True)
    assert drawer.toggle.count == 0 and not drawer._pending_bar.visible


def test_open_on_approval_can_be_disabled():
    runner = Runner()
    drawer = AgentDrawer(runner, open_on_approval=False)
    request(runner.state)
    assert not drawer.open and drawer.toggle.count == 1


def test_views_follow_settings_and_tab_param():
    runner = Runner()
    drawer = AgentDrawer(runner, show_approvals=False, open_on_approval=False)
    request(runner.state)
    assert not drawer._pending_bar.visible
    drawer.tab = "artifacts"
    assert shown(drawer) == "artifacts" and drawer._heading.object == "Artifacts"
    drawer._back.clicks += 1
    assert drawer.tab == "chat" and shown(drawer) == "chat"
    drawer.tab = "artifacts"
    drawer.show_artifacts = False
    assert drawer.tab == "chat" and shown(drawer) == "chat"


def test_artifacts_view_selects_artifacts():
    runner = Runner()
    applied = []
    drawer = AgentDrawer(runner, on_apply=lambda artifact, version: applied.append(version))
    state = runner.state
    state.apply(ev.ArtifactPublished(artifact_id="a", value="x", kind="text", title="First"))
    state.apply(ev.ArtifactPublished(artifact_id="b", value="y", kind="text"))
    assert drawer._artifacts_badge.visible and drawer._artifacts_badge.content == 2
    drawer._artifacts_button.clicks += 1
    assert drawer.tab == "artifacts" and not drawer._artifacts_badge.visible
    assert drawer._artifact_select.options == {"First": "a", "b": "b"}
    assert drawer._artifact_select.value == "b"
    drawer._artifact_select.value = "a"
    assert state.artifacts.active == "a"
    assert drawer.artifacts.active_view.on_apply is drawer.on_apply
    assert drawer.artifacts.active_view.state is state


def test_policy_from_runner_reaches_chat_and_queue():
    policy = ApprovalPolicy([ApprovalRule("save", approvers={"admin"})])
    drawer = AgentDrawer(Runner(approval_policy=policy))
    assert drawer.can_approve == policy.can_approve
    assert drawer.chat.can_approve == policy.can_approve
    assert drawer.approvals.can_approve == policy.can_approve


async def test_submit_and_new_conversation():
    runner = Runner()
    drawer = AgentDrawer(runner)
    await drawer.chat.submit("hello")
    assert runner.prompts == ["hello"]
    runner.state.apply(ev.ArtifactPublished(artifact_id="a", value="x", kind="text"))
    drawer.new_conversation()
    assert runner.cleared == 1
    assert runner.state.items == [] and len(runner.state.artifacts) == 0
    assert drawer.artifacts.store is runner.state.artifacts
    assert not drawer._artifacts_badge.visible
    runner.state.apply(ev.ArtifactPublished(artifact_id="c", value="z", kind="text"))
    assert drawer._artifact_select.options == {"c": "c"}


def test_new_conversation_waits_for_the_run():
    runner = Runner()
    drawer = AgentDrawer(runner)
    runner.state.apply(ev.RunStarted(run_id="r"))
    assert drawer._clear.disabled
    drawer.new_conversation()
    assert runner.cleared == 0 and runner.state.runs


def test_state_without_runner():
    state = RunState()
    drawer = AgentDrawer(state=state)
    assert drawer.chat.state is state and not drawer._clear.visible


def test_temporary_drawer_only_badges_new_approvals():
    runner = Runner()
    drawer = AgentDrawer(runner, variant="temporary")
    request(runner.state)
    assert not drawer.open and drawer.toggle.count == 1


def test_drawer_follows_a_new_runner_policy():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent

    from panel_agents.adapters.pydantic_ai import AgentRunner

    runner = AgentRunner(Agent("test"), RunState())
    drawer = AgentDrawer(runner)
    assert drawer.can_approve is None
    policy = ApprovalPolicy(["save"])
    runner.approval_policy = policy
    assert drawer.can_approve == policy.can_approve
    assert drawer.chat.can_approve == policy.can_approve and drawer.approvals.can_approve == policy.can_approve


def test_new_conversation_asks_before_dropping_documents():
    runner = Runner()
    drawer = AgentDrawer(runner)
    runner.state.apply(ev.ArtifactPublished(artifact_id="doc", value={}, kind="json", applied=True))
    drawer._ask_new_conversation()
    assert drawer._confirm.open and runner.cleared == 0
    drawer._confirm_new_conversation()
    assert not drawer._confirm.open and runner.cleared == 1 and len(runner.state.artifacts) == 0
