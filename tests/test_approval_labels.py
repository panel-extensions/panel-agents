import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from panel_agents import ApprovalPolicy, ApprovalRule, RunState, dump_events, load_state
from panel_agents import events as ev
from panel_agents.components.approval import ApprovalCard, ApprovalQueue, approval_labels
from panel_agents.components.tool_call import ToolCallCard

from .test_pydantic_ai import make_runner, resolve_on_request, streaming, tool_returns

PLAN = {"approve": "Approve", "deny": "Remove", "denied": "Removed"}


def make_state(*labels):
    state = RunState(flush_interval=0)
    events = [ev.RunStarted(run_id="r")]
    for i, value in enumerate(labels, 1):
        events.append(ev.ApprovalRequested(
            tool_call_id=f"t{i}", tool_name="plan_step", run_id="r",
            args={"question": f"Step {i}"}, labels=value,
        ))
    state.apply_many(events)
    return state


def test_default_labels():
    assert approval_labels() == {
        "approve": "Approve", "deny": "Deny", "approved": "Approved", "denied": "Denied", "edit": "Edit",
        "approve_edited": "Approve with edits", "approved_edited": "Approved with edits",
        "approve_all": "Approve all", "deny_all": "Deny all",
    }


@pytest.mark.parametrize("verb, past", [("Remove", "Removed"), ("Skip", "Skipped"), ("Run", "Ran")])
def test_labels_derive_batch_and_edit_wording(verb, past):
    labels = approval_labels({"approve": verb, "approved": past, "deny": verb, "denied": past})
    assert (labels["approved"], labels["denied"]) == (past, past)
    assert (labels["approve_all"], labels["deny_all"]) == (f"{verb} all", f"{verb} all")
    assert (labels["approve_edited"], labels["approved_edited"]) == (f"{verb} with edits", f"{past} with edits")


@pytest.mark.parametrize("labels", [{"deny": "Remove"}, {"approve": "Run"}, {"deny": "Skip", "approved": "Skipped"}])
def test_changed_verb_needs_its_past_tense(labels):
    with pytest.raises(ValueError, match="also needs"):
        ApprovalRule("plan_step", labels=labels)
    with pytest.raises(ValueError, match="also needs"):
        ApprovalQueue(make_state(None), labels=labels)


def test_default_verb_needs_no_past_tense():
    assert approval_labels({"approve": "Approve", "deny": "Remove", "denied": "Removed"})["approved"] == "Approved"


def test_explicit_batch_labels():
    labels = approval_labels({"deny": "Remove step", "denied": "Removed", "deny_all": "Remove all steps"})
    assert labels["deny_all"] == "Remove all steps"


def test_later_layer_resets_derived_labels():
    labels = approval_labels(
        {"deny": "Reject", "denied": "Rejected", "approve": "Run", "approved": "Ran"},
        {"deny": "Remove", "denied": "Removed"},
        {"approve": "Approve"},
    )
    assert labels == approval_labels()| {"deny": "Remove", "denied": "Removed", "deny_all": "Remove all"}
    assert approval_labels({"deny": "Remove", "denied": "Removed"}, None)["deny"] == "Remove"


def test_recorded_layer_without_past_tense_keeps_earlier_verb():
    labels = approval_labels({"deny": "Reject", "denied": "Rejected"}, {"deny": "Remove"})
    assert (labels["deny"], labels["denied"], labels["deny_all"]) == ("Reject", "Rejected", "Reject all")


def test_default_card_wording_is_unchanged():
    state = make_state(None)
    card = ApprovalCard(request=state.approvals["t1"], state=state, require_reason="deny")
    assert (card._approve.label, card._deny.label, card._edit_toggle.label) == ("Approve", "Deny", "Edit")
    assert card._edit_toggle.description == "Edit the arguments before approving"
    assert card._reason.helper_text == "Required to deny"
    card.can_approve = lambda request, user: False
    assert card._policy.object == "You are not permitted to approve or deny this tool call."


def test_card_labels():
    state = make_state(None)
    card = ApprovalCard(request=state.approvals["t1"], state=state, labels=PLAN, require_reason="deny")
    assert (card._approve.label, card._deny.label) == ("Approve", "Remove")
    assert card._reason.helper_text == "Required to remove"
    card.can_approve = lambda request, user: False
    assert card._policy.object == "You are not permitted to approve or remove this tool call."
    card.can_approve = None
    card.labels = {"deny": "Drop", "denied": "Dropped"}
    assert card._deny.label == "Drop"
    assert card._edit_toggle.description == "Edit the arguments first"


def test_policy_message_escapes_labels():
    state = make_state(None)
    card = ApprovalCard(request=state.approvals["t1"], state=state, labels={"deny": "Drop *now*", "denied": "Dropped"})
    card.can_approve = lambda request, user: False
    assert "drop \\*now\\*" in card._policy.object


def test_card_approve_with_edits_label():
    state = make_state(None)
    card = ApprovalCard(request=state.approvals["t1"], state=state, labels={"approve": "Run", "approved": "Ran"})
    card.editing = True
    card.form.value = {"question": "Step one"}
    assert card._approve.label == "Run with edits"


def test_unknown_label_raises():
    state = make_state(None)
    with pytest.raises(ValueError, match="Unknown approval label"):
        ApprovalCard(request=state.approvals["t1"], state=state, labels={"remove": "Remove"})
    with pytest.raises(ValueError, match="Unknown approval label"):
        ApprovalRule("plan_step", labels={"reject": "Remove"})
    with pytest.raises(ValueError, match="non-empty"):
        ApprovalQueue(state, labels={"deny": ""})


def test_rule_labels_are_read_only():
    rule = ApprovalRule("plan_step", labels=PLAN)
    with pytest.raises(TypeError):
        rule.labels["deny"] = "Drop"


def test_card_factory_gets_labels_only_when_set():
    seen = []

    def factory(request, state, **params):
        seen.append(params)
        return ApprovalCard(request=request, state=state, **params)

    ApprovalQueue(make_state(None), card_factory=factory)
    ApprovalQueue(make_state(None), card_factory=factory, labels=PLAN)
    assert ["labels" in p for p in seen] == [False, True]


def test_request_labels_override_card_labels():
    state = make_state(PLAN)
    card = ApprovalCard(request=state.approvals["t1"], state=state, labels={"deny": "Reject", "denied": "Rejected"})
    assert card._deny.label == "Remove"


def test_resolved_card_uses_labels():
    state = make_state(PLAN)
    card = ApprovalCard(request=state.approvals["t1"], state=state)
    assert card.deny()
    chip = card._layout.objects[0].objects[0].objects[0]
    assert chip.label == "Removed"


def test_queue_labels_propagate_to_cards():
    state = make_state(None, None)
    queue = ApprovalQueue(state, labels=PLAN)
    assert [c._deny.label for c in queue.cards] == ["Remove", "Remove"]
    assert (queue._approve_all.label, queue._deny_all.label) == ("Approve all", "Remove all")
    queue.labels = {"deny": "Reject", "denied": "Rejected"}
    assert [c._deny.label for c in queue.cards] == ["Reject", "Reject"]
    assert queue._deny_all.label == "Reject all"


def test_queue_default_batch_labels_unchanged():
    queue = ApprovalQueue(make_state(None, None))
    assert (queue._approve_all.label, queue._deny_all.label) == ("Approve all", "Deny all")


def test_queue_batch_uses_shared_request_labels():
    queue = ApprovalQueue(make_state(PLAN, PLAN))
    assert queue._deny_all.label == "Remove all"
    mixed = ApprovalQueue(make_state(PLAN, None))
    assert mixed._deny_all.label == "Deny all"


def test_queue_batch_message_uses_labels():
    state = make_state(PLAN, PLAN)
    queue = ApprovalQueue(state, require_reason="deny")
    assert queue.deny_all() == 0
    assert queue.message == "0 of 2 removed; 2 need attention."


def test_tool_call_badge_uses_request_labels():
    state = make_state(PLAN)
    state.resolve_approval("t1", False, reason=None)
    card = ToolCallCard(state.tool_calls["t1"], state=state)
    assert ">Removed<" in card._header.object
    assert card._error.title == "Removed"
    assert card._error.object.startswith("The call was removed")


def test_tool_call_badge_default():
    state = make_state(None)
    state.resolve_approval("t1", False)
    card = ToolCallCard(state.tool_calls["t1"], state=state)
    assert ">Denied<" in card._header.object
    assert card._error.title == "Denied"


def test_labels_round_trip_through_recording(tmp_path):
    state = make_state(PLAN)
    path = tmp_path / "run.jsonl"
    dump_events(state.events, path)
    replayed = load_state(path)
    assert replayed.approvals["t1"].labels == PLAN
    assert ApprovalCard(request=replayed.approvals["t1"], state=replayed)._deny.label == "Remove"


def test_unknown_recorded_label_is_ignored():
    state = make_state({"deny": "Remove", "denied": "Removed", "later_key": "x"})
    assert ApprovalCard(request=state.approvals["t1"], state=state)._deny.label == "Remove"


def plan_agent():
    def model(messages, info):
        if not tool_returns(messages):
            return ModelResponse(parts=[ToolCallPart("plan_step", {"question": "Revenue by region"}, tool_call_id="call-1")])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(streaming(model))

    @agent.tool_plain
    def plan_step(question: str) -> str:
        return question

    return agent


async def test_policy_labels_flow_into_request():
    policy = ApprovalPolicy([ApprovalRule("plan_step", labels=PLAN)])
    runner = make_runner(plan_agent(), approval_policy=policy)
    resolve_on_request(runner.state, approved=True)
    await runner.run("Plan it")
    [event] = [e for e in runner.state.events if isinstance(e, ev.ApprovalRequested)]
    assert event.labels == PLAN
    assert runner.state.approvals["call-1"].labels == PLAN
    assert RunState.from_events(runner.state.events).approvals["call-1"].labels == PLAN


async def test_policy_without_labels_leaves_request_unlabelled():
    runner = make_runner(plan_agent(), approval_policy=ApprovalPolicy(["plan_step"]))
    resolve_on_request(runner.state, approved=True)
    await runner.run("Plan it")
    assert runner.state.approvals["call-1"].labels is None
