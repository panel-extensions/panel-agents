"""The 1.x public API; changing these lists is a breaking change or a new feature."""

import importlib

import pytest

PUBLIC = {
    "panel_agents": [
        "Action", "ActionRegistry", "AgentChat", "AgentDrawer", "ApprovalCard", "ApprovalPolicy", "ApprovalQueue",
        "ApprovalRequest", "ApprovalRule", "Artifact", "ArtifactPane", "ArtifactStore", "ArtifactVersion", "ArtifactView",
        "AuditLog", "HistoryRecorder", "MessageState", "ModelCallCard", "ModelCallState", "Recorder", "RunInfo", "RunState",
        "RunTrace", "ToolCallCard", "ToolCallState", "current_run_id", "current_tool_call_id", "dump_events", "events",
        "load_events", "load_history", "load_state", "register_diff", "register_renderer",
    ],
    "panel_agents.actions": ["Action", "ActionRegistry"],
    "panel_agents.artifacts": [
        "Artifact", "ArtifactStore", "ArtifactVersion", "Change", "Diff", "StructuredDiff", "TableDiff", "TextDiff",
        "ValueDiff", "diff", "diff_structured", "diff_table", "diff_text", "diff_values", "infer_kind", "register_diff",
        "root_run_id",
    ],
    "panel_agents.audit": ["AuditLog"],
    "panel_agents.context": ["current_run_id", "current_tool_call_id"],
    "panel_agents.components": [
        "AgentChat", "AgentDrawer", "ApprovalCard", "ApprovalQueue", "ArtifactPane", "ArtifactView", "ModelCallCard",
        "RunTrace", "ToolCallCard", "approval_factory", "register_renderer", "render",
    ],
    "panel_agents.events": [
        "ApprovalRequested", "ApprovalResolved", "ArtifactApplied", "ArtifactPublished", "Custom", "Event",
        "GuardrailFired", "ModelRequestStarted", "RunError", "RunFinished", "RunStarted", "StateDelta", "StateSnapshot", "StepFinished",
        "StepStarted", "TextMessageContent", "TextMessageEnd", "TextMessageStart", "ThinkingContent", "ThinkingEnd",
        "ThinkingStart", "ToolCallArgs", "ToolCallEnd", "ToolCallResult", "ToolCallStart", "Usage",
    ],
    "panel_agents.policy": ["ApprovalPolicy", "ApprovalRule"],
    "panel_agents.security": ["inert_markdown", "set_trusted_image_origins", "trusted_image_origins", "untrusted_markdown"],
    "panel_agents.recording": [
        "Codec", "FORMAT_VERSION", "HistoryRecorder", "Recorder", "dump_events", "load_events", "load_history", "load_state",
        "register_codec",
    ],
    "panel_agents.state": ["ApprovalRequest", "MessageState", "ModelCallState", "RunInfo", "RunState", "ToolCallState"],
    "panel_agents.adapters.pydantic_ai": [
        "ActionToolset", "AgentRunner", "Cassette", "CassetteError", "Divergence", "ModelRequestSummary",
        "ModelTraceRecorder", "RecordedFailure", "artifact_return", "fingerprint", "load_messages", "load_model_calls",
        "merge_model_calls", "publish_artifact",
    ],
    "panel_agents.adapters.cassette": ["Cassette", "CassetteError", "Divergence", "RecordedFailure"],
    "panel_agents.adapters.model_trace": [
        "FORMAT_VERSION", "ModelRequestSummary", "ModelTraceRecorder", "fingerprint", "load_messages", "load_model_calls",
        "merge_model_calls",
    ],
    "panel_agents.mcp": ["create_server"],
}

OPTIONAL = {
    "panel_agents.adapters.pydantic_ai": "pydantic_ai", "panel_agents.adapters.model_trace": "pydantic_ai",
    "panel_agents.adapters.cassette": "pydantic_ai", "panel_agents.mcp": "fastmcp",
}


@pytest.mark.parametrize("module", list(PUBLIC))
def test_public_names(module):
    if module in OPTIONAL:
        pytest.importorskip(OPTIONAL[module])
    mod = importlib.import_module(module)
    assert sorted(mod.__all__) == sorted(PUBLIC[module])
    for name in mod.__all__:
        assert hasattr(mod, name), name


def test_core_does_not_import_agent_frameworks():
    import subprocess
    import sys

    code = "import sys, panel_agents; print(any(m.split('.')[0] in ('pydantic_ai', 'fastmcp') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_version_numbers_are_one_based():
    from panel_agents import RunState
    from panel_agents import events as ev

    public = [p for p in __import__("panel_agents").Artifact.param if not p.startswith("_")]
    assert sorted(public) == ["applied", "artifact_id", "current", "kind", "name", "redo_run_version", "redo_version", "title", "undo_version", "versions"]
    state = RunState(flush_interval=0)
    for i in range(2):
        state.apply(ev.ArtifactPublished(artifact_id="doc", value=i, applied=True))
    artifact = state.artifacts["doc"]
    assert [v.version for v in artifact.versions] == [1, 2]
    assert artifact.applied_version.version == 2
    state.apply_artifact_version("doc", 1)
    assert artifact.applied_version.version == 1
    assert state.events[-1].version == 1
