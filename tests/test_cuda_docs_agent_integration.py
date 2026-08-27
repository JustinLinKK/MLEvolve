from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import json

from agents.cuda_docs_context import get_cuda_docs_context
from agents.stage_repair import _build_repair_prompt
from engine.agent_search import AgentSearch
from localml_scheduler.cuda_docs.models import CudaDocsContext

ROOT = Path(__file__).resolve().parents[1]


def test_scheduler_ranking_admission_and_execution_have_no_cuda_docs_dependency() -> (
    None
):
    guarded_roots = (
        ROOT / "localml_scheduler" / "scheduler",
        ROOT / "localml_scheduler" / "execution",
    )
    forbidden = ("cuda_docs", "CudaDocsMCPClient", "search_cuda_docs")
    violations = []
    for guarded_root in guarded_roots:
        for path in guarded_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_llm_generation_layer_has_no_general_mcp_tool_loop() -> None:
    violations = []
    for path in (ROOT / "llm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "search_cuda_docs" in text or "CudaDocsMCPClient" in text:
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def _bare_agent(mode: str, *, enabled=True, rollout_mode="debug_live") -> AgentSearch:
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = SimpleNamespace(experiment=SimpleNamespace(mode=mode))
    agent.acfg = SimpleNamespace(
        hardware_context_enabled=True,
        cuda_docs=SimpleNamespace(enabled=enabled, rollout_mode=rollout_mode),
    )
    agent.cuda_docs_service = None
    return agent


def test_origin_baseline_disabled_and_off_modes_construct_no_service() -> None:
    scheduler = SimpleNamespace()
    for mode in ("origin", "baseline"):
        agent = _bare_agent(mode)
        status = agent._attach_cuda_docs(scheduler)
        assert status == {"ok": False, "reason": "experiment mode is network-free"}
        assert agent.cuda_docs_service is None
    disabled = _bare_agent("hardware_aware", enabled=False)
    assert disabled._attach_cuda_docs(scheduler)["reason"] == "cuda docs disabled"
    off = _bare_agent("hardware_aware", rollout_mode="off")
    assert off._attach_cuda_docs(scheduler)["reason"] == "cuda docs rollout is off"
    assert off.cuda_docs_service is None


class _RoleService:
    def __init__(self):
        self.calls = []

    def get_run_backend_brief(self, *, role):
        self.calls.append(("brief", role, {}))
        return CudaDocsContext.unavailable(reason="cached_miss", applicable=True)

    def get_context(self, *, role, **kwargs):
        self.calls.append(("context", role, kwargs))
        return CudaDocsContext.unavailable(reason="cached_miss", applicable=True)


def test_agent_role_adapter_never_sends_source_code_and_leaves_other_roles_local() -> (
    None
):
    service = _RoleService()
    agent = SimpleNamespace(cuda_docs_service=service)
    get_cuda_docs_context(agent, "draft")
    get_cuda_docs_context(
        agent,
        "debug",
        parent_node=SimpleNamespace(term_out="CUDA out of memory"),
    )
    get_cuda_docs_context(
        agent,
        "improve",
        hardware_context=SimpleNamespace(
            compact_context={"profile": {"risk": "gpu_memory_pressure"}}
        ),
    )
    proprietary = "secret_model = torch.cuda.FloatTensor([1])"
    get_cuda_docs_context(agent, "code_review", code=proprietary)
    evolution = get_cuda_docs_context(agent, "evolution")

    assert [call[1] for call in service.calls] == [
        "draft",
        "debug",
        "improve",
        "code_review",
    ]
    serialized_calls = json.dumps(service.calls)
    assert proprietary not in serialized_calls
    assert "secret_model" not in serialized_calls
    assert service.calls[-1][2]["topic"].startswith("verify CUDA API")
    assert evolution.reason == "role_uses_existing_local_context"


def test_agent_prompt_assembly_orders_hardware_docs_pipeline_then_instructions() -> (
    None
):
    from agents import debug_agent, draft_agent, improve_agent

    for module in (draft_agent, improve_agent):
        source = inspect.getsource(module.run)
        assembly = source[source.find("user_prompt =") :]
        hardware = assembly.find("hardware_section")
        cuda_docs = assembly.find("cuda_docs_section")
        pipeline = assembly.find("pipeline_decision_section")
        instructions = assembly.find("instructions")
        assert -1 not in {hardware, cuda_docs, pipeline, instructions}
        assert hardware < cuda_docs < pipeline < instructions

    debug_source = inspect.getsource(debug_agent.run)
    assert debug_source.find("cuda_docs_section") < debug_source.find(
        "repair_selected_stages"
    )
    assert "pipeline_decision = build_pipeline_decision" in debug_source


def test_selective_repair_receives_bounded_evidence_without_journal_mutation() -> None:
    agent = SimpleNamespace(
        cfg=SimpleNamespace(experiment=SimpleNamespace(mode="baseline")),
        acfg=SimpleNamespace(review=SimpleNamespace(), code=SimpleNamespace(temp=0.0)),
        task_desc="task",
    )
    node = SimpleNamespace(parent=None, pipeline_decision={}, stage_note_board=[])
    issue = SimpleNamespace(
        to_dict=lambda: {"owner": "model_design", "severity": "critical"}
    )
    evidence = (
        "CUDA Documentation Evidence\nSource: docs — https://docs.nvidia.com/cuda/"
    )
    prompt = _build_repair_prompt(
        agent,
        node,
        "print('ok')",
        "model_design",
        [issue],
        cuda_docs_evidence=evidence,
    )
    assert '"cuda_documentation_evidence"' in prompt
    assert "CUDA Documentation Evidence" in prompt
    assert "https://docs.nvidia.com/cuda/" in prompt
    assert not hasattr(node, "cuda_docs_response")
