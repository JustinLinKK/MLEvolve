from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import types

import pytest
import yaml

from agents.review_contracts import ReviewIssue, StageRepairResult
from config import PreflightConfig
from engine.agent_search import AgentSearch
from engine.preflight import (
    ModelPreflightGate,
    PreflightOutcome,
    admission_for_status,
    candidate_code_hash,
    derive_batch_scenarios,
    diagnostic_owner,
    diagnostic_to_review_issue,
    inspect_adapter,
    is_fresh_preflight,
    normalize_preflight_precision,
    preflight_diagnostics_require_rejection,
    select_target_profile,
)
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from engine.search_node import Journal
from utils.metric import MetricValue


def _cfg(tmp_path: Path, **overrides):
    settings = PreflightConfig(target_profile="nvidia/a10_24gb", **overrides)
    return SimpleNamespace(
        workspace_dir=tmp_path,
        exp_id="preflight-test",
        exp_name="preflight-test",
        experiment=SimpleNamespace(mode="hardware_aware"),
        preflight=settings,
        scheduler=SimpleNamespace(settings=None),
    )


def _node(code: str, *, node_id: str = "candidate-1") -> SearchNode:
    return SearchNode(code=code, plan="test", stage="draft", id=node_id)


def test_preflight_configuration_defaults():
    settings = PreflightConfig()
    assert settings.enabled is True
    assert settings.enabled_modes == ["hardware_aware"]
    assert settings.policy_mode == "balanced"
    assert settings.target_profile == "auto"
    assert settings.require_adapter_for_generated is True
    assert settings.max_repair_rounds == 1
    assert settings.fail_open_on_internal_error is True
    assert settings.abstract_timeout_seconds == 30
    assert settings.cpu_timeout_seconds == 90
    assert settings.maximum_cpu_memory_mb == 8192
    assert settings.maximum_processes == 32
    assert settings.maximum_output_bytes == 1_000_000
    assert settings.disable_network is True
    assert settings.allow_real_cpu_abstract_fallback is False


@pytest.mark.parametrize(
    ("name", "vram_mb", "expected"),
    [
        ("Tesla V100-SXM2-16GB", 16_384, "nvidia/v100_16gb"),
        ("Tesla V100-SXM2-32GB", 32_768, "nvidia/v100_32gb"),
        ("NVIDIA A10", 24_576, "nvidia/a10_24gb"),
        ("NVIDIA A100-SXM4-40GB", 40_960, "nvidia/a100_40gb"),
    ],
)
def test_automatic_bundled_profile_selection(name, vram_mb, expected):
    hardware = SimpleNamespace(gpu_name=name, total_vram_mb=vram_mb)
    assert (
        select_target_profile("auto", detected_hardware=hardware).manifest_profile
        == expected
    )


def test_project_owned_profile_selection_and_values():
    a100 = select_target_profile(
        "auto",
        detected_hardware=SimpleNamespace(
            gpu_name="NVIDIA A100 80GB PCIe", total_vram_mb=81_920
        ),
    )
    rtx = select_target_profile(
        "auto",
        detected_hardware=SimpleNamespace(
            gpu_name="NVIDIA GeForce RTX 5090", total_vram_mb=32_607
        ),
    )
    a100_value = yaml.safe_load(Path(a100.manifest_profile).read_text(encoding="utf-8"))
    rtx_value = yaml.safe_load(Path(rtx.manifest_profile).read_text(encoding="utf-8"))
    assert a100_value["vram_bytes"] == 80 * 1024**3
    assert rtx_value["architecture"] == "blackwell"
    assert {"fp8", "nvfp4"}.issubset(rtx_value["native_training_dtypes"])


def test_unknown_gpu_skips_hardware_checks_with_warning():
    selection = select_target_profile(
        "auto",
        detected_hardware=SimpleNamespace(gpu_name="Future GPU", total_vram_mb=12_345),
    )
    assert selection.hardware_checks_enabled is False
    assert "skipped" in selection.warning


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"precision_mode": "bf16"}, "bf16"),
        ({"precision_mode": "mxfp8_te"}, "fp8"),
        ({"precision_mode": "nvfp4_te"}, "nvfp4"),
        ({"uses_amp": True}, "fp16"),
        ({}, "fp32"),
    ],
)
def test_precision_normalization(metadata, expected):
    assert normalize_preflight_precision(metadata) == expected


def test_adapter_and_import_guard_detection():
    code = """
class CandidateAdapter:
    def build_model(self, context): pass
    def build_optimizer(self, model, context): pass
    def build_train_batch(self, scenario, device): pass
    def build_validation_batch(self, scenario, device): pass
    def training_step(self, model, batch, context): pass
    def validation_step(self, model, batch, context): pass

if __name__ == "__main__":
    main()
"""
    inspection = inspect_adapter(code)
    assert inspection.entrypoint_present
    assert inspection.complete
    assert inspection.main_guard_present
    assert inspection.unsafe_top_level_lines == ()


def test_import_safety_allows_pure_len_in_configuration_mapping():
    code = """
FEATURE_COLS = ["a", "b"]
ADAPTER_DEFAULTS = {"n_features": len(FEATURE_COLS)}

class CandidateAdapter:
    def build_model(self, context): pass
    def build_optimizer(self, model, context): pass
    def build_train_batch(self, scenario, device): pass
    def build_validation_batch(self, scenario, device): pass
    def training_step(self, model, batch, context): pass
    def validation_step(self, model, batch, context): pass

if __name__ == "__main__":
    pass
"""
    inspection = inspect_adapter(code)
    assert inspection.unsafe_top_level_lines == ()


def test_import_safety_detects_execution_outside_main_guard():
    inspection = inspect_adapter(
        "def main():\n    pass\nmain()\nif __name__ == '__main__':\n    main()\n"
    )
    assert inspection.main_guard_present
    assert inspection.unsafe_top_level_lines == (3,)


def test_import_safety_allows_lightweight_configuration_assignments():
    code = """
import os
import torch
from pathlib import Path

INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
OUTPUT_DIR = Path(os.path.join(".", "submission"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CandidateAdapter:
    def build_model(self, context): pass
    def build_optimizer(self, model, context): pass
    def build_train_batch(self, scenario, device): pass
    def build_validation_batch(self, scenario, device): pass
    def training_step(self, model, batch, context): pass
    def validation_step(self, model, batch, context): pass

if __name__ == "__main__":
    main()
"""
    inspection = inspect_adapter(code)
    assert inspection.main_guard_present
    assert inspection.unsafe_top_level_lines == ()


def test_import_safety_still_rejects_side_effecting_assignment_calls():
    code = """
RESULT = train_model()
if __name__ == "__main__":
    main()
"""
    inspection = inspect_adapter(code)
    assert inspection.unsafe_top_level_lines == (2,)


def test_import_safety_guidance_inlines_a_top_level_device_helper(tmp_path):
    code = """
import torch

def _resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = _resolve_device()

class CandidateAdapter:
    def build_model(self, context): pass
    def build_optimizer(self, model, context): pass
    def build_train_batch(self, scenario, device): pass
    def build_validation_batch(self, scenario, device): pass
    def training_step(self, model, batch, context): pass
    def validation_step(self, model, batch, context): pass

if __name__ == "__main__":
    main()
"""
    inspection = inspect_adapter(code)
    issues = ModelPreflightGate(_cfg(tmp_path))._contract_issues(inspection, code)
    assert len(issues) == 1
    assert 'DEVICE = torch.device("cuda"' in issues[0].repair_instruction
    assert "Do not add another main guard" in issues[0].repair_instruction


def test_import_safety_guidance_does_not_wrap_a_top_level_side_effect(tmp_path):
    code = """
def configure_precision():
    pass

configure_precision()

class CandidateAdapter:
    def build_model(self, context): pass
    def build_optimizer(self, model, context): pass
    def build_train_batch(self, scenario, device): pass
    def build_validation_batch(self, scenario, device): pass
    def training_step(self, model, batch, context): pass
    def validation_step(self, model, batch, context): pass

if __name__ == "__main__":
    main()
"""
    inspection = inspect_adapter(code)
    issues = ModelPreflightGate(_cfg(tmp_path))._contract_issues(inspection, code)
    assert len(issues) == 1
    assert "Calling the wrapper at module scope remains unsafe" in issues[0].repair_instruction
    assert "existing main guard" in issues[0].repair_instruction


def test_preflight_rejects_pandas_row_values_without_an_explicit_numeric_dtype(tmp_path):
    code = """
import torch

class CandidateAdapter:
    def build_model(self, context): pass
    def build_optimizer(self, model, context): pass
    def build_train_batch(self, scenario, device): pass
    def build_validation_batch(self, scenario, device): pass
    def training_step(self, model, batch, context): pass
    def validation_step(self, model, batch, context): pass

def encode_row(row, metadata_cols):
    return torch.tensor(row[metadata_cols].values, dtype=torch.float32)

if __name__ == "__main__":
    pass
"""
    inspection = inspect_adapter(code)
    issues = ModelPreflightGate(_cfg(tmp_path))._contract_issues(inspection, code)

    issue = next(issue for issue in issues if issue.category == "preflight_pandas_row_tensor")
    assert "to_numpy(dtype=np.float32)" in issue.repair_instruction
    assert "object dtype" in issue.evidence


def test_batch_scenarios_match_scheduler_offsets(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.scheduler.settings = SimpleNamespace(
        gpu_scheduler=SimpleNamespace(
            batch_options=SimpleNamespace(exponent_offsets=[-1, 0, 1]),
            submission_defaults=SimpleNamespace(batch_probe_max_batch_size=64),
        )
    )
    assert derive_batch_scenarios("BATCH_SIZE = 48", cfg) == [16, 32, 64]


def test_petfinder_manifest_exposes_the_real_multimodal_preflight_fixture(tmp_path):
    """Prevent image-plus-tabular candidates from receiving a vector-only fixture."""

    cfg = _cfg(tmp_path)
    cfg.exp_id = "petfinder-pawpularity-score"
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()

    manifest = ModelPreflightGate(cfg)._manifest(
        _node("BATCH_SIZE = 8"),
        candidate_dir,
        "nvidia/a10_24gb",
        "BATCH_SIZE = 8",
    )

    assert manifest["task"] == {
        "name": "petfinder-pawpularity-score",
        "input_rank": 3,
        "target_dtype": "float32",
    }
    assert manifest["scenarios"]["input_shapes"] == {"image": [3, 256, 256]}
    assert manifest["scenarios"]["fixture"] == {
        "image": [3, 256, 256],
        "tabular": [12],
    }


@pytest.mark.parametrize(
    ("code", "stage", "owner"),
    [
        ("SRC002", "static_source", "integration"),
        ("SHP003", "abstract_forward", "model_design"),
        ("GPU001", "hardware", "datatype_precision"),
        ("AUT001", "cpu_training", "training_evaluation"),
        ("MEM001", "memory", "training_evaluation"),
    ],
)
def test_diagnostic_ownership(code, stage, owner):
    assert diagnostic_owner(code, stage) == owner


def test_construction_key_error_repair_guidance_merges_partial_context():
    issue = diagnostic_to_review_issue(
        {
            "classification": "confirmed_candidate_failure",
            "code": "CON001",
            "stage": "construction",
            "exception_type": "KeyError",
            "message": "construction raised KeyError: 'precision'",
        }
    )
    assert issue is not None
    assert "partial mapping" in issue.repair_instruction
    assert "merge" in issue.repair_instruction


def test_offline_weight_error_repair_guidance_preserves_real_model():
    issue = diagnostic_to_review_issue(
        {
            "classification": "confirmed_candidate_failure",
            "code": "CON001",
            "stage": "construction",
            "exception_type": "LocalEntryNotFoundError",
            "message": "pretrained checkpoint is absent while network is disabled",
        }
    )
    assert issue is not None
    assert "pretrained=False" in issue.repair_instruction
    assert "same real model family" in issue.repair_instruction


def test_uncached_pretrained_dependency_rejects_inconclusive_preflight():
    diagnostics = [
        {
            "classification": "inconclusive",
            "code": "FIX001",
            "stage": "data_contract",
            "exception_type": "OSError",
            "message": (
                "fixture raised OSError: We couldn't connect to 'https://huggingface.co' "
                "to load the files, and couldn't find them in the cached files."
            ),
        }
    ]
    assert preflight_diagnostics_require_rejection(diagnostics)
    issue = diagnostic_to_review_issue(diagnostics[0])
    assert issue is not None
    assert "same real model family" in issue.repair_instruction


def test_source_pretrained_dependency_issues_rejects_adapter_mock_bypass(tmp_path):
    """A mock-only CandidateAdapter cannot hide a real uncached model download."""

    import engine.preflight as preflight

    code = '''
from transformers import AutoModel

def run_training():
    return AutoModel.from_pretrained("google/siglip2-so400m-patch16-256")

class CandidateAdapter:
    def build_model(self, context=None):
        return object()  # A preflight-only mock, unlike run_training().
'''
    issues = preflight.source_pretrained_dependency_issues(
        code, cache_root=tmp_path / "hub"
    )
    assert len(issues) == 1
    assert "google/siglip2-so400m-patch16-256" in issues[0].evidence


def test_source_pretrained_dependency_issues_resolves_module_string_constant(tmp_path):
    """A named model identifier must not bypass offline-weight admission."""

    import engine.preflight as preflight

    code = '''
from transformers import AutoModel

SIGLIP_MODEL_ID = "google/siglip2-so400m-patch16-256"

def run_training():
    return AutoModel.from_pretrained(SIGLIP_MODEL_ID)
'''
    issues = preflight.source_pretrained_dependency_issues(
        code, cache_root=tmp_path / "hub"
    )
    assert len(issues) == 1
    assert "google/siglip2-so400m-patch16-256" in issues[0].evidence


def test_none_criterion_repair_guidance_does_not_assume_context_persistence():
    issue = diagnostic_to_review_issue(
        {
            "classification": "confirmed_candidate_failure",
            "code": "AUT002",
            "stage": "cpu_training",
            "exception_type": "TypeError",
            "message": "training raised TypeError: 'NoneType' object is not callable",
            "stack_trace": 'loss = ctx["criterion"](preds, batch["targets"])',
        }
    )
    assert issue is not None
    assert "context mutations do not persist" in issue.repair_instruction
    assert "real criterion" in issue.repair_instruction


def test_read_only_shuffle_repair_guidance_copies_the_mutated_array():
    issue = diagnostic_to_review_issue(
        {
            "classification": "confirmed_candidate_failure",
            "code": "CON001",
            "stage": "construction",
            "exception_type": "ValueError",
            "message": "construction raised ValueError: array is read-only",
            "stack_trace": (
                "File \"candidate.py\", line 285, in make_split\n"
                "    rng.shuffle(ids)\n"
                "ValueError: array is read-only"
            ),
        }
    )
    assert issue is not None
    assert "rng.shuffle(ids)" in issue.repair_instruction
    assert "to_numpy(copy=True)" in issue.repair_instruction
    assert "DataFrame copy" in issue.repair_instruction


def test_balanced_admission_policy():
    assert admission_for_status("PASS", "balanced", True)
    assert admission_for_status("INCONCLUSIVE", "balanced", True)
    assert not admission_for_status("FAIL", "balanced", True)
    assert admission_for_status("INTERNAL_ERROR", "balanced", True)
    assert not admission_for_status("INTERNAL_ERROR", "balanced", False)
    assert not admission_for_status("FAIL", "audit", True)


def test_legacy_fallback_persists_schema_valid_report(tmp_path):
    node = _node("MODEL_FAMILY = 'legacy'\nBATCH_SIZE = 4\n")
    outcome = ModelPreflightGate(_cfg(tmp_path)).run(node, generated=False)
    assert outcome.status == "PASS"
    assert outcome.mode == "static_hardware_fallback"
    assert outcome.admitted
    assert Path(outcome.report_path).is_file()
    assert Path(outcome.summary_path).is_file()
    report = __import__("json").loads(
        Path(outcome.report_path).read_text(encoding="utf-8")
    )
    assert [stage["name"] for stage in report["stages"]] == [
        "static_source",
        "hardware",
    ]


def test_unsupported_precision_blocks_legacy_candidate(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.preflight.target_profile = "nvidia/v100_16gb"
    node = _node("PRECISION = 'bf16'\nBATCH_SIZE = 4\n")
    outcome = ModelPreflightGate(cfg).run(node, generated=False)
    assert outcome.status == "FAIL"
    assert not outcome.admitted
    assert "GPU001" in outcome.diagnostic_codes
    assert outcome.issues[0].owner == "datatype_precision"


def test_syntax_failure_blocks_candidate(tmp_path):
    outcome = ModelPreflightGate(_cfg(tmp_path)).run(
        _node("def broken(:\n    pass\n"), generated=False
    )
    assert outcome.status == "FAIL"
    assert not outcome.admitted
    assert "SRC001" in outcome.diagnostic_codes
    assert outcome.issues[0].owner == "integration"


def test_generated_candidate_missing_adapter_is_integration_failure(tmp_path):
    node = _node("if __name__ == '__main__':\n    print('train')\n")
    outcome = ModelPreflightGate(_cfg(tmp_path)).run(node, generated=True)
    assert outcome.status == "FAIL"
    assert not outcome.admitted
    assert "MLE_ADAPTER001" in outcome.diagnostic_codes
    assert {issue.owner for issue in outcome.issues} == {"integration"}


def test_full_cpu_candidate_passes_and_persists_all_stages(tmp_path):
    code = """
import torch
BATCH_SIZE = 2
MODEL_FAMILY = "linear_test"

class CandidateAdapter:
    def build_model(self, context):
        return torch.nn.Linear(4, 2, device=context["device"])
    def build_optimizer(self, model, context):
        return torch.optim.SGD(model.parameters(), lr=0.01)
    def build_train_batch(self, scenario, device):
        size = scenario["batch_size"]
        return torch.ones(size, 4, device=device), torch.zeros(size, dtype=torch.long, device=device)
    def build_validation_batch(self, scenario, device):
        return self.build_train_batch(scenario, device)
    def training_step(self, model, batch, context):
        inputs, targets = batch
        return torch.nn.functional.cross_entropy(model(inputs), targets)
    def validation_step(self, model, batch, context):
        inputs, targets = batch
        return torch.nn.functional.cross_entropy(model(inputs), targets)

def main():
    pass

if __name__ == "__main__":
    main()
"""
    outcome = ModelPreflightGate(_cfg(tmp_path)).run(_node(code), generated=True)
    assert outcome.status == "PASS"
    assert outcome.admitted
    report = __import__("json").loads(
        Path(outcome.report_path).read_text(encoding="utf-8")
    )
    assert {stage["name"] for stage in report["stages"]} == {
        "static_source",
        "hardware",
        "construction",
        "data_contract",
        "abstract_forward",
        "cpu_training",
        "validation",
        "memory",
    }


def test_detached_loss_is_confirmed_and_blocked(tmp_path):
    code = """
import torch
BATCH_SIZE = 2

class CandidateAdapter:
    def build_model(self, context):
        return torch.nn.Linear(4, 2, device=context["device"])
    def build_optimizer(self, model, context):
        return torch.optim.SGD(model.parameters(), lr=0.01)
    def build_train_batch(self, scenario, device):
        size = scenario["batch_size"]
        return torch.ones(size, 4, device=device), torch.zeros(size, dtype=torch.long, device=device)
    def build_validation_batch(self, scenario, device):
        return self.build_train_batch(scenario, device)
    def training_step(self, model, batch, context):
        model(batch[0])
        return torch.tensor(1.0)
    def validation_step(self, model, batch, context):
        return model(batch[0]).sum()

if __name__ == "__main__":
    pass
"""
    cfg = _cfg(tmp_path)
    cfg.scheduler.settings = SimpleNamespace(
        gpu_scheduler=SimpleNamespace(
            batch_options=SimpleNamespace(exponent_offsets=[0]),
            submission_defaults=SimpleNamespace(batch_probe_max_batch_size=None),
        )
    )
    outcome = ModelPreflightGate(cfg).run(_node(code), generated=True)
    assert outcome.status == "FAIL"
    assert not outcome.admitted
    assert "AUT001" in outcome.diagnostic_codes
    assert any(issue.owner == "training_evaluation" for issue in outcome.issues)


def test_checker_crash_fails_open_without_candidate_issue(tmp_path, monkeypatch):
    import model_preflight

    monkeypatch.setattr(
        model_preflight,
        "check",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    outcome = ModelPreflightGate(_cfg(tmp_path)).run(
        _node("MODEL_FAMILY = 'legacy'\n"), generated=False
    )
    assert outcome.status == "INTERNAL_ERROR"
    assert outcome.admitted
    assert outcome.issues == []
    assert outcome.internal_error == "RuntimeError: boom"


def test_stale_hash_detection():
    node = _node("x = 1\n")
    node.preflight_code_hash = candidate_code_hash(node.code)
    assert is_fresh_preflight(node)
    node.code = "x = 2\n"
    assert not is_fresh_preflight(node)


def test_successful_targeted_repair_is_rechecked(tmp_path, monkeypatch):
    issue = ReviewIssue(
        source="model_preflight",
        severity="critical",
        category="preflight_adapter_contract",
        owner="integration",
        evidence="missing adapter",
        repair_instruction="add adapter",
    )
    calls: list[str] = []

    def fake_run(_gate, node, *, generated, attempt=0):
        calls.append(node.code)
        if node.code == "broken":
            return PreflightOutcome(
                "FAIL",
                "static_hardware_fallback",
                candidate_code_hash(node.code),
                False,
                False,
                ["MLE_ADAPTER001"],
                issues=[issue],
            )
        return PreflightOutcome(
            "PASS", "full_cpu", candidate_code_hash(node.code), True, False
        )

    monkeypatch.setattr(ModelPreflightGate, "run", fake_run)
    monkeypatch.setattr(
        "agents.stage_repair.repair_selected_stages",
        lambda *args, **kwargs: (
            "fixed",
            [StageRepairResult(stage="integration", applied=True, patch_count=1)],
            {"stage_repair_calls": 1},
        ),
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = _cfg(tmp_path)
    agent.acfg = SimpleNamespace(
        review=SimpleNamespace(parallel_training_repairs=False, repair_retries=1)
    )
    agent.pipeline_logger = None
    agent.task_desc = "test"
    agent.refresh_hardware_context = lambda node: None
    node = _node("broken")
    node.review_status = "approved"

    assert AgentSearch._run_node_preflight(agent, node, generated=True)
    assert calls == ["broken", "fixed"]
    assert node.preflight_status == "PASS"
    assert node.preflight_repair_count == 1
    assert is_fresh_preflight(node)
    assert not any(
        item.get("source") == "model_preflight"
        for item in node.review_issues
    )


def test_preflight_recheck_replaces_only_previous_preflight_issues(
    tmp_path, monkeypatch
):
    stale_issue = ReviewIssue(
        source="model_preflight",
        severity="critical",
        category="preflight_con001",
        owner="model_design",
        evidence="old construction failure",
        repair_instruction="repair old construction failure",
    )
    current_issue = ReviewIssue(
        source="model_preflight",
        severity="critical",
        category="preflight_aut001",
        owner="training_evaluation",
        evidence="current training failure",
        repair_instruction="repair current training failure",
    )
    static_issue = ReviewIssue(
        source="static_review",
        severity="warning",
        category="training_loop",
        owner="training_evaluation",
        evidence="keep this independent warning",
        repair_instruction="optional cleanup",
    )
    outcomes = iter(
        [
            PreflightOutcome(
                "FAIL",
                "full_cpu",
                candidate_code_hash("broken"),
                False,
                False,
                ["CON001"],
                issues=[stale_issue],
            ),
            PreflightOutcome(
                "FAIL",
                "full_cpu",
                candidate_code_hash("fixed"),
                False,
                False,
                ["AUT001"],
                issues=[current_issue],
            ),
        ]
    )

    monkeypatch.setattr(
        ModelPreflightGate,
        "run",
        lambda *_args, **_kwargs: next(outcomes),
    )
    monkeypatch.setattr(
        "agents.stage_repair.repair_selected_stages",
        lambda *_args, **_kwargs: (
            "fixed",
            [StageRepairResult(stage="model_design", applied=True, patch_count=1)],
            {"stage_repair_calls": 1},
        ),
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = _cfg(tmp_path, max_repair_rounds=1)
    agent.acfg = SimpleNamespace(
        review=SimpleNamespace(parallel_training_repairs=False, repair_retries=1)
    )
    agent.pipeline_logger = None
    agent.task_desc = "test"
    agent.refresh_hardware_context = lambda node: None
    node = _node("broken")
    node.review_status = "approved"
    node.review_issues = [static_issue.to_dict()]

    assert not AgentSearch._run_node_preflight(agent, node, generated=True)
    assert node.review_issues == [static_issue.to_dict(), current_issue.to_dict()]


def test_preflight_uses_all_configured_repair_rounds(tmp_path, monkeypatch):
    """Catch stopping after one repair when the configured second repair can pass."""

    issue = ReviewIssue(
        source="model_preflight",
        severity="critical",
        category="preflight_import_error",
        owner="integration",
        evidence="import raised NameError",
        repair_instruction="repair the remaining import error",
    )
    gate_attempts: list[tuple[str, int]] = []

    def fake_run(_gate, node, *, generated, attempt=0):
        gate_attempts.append((node.code, attempt))
        if node.code != "fixed-2":
            return PreflightOutcome(
                "FAIL",
                "full_cpu",
                candidate_code_hash(node.code),
                False,
                False,
                ["SRC002"],
                issues=[issue],
            )
        return PreflightOutcome(
            "PASS", "full_cpu", candidate_code_hash(node.code), True, False
        )

    def fake_repair(_agent, _node, code, _issues):
        repair_number = int(code.rsplit("-", 1)[-1]) + 1
        return (
            f"fixed-{repair_number}",
            [StageRepairResult(stage="integration", applied=True, patch_count=1)],
            {"stage_repair_calls": 1},
        )

    monkeypatch.setattr(ModelPreflightGate, "run", fake_run)
    monkeypatch.setattr(
        "agents.stage_repair.repair_selected_stages",
        fake_repair,
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = _cfg(tmp_path, max_repair_rounds=2)
    agent.acfg = SimpleNamespace(
        review=SimpleNamespace(parallel_training_repairs=False, repair_retries=1)
    )
    agent.pipeline_logger = None
    agent.task_desc = "test"
    agent.refresh_hardware_context = lambda node: None
    node = _node("broken-0")
    node.review_status = "approved"

    assert AgentSearch._run_node_preflight(agent, node, generated=True)
    assert gate_attempts == [("broken-0", 0), ("fixed-1", 1), ("fixed-2", 2)]
    assert node.code == "fixed-2"
    assert node.preflight_repair_count == 2
    assert node.preflight_status == "PASS"


def test_failed_targeted_repair_is_rechecked_and_rejected(tmp_path, monkeypatch):
    issue = ReviewIssue(
        source="model_preflight",
        severity="critical",
        category="preflight_adapter_contract",
        owner="integration",
        evidence="missing adapter",
        repair_instruction="add adapter",
    )
    calls = 0

    def fake_run(_gate, node, *, generated, attempt=0):
        nonlocal calls
        calls += 1
        return PreflightOutcome(
            "FAIL",
            "static_hardware_fallback",
            candidate_code_hash(node.code),
            False,
            False,
            ["MLE_ADAPTER001"],
            issues=[issue],
        )

    monkeypatch.setattr(ModelPreflightGate, "run", fake_run)
    monkeypatch.setattr(
        "agents.stage_repair.repair_selected_stages",
        lambda *args, **kwargs: (
            "still broken",
            [
                StageRepairResult(
                    stage="integration", failure_reason="patch did not apply"
                )
            ],
            {"stage_repair_calls": 1},
        ),
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = _cfg(tmp_path)
    agent.acfg = SimpleNamespace(
        review=SimpleNamespace(parallel_training_repairs=False, repair_retries=1)
    )
    agent.pipeline_logger = None
    agent.task_desc = "test"
    agent.refresh_hardware_context = lambda node: None
    node = _node("still broken")
    node.review_status = "approved"

    assert not AgentSearch._run_node_preflight(agent, node, generated=True)
    assert calls == 2
    assert node.review_status == "rejected"
    assert node.preflight_repair_count == 1


def test_mixed_batch_submits_only_admitted_nodes(monkeypatch, tmp_path):
    from agents import result_parse_agent
    from engine import evaluation, execution, solution_manager

    rejected = _node("rejected", node_id="rejected")
    rejected.pending_execution = True
    rejected.review_status = "rejected"
    rejected.preflight_status = "FAIL"
    rejected.preflight_admitted = False

    admitted = _node("admitted", node_id="admitted")
    admitted.pending_execution = True
    admitted.review_status = "approved"
    admitted.preflight_status = "INCONCLUSIVE"
    admitted.preflight_admitted = True
    admitted.preflight_gpu_check_required = True
    admitted.preflight_code_hash = candidate_code_hash(admitted.code)

    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = _cfg(tmp_path)
    agent.pipeline_logger = None
    agent.journal = Journal()
    agent.journal_lock = threading.Lock()
    agent.current_step = 0
    agent.best_node = None
    agent._ensure_node_preflight_before_execution = types.MethodType(
        lambda self, node: bool(node.preflight_admitted), agent
    )
    agent._validate_node_precision_before_execution = types.MethodType(
        lambda self, node: True, agent
    )

    agent._finalize_review_rejected_node = types.MethodType(
        AgentSearch._finalize_review_rejected_node,
        agent,
    )
    monkeypatch.setattr(
        "agents.hardware_context.optimize_training_parameters_for_round",
        lambda *args: {},
    )

    def parse(_agent, *, node, exec_result):
        node.metric = MetricValue(0.5, maximize=True)
        node.is_buggy = False
        node.is_valid = True
        node.exec_time = exec_result.exec_time
        return node

    monkeypatch.setattr(result_parse_agent, "run", parse)
    monkeypatch.setattr(execution, "validate_executed_node", lambda *args: None)
    monkeypatch.setattr(evaluation, "check_improvement", lambda *args: False)
    monkeypatch.setattr(solution_manager, "update_best_solution", lambda *args: None)
    submitted: list[str] = []

    def execute_many(items):
        submitted.extend(item["id"] for item in items)
        return {
            item["id"]: ExecutionResult(["ok"], 0.01, None, {}, []) for item in items
        }

    results = AgentSearch.execute_deferred_nodes(
        agent, [rejected, admitted], execute_many
    )
    assert submitted == ["admitted"]
    assert {node.id for node in results} == {"rejected", "admitted"}
    assert [node.id for node in agent.journal.nodes] == ["admitted"]
