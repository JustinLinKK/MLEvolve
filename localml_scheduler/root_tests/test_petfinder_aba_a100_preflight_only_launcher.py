from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "localml_scheduler"
    / "deployments"
    / "run_petfinder_aba_a100_preflight_only.sh"
)


def test_launcher_is_strict_preflight_only_ablation() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'node_budget="${PREFLIGHT_NODE_BUDGET:-50}"' in launcher
    assert "agent.seed=42" in launcher
    assert "agent.initial_drafts=3" in launcher
    assert "agent.clean_task_description=true" in launcher
    assert (
        "agent.prompt_compatibility_profile=original_mlevolve_20260924"
        in launcher
    )
    assert "agent.use_stepwise_generation=false" in launcher
    assert "experiment.mode=hardware_aware" in launcher
    assert "preflight.enabled=true" in launcher
    assert "preflight.enabled_modes=[hardware_aware]" in launcher
    assert "preflight.target_profile=nvidia/a100_80gb" in launcher
    assert "preflight.prompt_contract_enabled=false" in launcher
    # This is a checker-only ablation against a baseline prompt that never
    # requested CandidateAdapter. Missing adapters must use the checker's
    # static fallback instead of triggering a generated-code rewrite.
    assert "preflight.require_adapter_for_generated=false" in launcher
    assert "scheduler.enabled=false" in launcher
    assert "hardware_knowledge.enabled=false" in launcher
    assert "agent.hardware_context_enabled=false" in launcher
    assert "agent.pipeline_decision_enabled=false" in launcher
    assert "agent.review.enabled=false" in launcher
    assert "agent.cuda_docs.enabled=false" in launcher
    assert "context_cache.enabled=false" in launcher
    assert "lesson_profiles.read_enabled=false" in launcher
    assert "lesson_profiles.write_enabled=false" in launcher
    assert "parallel_job_cap" not in launcher
    assert "prediction.mode" not in launcher


def test_launcher_separates_qwen_and_candidate_gpus() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES=0' in launcher
    assert 'nvidia-smi --id=0' in launcher
    assert 'nvidia-smi --id=1' in launcher
    assert 'curl --fail --silent --show-error --max-time 10' in launcher
    assert 'NVIDIA A100 80GB PCIe' in launcher
    assert 'Exclusive_Process' in launcher


def test_launcher_refuses_incorrect_prompt_surface_before_run() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "feature_prompt_contracts" in launcher
    assert "get_impl_guideline_from_agent" in launcher
    assert "CPU preflight adapter contract" in launcher
    assert "Backend design contract" in launcher
    assert "Scheduler Model Family Contract" in launcher
    assert 'required_absent = ["CPU preflight adapter contract", "CandidateAdapter"]' in launcher
    assert "prompt_contract_audit.json" in launcher
    assert launcher.index("prompt_contract_audit.json") < launcher.index('"$python_bin" -u run.py')


def test_launcher_persists_complete_runtime() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'cp -a "$volatile_root/." "$durable_root/"' in launcher
    assert 'cp -a "$runtime_root/pipeline" "$durable_root/"' in launcher
    assert 'printf \'%s\\n\' "$exit_status" > "$durable_root/exit_code"' in launcher
    assert "petfinder_aba_a100_preflight_only_current" in launcher
