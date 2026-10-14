from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE = REPO_ROOT / "localml_scheduler" / "deployments" / "run_petfinder_aba_a100_phase.sh"
QWEN = REPO_ROOT / "localml_scheduler" / "deployments" / "run_qwen38_int8_aba_a100.sh"
PAIR = REPO_ROOT / "localml_scheduler" / "deployments" / "run_petfinder_aba_a100_pair.sh"


def test_phase_launcher_is_a_scheduler_only_ab_comparison() -> None:
    launcher = PHASE.read_text(encoding="utf-8")

    assert "original_mlevolve_baseline_source" in launcher
    assert 'phase_repo="$original_baseline_repo"' in launcher
    assert 'phase_repo="$current_repo"' in launcher
    assert 'cd "$phase_repo"' in launcher
    assert 'MLEVOLVE_CONFIG="$phase_repo/config/config.yaml"' in launcher
    assert 'CUDA_VISIBLE_DEVICES=0' in launcher
    assert 'agent.seed=42' in launcher
    assert 'agent.initial_drafts=3' in launcher
    assert 'agent.use_stepwise_generation=false' in launcher
    assert 'vllm_client.default_completion_tokens=8192' in launcher
    assert 'experiment.mode=baseline' in launcher
    assert 'hardware_knowledge.enabled=false' in launcher
    assert 'preflight.enabled=false' in launcher
    assert 'context_cache.enabled=false' in launcher
    assert 'lesson_profiles.read_enabled=false' in launcher
    assert 'lesson_profiles.write_enabled=false' in launcher
    assert 'scheduler.enabled=false' in launcher
    assert 'scheduler.enabled=true' in launcher
    assert 'scheduler.settings.prediction.mode=ml_predictor' in launcher
    assert 'predictor_v3.2_transfer_A100.pt' in launcher
    assert 'parallel_job_cap' not in launcher


def test_original_baseline_does_not_receive_current_only_config_keys() -> None:
    launcher = PHASE.read_text(encoding="utf-8")

    assert "baseline_args=(" in launcher
    assert "scheduler_args=(" in launcher
    baseline_section = launcher.split("baseline_args=(", 1)[1].split(")", 1)[0]
    scheduler_section = launcher.split("scheduler_args=(", 1)[1].split(")", 1)[0]

    for current_only_key in (
        "experiment.mode",
        "context_cache.enabled",
        "hardware_knowledge.enabled",
        "preflight.enabled",
        "lesson_profiles.read_enabled",
        "agent.pipeline_decision_enabled",
        "agent.review.enabled",
        "agent.cuda_docs.enabled",
    ):
        assert current_only_key not in baseline_section
        assert current_only_key in scheduler_section


def test_experiment_gpu_must_support_multiple_cuda_processes() -> None:
    launcher = PHASE.read_text(encoding="utf-8")

    assert "compute_mode" in launcher
    assert "Exclusive_Process" in launcher


def test_qwen_and_experiment_use_different_a100s() -> None:
    qwen = QWEN.read_text(encoding="utf-8")
    phase = PHASE.read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES=1' in qwen
    assert 'CUDA_VISIBLE_DEVICES=0' in phase
    assert 'Qwen3.8-27B-INT8-W8A16-MTP' in qwen
    assert '--language-model-only' in qwen
    assert '--enable-prefix-caching' in qwen


def test_pair_runs_baseline_before_scheduler() -> None:
    launcher = PAIR.read_text(encoding="utf-8")

    assert 'for phase in baseline scheduler' in launcher
    assert 'petfinder_aba_a100_current_pair' in launcher
