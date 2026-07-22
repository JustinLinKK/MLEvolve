from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "compare_profile_scheduler.sh"
CONFIG = REPO_ROOT / "config.example.yaml"
PLOTTER = REPO_ROOT / "utils" / "plot_hardware_awareness_comparison.py"


def _run_dry_compare(run_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        str(SCRIPT),
        "unit-competition",
        "--config",
        str(CONFIG),
        "--dataset-root",
        str(run_root / "data"),
        "--run-root",
        str(run_root),
        "--skip-prepare",
        "--no-validation-server",
        "--skip-plots",
        "--dry-run",
    ]
    command.extend(args)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _command_text(run_root: Path, label: str) -> str:
    return (run_root / label / "command.txt").read_text(encoding="utf-8")


def test_profile_scheduler_dry_run_generates_matched_commands(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"

    result = _run_dry_compare(run_root)

    scheduler_off = _command_text(run_root, "scheduler_off")
    scheduler_on = _command_text(run_root, "scheduler_on")

    shared_fragments = [
        "config: " + str(CONFIG),
        "experiment.mode=hardware_aware",
        "hardware_knowledge.enabled=true",
        "hardware_knowledge.include_profile_evidence=true",
        "agent.steps=30",
        "agent.initial_drafts=3",
        "agent.seed=42",
        "agent.time_limit=10800",
        "exp_id=unit-competition",
        "MLEVOLVE_CONFIG=" + str(CONFIG),
    ]
    for fragment in shared_fragments:
        assert fragment in scheduler_off
        assert fragment in scheduler_on

    assert "scheduler.enabled=false" in scheduler_off
    assert "scheduler.runtime_root=" not in scheduler_off
    assert "MLEVOLVE_COMMAND_LABEL=scheduler_off" in scheduler_off

    assert "scheduler.enabled=true" in scheduler_on
    assert f"scheduler.runtime_root={run_root}/scheduler_on/scheduler_runtime" in scheduler_on
    assert "MLEVOLVE_COMMAND_LABEL=scheduler_on" in scheduler_on
    assert "Would collect profile scheduler evidence (preflight)" in result.stdout
    assert "Would collect profile scheduler evidence (postrun)" in result.stdout


def test_profile_scheduler_dry_run_accepts_budget_overrides(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"

    _run_dry_compare(
        run_root,
        "--steps",
        "12",
        "--initial-drafts",
        "4",
        "--seed",
        "123",
        "--agent-time-limit",
        "3600",
        "--timeout-seconds",
        "7200",
    )

    for label in ("scheduler_off", "scheduler_on"):
        command = _command_text(run_root, label)
        assert "agent.steps=12" in command
        assert "agent.initial_drafts=4" in command
        assert "agent.seed=123" in command
        assert "agent.time_limit=3600" in command

    manifest = (run_root / "manifest.txt").read_text(encoding="utf-8")
    assert "timeout_seconds: 7200" in manifest


def test_profile_scheduler_dry_run_can_run_scheduler_on_only_with_adaptive_defaults(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"

    _run_dry_compare(
        run_root,
        "--scheduler-on-only",
    )

    assert not (run_root / "scheduler_off" / "command.txt").exists()
    scheduler_on = _command_text(run_root, "scheduler_on")
    assert "scheduler.enabled=true" in scheduler_on
    assert "scheduler.settings.gpu_scheduler.mode=adaptive" in scheduler_on
    assert "scheduler.settings.prediction.mode=branch_profile" in scheduler_on
    assert "scheduler.settings.gpu_scheduler.max_packed_jobs_per_gpu=8" in scheduler_on
    assert "scheduler.settings.gpu_scheduler.candidate_window_size=16" in scheduler_on

    manifest = (run_root / "manifest.txt").read_text(encoding="utf-8")
    assert "scheduler_on_only: 1" in manifest
    assert "scheduler_off:" not in manifest
    assert "scheduler_on: dry-run" in manifest


def test_dynamic_plot_summary_handles_scheduler_modes(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"
    _write_metrics(
        run_root,
        "scheduler_off",
        {
            "mode": "hardware_aware",
            "experiment_mode": "hardware_aware",
            "command_label": "scheduler_off",
            "configured_scheduler_enabled": False,
            "scheduler_client_attached": False,
            "configured_hardware_knowledge_enabled": True,
            "hardware_knowledge_client_attached": True,
            "hardware_probe_source": "hardware_probe_subprocess",
            "hardware_probe_success": True,
            "hardware_knowledge_include_profile_evidence": True,
            "hardware_profile_evidence_used": False,
            "hardware_context_enabled": True,
            "total_wall_time_seconds": 20.0,
            "total_run_wall_time_seconds": 20.0,
            "total_candidate_execution_time_seconds": 8.0,
            "candidate_execution_makespan_seconds": 8.0,
            "total_llm_call_wall_time_seconds": 3.0,
            "total_training_wall_time_seconds": 5.0,
            "node_count": 2,
            "best_metric": 0.7,
        },
    )
    _write_metrics(
        run_root,
        "scheduler_on",
        {
            "mode": "hardware_aware",
            "experiment_mode": "hardware_aware",
            "command_label": "scheduler_on",
            "configured_scheduler_enabled": True,
            "scheduler_client_attached": True,
            "configured_hardware_knowledge_enabled": True,
            "hardware_knowledge_client_attached": True,
            "hardware_probe_source": "hardware_probe_subprocess",
            "hardware_probe_success": True,
            "hardware_knowledge_include_profile_evidence": True,
            "hardware_profile_evidence_used": True,
            "hardware_context_enabled": True,
            "scheduler_runtime_root": str(run_root / "scheduler_on" / "scheduler_runtime"),
            "total_wall_time_seconds": 15.0,
            "total_run_wall_time_seconds": 15.0,
            "total_candidate_execution_time_seconds": 6.0,
            "candidate_execution_makespan_seconds": 4.0,
            "total_llm_call_wall_time_seconds": 2.0,
            "total_training_wall_time_seconds": 3.0,
            "node_count": 3,
            "best_metric": 0.8,
        },
    )

    output_dir = run_root / "comparison_plots"
    subprocess.run(
        [
            "python",
            str(PLOTTER),
            "--run-root",
            str(run_root),
            "--output-dir",
            str(output_dir),
            "--modes",
            "scheduler_off",
            "scheduler_on",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((output_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "comparison_summary.md").read_text(encoding="utf-8")

    assert payload["mode_order"] == ["scheduler_off", "scheduler_on"]
    assert payload["delta_definition"]["scheduler_on_minus_scheduler_off"] == "scheduler_on - scheduler_off"
    assert payload["deltas"]["scheduler_on_minus_scheduler_off"]["total_wall_time_seconds"] == -5.0
    assert payload["deltas"]["scheduler_on_minus_scheduler_off"]["total_candidate_execution_time_seconds"] == -2.0
    assert payload["deltas"]["scheduler_on_minus_scheduler_off"]["node_count"] == 1.0
    assert "| Metric | Scheduler Off | Scheduler On | Scheduler On - Scheduler Off |" in markdown
    assert "| Scheduler enabled | False | True |  |" in markdown
    assert "| Hardware knowledge attached | True | True |  |" in markdown


def _write_metrics(run_root: Path, mode: str, metrics: dict[str, object]) -> None:
    log_dir = run_root / mode / "runs" / f"20260702_{mode}" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
