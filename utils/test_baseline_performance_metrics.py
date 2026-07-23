from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.search_node import Journal, SearchNode
from utils.compare_experiment_runs import compare_metrics, to_markdown
from utils.experiment_metrics import build_comparison_metrics, write_comparison_metrics
from utils.metric import MetricValue


BASELINE_SCRIPT = REPO_ROOT / "measure_baseline_performance.sh"


def _journal_with_valid_node(started: float) -> Journal:
    journal = Journal()
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(
        code=(
            "BATCH_SIZE = 16\n"
            "MODEL_NAME = 'vit_base_patch16_224'\n"
            "EPOCHS = 3\n"
            "IMG_SIZE = (224, 224)\n"
            "N_FOLDS = 5\n"
            "ENSEMBLE_SIZE = 2\n"
            "TTA_COUNT = 4\n"
        ),
        plan="draft",
        parent=root,
        stage="draft",
    )
    node.metric = MetricValue(0.8, maximize=True)
    node.is_buggy = False
    node.is_valid = True
    node.exec_time = 7.0
    node.ctime = started + 5
    journal.append(root)
    journal.append(node)
    return journal


def test_baseline_comparison_metrics_generation(tmp_path: Path) -> None:
    started = time.time() - 20
    cfg = SimpleNamespace(exp_name="run-baseline", exp_id="task-a", log_dir=tmp_path)
    journal = _journal_with_valid_node(started)

    metrics = build_comparison_metrics(cfg, journal, started_at=started, finished_at=started + 20)
    path = write_comparison_metrics(metrics, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["mode"] == "baseline"
    assert loaded["node_count"] == 1
    assert loaded["valid_count"] == 1
    assert loaded["buggy_count"] == 0
    assert loaded["best_metric"] == 0.8
    assert loaded["total_wall_time_seconds"] == 20
    assert loaded["time_to_best_seconds"] == 12
    assert loaded["model_key"] == "vit_base_patch16_224"
    assert loaded["proposed_epochs"] == 3
    assert loaded["input_resolution"] == 224
    assert loaded["fold_count"] == 5
    assert loaded["ensemble_count"] == 2
    assert loaded["tta_count"] == 4


class _SchedulerStub:
    def list_jobs(self) -> list[dict[str, object]]:
        return [
            {
                "submitted_at": "2026-01-01T00:00:00+00:00",
                "started_at": "2026-01-01T00:00:05+00:00",
                "finished_at": "2026-01-01T00:00:15+00:00",
                "metadata": {
                    "placement_backend": "cuda_process",
                    "placement_mode": "exclusive",
                },
                "packing": {"eligible": True},
            }
        ]

    def list_events(self) -> list[dict[str, object]]:
        return [
            {
                "event_type": "batch_probe_started",
                "job_id": "job-1",
                "created_at": "2026-01-01T00:00:06+00:00",
            },
            {
                "event_type": "batch_probe_selected",
                "job_id": "job-1",
                "created_at": "2026-01-01T00:00:09+00:00",
                "payload": {"peak_vram_mb": 1234},
            },
            {
                "event_type": "batch_probe_trial",
                "job_id": "job-1",
                "created_at": "2026-01-01T00:00:07+00:00",
                "payload": {"backend_name": "cuda_process"},
            },
        ]


class _FailingSchedulerStub:
    def list_jobs(self) -> list[dict[str, object]]:
        raise RuntimeError("jobs unavailable")

    def list_events(self) -> list[dict[str, object]]:
        raise RuntimeError("events unavailable")


def test_scheduler_metrics_are_collected_and_failures_are_graceful(tmp_path: Path) -> None:
    started = time.time() - 20
    cfg = SimpleNamespace(exp_name="run-baseline", exp_id="task-a", log_dir=tmp_path)
    journal = _journal_with_valid_node(started)

    metrics = build_comparison_metrics(
        cfg,
        journal,
        started_at=started,
        finished_at=started + 20,
        scheduler_client=_SchedulerStub(),
    )

    assert metrics["queue_wait_seconds"] == 5
    assert metrics["probe_time_seconds"] == 3
    assert metrics["total_job_execution_time_seconds"] == 10
    assert metrics["median_job_execution_time_seconds"] == 10
    assert metrics["placement_backend"] == "cuda_process"
    assert metrics["placement_mode"] == "exclusive"
    assert metrics["packed_dispatch_count"] == 1
    assert metrics["batch_probe_trial_count"] == 1
    assert metrics["peak_vram_mb"] == 1234

    fallback = build_comparison_metrics(
        cfg,
        journal,
        started_at=started,
        finished_at=started + 20,
        scheduler_client=_FailingSchedulerStub(),
    )

    assert fallback["queue_wait_seconds"] == 0
    assert fallback["probe_time_seconds"] == 0
    assert fallback["scheduler_backend_distribution"] == {}


def test_compare_experiment_runs_json_and_markdown() -> None:
    comparison = compare_metrics(
        {"run_id": "run-base", "mode": "baseline", "node_count": 2, "best_metric": 0.4},
        {"run_id": "run-hw", "mode": "hardware_aware", "node_count": 3, "best_metric": 0.7},
    )
    markdown = to_markdown(comparison)

    assert comparison["baseline_run_id"] == "run-base"
    assert comparison["hardware_run_id"] == "run-hw"
    assert comparison["metrics"]["node_count"]["delta"] == 1
    assert comparison["metrics"]["best_metric"]["delta"] == 0.29999999999999993
    assert "| `best_metric` |" in markdown


def test_measure_baseline_performance_dry_run_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "baseline-measure"
    result = subprocess.run(
        [
            "bash",
            str(BASELINE_SCRIPT),
            "unit-competition",
            "--run-root",
            str(run_root),
            "--no-validation-server",
            "--dry-run",
            "--steps",
            "7",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    command = (run_root / "baseline" / "command.txt").read_text(encoding="utf-8")
    assert "==> Running baseline" in result.stdout
    assert "agent.steps=7" in command
    assert "experiment.mode" not in command
    assert (run_root / "baseline" / "exit_code.txt").read_text(encoding="utf-8").strip() == "dry-run"
    assert not (run_root / "hardware_aware").exists()
