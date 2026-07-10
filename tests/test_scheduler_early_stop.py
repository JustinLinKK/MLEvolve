from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import CheckpointPolicy, JobMetricSample, JobStatus, TrainingJob, utc_now
from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint
from localml_scheduler.client import SchedulerClient
from localml_scheduler.scheduler.early_stop import analyze_metric_plateau
from localml_scheduler.scheduler.training_plot import render_training_process
from localml_scheduler.storage.state_store import StateStore


def _sample(job_id: str, step: int, **metrics: float) -> JobMetricSample:
    return JobMetricSample(job_id=job_id, created_at=utc_now(), epoch=0, global_step=step, metrics=metrics)


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.05) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


def test_plateau_detection_maximize_minimize_and_lr_only() -> None:
    maximize_samples = [_sample("job-a", index, accuracy=value, lr=0.01) for index, value in enumerate([0.7, 0.8, 0.8, 0.80001, 0.8], start=1)]
    maximize_decision = analyze_metric_plateau(
        maximize_samples,
        warmup_samples=2,
        patience_samples=2,
        min_delta=1e-3,
    )
    assert maximize_decision.should_stop is True
    assert maximize_decision.metric_key == "accuracy"
    assert maximize_decision.direction == "maximize"

    minimize_samples = [_sample("job-b", index, val_loss=value) for index, value in enumerate([1.0, 0.8, 0.8, 0.80001], start=1)]
    minimize_decision = analyze_metric_plateau(
        minimize_samples,
        warmup_samples=2,
        patience_samples=2,
        min_delta=1e-3,
    )
    assert minimize_decision.should_stop is True
    assert minimize_decision.metric_key == "val_loss"
    assert minimize_decision.direction == "minimize"

    lr_only = analyze_metric_plateau(
        [_sample("job-c", index, lr=0.01) for index in range(1, 6)],
        warmup_samples=1,
        patience_samples=1,
    )
    assert lr_only.should_stop is False
    assert lr_only.reason == "no objective metric available"


def test_sqlite_metric_samples_and_plot_render(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        graph_db={"enabled": False, "mode": "off"},
        hardware_feature_db={"enabled": False},
        hardware_knowledge_graph={"enabled": False},
        log_db={"enabled": False},
        redis_cache={"enabled": False},
    )
    store = StateStore(settings)
    store.record_job_metric_sample(
        job_id="job-plot",
        created_at=utc_now(),
        epoch=0,
        global_step=1,
        metrics={"loss": 1.0, "accuracy": 0.5, "lr": 0.01},
    )
    store.record_job_metric_sample(
        job_id="job-plot",
        created_at=utc_now(),
        epoch=0,
        global_step=2,
        metrics={"loss": 0.9, "accuracy": 0.6, "lr": 0.005},
    )
    samples = store.list_job_metric_samples("job-plot")

    assert len(samples) == 2
    assert samples[0].metrics["loss"] == 1.0

    artifact = render_training_process(samples, tmp_path / "plot")
    assert Path(artifact["plot_path"]).exists()
    assert Path(artifact["summary_path"]).exists()


def test_scheduler_early_stops_plateauing_toy_job(tmp_path: Path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        scheduler_poll_interval_seconds=0.05,
        gpu_scheduler={
            "mode": "serial_basic",
            "backend_priority": ["exclusive"],
            "checkpoint_preemption_enabled": False,
            "early_stop": {
                "enabled": True,
                "warmup_samples": 2,
                "patience_samples": 2,
                "min_delta": 1e-4,
                "min_runtime_seconds": 0.0,
                "min_global_step": 3,
                "metric_key": "loss",
                "direction": "minimize",
                "plot_enabled": True,
            },
            "stream": {"enabled": False},
            "cuda_process": {"enabled": False},
            "mps": {"enabled": False},
        },
        graph_db={"enabled": False, "mode": "off"},
        hardware_feature_db={"enabled": False},
        hardware_knowledge_graph={"enabled": False},
        log_db={"enabled": False},
        redis_cache={"enabled": False},
    )
    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)
    try:
        baseline = create_toy_baseline_checkpoint(tmp_path / "baseline.pt", seed=3)
        job = api.submit(
            TrainingJob.create(
                runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                baseline_model_id="toy",
                baseline_model_path=baseline,
                max_steps=50,
                runner_kwargs={"sleep_per_step": 0.02, "reported_loss_override": 1.0},
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, save_every_epoch=True),
            )
        )
        wait_for(lambda: (api.inspect(job.job_id) or job).status.is_terminal, timeout=20.0)
        final_job = api.inspect(job.job_id)
        assert final_job is not None
        assert final_job.status == JobStatus.EARLY_STOPPED
        assert final_job.metadata["scheduler_early_stop_decision"]["metric_key"] == "loss"
        assert final_job.metadata["scheduler_early_stop_decision"]["should_stop"] is True
        assert final_job.metadata["scheduler_early_stop_global_step"] < 50
        assert Path(final_job.metadata["scheduler_early_stop_plot_path"]).exists()
        assert api.list_events(job_id=job.job_id, event_type="scheduler_early_stop_requested")
        assert api.list_events(job_id=job.job_id, event_type="job_early_stopped")
    finally:
        service.stop()


def test_scheduler_bridge_returns_early_stop_feedback(tmp_path: Path) -> None:
    from engine.executor import Interpreter

    result_path = tmp_path / "result.json"
    result_path.write_text(
        """
        {
          "term_out": ["partial run\\n"],
          "exec_time": 1.5,
          "exc_type": null,
          "exc_info": {},
          "exc_stack": [],
          "phase_timings": {},
          "instrumentation": {"scheduler_early_stop": {"reason": "plateau", "plot_path": "/tmp/plot.png"}}
        }
        """,
        encoding="utf-8",
    )
    interpreter = Interpreter.__new__(Interpreter)
    interpreter.pipeline_logger = None
    interpreter.scheduler_client = None
    prepared = SimpleNamespace(
        job_id="job-early",
        node_id="node-early",
        result_path=result_path,
        runner_kwargs={},
        scheduler_mode="serial_basic",
        detected_batch_size=None,
        proposed_epochs=None,
        model_key=None,
        input_resolution=None,
        fold_count=None,
        ensemble_count=None,
        tta_count=None,
        framework=None,
        uses_amp=None,
        requires_gpu=None,
        script_signature="sig",
        start_time=time.time() - 2,
    )
    final_job = SimpleNamespace(
        status=JobStatus.EARLY_STOPPED,
        status_reason="early stop: loss plateaued",
        to_dict=lambda: {
            "status": "EARLY_STOPPED",
            "status_reason": "early stop: loss plateaued",
            "metadata": {
                "scheduler_early_stop_decision": {"metric_key": "loss", "should_stop": True},
                "scheduler_early_stop_plot_path": "/tmp/plot.png",
                "scheduler_early_stop_summary_path": "/tmp/summary.json",
            },
        },
    )

    result = interpreter._scheduler_execution_result_from_final(prepared, final_job)

    assert result.instrumentation["scheduler_early_stop"]["plot_path"] == "/tmp/plot.png"
    assert "Training process plot: /tmp/plot.png" in "".join(result.term_out)
