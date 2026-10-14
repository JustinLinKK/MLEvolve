from types import SimpleNamespace

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain.jobs import TrainingJob
from localml_scheduler.execution import worker_entry
from localml_scheduler.execution.worker_entry import _runtime_backend_name
from localml_scheduler.execution.worker_runtime import load_runtime_settings


def test_worker_uses_placement_backend_for_lifecycle_events() -> None:
    job = TrainingJob(job_id="mps-job", metadata={"placement_backend": "mps_process"})

    assert _runtime_backend_name(job) == "mps_process"


def test_worker_defaults_to_exclusive_without_scheduler_placement() -> None:
    job = TrainingJob(job_id="exclusive-job")

    assert _runtime_backend_name(job) == "exclusive"


def test_worker_applies_job_seed_before_running_candidate(monkeypatch) -> None:
    job = TrainingJob.create(
        runner_target="example.runner",
        baseline_model_id="baseline",
        baseline_model_path="/tmp/baseline.pt",
        seed=42,
    )
    context = SimpleNamespace(job=job)
    seeded: list[int] = []

    monkeypatch.setattr(worker_entry, "load_runtime_settings", lambda _root: SimpleNamespace(
        events_jsonl_path="/tmp/events.jsonl", scheduler_log_path="/tmp/scheduler.log"
    ))
    monkeypatch.setattr(worker_entry, "StateStore", lambda _settings: object())
    monkeypatch.setattr(worker_entry, "SchedulerLogStore", lambda _settings: object())
    monkeypatch.setattr(worker_entry, "EventLogger", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(worker_entry, "setup_scheduler_logger", lambda _path: SimpleNamespace(info=lambda *_args: None))
    monkeypatch.setattr(worker_entry, "create_runner_context", lambda *_args: (context, job))
    monkeypatch.setattr(worker_entry, "mark_job_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_entry, "run_batch_probe_preflight", lambda value: value.job)
    monkeypatch.setattr(worker_entry, "resolve_runner", lambda _context: lambda _value: {"ok": True})
    monkeypatch.setattr(worker_entry, "mark_job_completed", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(worker_entry, "set_global_seed", seeded.append)

    assert worker_entry._run_job("/tmp/runtime", job.job_id) == 0
    assert seeded == [42]


def test_client_persists_explicit_settings_for_worker_subprocesses(tmp_path) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
        hardware_knowledge_graph={"enabled": False},
        prediction={"mode": "ml_predictor"},
    )

    client = SchedulerClient(settings)
    client.submit(
        TrainingJob.create(
            runner_target="example.runner",
            baseline_model_id="baseline",
            baseline_model_path="/tmp/baseline.pt",
        )
    )
    worker_settings = load_runtime_settings(tmp_path)

    assert worker_settings.graph_db.enabled is False
    assert worker_settings.hardware_feature_db.enabled is False
    assert worker_settings.hardware_knowledge_graph.enabled is False
    assert worker_settings.prediction.mode == "ml_predictor"
