from __future__ import annotations

import logging
from types import SimpleNamespace

from localml_scheduler.config import SchedulerSettings
from run import _submit_startpoint_probe_jobs


class _FakePipelineLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))


class _FakeSchedulerClient:
    def __init__(self, settings):
        self.settings = settings
        self.submitted_jobs = []

    def submit_many(self, jobs):
        self.submitted_jobs.extend(jobs)
        return jobs


def test_submit_startpoint_probe_jobs_builds_async_scheduler_jobs(tmp_path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
    client = _FakeSchedulerClient(settings)
    pipeline = _FakePipelineLogger()
    cfg = SimpleNamespace(exp_name="unit-exp")
    specs = [
        {
            "rank": 1,
            "task_id": "unit-task",
            "model_key": "safe/model",
            "display_name": "Safe Model",
            "modality": "vision",
            "shape_hints": {"modality": "vision", "input_resolution": 256},
        }
    ]

    job_ids = _submit_startpoint_probe_jobs(
        scheduler_client=client,
        scheduler_settings=settings,
        cfg=cfg,
        startpoint_specs=specs,
        pipeline_logger=pipeline,
        logger=logging.getLogger("test"),
    )

    assert job_ids == [client.submitted_jobs[0].job_id]
    submitted = client.submitted_jobs[0]
    assert submitted.task_type == "mlevolve_startpoint_probe"
    assert submitted.priority == 100
    assert submitted.batch_probe.profile_key
    assert submitted.batch_probe.shape_signature_override
    assert submitted.batch_probe.search_mode == "power_of_two"
    assert pipeline.events[0][0] == "scheduler_startpoint_probes_submitted"
