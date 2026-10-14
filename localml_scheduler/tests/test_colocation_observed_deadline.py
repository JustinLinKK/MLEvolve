"""Regression coverage for live-trial evidence windows on slow real workloads."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from localml_scheduler.scheduler.colocation_evidence import ColocationEvidenceMixin
from localml_scheduler.scheduler.service_state import ColocationTrialState


class _EvidenceHarness(ColocationEvidenceMixin):
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            gpu_scheduler=SimpleNamespace(
                colocation=SimpleNamespace(
                    trial_epochs=2,
                    trial_evidence_timeout_min_seconds=300.0,
                    trial_evidence_timeout_max_seconds=1800.0,
                )
            )
        )


def test_observed_slow_epoch_extends_unexpired_trial_deadline() -> None:
    """A bad sub-second predictor estimate cannot cancel a real multi-minute trial."""
    started = datetime(2026, 9, 16, tzinfo=timezone.utc)
    trial = ColocationTrialState(
        trial_id="trial",
        candidate_job_id="new",
        preexisting_job_ids=("old",),
        started_at=started.isoformat(),
        start_epoch=0,
        target_epoch=2,
        backend_name="cuda_process",
        profile_key="pair",
        candidate_solo_epoch_seconds=0.4,
        pretrial_epoch_seconds={"old": 0.4},
        member_start_epochs={"old": 0, "new": 0},
        evidence_deadline_at=(started + timedelta(seconds=300)).isoformat(),
    )
    old = SimpleNamespace(
        job_id="old",
        metadata={
            "runtime_epoch_timing_history": [
                {
                    "epoch": 1,
                    "seconds": 210.0,
                    "started_at": (started + timedelta(seconds=1)).isoformat(),
                    "finished_at": (started + timedelta(seconds=211)).isoformat(),
                }
            ]
        },
    )
    new = SimpleNamespace(job_id="new", metadata={"runtime_epoch_timing_history": []})

    refreshed = _EvidenceHarness()._extended_trial_deadline_from_observed_epochs(
        trial,
        [old, new],
        now=started + timedelta(seconds=250),
    )

    assert refreshed is not None
    assert datetime.fromisoformat(refreshed) > datetime.fromisoformat(trial.evidence_deadline_at)
    assert datetime.fromisoformat(refreshed) <= started + timedelta(seconds=1800)
