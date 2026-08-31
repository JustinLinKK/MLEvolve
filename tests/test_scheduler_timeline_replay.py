from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchProbeSpec,
    JobStatus,
    PackingSpec,
    ResourceRequirements,
    RuntimeProbeSpec,
    TrainingJob,
)
from localml_scheduler.execution.process_utils import (
    start_new_session_kwargs,
    terminate_process_tree,
)
from scheduler_benchmark_test.replay_multiprocess_baseline import (
    replay_multiprocess_baseline,
)
from scheduler_benchmark_test.replay_model_sources import (
    build_scheduler_stress_fixture,
    materialize_sources,
    validate_smoke_sources,
)
from scheduler_benchmark_test.replay_scheduler_timeline import replay_fixture
from scheduler_benchmark_test.timeline_fixture import extract_fixture, load_fixture
from scheduler_benchmark_test.validate_replay_fixture import validate_fixture

ROOT = Path(__file__).resolve().parents[1]


def _run_wrapper(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)


def _wait_for(predicate, *, timeout: float = 5.0, interval: float = 0.05) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition was not met before timeout")


def _process_is_running(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    return bool(status and not status.startswith("Z"))


def _make_runtime(tmp_path: Path) -> Path:
    runtime_root = tmp_path / "runtime"
    workspace = tmp_path / "original_workspace"
    workspace.mkdir(parents=True)
    (workspace / "input").mkdir()
    script = workspace / "candidate.py"
    script.write_text("print('candidate')\n", encoding="utf-8")

    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.02,
        gpu_scheduler={
            "packing_backend": "cuda_process",
            "exclusive_fallback_enabled": True,
            "cuda_process": {"enabled": False},
            "mps": {"enabled": False},
        },
        log_db={"enabled": False},
    )
    settings.ensure_runtime_layout()
    (runtime_root / "scheduler_settings.json").write_text(
        json.dumps(settings.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    client = SchedulerClient(settings)

    job1 = _job("job-1", script, workspace, task_type="mlevolve_script", priority=1)
    job1.metadata.update({"placement_backend": "cuda_process", "resolved_batch_size": 64})
    job2 = _job(
        "job-2",
        script,
        workspace,
        task_type="mlevolve_model_family_probe",
        priority=100,
    )
    job2.metadata.update(
        {"kind": "mlevolve_model_family_probe", "placement_mode": "exclusive"}
    )

    client.submit(job1)
    client.cancel(job1.job_id)
    client.submit(job2)
    client.cancel(job2.job_id)
    client.store.set_job_status(
        job1.job_id, JobStatus.COMPLETED, reason="original complete"
    )
    client.store.set_job_status(
        job2.job_id, JobStatus.FAILED, reason="original failed", hold=True
    )
    _rewrite_command_times(settings.db_path, [0.0, 10.0, 20.0, 30.0])
    return runtime_root


def _job(
    job_id: str, script: Path, workspace: Path, *, task_type: str, priority: int
) -> TrainingJob:
    return TrainingJob.create(
        job_id=job_id,
        runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
        baseline_model_id=f"baseline-{job_id}",
        baseline_model_path=str(script),
        runner_kwargs={
            "script_path": str(script),
            "working_dir": str(workspace),
            "result_path": str(
                workspace / "working" / "scheduler_results" / f"result_{job_id}.json"
            ),
            "batch_size": 8,
        },
        priority=priority,
        task_type=task_type,
        resource_requirements=ResourceRequirements(requires_gpu=True),
        packing=PackingSpec(
            eligible=True,
            signature=f"sig-{job_id}",
            family="unit-family",
            backend_allowlist=["cuda_process", "exclusive"],
        ),
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
            model_key="unit-model",
        ),
        runtime_probe=RuntimeProbeSpec(enabled=True),
    )


def _rewrite_command_times(db_path: Path, offsets: list[float]) -> None:
    base = datetime(2026, 7, 5, 0, 0, tzinfo=timezone.utc)
    with sqlite3.connect(str(db_path)) as connection:
        command_ids = [
            row[0]
            for row in connection.execute(
                "SELECT command_id FROM commands ORDER BY command_id"
            )
        ]
        for command_id, offset in zip(command_ids, offsets, strict=True):
            timestamp = (base + timedelta(seconds=offset)).isoformat()
            connection.execute(
                "UPDATE commands SET created_at = ?, processed_at = ? WHERE command_id = ?",
                (timestamp, timestamp, command_id),
            )
        connection.commit()


def test_extract_fixture_resets_jobs_and_marks_final_cleanup(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"

    extract_fixture(runtime_root, fixture_dir)
    actions, jobs_by_id, baseline, settings = load_fixture(fixture_dir)

    assert [action["relative_seconds"] for action in actions] == [0.0, 10.0, 20.0, 30.0]
    assert [action["final_cleanup"] for action in actions] == [
        False,
        False,
        False,
        True,
    ]
    assert baseline["command_count"] == 4
    assert baseline["submit_count"] == 2
    assert baseline["mid_run_cancel_count"] == 1
    assert baseline["final_cleanup_cancel_count"] == 1
    assert baseline["task_type_counts"]["mlevolve_script"] == 1
    assert baseline["task_type_counts"]["mlevolve_model_family_probe"] == 1

    job1 = jobs_by_id["job-1"]
    assert job1["status"] == "PENDING"
    assert job1["status_timestamps"] == {}
    assert job1["started_at"] is None
    assert job1["finished_at"] is None
    assert job1["metadata"]["replay_original"]["status"] == "COMPLETED"
    assert "placement_backend" not in job1["metadata"]
    assert "resolved_batch_size" not in job1["metadata"]
    assert (
        job1["config"]["runner_target"]
        == "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job"
    )
    assert job1["config"]["runner_kwargs"]["script_path"].endswith("candidate.py")
    assert job1["batch_probe"]["enabled"] is True
    assert job1["packing"]["signature"] == "sig-job-1"
    assert settings["log_db"]["enabled"] is False


def test_replay_noop_writes_metrics_and_respects_cleanup_gate(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    dry_output = tmp_path / "dry"
    replay_fixture(
        fixture=fixture_dir,
        output_root=dry_output,
        runner_mode="noop",
        dry_run=True,
        no_sleep=True,
    )
    dry_metrics = json.loads(
        (dry_output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert dry_metrics["replay_action_count"] == 3
    assert dry_metrics["replay_submit_action_count"] == 2
    assert dry_metrics["replay_cancel_action_count"] == 1

    dry_with_cleanup = tmp_path / "dry_with_cleanup"
    replay_fixture(
        fixture=fixture_dir,
        output_root=dry_with_cleanup,
        runner_mode="noop",
        dry_run=True,
        no_sleep=True,
        include_final_cleanup_cancels=True,
    )
    cleanup_metrics = json.loads(
        (dry_with_cleanup / "logs" / "comparison_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert cleanup_metrics["replay_action_count"] == 4
    assert cleanup_metrics["replay_cancel_action_count"] == 2

    output = tmp_path / "replay"
    replay_fixture(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        no_sleep=True,
        post_actions_wait_seconds=5,
    )

    assert (output / "scheduler_runtime" / "db" / "scheduler.sqlite3").exists()
    assert (output / "logs" / "comparison_metrics.json").exists()
    assert (output / "replay_summary.json").exists()
    assert (output / "scheduler_replay" / "runs").exists()
    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["replay_runner_mode"] == "noop"
    assert metrics["submitted_job_count"] == 2
    assert metrics["scheduler_job_count"] == 2
    assert metrics["completed_job_count"] >= 1


def test_replay_until_seconds_truncates_actions(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    output = tmp_path / "until"
    replay_fixture(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        dry_run=True,
        until_seconds=0.0,
        no_sleep=True,
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["replay_action_count"] == 1
    assert metrics["replay_submit_action_count"] == 1
    assert metrics["replay_cancel_action_count"] == 0


def test_replay_real_skips_missing_script_unless_strict(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)
    actions, jobs_by_id, _baseline, _settings = load_fixture(fixture_dir)

    job = jobs_by_id["job-1"]
    missing_script = fixture_dir / "missing_candidate.py"
    job["baseline_model_path"] = str(missing_script)
    job["config"]["runner_kwargs"]["script_path"] = str(missing_script)
    selected_actions = [
        action
        for action in actions
        if action["job_id"] == "job-1" and action["action"] == "SUBMIT"
    ]
    (fixture_dir / "timeline.json").write_text(
        json.dumps({"actions": selected_actions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (fixture_dir / "jobs.jsonl").write_text(
        json.dumps(job, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "missing_script"
    result = replay_fixture(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="real",
        no_sleep=True,
        post_actions_wait_seconds=0,
    )

    assert result.submitted_job_ids == []
    assert len(result.skipped_actions) == 1
    assert (
        "Replay script path does not exist" in result.skipped_actions[0]["skip_reason"]
    )
    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["submitted_job_count"] == 0
    assert metrics["scheduler_job_count"] == 0
    assert metrics["replay_skipped_action_count"] == 1

    with pytest.raises(FileNotFoundError):
        replay_fixture(
            fixture=fixture_dir,
            output_root=tmp_path / "missing_script_strict",
            runner_mode="real",
            no_sleep=True,
            post_actions_wait_seconds=0,
            strict_missing_jobs=True,
        )


def test_scheduler_replay_ignore_cancels_and_wait_for_all(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    dry_output = tmp_path / "ignore_cancels_dry"
    replay_fixture(
        fixture=fixture_dir,
        output_root=dry_output,
        runner_mode="noop",
        dry_run=True,
        cancel_policy="ignore",
        include_final_cleanup_cancels=True,
    )
    dry_metrics = json.loads(
        (dry_output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert dry_metrics["replay_action_count"] == 2
    assert dry_metrics["replay_submit_action_count"] == 2
    assert dry_metrics["replay_cancel_action_count"] == 0
    assert dry_metrics["replay_cancel_policy"] == "ignore"

    output = tmp_path / "wait_for_all"
    replay_fixture(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        no_sleep=True,
        post_actions_wait_seconds=0,
        wait_for_all=True,
        cancel_policy="ignore",
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["replay_wait_for_all"] is True
    assert metrics["replay_cancel_policy"] == "ignore"
    assert metrics["submitted_job_count"] == 2
    assert metrics["completed_job_count"] == 2
    assert metrics["cancelled_job_count"] == 0


def test_replay_clean_scheduler_state_removes_stale_runtime_db(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    output = tmp_path / "clean_scheduler_state"
    stale_marker = output / "scheduler_runtime" / "db" / "stale_profile_marker.txt"
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("old profile cache", encoding="utf-8")

    replay_fixture(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        no_sleep=True,
        post_actions_wait_seconds=0,
        wait_for_all=True,
        cancel_policy="ignore",
        clean_scheduler_state=True,
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["replay_clean_scheduler_state"] is True
    assert not stale_marker.exists()
    assert (output / "scheduler_runtime" / "db" / "scheduler.sqlite3").exists()


def test_replay_strips_archive_bookkeeping_before_job_reconstruction(
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)
    actions, jobs_by_id, _baseline, _settings = load_fixture(fixture_dir)

    jobs_by_id["job-1"][
        "pre_archive_baseline_model_path"
    ] = "/previous/run/candidate.py"
    (fixture_dir / "jobs.jsonl").write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs_by_id.values()),
        encoding="utf-8",
    )

    output = tmp_path / "archive_bookkeeping"
    replay_fixture(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        no_sleep=True,
        post_actions_wait_seconds=0,
        wait_for_all=True,
        cancel_policy="ignore",
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["submitted_job_count"] == sum(
        1 for action in actions if action["action"] == "SUBMIT"
    )
    assert metrics["completed_job_count"] == metrics["submitted_job_count"]


def test_multiprocess_baseline_noop_writes_metrics(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    output = tmp_path / "multiprocess"
    replay_multiprocess_baseline(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        parallelism=2,
        until_seconds=0.0,
        no_sleep=True,
        post_actions_wait_seconds=5,
    )

    assert (output / "logs" / "comparison_metrics.json").exists()
    assert (output / "logs" / "multiprocess_jobs.jsonl").exists()
    assert (output / "replay_summary.json").exists()
    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["mode"] == "multiprocess_baseline"
    assert metrics["configured_scheduler_enabled"] is False
    assert metrics["multiprocess_parallelism"] == 2
    assert metrics["multiprocess_job_filter"] == "script"
    assert metrics["submitted_job_count"] == 1
    assert metrics["completed_job_count"] == 1
    assert metrics["node_count"] == 1


def test_multiprocess_baseline_dry_run_filters_probe_jobs(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    output = tmp_path / "multiprocess_dry"
    replay_multiprocess_baseline(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        dry_run=True,
        include_final_cleanup_cancels=True,
        no_sleep=True,
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["replay_action_count"] == 4
    assert metrics["replay_submit_action_count"] == 2
    assert metrics["replay_skipped_action_count"] == 2


def test_multiprocess_wait_for_all_and_ignore_cancels(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    extract_fixture(runtime_root, fixture_dir)

    output = tmp_path / "multiprocess_wait_for_all"
    replay_multiprocess_baseline(
        fixture=fixture_dir,
        output_root=output,
        runner_mode="noop",
        parallelism=1,
        job_filter="all",
        no_sleep=True,
        post_actions_wait_seconds=0,
        wait_for_all=True,
        cancel_policy="ignore",
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["replay_wait_for_all"] is True
    assert metrics["replay_cancel_policy"] == "ignore"
    assert metrics["replay_action_count"] == 2
    assert metrics["replay_cancel_action_count"] == 0
    assert metrics["submitted_job_count"] == 2
    assert metrics["completed_job_count"] == 2
    assert metrics["cancelled_job_count"] == 0


def test_validate_fixture_writes_clean_known_good_fixture(tmp_path: Path) -> None:
    runtime_root = _make_runtime(tmp_path)
    fixture_dir = tmp_path / "fixture"
    clean_dir = tmp_path / "clean_fixture"
    extract_fixture(runtime_root, fixture_dir)

    result = validate_fixture(
        fixture=fixture_dir,
        clean_fixture=clean_dir,
        task_filter="all",
    )

    assert result.summary["job_count"] == 2
    assert result.summary["kept_job_count"] == 1
    assert result.summary["excluded_job_count"] == 1
    assert result.report_path.exists()

    actions, jobs_by_id, baseline, settings = load_fixture(clean_dir)
    assert [action["action"] for action in actions] == ["SUBMIT"]
    assert actions[0]["relative_seconds"] == 0.0
    assert actions[0]["original_relative_seconds"] == 0.0
    assert list(jobs_by_id) == ["job-1"]
    job = jobs_by_id["job-1"]
    assert job["metadata"]["fixture_validation"]["known_good"] is True
    assert job["metadata"]["fixture_validation"]["original_status"] == "COMPLETED"
    assert baseline["job_count"] == 1
    assert baseline["submit_count"] == 1
    assert baseline["cancel_count"] == 0
    assert settings["log_db"]["enabled"] is False


def test_materialize_replay_sources_rewrites_fixtures_and_validates_smoke(
    tmp_path: Path,
) -> None:
    fixture_dir = tmp_path / "fixture"
    workspace = tmp_path / "source_run" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "input").mkdir()
    good_script = workspace / "runfile_1_good.py"
    good_script.write_text("print('good replay source')\n", encoding="utf-8")
    repaired_script = workspace / "runfile_29_4c400159969344d480b54aba0554b381_fix.py"
    repaired_script.write_text(
        "print('before')\n=======\nprint('after')\n", encoding="utf-8"
    )
    missing_script = (
        workspace / "runfile_26_66b11d68876c4a768709a5a91ba8fa41_missing.py"
    )
    prompt_dir = workspace.parent / "logs" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "66b11d68876c4a768709a5a91ba8fa41.improve.prompt.md").write_text(
        json.dumps(
            {
                "system": "",
                "user": "",
                "assistant": (
                    "```python\n"
                    "import os\n\n"
                    "def run_pipeline():\n"
                    "    print('recovered replay source')\n\n"
                    "if __name__ == '__main__':\n"
                    "    run_pipeline()\n"
                    "```\n"
                ),
            }
        ),
        encoding="utf-8",
    )

    settings = SchedulerSettings(
        runtime_root=tmp_path / "runtime",
        log_db={"enabled": False},
    )
    jobs = [
        _job(
            "job-good", good_script, workspace, task_type="mlevolve_script", priority=1
        ).to_dict(),
        _job(
            "job-repaired",
            repaired_script,
            workspace,
            task_type="mlevolve_script",
            priority=1,
        ).to_dict(),
        _job(
            "job-recovered",
            missing_script,
            workspace,
            task_type="mlevolve_script",
            priority=1,
        ).to_dict(),
    ]
    jobs[1]["metadata"]["node_id"] = "4c400159969344d480b54aba0554b381"
    jobs[2]["metadata"]["node_id"] = "66b11d68876c4a768709a5a91ba8fa41"
    fixture_dir.mkdir()
    (fixture_dir / "jobs.jsonl").write_text(
        "".join(json.dumps(job, sort_keys=True, default=str) + "\n" for job in jobs),
        encoding="utf-8",
    )
    (fixture_dir / "timeline.json").write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "action": "SUBMIT",
                        "command_id": index,
                        "final_cleanup": False,
                        "job_id": job["job_id"],
                        "relative_seconds": float(index),
                    }
                    for index, job in enumerate(jobs, start=1)
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (fixture_dir / "baseline_summary.json").write_text(
        json.dumps(
            {
                "original_input_dir": str(workspace / "input"),
                "script_path_count": 3,
                "missing_script_path_count": 1,
                "missing_script_paths": [str(missing_script)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (fixture_dir / "scheduler_settings.replay.json").write_text(
        json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive_root = tmp_path / "replay_model_sources" / "unit"
    result = materialize_sources(fixtures=[fixture_dir], archive_root=archive_root)

    resolutions = {
        Path(record["original_script_path"]).name: record["source_resolution"]
        for record in result.manifest["records"]
    }
    assert resolutions[good_script.name] == "original"
    assert resolutions[repaired_script.name] == "repaired_original"
    assert resolutions[missing_script.name] == "recovered_from_prompt"

    _actions, jobs_by_id, baseline, _settings = load_fixture(fixture_dir)
    assert baseline["missing_script_path_count"] == 0
    assert baseline["replay_source_archive"] == str(archive_root.resolve())
    for job in jobs_by_id.values():
        script_path = Path(job["config"]["runner_kwargs"]["script_path"])
        assert archive_root.resolve() in script_path.parents
        assert script_path.exists()
        assert "runs" not in script_path.parts
        assert job["baseline_model_path"] == str(script_path)

    smoke = validate_smoke_sources(
        fixtures=[fixture_dir], archive_root=archive_root, timeout_seconds=10
    )
    assert smoke.report["ok"] is True
    assert smoke.report["summary"]["source_count"] == 3


def test_build_scheduler_stress_fixture_creates_cold_two_epoch_jobs(
    tmp_path: Path,
) -> None:
    runtime_root = _make_runtime(tmp_path)
    source_fixture = tmp_path / "fixture"
    stress_fixture = tmp_path / "stress_fixture"
    archive_root = tmp_path / "replay_model_sources" / "unit"
    extract_fixture(runtime_root, source_fixture)

    result = build_scheduler_stress_fixture(
        source_fixture=source_fixture,
        output_fixture=stress_fixture,
        archive_root=archive_root,
        max_epochs=2,
    )

    actions, jobs_by_id, baseline, settings = load_fixture(stress_fixture)
    assert result.summary["job_count"] == 1
    assert result.summary["normal_timeout_field_count"] == 0
    assert result.summary["batch_probe_enabled_count"] == 1
    assert baseline["scheduler_stress_fixture"] is True
    assert baseline["stress_max_epochs"] == 2
    assert "clean scheduler state" in baseline["stress_profile_policy"]
    assert [action["action"] for action in actions] == ["SUBMIT"]
    assert list(jobs_by_id) == ["job-1"]

    job = jobs_by_id["job-1"]
    runner_kwargs = job["config"]["runner_kwargs"]
    assert job["task_type"] == "mlevolve_script"
    assert job["max_epochs"] == 2
    assert job["config"]["max_epochs"] == 2
    assert runner_kwargs["max_epochs"] == 2
    assert runner_kwargs["probe_max_epochs"] == 2
    assert "timeout" not in runner_kwargs
    assert archive_root.resolve() in Path(runner_kwargs["script_path"]).parents
    assert archive_root.resolve() in Path(job["baseline_model_path"]).parents
    assert Path(runner_kwargs["script_path"]).exists()
    assert Path(job["baseline_model_path"]).exists()
    assert "pre_archive_baseline_model_path" not in job
    assert job["batch_probe"]["enabled"] is True
    assert job["batch_probe"]["model_key"] == "unit-family"
    assert job["batch_probe"]["shape_hints"]["model_family"] == "unit-family"
    assert job["metadata"]["scheduler_stress_fixture"] is True
    assert job["metadata"]["scheduler_stress_max_epochs"] == 2
    assert (
        job["metadata"]["scheduler_stress_timeout_policy"]
        == "no_normal_execution_timeout"
    )
    assert settings["gpu_scheduler"]["batch_probe_enabled"] is True
    assert settings["gpu_scheduler"].get("parallel_job_cap") is None


def test_scheduler_replay_wrapper_quick_preset_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "scheduler_quick"

    result = _run_wrapper(
        [
            "bash",
            "scheduler_benchmark_test/run_histopath_scheduler_replay.sh",
            "--preset",
            "quick",
            "--dry-run",
            "--output-root",
            str(output),
            "--skip-plots",
        ]
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert "Preset: quick" in result.stdout
    assert "clean_scripts_completed" not in result.stdout
    assert "Runner mode: real" in result.stdout
    assert "Speedup: 100" in result.stdout
    assert "Wait for all: 1" in result.stdout
    assert "Cancel policy: ignore" in result.stdout
    assert "No sleep: 0" in result.stdout
    assert metrics["replay_dry_run"] is True
    assert metrics["replay_runner_mode"] == "real"
    assert metrics["replay_wait_for_all"] is True
    assert metrics["replay_cancel_policy"] == "ignore"
    assert metrics["replay_action_count"] == 38
    assert metrics["replay_submit_action_count"] == 38
    assert metrics["replay_cancel_action_count"] == 0


def test_scheduler_replay_wrapper_stress_preset_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "scheduler_stress"

    result = _run_wrapper(
        [
            "bash",
            "scheduler_benchmark_test/run_histopath_scheduler_replay.sh",
            "--preset",
            "stress",
            "--dry-run",
            "--output-root",
            str(output),
            "--skip-plots",
        ]
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert "Preset: stress" in result.stdout
    assert "stress_test_data" in result.stdout
    assert "scheduler_stress_2epoch" in result.stdout
    assert "Runner mode: real" in result.stdout
    assert "Speedup: 1" in result.stdout
    assert "Post-actions wait: 0" in result.stdout
    assert "Wait for all: 1" in result.stdout
    assert "Cancel policy: ignore" in result.stdout
    assert "No sleep: 1" in result.stdout
    assert "Clean scheduler state: 1" in result.stdout
    assert metrics["replay_dry_run"] is True
    assert metrics["replay_runner_mode"] == "real"
    assert metrics["replay_wait_for_all"] is True
    assert metrics["replay_cancel_policy"] == "ignore"
    assert metrics["replay_clean_scheduler_state"] is True
    assert metrics["replay_action_count"] == 12
    assert metrics["replay_submit_action_count"] == 12
    assert metrics["replay_cancel_action_count"] == 0

    fixture = (
        ROOT
        / "scheduler_benchmark_test"
        / "stress_test_data"
        / "histopathologic-cancer-detection_20260704_212842_scheduler_stress_2epoch"
    )
    _actions, jobs_by_id, _baseline, _settings = load_fixture(fixture)
    shape_hints_by_family: dict[str, set[str]] = {}
    counts_by_family: dict[str, int] = {}
    for job in jobs_by_id.values():
        family = job["metadata"]["model_family"]
        counts_by_family[family] = counts_by_family.get(family, 0) + 1
        shape_hints_by_family.setdefault(family, set()).add(
            json.dumps(job["batch_probe"]["shape_hints"], sort_keys=True)
        )
        assert "script_signature" not in job["batch_probe"]["shape_hints"]
    for family, count in counts_by_family.items():
        if count > 1:
            assert len(shape_hints_by_family[family]) == 1
    for source_path in (fixture / "sources").glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "_MlevolveProbeImageBackbone" not in source
        assert "_mlevolve_probe_or_load_automodel" not in source


def test_multiprocess_wrapper_quick_preset_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "baseline_quick"

    result = _run_wrapper(
        [
            "bash",
            "scheduler_benchmark_test/run_histopath_multiprocess_baseline.sh",
            "--preset",
            "quick",
            "--dry-run",
            "--output-root",
            str(output),
            "--skip-plots",
        ]
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert "Preset: quick" in result.stdout
    assert "clean_scripts_completed" not in result.stdout
    assert "Runner mode: real" in result.stdout
    assert "Job filter: all" in result.stdout
    assert "Speedup: 100" in result.stdout
    assert "Wait for all: 1" in result.stdout
    assert "Cancel policy: ignore" in result.stdout
    assert "No sleep: 0" in result.stdout
    assert metrics["replay_dry_run"] is True
    assert metrics["replay_runner_mode"] == "real"
    assert metrics["replay_wait_for_all"] is True
    assert metrics["replay_cancel_policy"] == "ignore"
    assert metrics["multiprocess_parallelism"] == 2
    assert metrics["multiprocess_job_filter"] == "all"
    assert metrics["replay_action_count"] == 38
    assert metrics["replay_submit_action_count"] == 38
    assert metrics["replay_cancel_action_count"] == 0
    assert metrics["replay_skipped_action_count"] == 0


def test_replay_wrapper_smoke_preset_sets_noop_no_sleep(tmp_path: Path) -> None:
    output = tmp_path / "scheduler_smoke"

    result = _run_wrapper(
        [
            "bash",
            "scheduler_benchmark_test/run_histopath_scheduler_replay.sh",
            "--preset",
            "smoke",
            "--dry-run",
            "--output-root",
            str(output),
            "--skip-plots",
        ]
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert "Runner mode: noop" in result.stdout
    assert "Post-actions wait: 5" in result.stdout
    assert "Wait for all: 0" in result.stdout
    assert "Cancel policy: replay" in result.stdout
    assert "No sleep: 1" in result.stdout
    assert metrics["replay_runner_mode"] == "noop"
    assert metrics["replay_action_count"] == 12


def test_replay_wrapper_explicit_flags_override_preset_defaults(tmp_path: Path) -> None:
    output = tmp_path / "scheduler_override"

    result = _run_wrapper(
        [
            "bash",
            "scheduler_benchmark_test/run_histopath_scheduler_replay.sh",
            "--speedup",
            "7",
            "--preset",
            "quick",
            "--runner-mode",
            "noop",
            "--post-actions-wait-seconds",
            "3",
            "--no-wait-for-all",
            "--cancel-policy",
            "replay",
            "--fixture-dir",
            "scheduler_benchmark_test/fixtures/histopathologic-cancer-detection_20260704_212842",
            "--dry-run",
            "--output-root",
            str(output),
            "--skip-plots",
        ]
    )

    metrics = json.loads(
        (output / "logs" / "comparison_metrics.json").read_text(encoding="utf-8")
    )
    assert "Runner mode: noop" in result.stdout
    assert "Speedup: 7" in result.stdout
    assert "Post-actions wait: 3" in result.stdout
    assert "Wait for all: 0" in result.stdout
    assert "Cancel policy: replay" in result.stdout
    assert "clean_scripts_completed" not in result.stdout
    assert metrics["replay_runner_mode"] == "noop"
    assert metrics["replay_wait_for_all"] is False
    assert metrics["replay_cancel_policy"] == "replay"
    assert metrics["replay_action_count"] == 44
    assert metrics["replay_submit_action_count"] == 38


def test_terminate_process_tree_stops_child_processes(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("process-group cleanup is POSIX-specific")
    marker = tmp_path / "child.pid"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')",
                "time.sleep(60)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(marker)], **start_new_session_kwargs()
    )
    try:
        _wait_for(marker.exists)
        child_pid = int(marker.read_text(encoding="utf-8"))
        assert _process_is_running(child_pid)

        terminate_process_tree(proc, timeout=0.5)

        assert proc.poll() is not None
        _wait_for(lambda: not _process_is_running(child_pid), timeout=3.0)
    finally:
        terminate_process_tree(proc, timeout=0.5)


def test_scheduler_stop_terminates_raw_script_child_tree(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("process-group cleanup is POSIX-specific")
    runtime_root = tmp_path / "runtime"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "child.pid"
    script = workspace / "candidate_spawn_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path('child.pid').write_text(str(child.pid), encoding='utf-8')",
                "time.sleep(60)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.02,
        gpu_scheduler={
            "packing_backend": "cuda_process",
            "exclusive_fallback_enabled": True,
            "cuda_process": {"enabled": False},
            "mps": {"enabled": False},
        },
        log_db={"enabled": False},
    )
    settings.ensure_runtime_layout()
    client = SchedulerClient(settings)
    service = client.create_service().start(background=True)
    child_pid: int | None = None
    try:
        job = TrainingJob.create(
            job_id="raw-spawn-child",
            runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            baseline_model_id="baseline-raw",
            baseline_model_path=str(script),
            runner_kwargs={
                "script_path": str(script),
                "working_dir": str(workspace),
                "result_path": str(
                    workspace / "working" / "scheduler_results" / "result_raw.json"
                ),
                "timeout": 60,
            },
            resource_requirements=ResourceRequirements(requires_gpu=False),
            batch_probe=BatchProbeSpec(enabled=False),
            runtime_probe=RuntimeProbeSpec(enabled=False),
        )
        client.submit(job)
        _wait_for(marker.exists, timeout=10.0)
        child_pid = int(marker.read_text(encoding="utf-8"))
        assert _process_is_running(child_pid)
    finally:
        service.stop()

    if child_pid is not None:
        _wait_for(lambda: not _process_is_running(child_pid), timeout=3.0)
