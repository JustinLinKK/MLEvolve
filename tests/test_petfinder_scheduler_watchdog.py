from __future__ import annotations

import calendar
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = REPO_ROOT / "deployments" / "monitor_petfinder_scheduler_progress.sh"


def _make_state(tmp_path: Path, pid: int) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run-root"
    phase = "scheduler_profile_hkwd"
    state_dir = tmp_path / "state"
    watchdog_state = tmp_path / "watchdog-state"
    run_root.mkdir()
    state_dir.mkdir()
    watchdog_state.mkdir()
    (state_dir / "comparison_root").write_text(f"{run_root}\n")
    (state_dir / "active_phase").write_text(f"{phase}\n")
    (state_dir / "scheduler.pid").write_text(f"{pid}\n")
    (run_root / f"{phase}.runner.out").write_text("scheduler started\n")
    (watchdog_state / "monitored_root").write_text(f"{run_root}\n")
    return run_root, state_dir, watchdog_state


def _run_once(state_dir: Path, watchdog_state: Path, now: int) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "RUN_ONCE": "1",
        "STATE_DIR": str(state_dir),
        "WATCHDOG_STATE_DIR": str(watchdog_state),
        "STALL_SECONDS": "60",
        "NOW_EPOCH": str(now),
        "STOP_GRACE_SECONDS": "2",
    }
    return subprocess.run(
        ["bash", str(WATCHDOG)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_stops_only_verified_scheduler_after_no_effective_node_progress(tmp_path: Path) -> None:
    marker = f"petfinder_scheduler_profile_hkwd_test_{os.getpid()}"
    process = subprocess.Popen(["bash", "-c", f"exec -a '{marker} run.py' sleep 60"])
    try:
        run_root, state_dir, watchdog_state = _make_state(tmp_path, process.pid)
        now = int(time.time())
        (watchdog_state / "last_count").write_text("0\n")
        (watchdog_state / "last_progress_epoch").write_text(f"{now - 61}\n")

        result = _run_once(state_dir, watchdog_state, now)

        assert result.returncode == 42, result.stderr
        process.wait(timeout=5)
        assert (state_dir / "status").read_text().strip() == "stalled_stopped"
        event = json.loads((watchdog_state / "STALL_DETECTED.json").read_text())
        assert event["effective_nodes"] == 0
        assert event["scheduler_pid"] == process.pid
        assert event["reason"] == "no_effective_node_progress"
        assert list((run_root / "watchdog_stalls").glob("*/diagnostics.txt"))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_refuses_to_stop_an_unrelated_process(tmp_path: Path) -> None:
    process = subprocess.Popen(["sleep", "60"])
    try:
        _, state_dir, watchdog_state = _make_state(tmp_path, process.pid)
        now = int(time.time())
        (watchdog_state / "last_count").write_text("0\n")
        (watchdog_state / "last_progress_epoch").write_text(f"{now - 61}\n")

        result = _run_once(state_dir, watchdog_state, now)

        assert result.returncode == 43
        assert process.poll() is None
        assert "refusing to stop unverified process" in result.stderr
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_new_watchdog_uses_run_start_as_initial_no_progress_time(tmp_path: Path) -> None:
    marker = f"petfinder_scheduler_profile_hkwd_initial_{os.getpid()}"
    process = subprocess.Popen(["bash", "-c", f"exec -a '{marker} run.py' sleep 60"])
    try:
        run_root, state_dir, watchdog_state = _make_state(tmp_path, process.pid)
        (watchdog_state / "monitored_root").unlink()
        run_started = calendar.timegm(datetime(2026, 9, 1, 23, 12, 43).timetuple())
        (run_root / "scheduler_profile_hkwd.runner.out").write_text(
            '[2026-09-01 23:12:43,145] INFO: Starting run "petfinder"\n'
        )

        result = _run_once(state_dir, watchdog_state, run_started + 61)

        assert result.returncode == 42, result.stderr
        process.wait(timeout=5)
        event = json.loads((watchdog_state / "STALL_DETECTED.json").read_text())
        assert event["seconds_without_progress"] == 61
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_live_scheduler_job_heartbeat_prevents_false_stall(tmp_path: Path) -> None:
    marker = f"petfinder_scheduler_profile_hkwd_heartbeat_{os.getpid()}"
    process = subprocess.Popen(["bash", "-c", f"exec -a '{marker} run.py' sleep 60"])
    try:
        run_root, state_dir, watchdog_state = _make_state(tmp_path, process.pid)
        now = int(time.time())
        (watchdog_state / "last_count").write_text("0\n")
        (watchdog_state / "last_progress_epoch").write_text(f"{now - 61}\n")
        job_dir = (
            run_root
            / "scheduler_profile_hkwd"
            / "scheduler_runtime"
            / "data"
            / "jobs"
            / "job-1"
        )
        job_dir.mkdir(parents=True)
        heartbeat = job_dir / "heartbeat.json"
        heartbeat.write_text('{"epoch": 2}\n')
        os.utime(heartbeat, (now, now))

        result = _run_once(state_dir, watchdog_state, now)

        assert result.returncode == 0, result.stderr
        assert process.poll() is None
        assert not (watchdog_state / "STALL_DETECTED.json").exists()
        assert (watchdog_state / "last_progress_epoch").read_text().strip() == str(now)
    finally:
        process.terminate()
        process.wait(timeout=5)
