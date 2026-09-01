from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR = REPO_ROOT / "deployments" / "monitor_qwen38_a100_vllm.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _run_monitor(tmp_path: Path, *, healthy: bool, pid: int, marker: str) -> Path:
    calls = tmp_path / "bootstrap.calls"
    curl = tmp_path / "curl"
    bootstrap = tmp_path / "bootstrap"
    _write_executable(curl, f"exit {0 if healthy else 1}\n")
    _write_executable(bootstrap, f"printf 'called\\n' >> {calls!s}\n")
    pid_file = tmp_path / "server.pid"
    pid_file.write_text(f"{pid}\n")
    env = os.environ | {
        "RUN_ONCE": "1",
        "CURL_BIN": str(curl),
        "BOOTSTRAP_SCRIPT": str(bootstrap),
        "PID_FILE": str(pid_file),
        "PROCESS_MARKER": marker,
        "HEALTH_URL": "http://unused/health",
    }
    subprocess.run(["bash", str(MONITOR)], env=env, check=True)
    return calls


def test_restarts_only_when_server_process_is_dead(tmp_path: Path) -> None:
    calls = _run_monitor(
        tmp_path,
        healthy=False,
        pid=999_999_999,
        marker="vllm.entrypoints.cli.main",
    )
    assert calls.read_text().splitlines() == ["called"]


def test_does_not_restart_while_matching_server_is_loading(tmp_path: Path) -> None:
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        calls = _run_monitor(
            tmp_path,
            healthy=False,
            pid=sleeper.pid,
            marker="sleep 30",
        )
        assert not calls.exists()
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_does_not_restart_healthy_server(tmp_path: Path) -> None:
    calls = _run_monitor(
        tmp_path,
        healthy=True,
        pid=999_999_999,
        marker="vllm.entrypoints.cli.main",
    )
    assert not calls.exists()


def test_preserves_server_log_before_restarting_dead_process(tmp_path: Path) -> None:
    calls = tmp_path / "bootstrap.calls"
    crash_dir = tmp_path / "crashes"
    server_log = tmp_path / "server.log"
    server_log.write_text("CUDA unspecified launch failure\n")
    curl = tmp_path / "curl"
    bootstrap = tmp_path / "bootstrap"
    _write_executable(curl, "exit 1\n")
    _write_executable(bootstrap, f"printf 'called\\n' >> {calls!s}\n")
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("999999999\n")
    env = os.environ | {
        "RUN_ONCE": "1",
        "CURL_BIN": str(curl),
        "BOOTSTRAP_SCRIPT": str(bootstrap),
        "PID_FILE": str(pid_file),
        "PROCESS_MARKER": "vllm.entrypoints.cli.main",
        "HEALTH_URL": "http://unused/health",
        "SERVER_LOG": str(server_log),
        "CRASH_LOG_DIR": str(crash_dir),
    }
    subprocess.run(["bash", str(MONITOR)], env=env, check=True)
    saved_logs = list(crash_dir.glob("vllm-a100-crash-*.log"))
    assert len(saved_logs) == 1
    assert saved_logs[0].read_text() == server_log.read_text()
