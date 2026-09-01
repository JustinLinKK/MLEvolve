from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "deployments" / "qwen38_a100_container_entrypoint.sh"


def _write_step(path: Path, label: str, *, stay_foreground: bool = False) -> None:
    body = ["#!/usr/bin/env bash", "set -euo pipefail", f"printf '%s\\n' {label!r} >> \"$TRACE_FILE\""]
    if stay_foreground:
        body.append("printf '%s\\n' \"$PPID\" > \"$MONITOR_PARENT_FILE\"")
    path.write_text("\n".join(body) + "\n")
    path.chmod(0o755)


def test_entrypoint_initializes_bootstraps_then_executes_monitor(tmp_path: Path) -> None:
    trace = tmp_path / "trace.txt"
    monitor_parent = tmp_path / "monitor-parent.txt"
    init = tmp_path / "init.sh"
    bootstrap = tmp_path / "bootstrap.sh"
    monitor = tmp_path / "monitor.sh"
    _write_step(init, "init")
    _write_step(bootstrap, "bootstrap")
    _write_step(monitor, "monitor", stay_foreground=True)

    env = os.environ | {
        "INIT_SCRIPT": str(init),
        "BOOTSTRAP_SCRIPT": str(bootstrap),
        "MONITOR_SCRIPT": str(monitor),
        "TRACE_FILE": str(trace),
        "MONITOR_PARENT_FILE": str(monitor_parent),
    }
    result = subprocess.run(["bash", str(ENTRYPOINT)], env=env, check=False)

    assert result.returncode == 0
    assert trace.read_text().splitlines() == ["init", "bootstrap", "monitor"]
    assert monitor_parent.read_text().strip() == str(os.getpid())
