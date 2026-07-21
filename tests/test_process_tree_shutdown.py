from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from engine.executor import Interpreter
from localml_scheduler.execution.process_utils import signal_process_tree, start_new_session_kwargs


def _running_non_zombie(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    stat = result.stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def test_signal_process_tree_reaches_child_processes(tmp_path) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, time\n"
        f"child_pid_path = pathlib.Path({str(child_pid_path)!r})\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        "child_pid_path.write_text(str(child.pid), encoding='utf-8')\n"
        "try:\n"
        "    time.sleep(60)\n"
        "finally:\n"
        "    child.terminate()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        **start_new_session_kwargs(),
    )
    try:
        deadline = time.time() + 5
        while not child_pid_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _running_non_zombie(child_pid)

        signal_process_tree(proc, signal.SIGTERM)
        proc.wait(timeout=5)

        deadline = time.time() + 5
        while _running_non_zombie(child_pid) and time.time() < deadline:
            time.sleep(0.05)
        assert not _running_non_zombie(child_pid)
    finally:
        if proc.poll() is None:
            signal_process_tree(proc, signal.SIGKILL)
            proc.wait(timeout=5)


def test_interpreter_terminate_all_subprocesses_uses_process_tree(monkeypatch, tmp_path) -> None:
    interpreter = Interpreter(tmp_path, max_parallel_run=1)

    class FakeProc:
        pid = os.getpid()

        def poll(self):
            return None

    fake_proc = FakeProc()
    calls = []
    monkeypatch.setattr("engine.executor.terminate_process_tree", lambda proc, timeout=2.0: calls.append((proc, timeout)))

    with interpreter._procs_lock:
        interpreter._active_procs[0] = fake_proc

    interpreter.terminate_all_subprocesses()

    assert calls == [(fake_proc, 2.0)]
    assert interpreter._active_procs == {}
