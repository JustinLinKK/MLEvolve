"""Measure N-way packing of real MLEvolve-generated training scripts.

Takes scripts the agent actually produced (extracted from a recorded trace) and
runs N concurrent copies on one GPU, each in its own working directory so their
submission/checkpoint writes cannot collide.

Reports per-copy wall time, the resulting slowdown against the solo run, and
device SM utilization, which is what decides whether 4-5 of these jobs fill the
GPU's compute.

Usage:
    python measure_script_packing.py <script.py> [--max-n 5] [--repeats 1]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

GATE = 1.15


def sample_sm(seconds: float, stop_event: threading.Event) -> dict:
    """Average/peak SM% and VRAM over a window via nvidia-smi dmon."""
    proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    sm_values: list[int] = []
    start = time.time()
    try:
        while not stop_event.is_set() and time.time() - start < seconds:
            line = proc.stdout.readline()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and re.fullmatch(r"\d+", parts[1]):
                sm_values.append(int(parts[1]))
    finally:
        proc.terminate()
    return {
        "avg_sm": round(sum(sm_values) / len(sm_values), 1) if sm_values else None,
        "peak_sm": max(sm_values) if sm_values else None,
        "samples": len(sm_values),
    }


def run_n(script: Path, n: int, python_bin: str, timeout: int, data_dir: Path | None = None) -> dict:
    """Run n copies of `script` concurrently, each in an isolated cwd.

    Agent scripts read their data through a relative `./input`, so when a data
    directory is supplied it is linked into every working directory.

    Returns per-copy wall times and device SM utilization for the window.
    """
    workdirs = [Path(tempfile.mkdtemp(prefix=f"pack_{n}_{i}_")) for i in range(n)]
    if data_dir is not None:
        for workdir in workdirs:
            (workdir / "input").symlink_to(data_dir)
            (workdir / "working").mkdir(exist_ok=True)
            (workdir / "submission").mkdir(exist_ok=True)
    procs = []
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    }

    stop_event = threading.Event()
    sm_result: dict = {}

    def sampler():
        sm_result.update(sample_sm(timeout, stop_event))

    sm_thread = threading.Thread(target=sampler, daemon=True)

    started = time.time()
    sm_thread.start()
    for workdir in workdirs:
        target = workdir / script.name
        shutil.copy(script, target)
        procs.append(
            subprocess.Popen(
                [python_bin, "-u", str(target)],
                cwd=workdir, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        )

    wall_times = []
    failures = 0
    for proc in procs:
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate()
            failures += 1
        if proc.returncode != 0:
            failures += 1
        wall_times.append(time.time() - started)

    stop_event.set()
    sm_thread.join(timeout=5)
    for workdir in workdirs:
        shutil.rmtree(workdir, ignore_errors=True)

    return {
        "n": n,
        "wall_s": round(max(wall_times), 2),
        "mean_wall_s": round(sum(wall_times) / len(wall_times), 2),
        "failures": failures,
        **sm_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    print(f"script: {args.script}")
    rows = []
    baseline = None
    for n in range(1, args.max_n + 1):
        row = run_n(args.script, n, args.python, args.timeout, args.data_dir)
        if n == 1:
            baseline = row["mean_wall_s"]
        row["slowdown"] = round(row["mean_wall_s"] / baseline, 3) if baseline else None
        row["pass"] = row["slowdown"] is not None and row["slowdown"] <= GATE
        rows.append(row)
        print(
            f"  N={n}: wall={row['mean_wall_s']:8.2f}s  "
            f"slowdown={row['slowdown']}x  "
            f"SM={row.get('avg_sm')}%  peakSM={row.get('peak_sm')}%  "
            f"failures={row['failures']}  {'PASS' if row['pass'] else 'FAIL'}",
            flush=True,
        )

    print()
    print("JSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
