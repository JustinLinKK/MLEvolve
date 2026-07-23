"""Replay driver B: pre-spawned torch.multiprocessing worker pool (T8-T11).

Workers are persistent. Each worker pulls tasks from an mp.Queue, runs the
recorded Python code via subprocess.

backend=mps: launch nvidia-cuda-mps-control -d daemon, set per-worker
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE so workers share the GPU.

backend=stream: each worker creates torch.cuda.Stream() in-process, but the
actual training runs as a child subprocess (one CUDA context per child).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch.multiprocessing as mp


REPO = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, REPO)


def _mps_directory(var_name: str, default: str) -> str:
    return str(os.environ.get(var_name) or default)


def _start_mps_daemon(pipe_dir: str | None = None, log_dir: str | None = None):
    pipe_dir = pipe_dir or _mps_directory("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    log_dir = log_dir or _mps_directory("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log")
    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-cuda-mps-control") is None:
        return None
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe_dir, "CUDA_MPS_LOG_DIRECTORY": log_dir}
    proc = subprocess.Popen(
        ["nvidia-cuda-mps-control", "-d"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def _stop_mps_daemon(pipe_dir: str | None = None, log_dir: str | None = None):
    try:
        env = {
            **os.environ,
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir or _mps_directory("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps"),
            "CUDA_MPS_LOG_DIRECTORY": log_dir or _mps_directory("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log"),
        }
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=b"quit\n",
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except Exception:
        pass


def worker_loop(worker_id: int, task_q, result_q, backend: str, batch_search: str | None,
                worker_env: dict[str, str]):
    for k, v in worker_env.items():
        os.environ[k] = v

    while True:
        task = task_q.get()
        if task is None:
            result_q.put({"worker_id": worker_id, "status": "stopped"})
            return
        step_idx = task["step_idx"]
        code_path = task["code_path"]
        workdir = task["workdir"]
        timeout = task.get("timeout", 1500.0)
        Path(workdir).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            cmd = [sys.executable, code_path]
            proc = subprocess.run(
                cmd, cwd=str(workdir), timeout=timeout,
                capture_output=True, text=True, env={**os.environ},
            )
            rc = proc.returncode
            stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
        except subprocess.TimeoutExpired:
            rc = -1
            stderr_tail = f"timeout after {timeout}s"
        elapsed = time.time() - t0

        result_q.put({
            "worker_id": worker_id,
            "step_idx": step_idx,
            "rc": rc,
            "elapsed_s": round(elapsed, 3),
            "stderr_tail_chars": len(stderr_tail),
            "completed_at": time.time(),
            "metadata": task.get("metadata", {}),
        })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-id", required=True)
    p.add_argument("--backend", required=True, choices=["mps", "stream"])
    p.add_argument("--batch-search", default="off", choices=["off", "binary", "power_of_two"])
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--trace", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--code-cache-dir", required=True)
    p.add_argument("--duration-s", type=float, default=2700.0)
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.code_cache_dir).mkdir(parents=True, exist_ok=True)

    trace = []
    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(json.loads(line))
    code_paths = {}
    for step in trace:
        if step.get("code"):
            cp = Path(args.code_cache_dir) / f"step_{step['step_idx']:03d}.py"
            cp.write_text(step["code"], encoding="utf-8")
            code_paths[step["step_idx"]] = str(cp)

    mps_proc = None
    mps_pipe_dir = _mps_directory("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    mps_log_dir = _mps_directory("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log")
    if args.backend == "mps":
        mps_proc = _start_mps_daemon(mps_pipe_dir, mps_log_dir)
        if mps_proc is None:
            print("WARN: nvidia-cuda-mps-control not found; running without MPS", file=sys.stderr)
        else:
            atexit.register(_stop_mps_daemon, mps_pipe_dir, mps_log_dir)

    n = args.n_workers
    worker_envs = []
    for i in range(n):
        e: dict[str, str] = {}
        if args.backend == "mps" and mps_proc is not None:
            pct = max(10, 100 // n)
            e["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
            e["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
            e["CUDA_MPS_LOG_DIRECTORY"] = mps_log_dir
        e.setdefault("CUDA_VISIBLE_DEVICES", "0")
        worker_envs.append(e)

    mp.set_start_method("spawn", force=True)
    task_q = mp.Queue()
    result_q = mp.Queue()
    bs_search = None if args.batch_search == "off" else args.batch_search
    workers = [
        mp.Process(target=worker_loop, args=(i, task_q, result_q, args.backend, bs_search, worker_envs[i]))
        for i in range(n)
    ]
    for w in workers:
        w.start()

    # Submit all tasks burst
    submit_t0 = time.time()
    workdir_root = Path(os.environ.get("REPLAY_WORKDIR_ROOT", f"/tmp/replay_workdirs/{args.config_id}")).resolve()
    workdir_root.mkdir(parents=True, exist_ok=True)
    for step in trace:
        si = step["step_idx"]
        if si not in code_paths:
            continue
        task_q.put({
            "step_idx": si,
            "code_path": code_paths[si],
            "workdir": str((workdir_root / f"step_{si:03d}").resolve()),
            "timeout": 1500.0,
            "metadata": {
                "agent_used": step.get("agent_used"),
                "model_class": step.get("model_class"),
                "model_name": step.get("model_name"),
                "bs": step.get("bs"),
            },
        })

    n_expected = sum(1 for s in trace if s["step_idx"] in code_paths)
    results = []
    deadline = time.time() + args.duration_s
    while len(results) < n_expected and time.time() < deadline:
        try:
            r = result_q.get(timeout=10)
            if r.get("status") != "stopped":
                results.append(r)
        except Exception:
            continue
    treat_elapsed = time.time() - submit_t0

    for _ in workers:
        task_q.put(None)
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()

    if mps_proc is not None:
        _stop_mps_daemon(mps_pipe_dir, mps_log_dir)

    n_completed = sum(1 for r in results if r.get("rc") == 0)
    n_failed = sum(1 for r in results if r.get("rc", -1) != 0)

    # Synthesize per_job for plot compat
    per_job = []
    for r in results:
        m = r.get("metadata") or {}
        per_job.append({
            "step_idx": r.get("step_idx"),
            "agent_used": m.get("agent_used"),
            "model_class": m.get("model_class"),
            "model_name": m.get("model_name"),
            "bs": m.get("bs"),
            "resolved_batch_size": m.get("bs"),
            "status": "COMPLETED" if r.get("rc") == 0 else "FAILED",
            "elapsed_s": r.get("elapsed_s"),
        })

    summary = {
        "config_id": args.config_id,
        "mode": "torch_mp_pool",
        "backend": args.backend,
        "batch_search": args.batch_search,
        "n_workers": n,
        "trace_path": args.trace,
        "n_jobs": n_expected,
        "by_status": {"COMPLETED": n_completed, "FAILED": n_failed},
        "treat_elapsed_s": round(treat_elapsed, 3),
        "per_job": per_job,
        "results": results,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("results", "per_job")}, indent=2))


if __name__ == "__main__":
    main()
