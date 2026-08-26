"""Replay a Poisson job trace under three execution conditions and record metrics.

  mp_only            no scheduler; multiprocessing with a fixed parallel cap
  scheduler_profile  localml_scheduler, prediction mode = branch_profile
  scheduler_ml       localml_scheduler, prediction mode = ml_predictor
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class GpuSampler:
    def __init__(self, index: int = 0, interval: float = 0.5) -> None:
        self.index = index
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.available = True
        except Exception:
            self._nvml = None
            self._handle = None

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append({"t": time.time(), "mem_used_mib": mem.used / (1024 * 1024),
                    "power_w": power_mw / 1000.0, "sm_util": float(util.gpu), "mem_util": float(util.memory)})
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if not self.samples:
            return {}
        def agg(key):
            v = [s[key] for s in self.samples]
            return {f"avg_{key}": sum(v) / len(v), f"peak_{key}": max(v), f"min_{key}": min(v)}
        out: dict[str, float] = {"sample_count": len(self.samples), "sample_interval_s": self.interval}
        for key in ("mem_used_mib", "power_w", "sm_util", "mem_util"):
            out.update(agg(key))
        energy = 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            energy += (a["power_w"] + b["power_w"]) / 2.0 * (b["t"] - a["t"])
        out["energy_wh"] = energy / 3600.0
        return out


def idle_gpu_baseline(index: int = 0, seconds: float = 3.0) -> dict[str, float]:
    sampler = GpuSampler(index=index, interval=0.2)
    sampler.start()
    time.sleep(seconds)
    return sampler.stop()


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_mp_only(trace, outdir: Path, *, max_parallel: int, python: str) -> dict:
    spec_dir = outdir / "specs"
    result_dir = outdir / "results"
    spec_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict] = {}
    running: list[tuple[subprocess.Popen, dict, float]] = []
    lock = threading.Lock()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def reap(block: bool) -> None:
        while True:
            with lock:
                still = []
                for proc, step, submitted in running:
                    if proc.poll() is None:
                        still.append((proc, step, submitted))
                        continue
                    idx = int(step["step_idx"])
                    finished = time.time()
                    payload = {"status": "failed" if proc.returncode else "succeeded"}
                    rpath = result_dir / f"step_{idx:03d}.json"
                    if rpath.is_file():
                        payload.update(json.loads(rpath.read_text()))
                    records[idx].update({"finished_at": finished, "returncode": proc.returncode,
                        "status": payload["status"], "training_seconds": payload.get("training_seconds"),
                        "startup_seconds": payload.get("startup_seconds"),
                        "peak_reserved_mib": payload.get("peak_reserved_mib"),
                        "peak_allocated_mib": payload.get("peak_allocated_mib")})
                running[:] = still
            if not block or len(running) < max_parallel:
                return
            time.sleep(0.1)

    t0 = time.time()
    for step in trace:
        idx = int(step["step_idx"])
        target = t0 + float(step["arrival_offset_s"])
        while time.time() < target:
            reap(block=False)
            time.sleep(min(0.05, max(0.0, target - time.time())))
        reap(block=False)
        while len(running) >= max_parallel:
            reap(block=True)
        spec_path = spec_dir / f"step_{idx:03d}.json"
        spec_path.write_text(json.dumps(step))
        result_path = result_dir / f"step_{idx:03d}.json"
        started = time.time()
        proc = subprocess.Popen(
            [python, "-m", "scheduler_benchmark_test.stress_bench.mp_worker",
             "--spec", str(spec_path), "--result", str(result_path)],
            cwd=str(REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        records[idx] = {"step_idx": idx, "job_id": step["job_id"], "architecture": step["architecture"],
            "family": step["family"], "precision": step["precision"],
            "arrival_offset_s": float(step["arrival_offset_s"]),
            "arrived_at": t0 + float(step["arrival_offset_s"]), "dispatched_at": started}
        with lock:
            running.append((proc, step, started))
    while running:
        reap(block=True)
        time.sleep(0.1)
    return {"t0": t0, "per_job": [records[k] for k in sorted(records)]}


def run_scheduler(trace, outdir: Path, *, prediction_mode: str, max_parallel: int,
                  gpu_vram_gib: float, timeout_s: float, runtime_root: Path,
                  ) -> dict:
    from localml_scheduler.adapters.mlevolve import build_mlevolve_job
    from localml_scheduler.client import SchedulerClient
    from localml_scheduler.config import (
        GpuMemorySettings, GpuProfilingSettings, GpuSchedulerSettings,
        SchedulerSettings)
    from localml_scheduler.domain import CheckpointPolicy, ResourceRequirements
    from scheduler_benchmark_test.stress_bench.stress_runner import make_baseline_checkpoint

    runtime_root = Path(runtime_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    baseline_dir = outdir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    result_dir = outdir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    gpu = GpuSchedulerSettings()
    gpu.mode = "parallel_time_aware"
    gpu.packing_backend = "cuda_process"
    gpu.exclusive_fallback_enabled = True
    # VRAM constrains admission only; timing evidence determines placement value.
    gpu.memory = GpuMemorySettings(gpu_vram_gib=gpu_vram_gib,
        predicted_budget_fraction=0.85, live_admission_stop_fraction=0.92,
        live_admission_resume_fraction=0.87)
    gpu.profiling = GpuProfilingSettings(warmup_steps=3, solo_probe_steps=6, pair_probe_steps=4,
        reuse_profile_if_confidence_ge=0.8)
    gpu.parallel_job_cap = max_parallel
    gpu.batch_probe_enabled = False

    settings = SchedulerSettings(
        runtime_root=runtime_root, scheduler_poll_interval_seconds=0.2, gpu_scheduler=gpu,
        prediction={"mode": prediction_mode}, graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
        baseline_cache={"warm_queue_policy": "top_k", "warm_queue_top_k": 4, "entry_capacity": 8,
            "max_ram_percent": 0.2, "memory_budget_bytes": 8 * (1024 ** 3)})

    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)
    submitted: dict[str, dict] = {}
    t0 = time.time()
    try:
        for step in trace:
            idx = int(step["step_idx"])
            target = t0 + float(step["arrival_offset_s"])
            while time.time() < target:
                time.sleep(min(0.05, max(0.0, target - time.time())))
            baseline_path = baseline_dir / f"{step['job_id']}.pt"
            if not baseline_path.is_file():
                make_baseline_checkpoint(baseline_path, {"constructor_kwargs": step["constructor_kwargs"]})
            job = build_mlevolve_job(
                workflow_id=f"stress-poisson-{prediction_mode}", baseline_model_id=str(step["job_id"]),
                baseline_model_path=str(baseline_path),
                runner_target="scheduler_benchmark_test.stress_bench.stress_runner:run_stress_job",
                runner_kwargs={"source_path": step["source_path"], "constructor_kwargs": step["constructor_kwargs"],
                    "input_shape": step["input_shape"], "precision": step["precision"],
                    "epochs": int(step["epochs"]), "batches_per_epoch": int(step["batches_per_epoch"]),
                    "stream_data": bool(step.get("stream_data")), "result_dir": str(result_dir)},
                priority=5, task_type="stress",
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=None, save_every_epoch=False),
                resource_requirements=ResourceRequirements(requires_gpu=True, gpu_slots=1,
                    estimated_vram_mb=(int(step["estimated_vram_mb"]) if step.get("estimated_vram_mb") else None)),
                packing_family=str(step["family"]), packing_signature=str(step["packing_signature"]),
                packing_eligible=True, max_epochs=int(step["epochs"]),
                metadata={"step_idx": idx, "architecture": step["architecture"], "family": step["family"],
                    "precision": step["precision"], "perfseer_model": {"source_path": step["source_path"],
                        "entry": step["entry"], "input_shapes": [step["input_shape"]],
                        "input_dtypes": step["input_dtypes"], "precision": step["precision"],
                        "constructor_kwargs": step["constructor_kwargs"]}})
            job = api.submit(job)
            submitted[job.job_id] = {"step_idx": idx, "job_id": step["job_id"], "scheduler_job_id": job.job_id,
                "architecture": step["architecture"], "family": step["family"], "precision": step["precision"],
                "arrival_offset_s": float(step["arrival_offset_s"]), "arrived_at": target}
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            jobs = api.list_jobs()
            if jobs and all(j.status.is_terminal for j in jobs):
                break
            time.sleep(1.0)
        for job in api.list_jobs():
            record = submitted.get(job.job_id)
            if record is None:
                continue
            sidecar = result_dir / f"{job.job_id}.json"
            runner_result = json.loads(sidecar.read_text()) if sidecar.is_file() else None
            record.update({"status": job.status.value, "started_at_iso": job.started_at,
                "finished_at_iso": job.finished_at, "result": runner_result,
                "training_seconds": (runner_result or {}).get("training_seconds"),
                "peak_reserved_mib": (runner_result or {}).get("peak_reserved_mib"),
                "peak_allocated_mib": (runner_result or {}).get("peak_allocated_mib"),
                "placement_backend": (job.metadata or {}).get("placement_backend"),
                "metadata": job.metadata})
    finally:
        try:
            service.stop()
        except Exception:
            pass
    events: list[dict] = []
    events_path = settings.logs_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(errors="ignore").splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return {"t0": t0, "per_job": [submitted[k] for k in submitted], "events": events,
            "runtime_root": str(runtime_root)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["mp_only", "scheduler_profile", "scheduler_ml"])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--gpu-vram-gib", type=float, default=20.0)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runtime-root", default=None)
    args = ap.parse_args()
    trace = load_trace(Path(args.trace))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idle = idle_gpu_baseline()
    sampler = GpuSampler(interval=0.5)
    sampler.start()
    wall_start = time.time()
    if args.condition == "mp_only":
        payload = run_mp_only(trace, outdir, max_parallel=args.max_parallel, python=args.python)
    else:
        mode = "branch_profile" if args.condition == "scheduler_profile" else "ml_predictor"
        payload = run_scheduler(trace, outdir, prediction_mode=mode, max_parallel=args.max_parallel,
            gpu_vram_gib=args.gpu_vram_gib, timeout_s=args.timeout_s,
            runtime_root=Path(args.runtime_root or f"/tmp/sb_{args.condition}"))
    wall_end = time.time()
    gpu = sampler.stop()
    summary = {"condition": args.condition, "trace": str(args.trace), "job_count": len(trace),
        "max_parallel": args.max_parallel, "wall_seconds": wall_end - wall_start, "gpu": gpu,
        "gpu_idle_baseline": idle}
    payload["summary"] = summary
    (outdir / "raw.json").write_text(json.dumps(payload, indent=2, default=str))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
