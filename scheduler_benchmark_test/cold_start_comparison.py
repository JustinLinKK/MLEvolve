"""Bounded cold-profile CUDA comparison; simulated input gaps, not Petfinder quality.

Run: python -m scheduler_benchmark_test.cold_start_comparison --output runs/cold-start-5090
Use --cpu-smoke to exercise the same controller without allocating GPU memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sqlite3
import subprocess
import sys
import time
import shutil

import torch

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import CheckpointPolicy, PackingSpec, ResourceRequirements, TrainingJob


MODELS = [(256, "relu"), (384, "gelu"), (512, "silu"), (768, "tanh")]
GAP = 0.002


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def gpu_state():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True, timeout=5)
    name, total, used, busy = result.stdout.strip().splitlines()[0].split(", ")
    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=5).stdout
    return {"name": name, "total_mib": float(total), "used_mib": float(used), "busy_percent": float(busy), "processes": processes}


def calibrate(path):
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.10)
    samples = []
    for width, activation in MODELS:
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(1024, width), {
            "relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "tanh": torch.nn.Tanh,
        }[activation](), torch.nn.Linear(width, 1)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x = torch.randn(64, 1024, device="cuda")
        for step in range(26):
            if step == 2:
                torch.cuda.synchronize()
                started = time.perf_counter()
            time.sleep(GAP)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x).float().square().mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / 24)
        del model, optimizer, x, loss
        torch.cuda.empty_cache()
    steps = max(64, min(1024, round(2.0 / statistics.mean(samples))))
    write_json(path, {"step_seconds": samples, "steps_per_epoch": steps, "scheduler_received_timings": False})


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def draw(root, results):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    columns = min(4, len(results))
    groups = math.ceil(len(results) / columns)
    fig, axes = plt.subplots(2 * groups, columns, figsize=(max(7, 5 * columns), 7 * groups), squeeze=False)
    for number, result in enumerate(results):
        row, column = divmod(number, columns)
        axis, metric = axes[row, column], axes[groups + row, column]
        start = result["started_at"]
        events = result["events"]
        for index, job in enumerate(result["jobs"]):
            opened = None
            trial = None
            yielded = None
            for event in events:
                if event["job_id"] != job["job_id"]:
                    continue
                when = timestamp(event["created_at"]) - start
                kind = event["event_type"]
                if kind == "job_dispatched":
                    opened = when
                if kind in {"job_paused", "job_completed", "job_failed"} and opened is not None:
                    axis.barh(index, when - opened, left=opened, height=0.6, color=f"C{index}", alpha=0.65, zorder=1)
                    opened = None
                if kind == "colocation_trial_started":
                    trial = when
                if kind == "cold_start_trial_decided" and trial is not None:
                    axis.barh(index, when - trial, left=trial, height=0.68, fill=False, edgecolor="black", hatch="///", zorder=2)
                    trial = None
                if kind == "trial_worker_yielded":
                    yielded = when
                if kind == "trial_worker_released" and yielded is not None:
                    axis.barh(index, when - yielded, left=yielded, height=0.65, color="#6d28d9", zorder=3)
                    yielded = None
            if opened is not None:
                axis.barh(index, result["makespan_seconds"] - opened, left=opened, height=0.6, color=f"C{index}")
        axis.set_yticks(range(len(result["jobs"])), [job["job_id"] for job in result["jobs"]])
        axis.set_xlabel("Seconds since common arrival")
        axis.set_title(f'{result["name"]}\n{result["makespan_seconds"]:.2f}s; {result["updates_per_second"]:.1f} updates/s')
        axis.grid(axis="x", alpha=0.2)
        metric.plot(range(1, len(result["jobs"]) + 1), [job["rmse"] for job in result["jobs"]], "o-")
        metric.set_xticks(range(1, len(result["jobs"]) + 1))
        metric.set_xlabel("Model / node")
        metric.set_ylabel("Final synthetic RMSE")
        metric.grid(alpha=0.2)
    for unused in range(len(results), groups * columns):
        row, column = divmod(unused, columns)
        axes[row, column].set_visible(False)
        axes[groups + row, column].set_visible(False)
    fig.legend(handles=[Patch(facecolor="C0", alpha=0.65, label="Worker lifetime"), Patch(fill=False, hatch="///", label="Addition trial (includes useful work)"), Patch(color="#6d28d9", label="In-memory yield")], loc="lower center", ncol=3)
    fig.suptitle("Cold-profile comparison | 3 epochs | BF16 compute, FP32 AdamW | simulated 2 ms input gap")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)


def run_arm(root, name, steps, deadline, *, cpu=False):
    path = root / name
    path.mkdir()
    trial = "trial" in name
    settings = SchedulerSettings(
        runtime_root=path / "runtime", scheduler_poll_interval_seconds=0.05,
        graph_db={"enabled": False}, log_db={"enabled": False},
        hardware_knowledge_graph={"enabled": False}, hardware_feature_db={"enabled": False},
        baseline_cache={"warm_queue_top_k": 0},
        gpu_scheduler={"enabled": trial, "packing_backend": "cuda_process", "memory": {"gpu_vram_gib": 31},
                       "batch_probe_enabled": False, "colocation": {"trial_epochs": 1, "decision_replay": {"enabled": False}}},
    )
    client = SchedulerClient(settings)
    store = client.store
    with sqlite3.connect(settings.db_path) as connection:
        counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("runtime_profiles", "solo_profiles", "pair_profiles", "colocation_timing_profiles")}
    write_json(path / "empty_profiles.json", counts)
    assert not any(counts.values())
    baseline = path / "baseline.pt"
    torch.save({}, baseline)
    jobs = []
    for index, (width, activation) in enumerate(MODELS[:2] if cpu else MODELS):
        job = TrainingJob.create(
            "localml_scheduler.examples.cold_start_runner:run_training", f"model-{index}", str(baseline),
            job_id=f"node-{index + 1}", max_epochs=3, seed=42 + index,
            python_executable=sys.executable,
            runner_kwargs={"width": width if not cpu else 32, "activation": activation, "batch_size": 64 if not cpu else 8,
                           "steps_per_epoch": steps, "seed": 42 + index, "input_delay_seconds": GAP},
            resource_requirements=ResourceRequirements(requires_gpu=not cpu),
            packing=PackingSpec(eligible=True, signature=f"synthetic-{activation}-{width}", family=f"synthetic-{activation}", backend_allowlist=["cuda_process"]),
            checkpoint_policy=CheckpointPolicy(save_every_epoch=False, keep_last_n=8),
            metadata={"cooperative_trial": True, "precision_mode": "bf16", "experiment_mode": "origin", "skip_active_scheduler_probes": True},
        )
        jobs.append(job)
    arrived = datetime.now(timezone.utc).isoformat()
    for job in jobs:
        job.submitted_at = arrived
        client.submit(job)
    write_json(path / "jobs.json", [job.to_dict() for job in jobs])
    write_json(path / "settings.json", settings.to_dict())
    service = client.create_service()
    started = time.time()
    telemetry, progress = [], []
    next_telemetry = 0
    service.start(background=True)
    try:
        while True:
            current = [store.get_job(job.job_id) for job in jobs]
            progress.append({"at": time.time(), "jobs": {job.job_id: {
                "step": job.metadata.get("runtime_global_step", 0), "status": job.status.value,
                "reserved_mb": job.metadata.get("trial_reserved_mb", 0),
            } for job in current}})
            if all(job.status.is_terminal for job in current):
                break
            if time.time() >= deadline:
                raise TimeoutError("Five-minute GPU-test budget exhausted")
            if not service._thread.is_alive():
                raise RuntimeError("Scheduler controller exited")
            if not cpu and time.time() >= next_telemetry:
                telemetry.append({"at": time.time(), **gpu_state()})
                next_telemetry = time.time() + 0.5
            time.sleep(0.05)
    finally:
        service.stop()
        (path / "progress.jsonl").write_text("".join(json.dumps(item) + "\n" for item in progress))
        write_json(path / "telemetry.json", telemetry)
    events = [json.loads(line) for line in settings.events_jsonl_path.read_text().splitlines()]
    end = max(timestamp(event["created_at"]) for event in events if event["event_type"] in {"job_completed", "job_failed"})
    makespan = end - started
    records, startup = [], []
    for job in current:
        assert job.status.value == "COMPLETED", (job.job_id, job.status, job.to_dict())
        checkpoint = torch.load(job.latest_checkpoint_path, map_location="cpu", weights_only=False)["state"]
        assert checkpoint["global_step"] == 3 * steps
        assert all(int(item["step"]) == 3 * steps for item in checkpoint["optimizer_state"]["state"].values())
        assert all(torch.isfinite(value).all() for value in checkpoint["model_state"].values())
        completion = next(event for event in events if event["job_id"] == job.job_id and event["event_type"] == "job_completed")
        rmse = float(completion["payload"]["rmse"])
        assert math.isfinite(rmse)
        job_events = [event for event in events if event["job_id"] == job.job_id]
        starts = [timestamp(event["created_at"]) for event in job_events if event["event_type"] == "job_dispatched"]
        ready = [timestamp(event["created_at"]) for event in job_events if event["event_type"] == "trial_runner_ready"]
        startup.extend(max(0, r - s) for r, s in zip(ready, starts))
        records.append({"job_id": job.job_id, "rmse": rmse, "steps": checkpoint["global_step"], "queue_wait_seconds": min(starts) - started,
                        "control_seconds": job.metadata.get("trial_control_seconds", 0), "sync_seconds": job.metadata.get("trial_sync_seconds", 0),
                        "peak_allocator_mib": job.metadata.get("trial_peak_reserved_mb", 0)})
    active_training = []
    for previous, point in zip(progress, progress[1:]):
        active_training.append(sum(point["jobs"][job.job_id]["step"] > previous["jobs"][job.job_id]["step"] for job in jobs))
    decisions = [event for event in events if event["event_type"] == "cold_start_trial_decided"]
    additions = [event for event in events if event["event_type"] == "colocation_trial_started"]
    if trial:
        assert additions and decisions, "Unprofiled addition was never tried"
        first_completion = min(timestamp(event["created_at"]) for event in events if event["event_type"] == "job_completed")
        assert timestamp(additions[0]["created_at"]) < first_completion
        assert max(active_training) >= 2, "No overlapping useful progress"
    result = {
        "name": name, "started_at": started, "makespan_seconds": makespan, "jobs": records,
        "artifact_dir": str(path),
        "updates_per_second": len(jobs) * steps * 3 / makespan,
        "mean_queue_wait_seconds": statistics.mean(job["queue_wait_seconds"] for job in records),
        "trial_control_seconds_sum": sum(job["control_seconds"] for job in records),
        "measurement_sync_seconds_sum": sum(job["sync_seconds"] for job in records),
        "max_overlapping_progress": max(active_training), "startup_seconds_sum": sum(startup),
        "yield_seconds_sum": sum(event["payload"].get("yield_seconds", 0) for event in events),
        "checkpoint_seconds_sum": sum(event["payload"].get("checkpoint_seconds", 0) for event in events),
        "measurement_useful_seconds_sum": sum(event["payload"].get("elapsed_seconds", 0) for event in events if event["event_type"] == "trial_step_window"),
        "peak_device_mib": max((item["used_mib"] for item in telemetry), default=0),
        "mean_gpu_busy_percent": statistics.mean(item["busy_percent"] for item in telemetry) if telemetry else None,
        "decisions": [event["payload"] for event in decisions], "events": events,
    }
    assert sum(job["peak_allocator_mib"] for job in records) < 8 * 1024
    write_json(path / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--previous-run", type=Path, help="Include earlier diagnostic arms and count their time against the same GPU budget")
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.output)
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "source.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "python": sys.executable,
                                      "torch": torch.__version__, "cuda": torch.version.cuda, "models": MODELS, "gap_seconds": GAP,
                                      "mps_hardware_validation": "unexecuted; MPS control binary absent in WSL", "cpu_smoke": args.cpu_smoke})
    (root / "working_diff.patch").write_text(subprocess.check_output(["git", "diff", "--", ".", ":(exclude)PerfSeer-predictor", ":(exclude)nn-model-preflight-checker"], text=True))
    for source in ("localml_scheduler/examples/cold_start_runner.py", "localml_scheduler/execution/script_context.py", "localml_scheduler/execution/trial_control.py", "localml_scheduler/scheduler/cold_start.py", "scheduler_benchmark_test/cold_start_comparison.py"):
        destination = root / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    started = time.time()
    previous_seconds = 0.0
    results = []
    if args.previous_run:
        previous_seconds = json.loads((args.previous_run / "budget.json").read_text())["elapsed_seconds"]
        results = json.loads((args.previous_run / "results.json").read_text())
        for result in results:
            result["artifact_dir"] = str(args.previous_run.resolve() / result["name"])
            result["name"] = "Diagnostic " + result["name"]
    deadline = started + 300 - previous_seconds
    if args.cpu_smoke:
        steps, arms = 240, ["cpu-trial"]
    else:
        before = gpu_state()
        write_json(root / "gpu_before.json", before)
        assert "5090" in before["name"]
        assert before["total_mib"] - before["used_mib"] >= 10 * 1024
        subprocess.run([sys.executable, "-m", "scheduler_benchmark_test.cold_start_comparison", "--calibrate", "--output", str(root / "calibration.json")], check=True, timeout=25)
        steps = json.loads((root / "calibration.json").read_text())["steps_per_epoch"]
        arms = ["pair1-sequential", "pair1-trial", "pair2-trial", "pair2-sequential"]
    for name in arms:
        if not args.cpu_smoke:
            write_json(root / f"gpu-before-{name}.json", gpu_state())
        result = run_arm(root, name, steps, deadline, cpu=args.cpu_smoke)
        results.append(result)
        write_json(root / "results.json", results)
        draw(root, results)
        print(json.dumps({key: result[key] for key in ("name", "makespan_seconds", "updates_per_second", "max_overlapping_progress")}), flush=True)
    write_json(root / "budget.json", {"elapsed_seconds": previous_seconds + time.time() - started, "prior_experiments_seconds": previous_seconds, "limit_seconds": 300, "steps_per_epoch": steps})


if __name__ == "__main__":
    main()
