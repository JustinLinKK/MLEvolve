"""Calibrated RTX 5090 pressure benchmark for LocalML Scheduler.

The module deliberately keeps the benchmark definition, calibration evidence,
mode orchestration, telemetry, and analysis in one importable place.  The
public entry point is ``run_rtx5090_pressure_benchmark.sh``; this module also
exposes small pure functions used by unit tests.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
MODEL_SOURCE = (
    REPO
    / "scheduler_benchmark_test"
    / "fixtures"
    / "stress_test_data_v1.0"
    / "model_source.py"
)
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
SCENARIO_COLORS = {
    "compute_heavy": "#7b2cbf",
    "near_exclusive": "#d00000",
    "light_pack": "#2a9d8f",
    "short_flow": "#f4a261",
    "boundary": "#457b9d",
    "asymmetric": "#e9c46a",
    "compute_bandwidth": "#264653",
}
CALIBRATION_BASE_MIB_GUESS = {
    "compute-heavy": 2560,
    "near-exclusive": 96,
    "light": 192,
    "short": 176,
    "boundary": 384,
    "asym-large": 608,
    "asym-small": 112,
    "compute-only": 1792,
    "bandwidth": 864,
}


@dataclass(frozen=True, slots=True)
class TraceTemplate:
    name: str
    release_s: float
    target_minutes: float
    scenario: str
    archetype: str
    vram_fraction: float
    architecture: str
    width: int
    depth: int
    activation: str
    batch_size: int
    image_size: int
    compute_repeats: int = 1
    bandwidth_mib: int = 0
    step_delay_ms: float = 0.0


TRACE_TEMPLATES = (
    TraceTemplate(
        "compute-heavy-long",
        0,
        14,
        "compute_heavy",
        "compute-heavy",
        0.18,
        "resnet",
        64,
        4,
        "gelu",
        64,
        96,
        3,
    ),
    TraceTemplate(
        "compute-heavy-short",
        0,
        10,
        "compute_heavy",
        "compute-heavy",
        0.18,
        "resnet",
        64,
        4,
        "gelu",
        64,
        96,
        3,
    ),
    TraceTemplate(
        "near-exclusive-a",
        60,
        10,
        "near_exclusive",
        "near-exclusive",
        0.80,
        "patch_mlp",
        24,
        2,
        "gelu",
        32,
        64,
        1,
        0,
        4,
    ),
    TraceTemplate(
        "light-1",
        120,
        4.5,
        "light_pack",
        "light",
        0.09,
        "mobilenet_v3",
        12,
        2,
        "silu",
        32,
        64,
        1,
        0,
        28,
    ),
    TraceTemplate(
        "light-2",
        120,
        4.5,
        "light_pack",
        "light",
        0.09,
        "mobilenet_v3",
        12,
        2,
        "silu",
        32,
        64,
        1,
        0,
        28,
    ),
    TraceTemplate(
        "light-3",
        120,
        4.5,
        "light_pack",
        "light",
        0.09,
        "mobilenet_v3",
        12,
        2,
        "silu",
        32,
        64,
        1,
        0,
        28,
    ),
    TraceTemplate(
        "light-4",
        120,
        4.5,
        "light_pack",
        "light",
        0.09,
        "mobilenet_v3",
        12,
        2,
        "silu",
        32,
        64,
        1,
        0,
        28,
    ),
    TraceTemplate(
        "short-a",
        360,
        2,
        "short_flow",
        "short",
        0.10,
        "conv_mlp",
        24,
        2,
        "relu",
        48,
        64,
    ),
    TraceTemplate(
        "boundary-1",
        720,
        8,
        "boundary",
        "boundary",
        0.41,
        "densenet",
        24,
        2,
        "relu",
        48,
        80,
        1,
        0,
        4,
    ),
    TraceTemplate(
        "boundary-2",
        720,
        8,
        "boundary",
        "boundary",
        0.41,
        "densenet",
        24,
        2,
        "relu",
        48,
        80,
        1,
        0,
        4,
    ),
    TraceTemplate(
        "asymmetric-large",
        1080,
        9,
        "asymmetric",
        "asym-large",
        0.55,
        "efficient_residual",
        32,
        3,
        "silu",
        48,
        80,
        1,
        0,
        5,
    ),
    TraceTemplate(
        "asymmetric-small",
        1080,
        4,
        "asymmetric",
        "asym-small",
        0.12,
        "patch_gru",
        20,
        2,
        "silu",
        64,
        64,
        1,
        0,
        8,
    ),
    TraceTemplate(
        "compute-pair",
        1440,
        9,
        "compute_bandwidth",
        "compute-only",
        0.18,
        "resnet",
        72,
        4,
        "relu",
        64,
        96,
        4,
    ),
    TraceTemplate(
        "bandwidth-pair",
        1440,
        9,
        "compute_bandwidth",
        "bandwidth",
        0.24,
        "depthwise_mlp",
        20,
        2,
        "silu",
        64,
        64,
        1,
        512,
    ),
    TraceTemplate(
        "near-exclusive-b",
        1800,
        10,
        "near_exclusive",
        "near-exclusive",
        0.80,
        "patch_mlp",
        24,
        2,
        "gelu",
        32,
        64,
        1,
        0,
        4,
    ),
    TraceTemplate(
        "short-b",
        2160,
        2,
        "short_flow",
        "short",
        0.10,
        "conv_mlp",
        24,
        2,
        "relu",
        48,
        64,
    ),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    temporary.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for record in records for key in record})
    with path.open("w", newline="") as handle:
        if not columns:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def trace_target_minutes() -> float:
    return sum(item.target_minutes for item in TRACE_TEMPLATES)


def build_trace(total_vram_mib: int, *, smoke: bool = False) -> list[dict[str, Any]]:
    """Return the immutable 16-job production trace before calibration."""
    duration_scale = 0.005 if smoke else 1.0
    release_scale = 0.005 if smoke else 1.0
    trace: list[dict[str, Any]] = []
    for index, template in enumerate(TRACE_TEMPLATES):
        target_seconds = max(0.75, template.target_minutes * 60.0 * duration_scale)
        if smoke and template.name == "compute-heavy-long":
            target_seconds = 12.0
        elif smoke and template.name == "compute-heavy-short":
            target_seconds = 3.0
        elif smoke and template.scenario == "light_pack":
            target_seconds = 30.0
        target_vram = int(round(total_vram_mib * template.vram_fraction))
        # The probe corrects this conservative first guess against observed
        # model/input/optimizer memory.
        ballast = max(
            0,
            target_vram - int(CALIBRATION_BASE_MIB_GUESS.get(template.archetype, 384)),
        )
        trace.append(
            {
                "step_idx": index,
                "job_id": template.name,
                "release_s": template.release_s * release_scale,
                "arrival_offset_s": template.release_s * release_scale,
                "target_minutes": template.target_minutes,
                "target_seconds": target_seconds,
                "scenario": template.scenario,
                "scenario_color": SCENARIO_COLORS[template.scenario],
                "archetype": template.archetype,
                "target_vram_fraction": template.vram_fraction,
                "target_vram_mib": target_vram,
                "estimated_vram_mb": target_vram,
                "source_path": str(MODEL_SOURCE),
                "entry": "build_model",
                "architecture": template.architecture,
                "family": template.scenario,
                "constructor_kwargs": {
                    "architecture": template.architecture,
                    "width": template.width,
                    "depth": template.depth,
                    "activation": template.activation,
                },
                "input_shape": [
                    template.batch_size,
                    3,
                    template.image_size,
                    template.image_size,
                ],
                "input_dtypes": ["float32"],
                "batch_size": template.batch_size,
                "precision": "bf16_amp",
                "epochs": 3 if smoke else 20,
                "batches_per_epoch": 1,
                "stream_data": True,
                "memory_ballast_mib": ballast,
                "compute_repeats": template.compute_repeats,
                "bandwidth_mib": template.bandwidth_mib,
                "step_delay_ms": template.step_delay_ms,
                "random_seed": 5090 + index,
                "manage_tf32": False,
                "packing_signature": f"rtx5090:{template.archetype}",
                "backend_allowlist": ["stream"],
            }
        )
    return trace


def validate_trace(trace: list[dict[str, Any]]) -> None:
    if len(trace) != 16:
        raise ValueError(f"RTX 5090 trace must contain 16 jobs, got {len(trace)}")
    if not math.isclose(trace_target_minutes(), 113.0):
        raise ValueError("target solo work must total 113 minutes")
    releases = [float(item["release_s"]) for item in trace]
    if releases != sorted(releases):
        raise ValueError("trace releases must be monotonic")
    names = [str(item["job_id"]) for item in trace]
    if len(set(names)) != len(names):
        raise ValueError("logical job names must be unique")
    if any(item.get("backend_allowlist") != ["stream"] for item in trace):
        raise ValueError("packed scheduler work must allow only the stream backend")


def hardware_manifest(device_index: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "device_index": device_index,
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            payload.update(
                {
                    "gpu_name": props.name,
                    "total_vram_mib": int(props.total_memory // (1024 * 1024)),
                    "compute_capability": list(
                        torch.cuda.get_device_capability(device_index)
                    ),
                }
            )
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.used,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        fields = [item.strip() for item in query.stdout.strip().split(",")]
        if len(fields) >= 7:
            payload["nvidia_smi"] = {
                "name": fields[0],
                "uuid": fields[1],
                "driver": fields[2],
                "memory_total_mib": float(fields[3]),
                "memory_used_mib": float(fields[4]),
                "power_limit_w": float(fields[5]),
                "temperature_c": float(fields[6]),
            }
    except Exception as exc:
        payload["nvidia_smi_error"] = repr(exc)
    try:
        payload["git"] = {
            "branch": subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip(),
            "head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip(),
        }
    except Exception:
        pass
    return payload


class TelemetrySampler:
    """Write 0.5-second NVML observations while retaining summary data."""

    def __init__(self, path: Path, *, device_index: int = 0, interval: float = 0.5):
        self.path = path
        self.device_index = device_index
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._nvml is not None and self._handle is not None

    def start(self) -> None:
        if self.available:
            self._thread = threading.Thread(
                target=self._run, name="rtx5090-nvml", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "elapsed_s": 0.0,
                        "memory_used_mib": memory.used / (1024 * 1024),
                        "memory_total_mib": memory.total / (1024 * 1024),
                        "gpu_util_percent": float(utilization.gpu),
                        "memory_util_percent": float(utilization.memory),
                        "power_w": self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                        / 1000.0,
                        "temperature_c": float(
                            self._nvml.nvmlDeviceGetTemperature(
                                self._handle, self._nvml.NVML_TEMPERATURE_GPU
                            )
                        ),
                    }
                )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self, *, origin: float | None = None) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if origin is not None:
            for sample in self.samples:
                sample["elapsed_s"] = sample["timestamp"] - origin
        self.path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "timestamp",
            "elapsed_s",
            "memory_used_mib",
            "memory_total_mib",
            "gpu_util_percent",
            "memory_util_percent",
            "power_w",
            "temperature_c",
        ]
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.samples)
        if not self.samples:
            return {"sample_count": 0}
        energy_ws = sum(
            (left["power_w"] + right["power_w"])
            * 0.5
            * (right["timestamp"] - left["timestamp"])
            for left, right in zip(self.samples, self.samples[1:])
        )
        return {
            "sample_count": len(self.samples),
            "avg_gpu_util_percent": statistics.fmean(
                item["gpu_util_percent"] for item in self.samples
            ),
            "avg_power_w": statistics.fmean(item["power_w"] for item in self.samples),
            "peak_memory_used_mib": max(
                item["memory_used_mib"] for item in self.samples
            ),
            "energy_wh": energy_ws / 3600.0,
        }


def wait_for_idle_gpu(
    *,
    device_index: int,
    reference_temperature_c: float | None,
    timeout_s: float,
    strict: bool,
    reference_memory_mib: float = 0.0,
) -> dict[str, Any]:
    """Wait outside the measured cap for idle utilization and a stable temperature."""
    started = time.time()
    last: dict[str, float] = {}
    while time.time() - started < timeout_s:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            util, memory, temperature = [
                float(value.strip()) for value in result.stdout.strip().split(",")
            ]
            last = {
                "gpu_util_percent": util,
                "memory_used_mib": memory,
                "temperature_c": temperature,
            }
            temperature_ok = (
                reference_temperature_c is None
                or abs(temperature - reference_temperature_c) <= 5.0
            )
            if (
                util <= 5.0
                and memory <= max(1024.0, reference_memory_mib + 512.0)
                and temperature_ok
            ):
                return {**last, "wait_seconds": time.time() - started, "ready": True}
        except Exception as exc:
            last = {"error": repr(exc)}
        time.sleep(1.0)
    result = {**last, "wait_seconds": time.time() - started, "ready": False}
    if strict:
        raise RuntimeError(f"GPU did not reach idle/temperature band: {result}")
    return result


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _run_probe_subprocess(
    spec: dict[str, Any], directory: Path, label: str
) -> dict[str, Any]:
    spec_path = directory / f"{label}.spec.json"
    result_path = directory / f"{label}.result.json"
    log_path = directory / f"{label}.log"
    _write_json(spec_path, spec)
    with log_path.open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scheduler_benchmark_test.stress_bench.mp_worker",
                "--spec",
                str(spec_path),
                "--result",
                str(result_path),
            ],
            cwd=REPO,
            env=_worker_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=240,
            check=False,
        )
    result = _read_json(result_path, {})
    result.update({"returncode": completed.returncode, "log_path": str(log_path)})
    if completed.returncode or not result_path.exists():
        result["error"] = log_path.read_text(errors="replace")[-4000:]
    return result


def _calibration_representatives(
    trace: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for job in trace:
        representatives.setdefault(str(job["archetype"]), job)
    return representatives


def _probe_spec(job: dict[str, Any], *, steps: int = 2) -> dict[str, Any]:
    spec = dict(job)
    spec["epochs"] = 1
    spec["batches_per_epoch"] = max(1, int(steps))
    return spec


def _invoke_group_probe(
    specs: list[dict[str, Any]], directory: Path, label: str
) -> dict[str, Any]:
    input_path = directory / f"{label}.group.json"
    output_path = directory / f"{label}.group.result.json"
    log_path = directory / f"{label}.group.log"
    _write_json(input_path, specs)
    with log_path.open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scheduler_benchmark_test.rtx5090_pressure_benchmark",
                "group-worker",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=REPO,
            env=_worker_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    result = _read_json(output_path, {})
    result.update({"returncode": completed.returncode, "log_path": str(log_path)})
    if completed.returncode:
        result["error"] = log_path.read_text(errors="replace")[-4000:]
    return result


def group_worker(input_path: Path, output_path: Path) -> int:
    """Run a calibration group in one context using distinct CUDA streams."""
    import torch
    from scheduler_benchmark_test.stress_bench.stress_runner import train_stress_model

    specs = _read_json(input_path, [])
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    lock = threading.Lock()
    started = time.time()

    def run(spec: dict[str, Any], stream: Any) -> None:
        try:
            with torch.cuda.stream(stream):
                result = train_stress_model(
                    source_path=spec["source_path"],
                    constructor_kwargs=spec["constructor_kwargs"],
                    input_shape=spec["input_shape"],
                    precision=spec["precision"],
                    epochs=int(spec["epochs"]),
                    batches_per_epoch=int(spec["batches_per_epoch"]),
                    device=torch.device("cuda"),
                    stream_data=bool(spec.get("stream_data")),
                    memory_ballast_mib=int(spec.get("memory_ballast_mib") or 0),
                    compute_repeats=int(spec.get("compute_repeats") or 1),
                    bandwidth_mib=int(spec.get("bandwidth_mib") or 0),
                    step_delay_ms=float(spec.get("step_delay_ms") or 0),
                    random_seed=int(spec.get("random_seed") or 0),
                    manage_tf32=False,
                )
            stream.synchronize()
            with lock:
                results[str(spec["job_id"])] = result
        except Exception as exc:
            with lock:
                errors[str(spec["job_id"])] = repr(exc)

    if not torch.cuda.is_available():
        raise RuntimeError("group calibration requires CUDA")
    threads = []
    stream_ids: dict[str, int] = {}
    for spec in specs:
        stream = torch.cuda.Stream()
        stream_ids[str(spec["job_id"])] = int(stream.cuda_stream)
        thread = threading.Thread(target=run, args=(spec, stream), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    payload = {
        "wall_seconds": time.time() - started,
        "results": results,
        "errors": errors,
        "host_pid": os.getpid(),
        "stream_ids": stream_ids,
    }
    _write_json(output_path, payload)
    return 1 if errors else 0


def calibrate_trace(
    trace: list[dict[str, Any]],
    output_dir: Path,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound steps/VRAM, gather solo and group evidence, and qualify the trace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_per_case = 1 if smoke else 3
    attempts: list[dict[str, Any]] = []
    working = [dict(item) for item in trace]
    valid = False
    reasons: list[str] = []

    for tuning_round in range(1, 4):
        representatives = _calibration_representatives(working)
        solo: dict[str, list[dict[str, Any]]] = {}
        for archetype, job in representatives.items():
            samples: list[dict[str, Any]] = []
            for observation in range(observations_per_case):
                sample = _run_probe_subprocess(
                    _probe_spec(job, steps=20),
                    output_dir,
                    f"round{tuning_round}-solo-{archetype}-{observation}",
                )
                if sample.get("returncode"):
                    raise RuntimeError(
                        f"solo calibration failed for {archetype}: {sample.get('error')}"
                    )
                samples.append(sample)
                if observation == 0:
                    observed = float(sample.get("peak_reserved_mib") or 0)
                    correction = int(round(float(job["target_vram_mib"]) - observed))
                    corrected = max(0, int(job["memory_ballast_mib"]) + correction)
                    for candidate in working:
                        if candidate["archetype"] == archetype:
                            candidate["memory_ballast_mib"] = corrected
                    job["memory_ballast_mib"] = corrected
            solo[archetype] = samples

        # Freeze a fixed step count shared by every measured mode.
        for job in working:
            samples = solo[str(job["archetype"])]
            step_rates = [
                float(sample["training_seconds"]) / max(1, int(sample["global_steps"]))
                for sample in samples
            ]
            step_rate = statistics.median(step_rates)
            raw_steps = max(1, int(round(float(job["target_seconds"]) / step_rate)))
            desired_epochs = 3 if smoke else 20
            candidates = [
                (
                    abs((epochs * max(1, round(raw_steps / epochs))) - raw_steps),
                    epochs,
                    max(1, round(raw_steps / epochs)),
                )
                for epochs in range(1, min(desired_epochs, raw_steps) + 1)
            ]
            bounded = [item for item in candidates if item[0] / raw_steps <= 0.10]
            _, epochs, batches = min(
                bounded or candidates, key=lambda item: (-item[1], item[0])
            )
            job["epochs"] = epochs
            job["batches_per_epoch"] = batches
            job["calibrated_step_seconds"] = step_rate
            job["calibrated_solo_seconds"] = step_rate * epochs * batches
            relative_error = (
                abs(job["calibrated_solo_seconds"] - job["target_seconds"])
                / job["target_seconds"]
            )
            job["duration_relative_error"] = relative_error

        by_name = {item["job_id"]: item for item in working}
        group_definitions = {
            "compute_pair": [
                by_name["compute-heavy-long"],
                by_name["compute-heavy-short"],
            ],
            "light_four": [by_name[f"light-{index}"] for index in range(1, 5)],
            "boundary_pair": [by_name["boundary-1"], by_name["boundary-2"]],
        }
        group_results: dict[str, list[dict[str, Any]]] = {}
        for group_name, members in group_definitions.items():
            results = []
            for observation in range(observations_per_case):
                specs = [_probe_spec(member, steps=6) for member in members]
                result = _invoke_group_probe(
                    specs,
                    output_dir,
                    f"round{tuning_round}-{group_name}-{observation}",
                )
                results.append(result)
            group_results[group_name] = results

        reasons = []
        if any(float(job["duration_relative_error"]) > 0.10 for job in working):
            reasons.append("one or more calibrated durations exceed the ±10% bound")
        for archetype, samples in solo.items():
            peaks = [float(item.get("peak_reserved_mib") or 0) for item in samples]
            target = float(representatives[archetype]["target_vram_mib"])
            if not peaks or abs(statistics.median(peaks) - target) / target > 0.10:
                reasons.append(f"{archetype} VRAM target is unstable/outside ±10%")
        if any(
            item.get("returncode")
            for values in group_results.values()
            for item in values
        ):
            reasons.append("a required colocation group failed")
        light_solo = statistics.median(
            float(item["training_seconds"]) / max(1, int(item["global_steps"]))
            for item in solo["light"]
        )
        light_group = statistics.median(
            max(
                (
                    float(result.get("training_seconds") or math.inf)
                    / max(1, int(result.get("global_steps") or 1))
                    for result in (item.get("results") or {}).values()
                ),
                default=math.inf,
            )
            for item in group_results["light_four"]
        )
        light_speedup = (2.0 * light_solo) / light_group if light_group > 0 else 0.0
        compute_solo = statistics.median(
            float(item["training_seconds"]) / max(1, int(item["global_steps"]))
            for item in solo["compute-heavy"]
        )
        compute_group = statistics.median(
            max(
                (
                    float(result.get("training_seconds") or 0)
                    / max(1, int(result.get("global_steps") or 1))
                    for result in (item.get("results") or {}).values()
                ),
                default=0.0,
            )
            for item in group_results["compute_pair"]
        )
        compute_slowdown = compute_group / compute_solo if compute_solo > 0 else 0.0
        if light_speedup <= 1.10:
            reasons.append(
                f"four-light packing speedup too small ({light_speedup:.3f}x)"
            )
        if compute_slowdown <= 1.10:
            reasons.append(
                f"compute-heavy pairing is not harmful ({compute_slowdown:.3f}x)"
            )
        boundary_fraction = sum(
            float(item["target_vram_fraction"])
            for item in group_definitions["boundary_pair"]
        )
        if not 0.78 <= boundary_fraction < 0.85:
            reasons.append("boundary pair is not just below the 85% admission budget")
        near = by_name["near-exclusive-a"]
        if not 0.78 <= float(near["target_vram_fraction"]) <= 0.82:
            reasons.append("near-exclusive workload is not approximately 80% VRAM")
        if (
            float(near["target_vram_fraction"])
            + min(
                float(item["target_vram_fraction"])
                for item in working
                if item is not near
            )
            <= 0.85
        ):
            reasons.append(
                "near-exclusive workload can unexpectedly share the admission budget"
            )

        attempt = {
            "round": tuning_round,
            "solo": solo,
            "groups": group_results,
            "light_speedup_vs_mp2": light_speedup,
            "compute_pair_slowdown": compute_slowdown,
            "reasons": list(reasons),
            "valid": not reasons,
        }
        attempts.append(attempt)
        valid = not reasons
        if valid:
            break
        if tuning_round < 3:
            # Only bounded workload knobs are changed.  Release times, seeds,
            # target durations, scenarios, and memory targets stay frozen.
            for job in working:
                if (
                    "four-light packing speedup" in " ".join(reasons)
                    and job["archetype"] == "light"
                ):
                    job["step_delay_ms"] = float(job["step_delay_ms"]) + 20.0
                if (
                    "compute-heavy pairing" in " ".join(reasons)
                    and job["archetype"] == "compute-heavy"
                ):
                    job["compute_repeats"] = int(job["compute_repeats"]) + 2

    qualification = {
        "valid": valid,
        "reasons": reasons,
        "attempts": attempts,
        "solo_observations_per_archetype": observations_per_case,
        "group_observations_per_case": observations_per_case,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "admission_budget_fraction": 0.85,
    }
    _write_json(output_dir / "calibration.json", qualification)
    _write_json(output_dir / "calibrated_trace.json", working)
    with (output_dir / "calibrated_trace.jsonl").open("w") as handle:
        for item in working:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    if not valid and not smoke:
        raise RuntimeError(
            "benchmark qualification failed after three rounds: " + "; ".join(reasons)
        )
    return working, qualification


@dataclass(slots=True)
class BaselineProcess:
    process: subprocess.Popen
    job: dict[str, Any]
    attempt: int
    started_at: float
    result_path: Path
    log_path: Path


def _is_cuda_oom(returncode: int, result: dict[str, Any], log_text: str) -> bool:
    combined = " ".join(
        [str(result.get("error") or ""), str(result.get("exception") or ""), log_text]
    ).lower()
    return returncode != 0 and any(
        marker in combined
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cudnn_status_alloc_failed",
            "outofmemoryerror",
        )
    )


def run_mp2_baseline(
    trace: list[dict[str, Any]],
    output_dir: Path,
    *,
    timeout_s: float,
    python_executable: str = sys.executable,
    command_factory: Callable[[dict[str, Any], Path, Path], list[str]] | None = None,
) -> dict[str, Any]:
    """Run unconditional FIFO MP2 with the specified per-job OOM recovery.

    Admission never consults VRAM.  Once an OOM is observed, new launches stop,
    the surviving peer drains, and that logical job is retried exactly once
    alone.  MP2 resumes immediately after the retry.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = output_dir / "specs"
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    for path in (specs_dir, results_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    origin = time.time()
    deadline = origin + timeout_s
    next_release = 0
    ready: list[dict[str, Any]] = []
    running: list[BaselineProcess] = []
    recovery_queue: list[dict[str, Any]] = []
    recovery_active: BaselineProcess | None = None
    attempts: list[dict[str, Any]] = []
    retries_seen: set[str] = set()
    timed_out = False
    interrupted: BaseException | None = None

    def launch(job: dict[str, Any], attempt_no: int) -> BaselineProcess:
        stem = f"{int(job['step_idx']):02d}-{job['job_id']}-attempt{attempt_no}"
        spec_path = specs_dir / f"{stem}.json"
        result_path = results_dir / f"{stem}.json"
        log_path = logs_dir / f"{stem}.log"
        _write_json(spec_path, job)
        command = (
            command_factory(job, spec_path, result_path)
            if command_factory is not None
            else [
                python_executable,
                "-m",
                "scheduler_benchmark_test.stress_bench.mp_worker",
                "--spec",
                str(spec_path),
                "--result",
                str(result_path),
            ]
        )
        log_handle = log_path.open("w")
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=_worker_env(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        log_handle.close()
        return BaselineProcess(
            process, job, attempt_no, time.time(), result_path, log_path
        )

    def finish(
        worker: BaselineProcess, *, timeout_kill: bool = False
    ) -> dict[str, Any]:
        result = _read_json(worker.result_path, {})
        log_text = (
            worker.log_path.read_text(errors="replace")
            if worker.log_path.exists()
            else ""
        )
        oom = _is_cuda_oom(worker.process.returncode or 0, result, log_text)
        status = (
            "timeout"
            if timeout_kill
            else (
                "oom"
                if oom
                else ("succeeded" if worker.process.returncode == 0 else "failed")
            )
        )
        record = {
            "logical_job_id": worker.job["job_id"],
            "step_idx": worker.job["step_idx"],
            "scenario": worker.job["scenario"],
            "attempt": worker.attempt,
            "retry": worker.attempt > 1,
            "backend": "multiprocess",
            "pid": worker.process.pid,
            "released_at": origin + float(worker.job["release_s"]),
            "started_at": worker.started_at,
            "finished_at": time.time(),
            "returncode": worker.process.returncode,
            "status": status,
            "oom": oom,
            "result_path": str(worker.result_path),
            "log_path": str(worker.log_path),
            "training_seconds": result.get("training_seconds"),
            "peak_reserved_mib": result.get("peak_reserved_mib"),
        }
        attempts.append(record)
        return record

    try:
        while True:
            now = time.time()
            if now >= deadline:
                timed_out = True
                break

            while (
                next_release < len(trace)
                and origin + float(trace[next_release]["release_s"]) <= now
            ):
                ready.append(trace[next_release])
                next_release += 1

            for worker in list(running):
                if worker.process.poll() is None:
                    continue
                running.remove(worker)
                record = finish(worker)
                if (
                    record["oom"]
                    and worker.attempt == 1
                    and str(worker.job["job_id"]) not in retries_seen
                ):
                    retries_seen.add(str(worker.job["job_id"]))
                    recovery_queue.append(worker.job)
                if recovery_active is worker:
                    recovery_active = None

            # A recovery begins only after every surviving MP2 peer drains.
            if recovery_active is None and recovery_queue and not running:
                recovery_job = recovery_queue.pop(0)
                recovery_active = launch(recovery_job, 2)
                running.append(recovery_active)
            elif recovery_active is None and not recovery_queue:
                while ready and len(running) < 2:
                    running.append(launch(ready.pop(0), 1))

            if (
                next_release >= len(trace)
                and not ready
                and not running
                and not recovery_queue
            ):
                break
            time.sleep(0.05)
    except BaseException as exc:
        interrupted = exc
    finally:
        if timed_out or interrupted is not None:
            for worker in list(running):
                if worker.process.poll() is None:
                    try:
                        if os.name == "posix":
                            os.killpg(os.getpgid(worker.process.pid), signal.SIGTERM)
                        else:
                            worker.process.terminate()
                        worker.process.wait(timeout=3)
                    except Exception:
                        worker.process.kill()
                        worker.process.wait()
                finish(worker, timeout_kill=True)

    if interrupted is not None:
        raise interrupted

    logical: list[dict[str, Any]] = []
    for job in trace:
        job_attempts = [
            item for item in attempts if item["logical_job_id"] == job["job_id"]
        ]
        success = next(
            (item for item in reversed(job_attempts) if item["status"] == "succeeded"),
            None,
        )
        final = success or (job_attempts[-1] if job_attempts else None)
        logical.append(
            {
                "logical_job_id": job["job_id"],
                "step_idx": job["step_idx"],
                "scenario": job["scenario"],
                "release_s": job["release_s"],
                "status": (
                    final["status"]
                    if final
                    else ("timeout" if timed_out else "not_started")
                ),
                "started_at": final.get("started_at") if final else None,
                "finished_at": final.get("finished_at") if final else None,
                "attempt_count": len(job_attempts),
                "oom_count": sum(bool(item["oom"]) for item in job_attempts),
            }
        )
    return {
        "mode": "baseline",
        "origin": origin,
        "deadline": deadline,
        "timed_out": timed_out,
        "attempts": attempts,
        "logical_jobs": logical,
        "events": [],
        "policy": "unconditional_fifo_mp2_with_per_job_oom_recovery",
    }


def build_profile_snapshots(
    trace: list[dict[str, Any]],
    qualification: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze identical solo/memory evidence and warm-only colocation evidence."""
    final_attempt = qualification.get("attempts", [{}])[-1]
    solo_samples = final_attempt.get("solo", {})
    solo_profiles: list[dict[str, Any]] = []
    for job in trace:
        samples = solo_samples.get(job["archetype"], [])
        step_seconds = float(job["calibrated_step_seconds"])
        solo_profiles.append(
            {
                "logical_job_id": job["job_id"],
                "signature": job["packing_signature"],
                "batch_size": job["batch_size"],
                "avg_vram_mib": job["target_vram_mib"],
                "peak_vram_mib": (
                    statistics.median(
                        [
                            float(
                                item.get("peak_reserved_mib") or job["target_vram_mib"]
                            )
                            for item in samples
                        ]
                    )
                    if samples
                    else job["target_vram_mib"]
                ),
                "step_seconds": step_seconds,
                "seconds_per_epoch": step_seconds * int(job["batches_per_epoch"]),
                "total_seconds": float(job["calibrated_solo_seconds"]),
                "observations": max(1, len(samples)),
            }
        )

    by_name = {item["job_id"]: item for item in trace}
    group_members = {
        "compute_pair": ["compute-heavy-long", "compute-heavy-short"],
        "light_pair": ["light-1", "light-2"],
        "light_triple": ["light-1", "light-2", "light-3"],
        "light_four": ["light-1", "light-2", "light-3", "light-4"],
        "boundary_pair": ["boundary-1", "boundary-2"],
    }
    calibrated_groups = final_attempt.get("groups", {})
    colocation_profiles: list[dict[str, Any]] = []
    for group_name, names in group_members.items():
        source_group = "light_four" if group_name.startswith("light_") else group_name
        observations = calibrated_groups.get(source_group, [])
        if source_group == "compute_pair":
            slowdown = float(final_attempt.get("compute_pair_slowdown") or 1.5)
            decision = "rejected"
        elif source_group == "light_four":
            # Scale the measured four-way result conservatively for pair/triple profiles.
            speedup = max(1.01, float(final_attempt.get("light_speedup_vs_mp2") or 1.5))
            slowdown = min(1.20, max(1.01, 2.0 / speedup))
            decision = "accepted"
        else:
            slowdown = 1.08
            decision = "accepted"
        members = []
        timings = []
        for name in names:
            job = by_name[name]
            descriptor = {
                "signature": job["packing_signature"],
                "batch_size": job["batch_size"],
                "backend_name": "stream",
            }
            members.append(descriptor)
            timings.append(
                {
                    **descriptor,
                    "seconds_per_epoch": float(job["calibrated_step_seconds"])
                    * int(job["batches_per_epoch"])
                    * slowdown,
                    "observations": max(2, len(observations)),
                    "source": "rtx5090_calibration",
                }
            )
        colocation_profiles.append(
            {
                "name": group_name,
                "members": members,
                "member_timings": timings,
                "observations": max(2, len(observations)),
                "decision": decision,
                "slowdown": slowdown,
            }
        )
    cold = {
        "kind": "cold",
        "solo_memory_profiles": solo_profiles,
        "colocation_profiles": [],
    }
    warm = {
        "kind": "warm",
        "solo_memory_profiles": solo_profiles,
        "colocation_profiles": colocation_profiles,
    }
    _write_json(output_dir / "profiles-cold.json", cold)
    _write_json(output_dir / "profiles-warm.json", warm)
    return warm, cold


def _scheduler_settings(runtime_root: Path, total_vram_mib: int, *, smoke: bool):
    from localml_scheduler.config import (
        ColocationSettings,
        ExclusiveProbeSettings,
        GpuMemorySettings,
        GpuProfilingSettings,
        GpuSchedulerSettings,
        SchedulerSettings,
        StreamSettings,
    )

    gpu = GpuSchedulerSettings(
        mode="parallel_time_aware",
        backend_priority=["stream", "exclusive"],
        parallel_job_cap=4,
        batch_probe_enabled=False,
        memory=GpuMemorySettings(
            gpu_vram_gib=total_vram_mib / 1024.0,
            predicted_budget_fraction=0.85,
            live_admission_stop_fraction=0.90,
            live_admission_resume_fraction=0.85,
            admission_average_window_seconds=0.5 if smoke else 3.0,
        ),
        profiling=GpuProfilingSettings(
            warmup_steps=1, solo_probe_steps=2, pair_probe_steps=2
        ),
        colocation=ColocationSettings(
            min_gain=1.0,
            trial_epochs=1 if smoke else 2,
            trial_decision_timeout_seconds=5 if smoke else 30,
            trial_evidence_timeout_min_seconds=5 if smoke else 120,
            trial_evidence_timeout_max_seconds=30 if smoke else 900,
            profile_rejection_min_bad_trials=2,
            live_trial_enabled=True,
        ),
        exclusive_probe=ExclusiveProbeSettings(enabled=False),
        stream=StreamSettings(enabled=True),
    )
    return SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.05 if smoke else 0.2,
        gpu_scheduler=gpu,
        prediction={"mode": "branch_profile"},
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
        early_stopping={"enabled": False},
        baseline_cache={
            "entry_capacity": 0,
            "warm_queue_top_k": 0,
            "memory_budget_bytes": 0,
        },
    )


def _build_scheduler_jobs(
    trace: list[dict[str, Any]], result_dir: Path, baseline_dir: Path
):
    from localml_scheduler.adapters.mlevolve import build_mlevolve_job
    from localml_scheduler.domain import (
        CheckpointPolicy,
        ResourceRequirements,
        RuntimeProbeSpec,
    )
    from scheduler_benchmark_test.stress_bench.stress_runner import (
        make_baseline_checkpoint,
    )

    jobs = {}
    for item in trace:
        baseline = baseline_dir / f"{item['job_id']}.pt"
        if not baseline.exists():
            make_baseline_checkpoint(
                baseline, {"constructor_kwargs": item["constructor_kwargs"]}
            )
        runner_kwargs = {
            key: item[key]
            for key in (
                "source_path",
                "constructor_kwargs",
                "input_shape",
                "precision",
                "epochs",
                "batches_per_epoch",
                "stream_data",
                "memory_ballast_mib",
                "compute_repeats",
                "bandwidth_mib",
                "step_delay_ms",
                "random_seed",
                "manage_tf32",
                "batch_size",
            )
        }
        runner_kwargs["result_dir"] = str(result_dir)
        job = build_mlevolve_job(
            workflow_id="rtx5090-pressure",
            baseline_model_id=item["job_id"],
            baseline_model_path=str(baseline),
            runner_target="scheduler_benchmark_test.stress_bench.stress_runner:run_stress_job",
            runner_kwargs=runner_kwargs,
            priority=5,
            task_type="rtx5090_pressure",
            checkpoint_policy=CheckpointPolicy(
                save_every_n_steps=None, save_every_epoch=False
            ),
            resource_requirements=ResourceRequirements(
                requires_gpu=True,
                gpu_slots=1,
                estimated_vram_mb=int(item["target_vram_mib"]),
                estimated_avg_vram_mb=int(item["target_vram_mib"]),
            ),
            packing_family=item["scenario"],
            packing_signature=item["packing_signature"],
            # Compute-heavy jobs intentionally saturate the device and are
            # single-job exclusive fallbacks.  Other jobs remain eligible for
            # incremental stream trials/packing.
            packing_eligible=item["scenario"] != "compute_heavy",
            packing_max_slowdown_ratio=3.0,
            packing_backend_allowlist=["stream"],
            runtime_probe=RuntimeProbeSpec(enabled=True, strategy="epoch_1"),
            max_epochs=int(item["epochs"]),
            metadata={
                "logical_job_id": item["job_id"],
                "step_idx": item["step_idx"],
                "scenario": item["scenario"],
                "release_s": item["release_s"],
                "target_vram_fraction": item["target_vram_fraction"],
            },
        )
        jobs[item["job_id"]] = job
    return jobs


def _seed_scheduler_profiles(
    settings: Any, jobs: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, int]:
    from localml_scheduler.domain import (
        BatchSizeObservation,
        ColocationTimingProfile,
        RuntimeProfile,
        build_batch_size_observation_key,
        build_batch_probe_shape_signature,
    )
    from localml_scheduler.storage.state_store import StateStore

    store = StateStore(settings)
    hardware_key = store.hardware_key()
    seeded = {"runtime": 0, "memory": 0, "colocation": 0}
    for profile in snapshot["solo_memory_profiles"]:
        job = jobs[profile["logical_job_id"]]
        for backend in ("exclusive", "stream"):
            store.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=job.packing.signature,
                    hardware_key=hardware_key,
                    backend_name=backend,
                    resolved_batch_size=int(profile["batch_size"]),
                    strategy="epoch_1",
                    startup_seconds=0.0,
                    epoch_1_seconds=float(profile["seconds_per_epoch"]),
                    steps_per_epoch=int(job.config.runner_kwargs["batches_per_epoch"]),
                    avg_step_time_ms=float(profile["step_seconds"]) * 1000.0,
                    estimated_total_runtime_seconds=float(profile["total_seconds"]),
                    confidence=0.99,
                    observations=int(profile["observations"]),
                    source="rtx5090_calibration",
                    metadata={"frozen": True},
                )
            )
            shape_signature = build_batch_probe_shape_signature(job)
            store.upsert_batch_size_observation(
                BatchSizeObservation(
                    observation_key=build_batch_size_observation_key(
                        job.baseline_model_id,
                        shape_signature,
                        hardware_key,
                        backend,
                        int(profile["batch_size"]),
                    ),
                    model_key=job.baseline_model_id,
                    shape_signature=shape_signature,
                    hardware_key=hardware_key,
                    backend_name=backend,
                    batch_param_name="batch_size",
                    batch_size=int(profile["batch_size"]),
                    peak_vram_mb=int(profile["peak_vram_mib"]),
                    avg_vram_mb=float(profile["avg_vram_mib"]),
                    memory_total_mb=int(
                        settings.gpu_scheduler.memory.gpu_vram_gib * 1024
                    ),
                    avg_step_time_ms=float(profile["step_seconds"]) * 1000,
                    observations=int(profile["observations"]),
                    metadata={"estimate_source": "rtx5090_calibration", "frozen": True},
                )
            )
            seeded["runtime"] += 1
            seeded["memory"] += 1
    observed_at = datetime.now(timezone.utc).isoformat()
    for profile in snapshot.get("colocation_profiles", []):
        metadata: dict[str, Any] = {
            "evidence_policy": "fresh_member_epochs_v1",
            "calibration_group": profile["name"],
        }
        if profile["decision"] == "rejected":
            metadata["recent_trial_outcomes"] = [
                {
                    "trial_id": f"calibration-{profile['name']}-{index}",
                    "decision": "rejected",
                    "gain": 0.8,
                    "observed_at": observed_at,
                }
                for index in range(2)
            ]
        store.upsert_colocation_timing_profile(
            ColocationTimingProfile.create(
                hardware_key=hardware_key,
                members=profile["members"],
                member_timings=profile["member_timings"],
                observations=max(2, int(profile["observations"])),
                source="rtx5090_calibration",
                metadata=metadata,
            )
        )
        seeded["colocation"] += 1
    return seeded


def run_scheduler_mode(
    mode: str,
    trace: list[dict[str, Any]],
    snapshot: dict[str, Any],
    output_dir: Path,
    *,
    timeout_s: float,
    total_vram_mib: int,
    smoke: bool,
) -> dict[str, Any]:
    from localml_scheduler.client import SchedulerClient

    runtime_root = output_dir / "runtime"
    result_dir = output_dir / "results"
    baseline_dir = output_dir / "baselines"
    for directory in (runtime_root, result_dir, baseline_dir):
        directory.mkdir(parents=True, exist_ok=True)
    settings = _scheduler_settings(runtime_root, total_vram_mib, smoke=smoke)
    jobs = _build_scheduler_jobs(trace, result_dir, baseline_dir)
    seeded = _seed_scheduler_profiles(settings, jobs, snapshot)
    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)
    origin = time.time()
    deadline = origin + timeout_s
    submitted: dict[str, dict[str, Any]] = {}
    timed_out = False
    try:
        for item in trace:
            target = origin + float(item["release_s"])
            while time.time() < target and time.time() < deadline:
                time.sleep(min(0.05, target - time.time()))
            if time.time() >= deadline:
                timed_out = True
                break
            submitted_job = api.submit(jobs[item["job_id"]])
            submitted[submitted_job.job_id] = item
        while time.time() < deadline:
            current = api.list_jobs()
            if len(current) == len(submitted) and all(
                job.status.value in TERMINAL for job in current
            ):
                break
            time.sleep(0.2)
        else:
            timed_out = True
        if timed_out:
            # Stop owns the host and guarantees its process tree is not orphaned.
            service.stop()
    finally:
        try:
            service.stop()
        except Exception:
            pass

    attempts: list[dict[str, Any]] = []
    logical: list[dict[str, Any]] = []
    for job in api.list_jobs():
        item = submitted.get(job.job_id)
        if item is None:
            continue
        result_path = result_dir / f"{job.job_id}.json"
        result = _read_json(result_path, {})
        started = _iso_epoch(job.started_at)
        finished = _iso_epoch(job.finished_at)
        failure_text = f"{job.status_reason or ''} {result.get('error') or ''}".lower()
        oom = job.status.value == "FAILED" and "out of memory" in failure_text
        record = {
            "logical_job_id": item["job_id"],
            "scheduler_job_id": job.job_id,
            "step_idx": item["step_idx"],
            "scenario": item["scenario"],
            "attempt": 1,
            "retry": False,
            "backend": (job.metadata or {}).get("placement_backend"),
            "stream_host_pid": (job.metadata or {}).get("stream_host_pid")
            or result.get("stream_host_pid"),
            "cuda_stream_id": (job.metadata or {}).get("cuda_stream_id")
            or result.get("cuda_stream_id"),
            "released_at": origin + float(item["release_s"]),
            "started_at": started,
            "finished_at": finished,
            "status": job.status.value.lower(),
            "oom": oom,
            "status_reason": job.status_reason,
            "training_seconds": result.get("training_seconds"),
            "peak_reserved_mib": result.get("peak_reserved_mib"),
            "metadata": job.metadata,
            "result_path": str(result_path),
        }
        attempts.append(record)
        logical.append(
            {
                "logical_job_id": item["job_id"],
                "step_idx": item["step_idx"],
                "scenario": item["scenario"],
                "release_s": item["release_s"],
                "status": record["status"],
                "started_at": started,
                "finished_at": finished,
                "attempt_count": 1,
                "oom_count": 0,
            }
        )
    events = []
    events_path = settings.events_jsonl_path
    if events_path.exists():
        for line in events_path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    shared_assertions = validate_stream_placements(attempts)
    return {
        "mode": mode,
        "origin": origin,
        "deadline": deadline,
        "timed_out": timed_out,
        "attempts": attempts,
        "logical_jobs": logical,
        "events": events,
        "seeded_profiles": seeded,
        "profile_kind": snapshot["kind"],
        "stream_assertions": shared_assertions,
        "runtime_root": str(runtime_root),
    }


def _iso_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp()
    except ValueError:
        return None


def validate_stream_placements(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Assert real overlap means one host PID and distinct CUDA streams."""
    overlaps: list[dict[str, Any]] = []
    valid = True
    reasons: list[str] = []
    completed = [
        item
        for item in attempts
        if item.get("started_at")
        and item.get("finished_at")
        and item.get("backend") == "stream"
    ]
    for index, left in enumerate(completed):
        group = [left]
        for right in completed[index + 1 :]:
            if max(left["started_at"], right["started_at"]) < min(
                left["finished_at"], right["finished_at"]
            ):
                group.append(right)
        if len(group) < 2:
            continue
        pids = {item.get("stream_host_pid") for item in group}
        streams = [item.get("cuda_stream_id") for item in group]
        entry = {
            "jobs": [item["logical_job_id"] for item in group],
            "host_pids": sorted(str(value) for value in pids),
            "stream_ids": streams,
        }
        overlaps.append(entry)
        if None in pids or len(pids) != 1:
            valid = False
            reasons.append(f"overlap does not share one stream host: {entry['jobs']}")
        if None in streams or len(set(streams)) != len(streams):
            valid = False
            reasons.append(f"overlap does not use distinct streams: {entry['jobs']}")
    return {"valid": valid, "overlaps": overlaps, "reasons": sorted(set(reasons))}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _concurrency_residency(attempts: list[dict[str, Any]]) -> dict[str, float]:
    points: list[tuple[float, int]] = []
    for attempt in attempts:
        if attempt.get("started_at") is None or attempt.get("finished_at") is None:
            continue
        points.append((float(attempt["started_at"]), 1))
        points.append((float(attempt["finished_at"]), -1))
    points.sort(key=lambda item: (item[0], item[1]))
    residency: dict[str, float] = {}
    concurrency = 0
    previous = points[0][0] if points else 0.0
    for timestamp, delta in points:
        residency[str(concurrency)] = residency.get(str(concurrency), 0.0) + max(
            0.0, timestamp - previous
        )
        concurrency += delta
        previous = timestamp
    return residency


def _predicted_budget_violations(
    attempts: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    budget_fraction: float = 0.85,
) -> list[dict[str, Any]]:
    fractions = {item["job_id"]: float(item["target_vram_fraction"]) for item in trace}
    points = sorted(
        [
            (float(attempt[key]), delta, attempt["logical_job_id"])
            for attempt in attempts
            if attempt.get("started_at") is not None
            and attempt.get("finished_at") is not None
            for key, delta in (("started_at", 1), ("finished_at", -1))
        ],
        key=lambda item: (item[0], item[1]),
    )
    active: set[str] = set()
    violations: list[dict[str, Any]] = []
    for timestamp, delta, job_id in points:
        if delta < 0:
            active.discard(job_id)
        else:
            active.add(job_id)
            total = sum(fractions.get(item, 0.0) for item in active)
            if total > budget_fraction + 1e-9:
                violations.append(
                    {
                        "timestamp": timestamp,
                        "active_jobs": sorted(active),
                        "predicted_fraction": total,
                    }
                )
    return violations


def summarize_mode(raw: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    origin = float(raw.get("origin") or 0.0)
    jobs = list(raw.get("logical_jobs") or [])
    attempts = list(raw.get("attempts") or [])
    completed = [
        item
        for item in jobs
        if item.get("status") in {"succeeded", "completed"} and item.get("finished_at")
    ]
    waits = [
        max(0.0, float(item["started_at"]) - (origin + float(item["release_s"])))
        for item in completed
        if item.get("started_at")
    ]
    flows = [
        max(0.0, float(item["finished_at"]) - (origin + float(item["release_s"])))
        for item in completed
    ]
    makespan = (
        max((float(item["finished_at"]) for item in completed), default=origin) - origin
    )
    scenario_completion: dict[str, float] = {}
    for scenario in sorted({item["scenario"] for item in trace}):
        members = [
            item
            for item in jobs
            if item.get("scenario") == scenario and item.get("finished_at")
        ]
        if members:
            first_release = min(origin + float(item["release_s"]) for item in members)
            scenario_completion[scenario] = (
                max(float(item["finished_at"]) for item in members) - first_release
            )
    training_sum = sum(
        float(item.get("training_seconds") or 0.0)
        for item in attempts
        if item.get("status") in {"succeeded", "completed"}
    )
    solo_sum = sum(
        float(item.get("calibrated_solo_seconds") or 0.0)
        for item in trace
        if any(
            job["logical_job_id"] == item["job_id"]
            and job.get("status") in {"succeeded", "completed"}
            for job in jobs
        )
    )
    events = list(raw.get("events") or [])
    trial_events = [
        item for item in events if item.get("event_type") == "colocation_trial_started"
    ]
    profile_reuse_count = sum(
        bool(
            (item.get("payload") or {}).get("known_profile")
            or ((item.get("payload") or {}).get("objective_breakdown") or {}).get(
                "known_profile"
            )
        )
        for item in events
    )
    return {
        "mode": raw.get("mode"),
        "timed_out": bool(raw.get("timed_out")),
        "logical_jobs": len(jobs),
        "completed_jobs": len(completed),
        "makespan_seconds": makespan,
        "throughput_jobs_per_hour": (
            len(completed) * 3600.0 / makespan if makespan > 0 else 0.0
        ),
        "flow_seconds": {
            "mean": statistics.fmean(flows) if flows else None,
            "median": statistics.median(flows) if flows else None,
            "p95": _percentile(flows, 0.95),
        },
        "wait_seconds": {
            "mean": statistics.fmean(waits) if waits else None,
            "median": statistics.median(waits) if waits else None,
            "p95": _percentile(waits, 0.95),
        },
        "scenario_completion_seconds": scenario_completion,
        "summed_training_seconds": training_sum,
        "observed_slowdown": training_sum / solo_sum if solo_sum > 0 else None,
        "concurrency_residency_seconds": _concurrency_residency(attempts),
        "predicted_vram_budget_violations": _predicted_budget_violations(
            attempts, trace
        ),
        "oom_attempts": sum(bool(item.get("oom")) for item in attempts),
        "retry_attempts": sum(bool(item.get("retry")) for item in attempts),
        "gpu": raw.get("gpu", {}),
        "profiles_seeded": raw.get("seeded_profiles", {}),
        "profile_reuse_count": profile_reuse_count,
        "live_trial_event_count": len(trial_events),
        "live_trial_overhead_seconds": sum(
            max(0.0, finish - start) for start, finish in _trial_intervals(raw)
        ),
        "stream_assertions": raw.get("stream_assertions"),
    }


def _trial_intervals(raw: dict[str, Any]) -> list[tuple[float, float]]:
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in raw.get("events") or []:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        trial_id = str(payload.get("trial_id") or "")
        timestamp = _iso_epoch(event.get("created_at"))
        if not trial_id or timestamp is None:
            continue
        if event_type == "colocation_trial_started":
            starts[trial_id] = timestamp
        elif event_type.startswith("colocation_trial_") and trial_id in starts:
            intervals.append((starts.pop(trial_id), timestamp))
    return intervals


def _attempt_peak_concurrency(
    target: dict[str, Any], attempts: list[dict[str, Any]]
) -> int:
    start = float(target["started_at"])
    finish = float(target.get("finished_at") or start)
    points: list[tuple[float, int]] = []
    for other in attempts:
        if other.get("started_at") is None or other.get("finished_at") is None:
            continue
        overlap_start = max(start, float(other["started_at"]))
        overlap_finish = min(finish, float(other["finished_at"]))
        if overlap_start < overlap_finish:
            points.extend(((overlap_start, 1), (overlap_finish, -1)))
    active = peak = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def render_gantt(
    output_root: Path,
    trace: list[dict[str, Any]],
    raw_by_mode: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    modes = [mode for mode in ("baseline", "warm", "cold") if mode in raw_by_mode]
    if not modes:
        modes = ["baseline", "warm", "cold"]
    figure, axes = plt.subplots(
        len(modes), 1, figsize=(18, 14), sharex=True, squeeze=False
    )
    order = [item["job_id"] for item in trace]
    trace_by_name = {item["job_id"]: item for item in trace}
    for axis, mode in zip(axes[:, 0], modes, strict=True):
        raw = raw_by_mode.get(mode, {"origin": 0, "attempts": [], "events": []})
        origin = float(raw.get("origin") or 0.0)
        attempts = raw.get("attempts") or []
        for row, name in enumerate(order):
            job = trace_by_name[name]
            release = float(job["release_s"])
            job_attempts = [
                item for item in attempts if item.get("logical_job_id") == name
            ]
            for attempt in job_attempts:
                if attempt.get("started_at") is None:
                    continue
                start = float(attempt["started_at"]) - origin
                finish = (
                    float(
                        attempt.get("finished_at")
                        or raw.get("deadline")
                        or attempt["started_at"]
                    )
                    - origin
                )
                axis.plot(
                    [release, start],
                    [row, row],
                    color="#888888",
                    linewidth=0.8,
                    linestyle=":",
                    zorder=1,
                )
                color = "#d90429" if attempt.get("oom") else job["scenario_color"]
                hatch = "////" if attempt.get("retry") else None
                axis.barh(
                    row,
                    max(0.01, finish - start),
                    left=start,
                    height=0.62,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatch,
                    alpha=0.88,
                )
                backend = attempt.get("backend") or "?"
                stream = attempt.get("cuda_stream_id")
                concurrency = _attempt_peak_concurrency(attempt, attempts)
                host_pid = attempt.get("stream_host_pid")
                annotation = backend + f" c={concurrency}"
                if host_pid is not None:
                    annotation += f" p={host_pid}"
                if stream is not None:
                    annotation += f" s={str(stream)[-6:]}"
                axis.text(
                    start + max(0.02, (finish - start) * 0.02),
                    row,
                    annotation,
                    va="center",
                    ha="left",
                    fontsize=6,
                    color="white" if not attempt.get("oom") else "black",
                    clip_on=True,
                )
        for start, finish in _trial_intervals(raw):
            axis.axvspan(
                start - origin,
                finish - origin,
                color="#ffd166",
                alpha=0.16,
                hatch="..",
                zorder=0,
            )
        mode_deadline = max(0.0, float(raw.get("deadline") or (origin + 5400)) - origin)
        axis.axvline(mode_deadline, color="#d00000", linestyle="--", linewidth=1.5)
        axis.set_yticks(range(len(order)), order, fontsize=7)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(
            f"{mode}: fixed trace order (dotted segment = release-to-start wait)",
            loc="left",
            fontsize=11,
        )
        axis.set_ylabel("logical job")
    axes[-1, 0].set_xlabel("seconds since measured-mode start")
    axes[-1, 0].legend(
        handles=[
            *[
                Patch(facecolor=color, label=name)
                for name, color in SCENARIO_COLORS.items()
            ],
            Patch(facecolor="#d90429", label="OOM attempt"),
            Patch(facecolor="white", edgecolor="black", hatch="////", label="retry"),
            Patch(facecolor="#ffd166", alpha=0.25, hatch="..", label="live trial"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        fontsize=8,
    )
    figure.tight_layout()
    png = output_root / "gantt-comparison.png"
    pdf = output_root / "gantt-comparison.pdf"
    figure.savefig(png, dpi=170, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def analyze_results(output_root: Path, *, deadline_s: float = 5400.0) -> dict[str, Any]:
    trace = _read_json(
        output_root / "calibration" / "calibrated_trace.json"
    ) or _read_json(output_root / "trace.json", [])
    raw_by_mode: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for mode in ("baseline", "warm", "cold"):
        raw = _read_json(output_root / mode / "raw.json")
        if raw:
            raw_by_mode[mode] = raw
            summaries.append(summarize_mode(raw, trace))
    png, pdf = render_gantt(output_root, trace, raw_by_mode)
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deadline_seconds": deadline_s,
        "modes": summaries,
        "gantt_png": str(png),
        "gantt_pdf": str(pdf),
    }
    _write_json(output_root / "comparison-summary.json", comparison)
    for summary in summaries:
        _write_json(output_root / str(summary["mode"]) / "summary.json", summary)
    with (output_root / "comparison-summary.csv").open("w", newline="") as handle:
        columns = [
            "mode",
            "completed_jobs",
            "makespan_seconds",
            "throughput_jobs_per_hour",
            "mean_flow_seconds",
            "median_flow_seconds",
            "p95_flow_seconds",
            "mean_wait_seconds",
            "median_wait_seconds",
            "p95_wait_seconds",
            "summed_training_seconds",
            "observed_slowdown",
            "oom_attempts",
            "retry_attempts",
            "predicted_vram_budget_violation_count",
            "avg_gpu_util_percent",
            "avg_power_w",
            "energy_wh",
            "profile_reuse_count",
            "live_trial_event_count",
            "live_trial_overhead_seconds",
            "concurrency_residency_seconds",
            "scenario_completion_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **{key: summary.get(key) for key in columns},
                    "mean_flow_seconds": summary["flow_seconds"]["mean"],
                    "median_flow_seconds": summary["flow_seconds"]["median"],
                    "p95_flow_seconds": summary["flow_seconds"]["p95"],
                    "mean_wait_seconds": summary["wait_seconds"]["mean"],
                    "median_wait_seconds": summary["wait_seconds"]["median"],
                    "p95_wait_seconds": summary["wait_seconds"]["p95"],
                    "predicted_vram_budget_violation_count": len(
                        summary["predicted_vram_budget_violations"]
                    ),
                    "avg_gpu_util_percent": (summary.get("gpu") or {}).get(
                        "avg_gpu_util_percent"
                    ),
                    "avg_power_w": (summary.get("gpu") or {}).get("avg_power_w"),
                    "energy_wh": (summary.get("gpu") or {}).get("energy_wh"),
                    "concurrency_residency_seconds": json.dumps(
                        summary["concurrency_residency_seconds"], sort_keys=True
                    ),
                    "scenario_completion_seconds": json.dumps(
                        summary["scenario_completion_seconds"], sort_keys=True
                    ),
                }
            )
    lines = [
        "# RTX 5090 scheduler pressure benchmark",
        "",
        f"Generated: {comparison['generated_at']}",
        "",
        "Each measured mode is capped at 90 minutes; calibration and GPU cooldown are outside that cap.",
        "",
        "| Mode | Completed | Makespan (s) | Throughput (jobs/h) | Mean flow (s) | p95 wait (s) | OOM/retries | Energy (Wh) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        gpu = summary.get("gpu") or {}
        lines.append(
            f"| {summary['mode']} | {summary['completed_jobs']}/{summary['logical_jobs']} | {summary['makespan_seconds']:.2f} | {summary['throughput_jobs_per_hour']:.2f} | {_fmt(summary['flow_seconds']['mean'])} | {_fmt(summary['wait_seconds']['p95'])} | {summary['oom_attempts']}/{summary['retry_attempts']} | {_fmt(gpu.get('energy_wh'))} |"
        )
    lines.extend(["", "## Scenario completion (seconds)", ""])
    scenarios = sorted(
        {
            scenario
            for summary in summaries
            for scenario in summary["scenario_completion_seconds"]
        }
    )
    lines.append(
        "| Scenario | " + " | ".join(summary["mode"] for summary in summaries) + " |"
    )
    lines.append("|---|" + "---:|" * len(summaries))
    for scenario in scenarios:
        lines.append(
            "| "
            + scenario
            + " | "
            + " | ".join(
                _fmt(summary["scenario_completion_seconds"].get(scenario))
                for summary in summaries
            )
            + " |"
        )
    lines.extend(["", "## Evidence and safety", ""])
    for summary in summaries:
        lines.append(
            f"- {summary['mode']}: training sum={summary['summed_training_seconds']:.2f}s; observed slowdown={_fmt(summary['observed_slowdown'])}; concurrency residency={summary['concurrency_residency_seconds']}; predicted VRAM violations={len(summary['predicted_vram_budget_violations'])}; profiles seeded={summary['profiles_seeded']}; profile reuse={summary['profile_reuse_count']}; live trials={summary['live_trial_event_count']} ({summary['live_trial_overhead_seconds']:.2f}s); GPU util/power/energy={_fmt((summary.get('gpu') or {}).get('avg_gpu_util_percent'))}%/{_fmt((summary.get('gpu') or {}).get('avg_power_w'))}W/{_fmt((summary.get('gpu') or {}).get('energy_wh'))}Wh; stream assertions={summary.get('stream_assertions')}."
        )
    lines.extend(["", f"![Three-panel Gantt]({png.name})", ""])
    (output_root / "REPORT.md").write_text("\n".join(lines))
    return comparison


def validate_smoke_acceptance(
    trace: list[dict[str, Any]],
    raw_by_mode: dict[str, dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    for mode in ("baseline", "warm", "cold"):
        raw = raw_by_mode.get(mode, {})
        completed = sum(
            job.get("status") in {"completed", "succeeded"}
            for job in raw.get("logical_jobs") or []
        )
        if completed != 16:
            reasons.append(f"{mode} completed {completed}/16 logical jobs")
    baseline = raw_by_mode.get("baseline", {})
    oom_jobs = {
        item["logical_job_id"]
        for item in baseline.get("attempts") or []
        if item.get("oom")
    }
    for job_id in oom_jobs:
        if not any(
            item.get("logical_job_id") == job_id
            and item.get("retry")
            and item.get("status") == "succeeded"
            for item in baseline.get("attempts") or []
        ):
            reasons.append(f"baseline OOM job {job_id} has no successful solo retry")

    summary_by_mode = {item["mode"]: item for item in summaries}
    for mode in ("warm", "cold"):
        raw = raw_by_mode.get(mode, {})
        assertions = raw.get("stream_assertions") or {}
        if not assertions.get("valid") or not assertions.get("overlaps"):
            reasons.append(
                f"{mode} did not demonstrate valid shared-host stream overlap"
            )
        if any(item.get("oom") for item in raw.get("attempts") or []):
            reasons.append(f"{mode} encountered a CUDA OOM")
        if (summary_by_mode.get(mode) or {}).get("predicted_vram_budget_violations"):
            reasons.append(f"{mode} exceeded the predicted 85% VRAM budget")
        attempts = raw.get("attempts") or []
        points = sorted(
            [
                (float(item[key]), delta, item)
                for item in attempts
                if item.get("started_at") and item.get("finished_at")
                for key, delta in (("started_at", 1), ("finished_at", -1))
            ],
            key=lambda value: (value[0], value[1]),
        )
        active: list[dict[str, Any]] = []
        light_max = 0
        for _, delta, item in points:
            if delta < 0:
                active = [candidate for candidate in active if candidate is not item]
            else:
                active.append(item)
                light_max = max(
                    light_max,
                    sum(
                        str(candidate["logical_job_id"]).startswith("light-")
                        for candidate in active
                    ),
                )
        if light_max < 3:
            reasons.append(f"{mode} reached only {light_max}-way light-job concurrency")

        scheduler_ids = {
            item["scheduler_job_id"]: item["logical_job_id"]
            for item in attempts
            if item.get("scheduler_job_id")
        }
        compute_ids = {
            scheduler_id
            for scheduler_id, logical in scheduler_ids.items()
            if logical.startswith("compute-heavy-")
        }
        for event in raw.get("events") or []:
            if event.get("event_type") != "colocation_trial_accepted":
                continue
            payload = event.get("payload") or {}
            members = {
                str(event.get("job_id")),
                *[str(value) for value in payload.get("preexisting_job_ids") or []],
                *[
                    str(value)
                    for value in (payload.get("packed_epoch_seconds") or {}).keys()
                ],
            }
            if members & compute_ids:
                reasons.append(f"{mode} accepted colocation with a compute-heavy job")
        near_attempts = [
            item
            for item in attempts
            if str(item["logical_job_id"]).startswith("near-exclusive")
        ]
        for near in near_attempts:
            for other in attempts:
                if near is other or not all(
                    near.get(key) and other.get(key)
                    for key in ("started_at", "finished_at")
                ):
                    continue
                if max(float(near["started_at"]), float(other["started_at"])) < min(
                    float(near["finished_at"]), float(other["finished_at"])
                ):
                    reasons.append(
                        f"{mode} overlapped near-exclusive job {near['logical_job_id']}"
                    )
                    break
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _fmt(value: Any) -> str:
    comparison: dict[str, Any] | None = None
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _phase_marker(output_root: Path, phase: str) -> Path:
    return output_root / ".phases" / f"{phase}.complete.json"


def _mark_phase(output_root: Path, phase: str, payload: dict[str, Any]) -> None:
    _write_json(
        _phase_marker(output_root, phase),
        {"completed_at": datetime.now(timezone.utc).isoformat(), **payload},
    )


def run_full(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = hardware_manifest(args.device_index)
    _write_json(output_root / "hardware-software-manifest.json", manifest)
    capability = tuple(manifest.get("compute_capability") or ())
    if not manifest.get("cuda_available") or capability < (12, 0):
        raise RuntimeError(
            f"RTX 5090 CUDA capability 12.0+ is required, detected {manifest}"
        )
    if "5090" not in str(manifest.get("gpu_name", "")) and not args.allow_non_5090:
        raise RuntimeError(
            f"expected an RTX 5090, detected {manifest.get('gpu_name')!r}; pass --allow-non-5090 only for development"
        )
    total_vram_mib = int(manifest["total_vram_mib"])
    initial_trace = build_trace(total_vram_mib, smoke=args.smoke)
    validate_trace(initial_trace)
    _write_json(output_root / "trace.json", initial_trace)

    calibration_dir = output_root / "calibration"
    if (
        _phase_marker(output_root, "calibration").exists()
        and (calibration_dir / "calibrated_trace.json").exists()
    ):
        trace = _read_json(calibration_dir / "calibrated_trace.json")
        qualification = _read_json(calibration_dir / "calibration.json")
    else:
        trace, qualification = calibrate_trace(
            initial_trace, calibration_dir, smoke=args.smoke
        )
        _mark_phase(output_root, "calibration", {"valid": qualification["valid"]})
    warm_snapshot, cold_snapshot = build_profile_snapshots(
        trace, qualification, calibration_dir
    )
    snapshots = {"warm": warm_snapshot, "cold": cold_snapshot}

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in {"baseline", "warm", "cold"}]
    if invalid:
        raise ValueError(f"unknown modes: {invalid}")
    reference_temperature = (
        float((manifest.get("nvidia_smi") or {}).get("temperature_c") or 0) or None
    )
    reference_memory = float(
        (manifest.get("nvidia_smi") or {}).get("memory_used_mib") or 0
    )
    failures: list[str] = []
    try:
        for mode in modes:
            marker = _phase_marker(output_root, mode)
            raw_path = output_root / mode / "raw.json"
            if marker.exists() and raw_path.exists():
                continue
            mode_dir = output_root / mode
            if mode_dir.exists() and any(mode_dir.iterdir()):
                archive = (
                    output_root
                    / f"{mode}.incomplete-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                )
                suffix = 1
                while archive.exists():
                    archive = (
                        output_root
                        / f"{mode}.incomplete-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{suffix}"
                    )
                    suffix += 1
                mode_dir.replace(archive)
            mode_dir.mkdir(parents=True, exist_ok=True)
            cooldown = wait_for_idle_gpu(
                device_index=args.device_index,
                reference_temperature_c=reference_temperature,
                timeout_s=30 if args.smoke else args.cooldown_timeout,
                strict=not args.smoke,
                reference_memory_mib=reference_memory,
            )
            _write_json(mode_dir / "gpu-cooldown.json", cooldown)
            sampler = TelemetrySampler(
                mode_dir / "gpu-telemetry.csv",
                device_index=args.device_index,
                interval=0.5,
            )
            sampler.start()
            origin = time.time()
            try:
                if mode == "baseline":
                    raw = run_mp2_baseline(trace, mode_dir, timeout_s=args.mode_timeout)
                else:
                    raw = run_scheduler_mode(
                        mode,
                        trace,
                        snapshots[mode],
                        mode_dir,
                        timeout_s=args.mode_timeout,
                        total_vram_mib=total_vram_mib,
                        smoke=args.smoke,
                    )
            except Exception as exc:
                failures.append(f"{mode}: {exc!r}")
                raw = {
                    "mode": mode,
                    "origin": origin,
                    "deadline": origin + args.mode_timeout,
                    "timed_out": False,
                    "attempts": [],
                    "logical_jobs": [],
                    "events": [],
                    "error": repr(exc),
                }
            raw["gpu"] = sampler.stop(origin=float(raw.get("origin") or origin))
            raw["cooldown"] = cooldown
            _write_json(raw_path, raw)
            _write_json(mode_dir / "raw-job-attempts.json", raw.get("attempts") or [])
            _write_json(
                mode_dir / "logical-job-results.json", raw.get("logical_jobs") or []
            )
            _write_json(mode_dir / "scheduler-events.json", raw.get("events") or [])
            _write_records_csv(
                mode_dir / "raw-job-attempts.csv", raw.get("attempts") or []
            )
            _write_records_csv(
                mode_dir / "logical-job-results.csv", raw.get("logical_jobs") or []
            )
            if not raw.get("error"):
                _mark_phase(
                    output_root,
                    mode,
                    {"timed_out": raw.get("timed_out"), "error": None},
                )
    finally:
        comparison = analyze_results(output_root, deadline_s=args.mode_timeout)
    if args.smoke:
        raw_by_mode = {
            mode: _read_json(output_root / mode / "raw.json", {})
            for mode in ("baseline", "warm", "cold")
        }
        acceptance = validate_smoke_acceptance(
            trace, raw_by_mode, list((comparison or {}).get("modes") or [])
        )
        _write_json(output_root / "smoke-acceptance.json", acceptance)
        if not acceptance["valid"]:
            failures.append("smoke acceptance: " + "; ".join(acceptance["reasons"]))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser("full", help="run calibration and measured modes")
    full.add_argument("--output-root", required=True)
    full.add_argument("--modes", default="baseline,warm,cold")
    full.add_argument("--mode-timeout", type=float, default=5400.0)
    full.add_argument("--cooldown-timeout", type=float, default=900.0)
    full.add_argument("--device-index", type=int, default=0)
    full.add_argument("--smoke", action="store_true")
    full.add_argument("--allow-non-5090", action="store_true")
    analyze = subparsers.add_parser(
        "analyze", help="render summaries and partial Gantt"
    )
    analyze.add_argument("--output-root", required=True)
    analyze.add_argument("--deadline", type=float, default=5400.0)
    group = subparsers.add_parser("group-worker", help=argparse.SUPPRESS)
    group.add_argument("--input", required=True)
    group.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "group-worker":
        return group_worker(Path(args.input), Path(args.output))
    if args.command == "analyze":
        analyze_results(Path(args.output_root), deadline_s=args.deadline)
        return 0
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
