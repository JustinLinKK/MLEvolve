"""Lightweight CPU/RAM/GPU utilization monitoring for MLEvolve runs."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


logger = logging.getLogger("MLEvolve")


CSV_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "cpu_percent_avg",
    "cpu_percent_p95",
    "cpu_percent_max",
    "cpu_percent_per_core",
    "ram_used_mb",
    "ram_total_mb",
    "ram_percent",
    "gpu_index",
    "gpu_name",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_memory_percent",
    "gpu_util_percent",
    "gpu_memory_util_percent",
    "gpu_power_draw_w",
    "gpu_power_limit_w",
    "gpu_temperature_c",
    "sample_error",
]


@dataclass
class GpuSample:
    index: str
    name: str
    memory_total_mb: float
    memory_used_mb: float
    gpu_util_percent: float
    memory_util_percent: float
    power_draw_w: float | None
    power_limit_w: float | None
    temperature_c: float | None

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0


@dataclass
class HardwareSample:
    timestamp: str
    elapsed_seconds: float
    cpu_percent_per_core: list[float]
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    gpu_samples: list[GpuSample]
    sample_error: str = ""

    @property
    def cpu_avg(self) -> float:
        return _mean(self.cpu_percent_per_core)

    @property
    def cpu_p95(self) -> float:
        return _percentile(self.cpu_percent_per_core, 95.0)

    @property
    def cpu_max(self) -> float:
        return max(self.cpu_percent_per_core) if self.cpu_percent_per_core else 0.0


class HardwareMonitor:
    """Background hardware sampler with CSV output and markdown reporting."""

    def __init__(self, cfg: Any, log: logging.Logger | None = None):
        monitor_cfg = getattr(cfg, "monitor", None)
        self.enabled = bool(getattr(monitor_cfg, "enabled", True))
        self.interval_seconds = max(
            0.1, float(getattr(monitor_cfg, "interval_seconds", 5))
        )
        self.gpu_idle_util_threshold = float(
            getattr(monitor_cfg, "gpu_idle_util_threshold", 10)
        )
        self.gpu_idle_memory_threshold_mb = float(
            getattr(monitor_cfg, "gpu_idle_memory_threshold_mb", 1024)
        )
        self.cpu_idle_util_threshold = float(
            getattr(monitor_cfg, "cpu_idle_util_threshold", 20)
        )
        self.log_dir = Path(cfg.log_dir)
        self.samples_path = self.log_dir / "hardware_samples.csv"
        self.report_path = self.log_dir / "hardware_report.md"
        self.logger = log or logger

        self._samples: list[HardwareSample] = []
        self._warnings: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        self._start_monotonic = 0.0
        self._end_monotonic = 0.0
        self._start_wall = ""
        self._end_wall = ""
        self._nvidia_smi_path = shutil.which("nvidia-smi")
        self._gpu_discovery_status = "not checked"
        self._cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")

    def start(self) -> None:
        if not self.enabled:
            self.logger.info("Hardware monitor disabled by config.")
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start_monotonic = time.monotonic()
        self._start_wall = _utc_now()
        self._csv_file = self.samples_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        psutil.cpu_percent(interval=None, percpu=True)
        self._collect_and_record_sample()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="hardware-monitor",
            daemon=True,
        )
        self._thread.start()
        self.logger.info(
            "Hardware monitor started: interval=%ss, samples=%s, report=%s",
            self.interval_seconds,
            self.samples_path,
            self.report_path,
        )

    def stop(self) -> None:
        if not self.enabled:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(6.0, self.interval_seconds + 1.0))
        self._end_monotonic = time.monotonic()
        self._end_wall = _utc_now()

        try:
            self._collect_and_record_sample()
        except Exception as exc:
            self._warnings.append(f"Final sample failed: {exc}")
            self.logger.warning("Final hardware sample failed: %s", exc)

        try:
            self._write_report()
            self.logger.info("Hardware report written to %s", self.report_path)
        finally:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self._collect_and_record_sample()
            except Exception as exc:
                self._warnings.append(f"Sample failed: {exc}")
                self.logger.warning("Hardware monitor sample failed: %s", exc)

    def _collect_and_record_sample(self) -> None:
        sample = self._collect_sample()
        with self._lock:
            self._samples.append(sample)
            self._write_sample_rows(sample)

    def _collect_sample(self) -> HardwareSample:
        timestamp = _utc_now()
        elapsed = time.monotonic() - self._start_monotonic
        sample_error = ""

        cpu_per_core = [float(v) for v in psutil.cpu_percent(interval=None, percpu=True)]
        memory = psutil.virtual_memory()
        ram_used_mb = float(memory.used) / (1024 * 1024)
        ram_total_mb = float(memory.total) / (1024 * 1024)
        gpu_samples: list[GpuSample] = []

        try:
            gpu_samples = self._query_gpu_samples()
        except Exception as exc:
            sample_error = str(exc)
            if sample_error not in self._warnings:
                self._warnings.append(sample_error)

        return HardwareSample(
            timestamp=timestamp,
            elapsed_seconds=elapsed,
            cpu_percent_per_core=cpu_per_core,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            ram_percent=float(memory.percent),
            gpu_samples=gpu_samples,
            sample_error=sample_error,
        )

    def _query_gpu_samples(self) -> list[GpuSample]:
        if not self._nvidia_smi_path:
            self._gpu_discovery_status = "nvidia-smi not found"
            raise RuntimeError("nvidia-smi not found; GPU telemetry unavailable")

        query = (
            "index,name,memory.total,memory.used,utilization.gpu,"
            "utilization.memory,power.draw,power.limit,temperature.gpu"
        )
        cmd = [
            self._nvidia_smi_path,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            self._gpu_discovery_status = f"nvidia-smi failed: {message}"
            raise RuntimeError(self._gpu_discovery_status)

        rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
        if not rows:
            self._gpu_discovery_status = "no GPUs reported by nvidia-smi"
            return []

        self._gpu_discovery_status = f"{len(rows)} GPU(s) reported by nvidia-smi"
        samples: list[GpuSample] = []
        for row in rows:
            padded = [value.strip() for value in row] + [""] * 9
            samples.append(
                GpuSample(
                    index=padded[0],
                    name=padded[1],
                    memory_total_mb=_parse_float(padded[2]),
                    memory_used_mb=_parse_float(padded[3]),
                    gpu_util_percent=_parse_float(padded[4]),
                    memory_util_percent=_parse_float(padded[5]),
                    power_draw_w=_parse_optional_float(padded[6]),
                    power_limit_w=_parse_optional_float(padded[7]),
                    temperature_c=_parse_optional_float(padded[8]),
                )
            )
        return samples

    def _write_sample_rows(self, sample: HardwareSample) -> None:
        if self._csv_writer is None or self._csv_file is None:
            return

        gpu_samples = sample.gpu_samples or [None]
        for gpu in gpu_samples:
            row = {
                "timestamp": sample.timestamp,
                "elapsed_seconds": f"{sample.elapsed_seconds:.3f}",
                "cpu_percent_avg": f"{sample.cpu_avg:.2f}",
                "cpu_percent_p95": f"{sample.cpu_p95:.2f}",
                "cpu_percent_max": f"{sample.cpu_max:.2f}",
                "cpu_percent_per_core": ";".join(
                    f"{value:.1f}" for value in sample.cpu_percent_per_core
                ),
                "ram_used_mb": f"{sample.ram_used_mb:.1f}",
                "ram_total_mb": f"{sample.ram_total_mb:.1f}",
                "ram_percent": f"{sample.ram_percent:.2f}",
                "gpu_index": "",
                "gpu_name": "",
                "gpu_memory_used_mb": "",
                "gpu_memory_total_mb": "",
                "gpu_memory_percent": "",
                "gpu_util_percent": "",
                "gpu_memory_util_percent": "",
                "gpu_power_draw_w": "",
                "gpu_power_limit_w": "",
                "gpu_temperature_c": "",
                "sample_error": sample.sample_error,
            }
            if gpu is not None:
                row.update(
                    {
                        "gpu_index": gpu.index,
                        "gpu_name": gpu.name,
                        "gpu_memory_used_mb": f"{gpu.memory_used_mb:.1f}",
                        "gpu_memory_total_mb": f"{gpu.memory_total_mb:.1f}",
                        "gpu_memory_percent": f"{gpu.memory_percent:.2f}",
                        "gpu_util_percent": f"{gpu.gpu_util_percent:.2f}",
                        "gpu_memory_util_percent": f"{gpu.memory_util_percent:.2f}",
                        "gpu_power_draw_w": _format_optional(gpu.power_draw_w),
                        "gpu_power_limit_w": _format_optional(gpu.power_limit_w),
                        "gpu_temperature_c": _format_optional(gpu.temperature_c),
                    }
                )
            self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _write_report(self) -> None:
        with self._lock:
            samples = list(self._samples)
            warnings = list(self._warnings)

        duration_seconds = self._duration_seconds(samples)
        cpu_avgs = [sample.cpu_avg for sample in samples]
        cpu_maxes = [sample.cpu_max for sample in samples]
        ram_used = [sample.ram_used_mb for sample in samples]
        cpu_idle_count = sum(
            1 for sample in samples if sample.cpu_avg < self.cpu_idle_util_threshold
        )
        sample_count = len(samples)
        cpu_idle_percent = _ratio_percent(cpu_idle_count, sample_count)

        lines = [
            "# Hardware Utilization Report",
            "",
            "## Run Summary",
            "",
            f"- Start time (UTC): {self._start_wall or 'unknown'}",
            f"- End time (UTC): {self._end_wall or _utc_now()}",
            f"- Duration: {_format_duration(duration_seconds)}",
            f"- Sampling interval: {self.interval_seconds:g}s",
            f"- Samples: {sample_count}",
            f"- CUDA_VISIBLE_DEVICES: `{self._cuda_visible_devices}`",
            f"- GPU discovery: {self._gpu_discovery_status}",
            f"- Raw samples: `{self.samples_path.name}`",
            "",
            "## CPU and Memory",
            "",
            f"- CPU average utilization: {_format_percent(_mean(cpu_avgs))}",
            f"- CPU p95 utilization: {_format_percent(_percentile(cpu_avgs, 95.0))}",
            f"- CPU max observed core utilization: {_format_percent(max(cpu_maxes) if cpu_maxes else 0.0)}",
            f"- CPU idle sampled time (< {self.cpu_idle_util_threshold:g}% avg): {_format_percent(cpu_idle_percent)}",
            f"- RAM average used: {_format_mb(_mean(ram_used))}",
            f"- RAM max used: {_format_mb(max(ram_used) if ram_used else 0.0)}",
            f"- RAM total: {_format_mb(samples[-1].ram_total_mb if samples else 0.0)}",
            "",
            "## GPU Utilization",
            "",
        ]

        gpu_by_index = _group_gpu_samples(samples)
        if not gpu_by_index:
            lines.extend(
                [
                    "No GPU samples were captured.",
                    "",
                ]
            )
        else:
            for gpu_index, gpu_samples in gpu_by_index.items():
                name = gpu_samples[-1].name
                util_values = [gpu.gpu_util_percent for gpu in gpu_samples]
                mem_used_values = [gpu.memory_used_mb for gpu in gpu_samples]
                mem_percent_values = [gpu.memory_percent for gpu in gpu_samples]
                idle_compute_count = sum(
                    1
                    for gpu in gpu_samples
                    if gpu.gpu_util_percent < self.gpu_idle_util_threshold
                )
                idle_memory_samples = [
                    gpu
                    for gpu in gpu_samples
                    if gpu.gpu_util_percent < self.gpu_idle_util_threshold
                    and gpu.memory_used_mb >= self.gpu_idle_memory_threshold_mb
                ]
                idle_compute_percent = _ratio_percent(idle_compute_count, len(gpu_samples))
                idle_memory_percent = _ratio_percent(
                    len(idle_memory_samples), len(gpu_samples)
                )
                wasted_gib_hours = (
                    sum(gpu.memory_used_mb for gpu in idle_memory_samples)
                    / 1024.0
                    * self.interval_seconds
                    / 3600.0
                )

                lines.extend(
                    [
                        f"### GPU {gpu_index}: {name}",
                        "",
                        f"- GPU compute utilization avg/p95/max: {_format_percent(_mean(util_values))} / {_format_percent(_percentile(util_values, 95.0))} / {_format_percent(max(util_values) if util_values else 0.0)}",
                        f"- GPU memory used avg/p95/max: {_format_mb(_mean(mem_used_values))} / {_format_mb(_percentile(mem_used_values, 95.0))} / {_format_mb(max(mem_used_values) if mem_used_values else 0.0)}",
                        f"- GPU memory occupancy avg/p95/max: {_format_percent(_mean(mem_percent_values))} / {_format_percent(_percentile(mem_percent_values, 95.0))} / {_format_percent(max(mem_percent_values) if mem_percent_values else 0.0)}",
                        f"- Compute-idle sampled time (< {self.gpu_idle_util_threshold:g}% util): {_format_percent(idle_compute_percent)}",
                        f"- Idle-memory sampled time (< {self.gpu_idle_util_threshold:g}% util and >= {_format_mb(self.gpu_idle_memory_threshold_mb)} used): {_format_percent(idle_memory_percent)}",
                        f"- Estimated wasted GPU memory: {wasted_gib_hours:.3f} GiB-hours",
                        "",
                    ]
                )

        lines.extend(self._resource_waste_lines(samples, gpu_by_index, cpu_idle_percent))
        if warnings:
            lines.extend(["", "## Telemetry Warnings", ""])
            for warning in sorted(set(warnings)):
                lines.append(f"- {warning}")

        self.report_path.write_text("\n".join(lines) + "\n")

    def _duration_seconds(self, samples: list[HardwareSample]) -> float:
        if self._end_monotonic and self._start_monotonic:
            return max(0.0, self._end_monotonic - self._start_monotonic)
        if samples:
            return max(0.0, samples[-1].elapsed_seconds)
        return 0.0

    def _resource_waste_lines(
        self,
        samples: list[HardwareSample],
        gpu_by_index: dict[str, list[GpuSample]],
        cpu_idle_percent: float,
    ) -> list[str]:
        lines = ["## Resource Waste Signals", ""]
        signals: list[str] = []

        if cpu_idle_percent >= 75.0:
            signals.append(
                f"CPU appears underutilized: {cpu_idle_percent:.1f}% of samples were below {self.cpu_idle_util_threshold:g}% average utilization."
            )

        if not gpu_by_index:
            signals.append(
                "GPU telemetry is unavailable, so GPU memory or compute waste could not be evaluated."
            )
        else:
            for gpu_index, gpu_samples in gpu_by_index.items():
                util_values = [gpu.gpu_util_percent for gpu in gpu_samples]
                avg_util = _mean(util_values)
                idle_memory_count = sum(
                    1
                    for gpu in gpu_samples
                    if gpu.gpu_util_percent < self.gpu_idle_util_threshold
                    and gpu.memory_used_mb >= self.gpu_idle_memory_threshold_mb
                )
                idle_memory_percent = _ratio_percent(idle_memory_count, len(gpu_samples))
                if avg_util < self.gpu_idle_util_threshold:
                    signals.append(
                        f"GPU {gpu_index} compute utilization is consistently low: average {_format_percent(avg_util)}."
                    )
                if idle_memory_percent >= 25.0:
                    signals.append(
                        f"GPU {gpu_index} has likely idle memory reservation: {idle_memory_percent:.1f}% of samples had low compute with >= {_format_mb(self.gpu_idle_memory_threshold_mb)} allocated."
                    )

        if not signals:
            signals.append("No major resource waste signals crossed the configured thresholds.")

        lines.extend(f"- {signal}" for signal in signals)
        return lines


def _group_gpu_samples(samples: list[HardwareSample]) -> dict[str, list[GpuSample]]:
    grouped: dict[str, list[GpuSample]] = {}
    for sample in samples:
        for gpu in sample.gpu_samples:
            grouped.setdefault(gpu.index, []).append(gpu)
    return grouped


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_float(value: str) -> float:
    parsed = _parse_optional_float(value)
    return parsed if parsed is not None else 0.0


def _parse_optional_float(value: str) -> float | None:
    clean = value.strip()
    if not clean or clean.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(clean)
    except ValueError:
        return None


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _ratio_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def _format_mb(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024.0:.2f} GiB"
    return f"{value:.1f} MiB"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
