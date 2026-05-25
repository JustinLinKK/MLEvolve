from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from utils.hardware_monitor import (
    GpuSample,
    HardwareMonitor,
    HardwareSample,
    _compress_csv_rows,
)


def _row(index: int) -> dict[str, str]:
    return {
        "timestamp": f"t{index}",
        "elapsed_seconds": str(float(index)),
        "cpu_percent_avg": str(float(index)),
        "cpu_percent_p95": str(float(index)),
        "cpu_percent_max": str(float(index)),
        "cpu_percent_per_core": f"{index};{index + 1}",
        "ram_used_mb": str(1000 + index),
        "ram_total_mb": "64000",
        "ram_percent": str(float(index)),
        "gpu_index": "0",
        "gpu_name": "unit-gpu",
        "gpu_memory_used_mb": str(2000 + index),
        "gpu_memory_total_mb": "32768",
        "gpu_memory_percent": str(float(index)),
        "gpu_util_percent": str(float(index)),
        "gpu_memory_util_percent": str(float(index)),
        "gpu_power_draw_w": str(100 + index),
        "gpu_power_limit_w": "450",
        "gpu_temperature_c": str(40 + index),
        "sample_error": "",
        "sample_count": "1",
        "window_start": f"t{index}",
        "window_end": f"t{index}",
    }


def test_compress_csv_rows_preserves_row_cap_weights_and_maxima() -> None:
    rows = [_row(index) for index in range(25)]

    compressed = _compress_csv_rows(rows, target_rows=10)

    assert len(compressed) <= 10
    assert sum(int(row.get("sample_count") or "1") for row in compressed) == 25
    assert max(float(row["cpu_percent_max"]) for row in compressed) == 24.0


def test_hardware_monitor_writes_bounded_compressed_csv(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        log_dir=tmp_path,
        monitor=SimpleNamespace(
            enabled=True,
            interval_seconds=5,
            gpu_idle_util_threshold=10,
            gpu_idle_memory_threshold_mb=1024,
            cpu_idle_util_threshold=20,
            adaptive_compression=True,
            max_csv_rows=10,
            compress_to_rows=5,
        ),
    )
    monitor = HardwareMonitor(cfg)

    for index in range(18):
        sample = HardwareSample(
            timestamp=f"t{index}",
            elapsed_seconds=float(index),
            cpu_percent_per_core=[float(index), float(index + 1)],
            ram_used_mb=1000 + index,
            ram_total_mb=64000,
            ram_percent=float(index),
            gpu_samples=[
                GpuSample(
                    index="0",
                    name="unit-gpu",
                    memory_total_mb=32768,
                    memory_used_mb=2000 + index,
                    gpu_util_percent=float(index),
                    memory_util_percent=float(index),
                    power_draw_w=100 + index,
                    power_limit_w=450,
                    temperature_c=40 + index,
                )
            ],
        )
        monitor._write_sample_rows(sample)

    with (tmp_path / "hardware_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) <= 10
    assert "sample_count" in rows[0]
    assert sum(int(row.get("sample_count") or "1") for row in rows) == 18
