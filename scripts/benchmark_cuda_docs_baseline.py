#!/usr/bin/env python3
"""Reproduce the pre-CUDA-docs generation/throughput baseline from a trace."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from statistics import median, quantiles
import json


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[94]


def build_baseline(trace_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generation: dict[str, dict[str, float | int]] = {}
    for role in sorted({str(row.get("agent_used") or "unknown") for row in rows}):
        values = [
            float(row["gen_duration_s"])
            for row in rows
            if str(row.get("agent_used") or "unknown") == role
            and row.get("gen_duration_s") is not None
        ]
        generation[role] = {
            "samples": len(values),
            "median_seconds": median(values) if values else None,
            "p95_seconds": percentile_95(values),
        }
    starts = [
        float(row["gen_start_at"])
        for row in rows
        if row.get("gen_start_at") is not None
    ]
    ends = [
        float(row["exec_complete_at"])
        for row in rows
        if row.get("exec_complete_at") is not None
    ]
    span = max(ends) - min(starts) if starts and ends else None
    try:
        source_trace = str(trace_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        source_trace = str(trace_path)
    return {
        "schema_version": "cuda_docs_preintegration_baseline_v1",
        "source_trace": source_trace,
        "feature_state": "pre_cuda_docs_gateway",
        "node_count": len(rows),
        "generation_latency_by_role": generation,
        "node_throughput_per_hour": len(rows) / (span / 3600.0) if span else None,
        "local_hardware_context_latency_seconds": {
            "value": None,
            "status": "not_instrumented_in_source_trace",
        },
        "qdrant_latency_seconds": {
            "value": None,
            "status": "not_instrumented_in_source_trace",
        },
        "notes": (
            "The historic trace contains exact per-role generation and node timing, "
            "but predates tier-labelled local-context/Qdrant metrics. Those fields are "
            "explicitly null rather than replaced with synthetic measurements."
        ),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_baseline(args.trace)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
