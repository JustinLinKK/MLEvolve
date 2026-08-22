"""Summarize paired context-cache trials into JSONL, CSV, and Markdown.

The harness intentionally does not issue billable provider requests itself.
Callers feed normalized observations produced by the shared telemetry layer or
an opt-in integration runner, then this module validates and reports them.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] + (clean[upper] - clean[lower]) * fraction


GROUP_FIELDS = (
    "benchmark_mode",
    "provider",
    "upstream_provider",
    "model",
    "agent_role",
    "context_length",
    "reuse",
    "concurrency",
    "idle_gap_minutes",
    "routing",
    "cache_family_id",
)


def summarize_observations(
    observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in observations:
        key = tuple(row.get(field) for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        summary = dict(zip(GROUP_FIELDS, key))
        summary["trials"] = len(rows)
        for source, prefix in (("ttft_ms", "ttft"), ("end_to_end_ms", "end_to_end")):
            values = [row.get(source) for row in rows if row.get(source) is not None]
            summary[f"{prefix}_p50_ms"] = percentile(values, 0.5)
            summary[f"{prefix}_p95_ms"] = percentile(values, 0.95)
        for field in (
            "prompt_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_miss_tokens",
            "output_tokens",
            "cache_hit_ratio",
            "cost_usd",
            "db_retrieval_ms",
            "pack_build_ms",
        ):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            summary[f"{field}_median"] = statistics.median(values) if values else None
        summary["errors"] = sum(1 for row in rows if row.get("error_type"))
        summary["under_required_trials"] = len(rows) < 20
        summaries.append(summary)
    return summaries


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def comparison_metrics(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = tuple(
        field
        for field in GROUP_FIELDS
        if field not in {"benchmark_mode", "cache_family_id"}
    )
    indexed: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in summaries:
        key = tuple(row.get(field) for field in dimensions)
        indexed.setdefault(key, {})[str(row.get("benchmark_mode"))] = row
    comparisons = []
    for key, modes in indexed.items():
        baseline = modes.get("baseline")
        warm = modes.get("both") or modes.get("provider-only")
        local = modes.get("local-only") or modes.get("both")
        if baseline is None:
            continue
        baseline_local = sum(
            float(baseline.get(field) or 0)
            for field in ("db_retrieval_ms_median", "pack_build_ms_median")
        )
        local_time = (
            sum(
                float(local.get(field) or 0)
                for field in ("db_retrieval_ms_median", "pack_build_ms_median")
            )
            if local
            else None
        )
        comparisons.append(
            {
                **dict(zip(dimensions, key)),
                "ttft_speedup": _ratio(
                    baseline.get("ttft_p50_ms"),
                    warm.get("ttft_p50_ms") if warm else None,
                ),
                "end_to_end_speedup": _ratio(
                    baseline.get("end_to_end_p50_ms"),
                    warm.get("end_to_end_p50_ms") if warm else None,
                ),
                "local_retrieval_saved_ms": (
                    baseline_local - local_time if local_time is not None else None
                ),
            }
        )
    return comparisons


def write_outputs(
    observations: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = destination / f"context-cache-{stamp}.jsonl"
    summary_path = destination / f"context-cache-{stamp}-summary.csv"
    report_path = destination / f"context-cache-{stamp}.md"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in observations:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )
    summaries = summarize_observations(observations)
    columns = sorted({key for row in summaries for key in row})
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "commit_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **dict(environment or {}),
    }
    comparisons = comparison_metrics(summaries)
    lines = [
        "# MLEvolve Context-Cache Benchmark",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Results",
        "",
        f"- Raw observations: `{raw_path.name}`",
        f"- Condition summaries: `{summary_path.name}`",
        f"- Observations: {len(observations)}",
        f"- Conditions below 20 trials: {sum(bool(row['under_required_trials']) for row in summaries)}",
        "",
        "Provider cache hits are confirmed only by non-null, positive cache-read token metrics. "
        "Latency-only changes remain inferences.",
        "",
        "## Paired speedups",
        "",
        "```json",
        json.dumps(comparisons, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Limitations",
        "",
        "Idle-gap and live-provider trials are omitted unless supplied by an opt-in runner with credentials. "
        "Routing mode and the cold-marker method must be recorded on every live observation.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"jsonl": raw_path, "csv": summary_path, "markdown": report_path}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Normalized JSON Lines observations")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark-results/context-cache")
    )
    parser.add_argument("--environment-json", type=Path)
    args = parser.parse_args(argv)
    environment = (
        json.loads(args.environment_json.read_text(encoding="utf-8"))
        if args.environment_json
        else {}
    )
    paths = write_outputs(
        load_jsonl(args.input), args.output_dir, environment=environment
    )
    print(
        json.dumps(
            {key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
