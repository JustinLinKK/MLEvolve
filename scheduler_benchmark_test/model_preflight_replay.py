#!/usr/bin/env python3
"""Measure CPU preflight latency and hard-rejection rate on archived scripts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PreflightConfig
from engine.preflight import ModelPreflightGate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=REPO_ROOT / "replay_model_sources"
    )
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def evaluate(sources: Path, *, profile: str, artifact_root: Path) -> dict:
    cfg = SimpleNamespace(
        workspace_dir=artifact_root,
        exp_id="archived-preflight-replay",
        exp_name="archived-preflight-replay",
        experiment=SimpleNamespace(mode="hardware_aware"),
        preflight=PreflightConfig(target_profile=profile, max_repair_rounds=0),
        scheduler=SimpleNamespace(settings=None),
    )
    records = []
    for index, path in enumerate(sorted(sources.rglob("*.py"))):
        relative = path.relative_to(sources)
        node = SimpleNamespace(
            id=f"archived-{index:04d}",
            code=path.read_text(encoding="utf-8", errors="replace"),
        )
        started = time.monotonic()
        outcome = ModelPreflightGate(cfg).run(node, generated=False)
        records.append(
            {
                "source": str(relative),
                "status": outcome.status,
                "admitted": outcome.admitted,
                "mode": outcome.mode,
                "gpu_check_required": outcome.gpu_check_required,
                "diagnostic_codes": outcome.diagnostic_codes,
                "latency_seconds": time.monotonic() - started,
            }
        )
    latencies = [record["latency_seconds"] for record in records]
    sorted_latencies = sorted(latencies)
    p95_index = max(
        0, min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95) - 1)
    )
    hard_rejections = sum(not record["admitted"] for record in records)
    return {
        "source_root": str(sources.resolve()),
        "profile": profile,
        "candidate_count": len(records),
        "hard_rejection_count": hard_rejections,
        "hard_rejection_rate": hard_rejections / len(records) if records else 0.0,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "p95_latency_seconds": sorted_latencies[p95_index] if sorted_latencies else 0.0,
        "records": records,
    }


def main() -> int:
    args = _parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="mlevolve-preflight-replay-") as temp_dir:
        result = evaluate(
            args.sources.resolve(), profile=args.profile, artifact_root=Path(temp_dir)
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
