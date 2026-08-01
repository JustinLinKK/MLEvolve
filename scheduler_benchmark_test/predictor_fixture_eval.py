"""Evaluate the PerfSeer ML predictor on an archived scheduler stress fixture.

For every job in the fixture's ``jobs.jsonl`` this builds the same
``PredictionRequest`` the scheduler would issue, runs ``PerfSeerMLAdapter``,
and compares predicted runtime against the originally observed wall time
(``metadata.replay_original.started_at/finished_at``).

Usage:
    python -m scheduler_benchmark_test.predictor_fixture_eval \
        --fixture <fixture_dir> \
        --predictor-repo <Predictor repo root> \
        --checkpoint <SeerNet checkpoint .pt> \
        --device cuda \
        --output <out.json>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO)
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localml_scheduler.domain import TrainingJob
from localml_scheduler.prediction.ml_adapter import PerfSeerMLAdapter
from localml_scheduler.prediction.request_builder import build_prediction_request


def _observed_seconds(metadata: dict) -> float | None:
    original = metadata.get("replay_original") or {}
    started, finished = original.get("started_at"), original.get("finished_at")
    if not started or not finished:
        return None
    try:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        return max(0.0, delta.total_seconds())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--predictor-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hardware-key", default="a10")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    fixture = Path(args.fixture).expanduser().resolve()
    adapter = PerfSeerMLAdapter(
        enabled=True,
        checkpoint_path=args.checkpoint,
        repo_path=args.predictor_repo,
        device=args.device,
        cache_size=64,
    )
    print("adapter health:", adapter.health())

    rows = []
    for line in (fixture / "jobs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        job = TrainingJob.from_dict(payload)
        source = Path(str(job.baseline_model_path))
        if not source.is_file():
            candidate = fixture / "sources" / source.name
            if candidate.is_file():
                job.metadata["architecture_source"] = str(candidate)
        batch_size = int(
            job.metadata.get("detected_batch_size")
            or job.config.runner_kwargs.get("batch_size")
            or job.metadata.get("proposed_batch_size")
            or 32
        )
        request = build_prediction_request(
            job,
            hardware_key=args.hardware_key,
            backend="cuda",
            batch_size=batch_size,
        )
        prediction = adapter.predict(request)
        observed = _observed_seconds(job.metadata)
        row = {
            "job_id": job.job_id,
            "model_family": job.metadata.get("model_family"),
            "batch_size": batch_size,
            "precision": request.precision,
            "observed_wall_seconds": observed,
            "prediction": prediction.to_dict() if prediction else None,
        }
        rows.append(row)
        status = "OK" if prediction else "no-prediction"
        step = prediction.step_time_ms.mean if prediction and prediction.step_time_ms else None
        vram = prediction.peak_vram_used_mib.mean if prediction and prediction.peak_vram_used_mib else None
        util = prediction.avg_sm_util_percent.mean if prediction and prediction.avg_sm_util_percent else None
        print(
            f"{row['model_family']:<18} bs={batch_size:<4} {status:<14}"
            f" step_ms={step if step is None else round(step, 2)!s:<10}"
            f" vram_mib={vram if vram is None else round(vram)!s:<8}"
            f" util%={util if util is None else round(util, 1)!s:<7}"
            f" observed_s={observed if observed is None else round(observed)}"
        )

    predicted = sum(1 for r in rows if r["prediction"])
    summary = {
        "fixture": str(fixture),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "job_count": len(rows),
        "predicted_count": predicted,
        "adapter": adapter.to_dict(),
        "rows": rows,
    }
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("written:", out)
    print(f"predicted {predicted}/{len(rows)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
