"""Standalone worker for the no-scheduler multiprocess baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    launched_at = time.time()
    import torch
    from scheduler_benchmark_test.stress_bench.stress_runner import train_stress_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ready_at = time.time()
    outcome = train_stress_model(
        source_path=spec["source_path"], constructor_kwargs=spec["constructor_kwargs"],
        input_shape=spec["input_shape"], precision=spec["precision"],
        epochs=int(spec["epochs"]), batches_per_epoch=int(spec["batches_per_epoch"]),
        device=device, stream_data=bool(spec.get("stream_data")))
    outcome.update({
        "job_id": spec["job_id"], "step_idx": spec["step_idx"], "pid": os.getpid(),
        "process_launched_at": launched_at, "cuda_ready_at": ready_at,
        "startup_seconds": ready_at - launched_at, "finished_at": time.time()})
    Path(args.result).write_text(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
