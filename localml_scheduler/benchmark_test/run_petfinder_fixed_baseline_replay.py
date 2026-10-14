"""Run recorded PetFinder candidates through dynamic Scheduler admission.

This is a scheduler-only experiment: the candidate code is materialized from
the recorded original baseline, so code generation cannot confound either the
first metric or the execution timeline.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings

from localml_scheduler.benchmark_test.replay_petfinder_baseline_candidates import (
    BaselineCandidate,
    build_replay_job,
)


def _candidate_sources(source_dir: Path) -> list[tuple[BaselineCandidate, Path]]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    sources: list[tuple[BaselineCandidate, Path]] = []
    for item in manifest:
        path = source_dir / str(item["path"])
        sources.append(
            (
                BaselineCandidate(
                    node_id=str(item["node_id"]),
                    code=path.read_text(encoding="utf-8"),
                    is_buggy=bool(item["is_buggy_in_original"]),
                    execution_seconds=float(item["original_execution_seconds"]),
                ),
                path,
            )
        )
    return sources


def _first_scored_candidate_first(
    sources: list[tuple[BaselineCandidate, Path]],
) -> list[tuple[BaselineCandidate, Path]]:
    """Return the exact recorded order for a node-aligned scheduler replay."""
    return sources


def _selected_candidate_sources(
    source_dir: Path, *, limit: int
) -> list[tuple[BaselineCandidate, Path]]:
    """Select a source prefix without reordering the recorded baseline."""
    return _first_scored_candidate_first(_candidate_sources(source_dir))[:limit]


def _prepare_workspace(path: Path, data_root: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / "input").symlink_to(data_root, target_is_directory=True)


def _materialize_candidate_source(source_path: Path, workspace: Path) -> Path:
    """Execute a byte-identical workspace copy, never mutate frozen evidence."""
    destination = workspace / "candidate.py"
    shutil.copyfile(source_path, destination)
    if sha256(source_path.read_bytes()).digest() != sha256(destination.read_bytes()).digest():
        raise RuntimeError(f"candidate copy diverged from frozen source: {source_path}")
    return destination


def _settings(
    runtime_root: Path,
    *,
    predictor_source_root: Path,
    predictor_artifact_path: Path,
) -> SchedulerSettings:
    """Dynamic CUDA-process admission; no fixed parallel-job cap is configured."""
    return SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.25,
        # These jobs execute self-contained raw scripts.  They do not call the
        # model-cache loader, so warming their Python source as a baseline
        # model is both irrelevant and can repeatedly fail before admission.
        baseline_cache={"warm_queue_top_k": 0},
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
        hardware_knowledge_graph={"enabled": False},
        prediction={
            "mode": "ml_predictor",
            # Predictor source capture is advisory for generated raw scripts:
            # an unsupported multimodal wrapper must fall back to the measured
            # branch/runtime profile, not turn a valid baseline node terminal.
            "require_ml_prediction": False,
            "ml": {
                "v32_source_root": str(predictor_source_root),
                "v32_artifact_path": str(predictor_artifact_path),
                # Paired graph capture on CPU repeatedly exceeds the bounded
                # conversion deadline for generated PetFinder vision models.
                # The A10 measurement is 21 seconds on CUDA for the same
                # candidate, so use the accelerator for this short-lived
                # Predictor operation; Scheduler admission remains governed
                # by live device telemetry rather than a parallel-job cap.
                "v32_capture_device": "cuda",
                "v32_async_source_capture": True,
                "source_conversion_timeout_seconds": 180,
            },
        },
        gpu_scheduler={
            "mode": "parallel_time_aware",
            "packing_backend": "cuda_process",
            "exclusive_fallback_enabled": True,
            "cuda_process": {"enabled": True},
            "memory": {
                "predicted_budget_fraction": 0.85,
                "live_admission_stop_fraction": 0.90,
                "live_admission_resume_fraction": 0.85,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-python",
        type=Path,
        required=True,
        help="Historical interpreter used to execute each recorded candidate.",
    )
    parser.add_argument("--predictor-v32-source-root", type=Path, required=True)
    parser.add_argument("--predictor-v32-artifact", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--time-limit-seconds", type=int, default=3600)
    args = parser.parse_args()

    sources = _selected_candidate_sources(args.sources, limit=args.limit)
    if len(sources) != args.limit:
        raise ValueError(f"need {args.limit} recorded sources, found {len(sources)}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if not args.candidate_python.is_file():
        raise FileNotFoundError(args.candidate_python)
    if not args.predictor_v32_source_root.is_dir():
        raise FileNotFoundError(args.predictor_v32_source_root)
    if not args.predictor_v32_artifact.is_file():
        raise FileNotFoundError(args.predictor_v32_artifact)
    args.output_root.mkdir(parents=True, exist_ok=False)
    settings = _settings(
        args.output_root / "scheduler_runtime",
        predictor_source_root=args.predictor_v32_source_root,
        predictor_artifact_path=args.predictor_v32_artifact,
    )
    api = SchedulerClient(settings)
    service = api.create_service().start(background=True)
    submitted = []
    started_at = time.time()
    try:
        for index, (candidate, source_path) in enumerate(sources, start=1):
            workspace = args.output_root / "workspace" / f"{index:02d}_{candidate.node_id}"
            _prepare_workspace(workspace, args.data_root)
            candidate_path = _materialize_candidate_source(source_path, workspace)
            submitted.append(
                api.submit(
                    build_replay_job(
                        candidate=candidate,
                        source_path=candidate_path,
                        workspace=workspace,
                        workflow_id="petfinder-fixed-baseline-scheduler-replay",
                        python_executable=str(args.candidate_python),
                    )
                )
            )
        deadline = started_at + args.time_limit_seconds
        while time.time() < deadline:
            current = [api.inspect(job.job_id) for job in submitted]
            if all(job is not None and job.status.is_terminal for job in current):
                break
            time.sleep(1.0)
        final = [api.inspect(job.job_id) for job in submitted]
        (args.output_root / "summary.json").write_text(
            json.dumps(
                {
                    "started_at_unix": started_at,
                    "finished_at_unix": time.time(),
                    "requested_nodes": args.limit,
                    "jobs": [job.to_dict() if job is not None else None for job in final],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
