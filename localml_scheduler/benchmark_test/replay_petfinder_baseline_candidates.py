"""Extract the fixed candidate sequence used by a historical PetFinder run.

Replaying these sources isolates scheduler behavior.  It intentionally avoids
asking an LLM to regenerate a different candidate sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from engine.executor import _build_perfseer_model_metadata, _count_labeled_training_rows
from engine.script_introspection import introspect_training_script
from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.domain import (
    BatchProbeSpec,
    CheckpointPolicy,
    ResourceRequirements,
    RuntimeProbeSpec,
)


@dataclass(frozen=True)
class BaselineCandidate:
    node_id: str
    code: str
    is_buggy: bool
    execution_seconds: float


def load_budgeted_candidates(journal_path: Path, *, limit: int) -> list[BaselineCandidate]:
    """Load the baseline's historical 60-second-budget candidate sequence."""
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    selected: list[BaselineCandidate] = []
    for raw in payload.get("nodes", []):
        if raw.get("stage") == "root":
            continue
        code = str(raw.get("code") or "")
        if not code:
            continue
        is_buggy = bool(raw.get("is_buggy"))
        try:
            execution_seconds = float(raw.get("exec_time") or 0.0)
        except (TypeError, ValueError):
            execution_seconds = 0.0
        if is_buggy and execution_seconds < 60.0:
            continue
        selected.append(
            BaselineCandidate(
                node_id=str(raw.get("id") or "unknown"),
                code=code,
                is_buggy=is_buggy,
                execution_seconds=execution_seconds,
            )
        )
        if len(selected) >= limit:
            break
    return selected


def materialize_candidates(
    journal_path: Path, *, destination: Path, limit: int
) -> list[BaselineCandidate]:
    """Write immutable per-node source files and a compact manifest."""
    candidates = load_budgeted_candidates(journal_path, limit=limit)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        path = destination / f"{index:02d}_{candidate.node_id}.py"
        path.write_text(candidate.code, encoding="utf-8")
        manifest.append(
            {
                "index": index,
                "node_id": candidate.node_id,
                "path": path.name,
                "is_buggy_in_original": candidate.is_buggy,
                "original_execution_seconds": candidate.execution_seconds,
            }
        )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return candidates


def build_replay_job(
    *,
    candidate: BaselineCandidate,
    source_path: Path,
    workspace: Path,
    workflow_id: str,
    python_executable: str | None = None,
):
    """Construct a Scheduler job without changing candidate hyperparameters.

    Batch-size probing is deliberately off: changing a recorded source's batch
    size would no longer be a same-candidate reproducibility replay.
    """
    result_path = workspace / "scheduler_results" / f"{candidate.node_id}.json"
    script_metadata = introspect_training_script(candidate.code)
    try:
        planned_epochs = int(script_metadata.get("proposed_epochs"))
    except (TypeError, ValueError):
        planned_epochs = None
    if planned_epochs is not None and planned_epochs <= 0:
        planned_epochs = None
    try:
        candidate_batch_size = int(script_metadata.get("proposed_batch_size"))
    except (TypeError, ValueError):
        candidate_batch_size = None
    if candidate_batch_size is not None and candidate_batch_size <= 0:
        candidate_batch_size = None
    requires_gpu = bool(script_metadata.get("requires_gpu"))
    predictor_metadata = _build_perfseer_model_metadata(
        candidate.code,
        source_path=source_path,
        input_resolution=script_metadata.get("input_resolution"),
        precision_mode=script_metadata.get("precision_mode"),
        dataset_size=_count_labeled_training_rows(workspace / "input" / "train.csv"),
    )
    if predictor_metadata is None and requires_gpu:
        raise ValueError(
            "frozen replay candidate has no statically usable Predictor v3.2 contract: "
            f"{candidate.node_id}"
        )
    metadata = {
        "fixed_candidate_replay": True,
        "mlevolve_node_id": candidate.node_id,
        "original_candidate_runtime_bug": candidate.is_buggy,
        "original_execution_seconds": candidate.execution_seconds,
        "preserve_candidate_batch_contract": True,
        "proposed_epochs": planned_epochs,
    }
    if predictor_metadata is not None:
        metadata["perfseer_model"] = predictor_metadata
    else:
        metadata["predictor_not_applicable"] = "cpu_candidate"
    return build_mlevolve_job(
        workflow_id=workflow_id,
        baseline_model_id=f"petfinder-recorded-{candidate.node_id}",
        baseline_model_path=str(source_path),
        runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
        runner_kwargs={
            "script_path": str(source_path),
            "working_dir": str(workspace),
            "result_path": str(result_path),
            "timeout": 7200,
            "candidate_python_executable": python_executable,
            **({"batch_size": candidate_batch_size} if candidate_batch_size is not None else {}),
        },
        task_type="mlevolve_recorded_candidate",
        resource_requirements=ResourceRequirements(requires_gpu=requires_gpu, gpu_slots=1),
        checkpoint_policy=CheckpointPolicy(save_every_epoch=False),
        batch_probe=BatchProbeSpec(enabled=False),
        # Raw scripts cannot be checkpoint-resumed for trial packing, but they
        # do print one metric per epoch.  Stream that real epoch timing so the
        # scheduler replaces a cold Predictor estimate for later admissions.
        runtime_probe=RuntimeProbeSpec(enabled=True, strategy="epoch_1"),
        packing_family="petfinder-recorded-candidate",
        packing_signature=f"petfinder-recorded:{candidate.node_id}",
        packing_eligible=requires_gpu,
        packing_backend_allowlist=["cuda_process"] if requires_gpu else [],
        max_epochs=planned_epochs,
        seed=42,
        metadata=metadata,
    )
