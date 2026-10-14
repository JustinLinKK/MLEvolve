from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_load_budgeted_candidates_keeps_successes_and_long_runtime_failures(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.replay_petfinder_baseline_candidates import (
        load_budgeted_candidates,
    )

    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"stage": "root"},
                    {"id": "quick-error", "code": "x=1", "is_buggy": True, "exec_time": 12},
                    {"id": "long-error", "code": "x=2", "is_buggy": True, "exec_time": 61},
                    {"id": "success", "code": "x=3", "is_buggy": False, "exec_time": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_budgeted_candidates(journal, limit=10)

    assert [candidate.node_id for candidate in candidates] == ["long-error", "success"]
    assert [candidate.code for candidate in candidates] == ["x=2", "x=3"]


def test_build_replay_job_preserves_the_recorded_candidate_batch_contract(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.replay_petfinder_baseline_candidates import (
        BaselineCandidate,
        build_replay_job,
    )

    source = tmp_path / "candidate.py"
    source.write_text(
        """
import torch.nn as nn
IMG_SIZE = 32
BATCH_SIZE = 8
EPOCHS = 4
class ReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, images):
        return images
DataLoader([], batch_size=BATCH_SIZE)
print('Final Validation Score: 1.0')
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    job = build_replay_job(
        candidate=BaselineCandidate("baseline-node", source.read_text(), False, 61.0),
        source_path=source,
        workspace=workspace,
        workflow_id="fixed-baseline-replay",
        python_executable="/opt/baseline/bin/python",
    )

    assert job.config.seed == 42
    assert job.config.python_executable is None
    assert job.config.runner_kwargs["candidate_python_executable"] == "/opt/baseline/bin/python"
    assert job.config.runner_kwargs["batch_size"] == 8
    assert job.max_epochs == 4
    assert job.metadata["proposed_epochs"] == 4
    assert job.runtime_probe.enabled
    assert job.runtime_probe.strategy == "epoch_1"
    assert not job.batch_probe.enabled
    assert job.packing.eligible
    assert job.metadata["fixed_candidate_replay"] is True
    assert job.metadata["mlevolve_node_id"] == "baseline-node"


def test_fixed_replay_attaches_v32_predictor_contract_for_frozen_multimodal_source(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.replay_petfinder_baseline_candidates import (
        BaselineCandidate,
        build_replay_job,
    )

    source = tmp_path / "candidate.py"
    source.write_text(
        """
import torch.nn as nn
FEATURE_COLS = ["a", "b", "c"]
IMG_SIZE = 384
class FrozenModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
    def forward(self, images, meta):
        return images
""",
        encoding="utf-8",
    )
    job = build_replay_job(
        candidate=BaselineCandidate("baseline-node", source.read_text(), False, 61.0),
        source_path=source,
        workspace=tmp_path,
        workflow_id="fixed-baseline-replay",
        python_executable="/opt/baseline/bin/python",
    )

    assert job.metadata["perfseer_model"]["input_shapes"] == [
        ["$batch", 3, 384, 384],
        ["$batch", 3],
    ]
    assert job.metadata["perfseer_model"]["source_path"].endswith(
        "candidate_perfseer_v32.py"
    )


def test_fixed_replay_keeps_cpu_candidate_without_a_gpu_predictor_contract(
    tmp_path: Path,
) -> None:
    """A recorded LightGBM-only node must remain in the matched source sequence."""
    from localml_scheduler.benchmark_test.replay_petfinder_baseline_candidates import (
        BaselineCandidate,
        build_replay_job,
    )

    source = tmp_path / "candidate.py"
    source.write_text(
        """
import lightgbm as lgb
final_model = lgb.LGBMRegressor()
print('Final Validation Score: 1.0')
""",
        encoding="utf-8",
    )

    job = build_replay_job(
        candidate=BaselineCandidate("cpu-baseline-node", source.read_text(), False, 61.0),
        source_path=source,
        workspace=tmp_path / "workspace",
        workflow_id="fixed-baseline-replay",
        python_executable="/opt/baseline/bin/python",
    )

    assert not job.resource_requirements.requires_gpu
    assert "perfseer_model" not in job.metadata
    assert job.metadata["predictor_not_applicable"] == "cpu_candidate"


def test_raw_script_runner_uses_the_candidate_interpreter_without_replacing_worker_python() -> None:
    from localml_scheduler.adapters.mlevolve_runner import _candidate_python_executable

    context = SimpleNamespace(
        job=SimpleNamespace(
            config=SimpleNamespace(
                python_executable=None,
                runner_kwargs={"candidate_python_executable": "/opt/baseline/bin/python"},
            )
        )
    )

    assert _candidate_python_executable(context) == "/opt/baseline/bin/python"


def test_fixed_candidate_replay_disables_irrelevant_baseline_cache_warming(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.run_petfinder_fixed_baseline_replay import _settings

    settings = _settings(
        tmp_path / "runtime",
        predictor_source_root=tmp_path / "predictor-source",
        predictor_artifact_path=tmp_path / "predictor.pt",
    )

    assert settings.baseline_cache.warm_queue_top_k == 0
    assert settings.prediction.mode == "ml_predictor"
    assert settings.prediction.require_ml_prediction is False
    # CPU paired-graph capture can exceed its bounded deadline for generated
    # vision candidates, which leaves the dynamic scheduler without an epoch
    # estimate.  CUDA capture is short-lived and is verified separately
    # against the A10 before a live scheduler replay.
    assert settings.prediction.v32_capture_device == "cuda"
    assert settings.prediction.v32_async_source_capture is True
    assert settings.prediction.conversion_timeout_seconds == 180
    assert settings.graph_db.enabled is False
    assert settings.hardware_feature_db.enabled is False
    assert settings.hardware_knowledge_graph.enabled is False


def test_fixed_replay_copies_candidate_before_predictor_metadata_generation(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.run_petfinder_fixed_baseline_replay import (
        _materialize_candidate_source,
    )

    source = tmp_path / "frozen.py"
    source.write_text("print('frozen candidate')\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    materialized = _materialize_candidate_source(source, workspace)

    assert materialized == workspace / "candidate.py"
    assert materialized.read_bytes() == source.read_bytes()
    assert source.read_text(encoding="utf-8") == "print('frozen candidate')\n"


def test_zero_cache_warm_top_k_selects_no_models() -> None:
    from localml_scheduler.domain import TrainingJob
    from localml_scheduler.model_cache.warming import select_models_to_warm

    job = TrainingJob.create(
        runner_target="example.runner",
        baseline_model_id="raw-python-source",
        baseline_model_path="candidate.py",
    )

    assert select_models_to_warm([job], top_k=0) == []


def test_fixed_replay_preserves_the_recorded_candidate_order(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.run_petfinder_fixed_baseline_replay import (
        _first_scored_candidate_first,
    )
    from localml_scheduler.benchmark_test.replay_petfinder_baseline_candidates import BaselineCandidate

    buggy = BaselineCandidate("known-bug", "", True, 100.0)
    first_scored = BaselineCandidate("first-score", "", False, 100.0)
    later = BaselineCandidate("later", "", False, 100.0)
    ordered = _first_scored_candidate_first(
        [(buggy, tmp_path / "bug.py"), (first_scored, tmp_path / "score.py"), (later, tmp_path / "later.py")]
    )

    assert [candidate.node_id for candidate, _ in ordered] == ["known-bug", "first-score", "later"]


def test_fixed_replay_applies_the_limit_to_the_recorded_source_order(
    tmp_path: Path,
) -> None:
    from localml_scheduler.benchmark_test.run_petfinder_fixed_baseline_replay import (
        _selected_candidate_sources,
    )

    for name in ("known-bug.py", "first-score.py", "later.py"):
        (tmp_path / name).write_text("print('candidate')\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "node_id": "known-bug",
                    "path": "known-bug.py",
                    "is_buggy_in_original": True,
                    "original_execution_seconds": 100.0,
                },
                {
                    "node_id": "first-score",
                    "path": "first-score.py",
                    "is_buggy_in_original": False,
                    "original_execution_seconds": 100.0,
                },
                {
                    "node_id": "later",
                    "path": "later.py",
                    "is_buggy_in_original": False,
                    "original_execution_seconds": 100.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    selected = _selected_candidate_sources(tmp_path, limit=1)

    assert [candidate.node_id for candidate, _ in selected] == ["known-bug"]
