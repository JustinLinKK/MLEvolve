from __future__ import annotations

from pathlib import Path
import json

from scripts.benchmark_cuda_docs_baseline import build_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_docs_baseline_artifact_is_reproducible() -> None:
    artifact = json.loads(
        (ROOT / "records" / "cuda_docs_integration_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    reproduced = build_baseline(ROOT / artifact["source_trace"])
    assert reproduced == artifact
    assert artifact["node_count"] == 60
    assert artifact["generation_latency_by_role"]["debug"]["samples"] == 3
    assert artifact["local_hardware_context_latency_seconds"]["value"] is None
    assert artifact["qdrant_latency_seconds"]["value"] is None
