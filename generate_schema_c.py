"""Generate schema/optimization_recipe_chunks.yaml (Schema C) by deriving from Schema A.

For each Schema A record with non-empty `recommended_patterns`:
  - emit one optimization_recipe_chunk_v1 record
  - link source_chunk_ids to Schema B chunks sharing hardware_feature_keys

Auto-derive mapping mirrors project's convert_hardware_feature_records
(localml_scheduler/code_knowledge/records.py:154-205) but writes the yaml so
verify can re-check it as a static artifact.
"""
from __future__ import annotations
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(REPO))
from schema_io import load_records, save_record  # noqa: E402


def schema_b_index(b_records: list[dict]) -> list[tuple[str, set[str]]]:
    """Return [(chunk_id, set(hardware_feature_keys)), ...] for fast overlap lookup."""
    out = []
    for rec in b_records:
        keys = set(rec.get("hardware_feature_keys") or [])
        out.append((rec["chunk_id"], keys))
    return out


def derive_recipe(a_rec: dict, b_index: list[tuple[str, set[str]]]) -> dict:
    a_features = set(a_rec.get("features") or [])
    # link B chunks that share at least one hardware_feature_key with this A
    source_chunk_ids = [cid for cid, b_keys in b_index if a_features & b_keys]

    return {
        "schema_version": "optimization_recipe_chunk_v1",
        "recipe_id": f"hardware_feature_recipe:{a_rec['record_id']}",
        "title": f"{a_rec['title']} optimization recipe",
        "problem_statement": (
            f"Optimize {' / '.join(a_rec.get('workload_types', []))} "
            f"on {a_rec['title']}"
        ),
        "solution_summary": "; ".join(a_rec.get("recommended_patterns", [])),
        "text": (
            "Recommended patterns:\n"
            + "\n".join(f"- {p}" for p in a_rec.get("recommended_patterns", []))
            + "\n\nAvoid patterns:\n"
            + "\n".join(f"- {p}" for p in a_rec.get("avoid_patterns", []))
        ),
        "recommended_patterns": list(a_rec.get("recommended_patterns", [])),
        "avoid_patterns": list(a_rec.get("avoid_patterns", [])),
        "source_chunk_ids": source_chunk_ids,
        "source_job_ids": [],  # populate later from Graph DB
        "framework": (a_rec.get("frameworks") or ["pytorch"])[0].split(">")[0],
        "framework_version": "2.x",
        "technology_keys": list(a_features),
        "hardware_keys": list(a_rec.get("accelerator_names") or []),
        "hardware_feature_keys": list(a_features),
        "workload_types": list(a_rec.get("workload_types") or []),
        "optimization_targets": ["improve_throughput", "reduce_vram", "enable_tensor_core"],
        "profile_symptoms": ["precision_not_optimized", "low_sm_utilization"],
        "api_symbols": [],
        "precision_modes": [
            p for p in ["bf16", "fp16", "fp8", "tf32"] if p in a_features
        ],
        "risk_level": "medium",
        "confidence": 0.85,
        "deprecated": False,
    }


def main() -> int:
    a_records = load_records("hardware_feature_records")
    b_records = load_records("code_doc_chunks")
    b_index = schema_b_index(b_records)
    written = 0
    for a_rec in a_records:
        if not a_rec.get("recommended_patterns"):
            continue
        rec = derive_recipe(a_rec, b_index)
        save_record("optimization_recipe_chunks", rec["recipe_id"], rec)
        print(f"  + {rec['recipe_id']} ({len(rec['source_chunk_ids'])} source chunks linked)")
        written += 1
    print(f"wrote {written} recipes to schema/C/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
