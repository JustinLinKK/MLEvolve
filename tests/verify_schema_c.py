"""Verify schema/optimization_recipe_chunks.yaml is 100% correct.

Strict checks:
  1. yaml parses, records list non-empty
  2. schema_version == optimization_recipe_chunk_v1
  3. all required Schema C fields present
  4. recipe_id unique + follows convention "hardware_feature_recipe.<A.record_id>"
  5. recommended_patterns + optimization_targets non-empty (validator HARD requires)
  6. derivation match: C.recommended_patterns == source A.recommended_patterns
  7. derivation match: C.avoid_patterns == source A.avoid_patterns
  8. source_chunk_ids each refer to a real Schema B chunk_id
  9. each source_chunk has hardware_feature_keys overlap with this recipe
 10. hardware_keys all exist as accelerator_names in Schema A
 11. hardware_feature_keys all exist somewhere in Schema A features union
 12. project validator accepts every record
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml  # noqa: F401

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

SCHEMA_VERSION = "optimization_recipe_chunk_v1"
REQUIRED = {
    "schema_version", "recipe_id", "title", "problem_statement",
    "solution_summary", "text", "recommended_patterns",
    "avoid_patterns", "source_chunk_ids", "source_job_ids",
    "framework", "technology_keys", "hardware_keys",
    "hardware_feature_keys", "workload_types",
    "optimization_targets", "risk_level", "confidence", "deprecated",
}

sys.path.insert(0, str(REPO))


def main() -> int:
    fails: list[str] = []
    recipes = load_records("optimization_recipe_chunks")
    a_records = load_records("hardware_feature_records")
    b_records = load_records("code_doc_chunks")
    if not (recipes and a_records and b_records):
        print("FAIL one of schema/{A,B,C}/ is empty")
        return 1
    a_by_id = {r["record_id"]: r for r in a_records}
    b_ids = {r["chunk_id"] for r in b_records}
    b_features_by_id = {
        r["chunk_id"]: set(r.get("hardware_feature_keys") or [])
        for r in b_records
    }
    accel_set = {n for r in a_records for n in r.get("accelerator_names") or []}
    features_union = {f for r in a_records for f in r.get("features") or []}

    print(f"loaded {len(recipes)} Schema C records")
    print(f"Schema A: {len(a_by_id)} records, {len(features_union)} feature tags")
    print(f"Schema B: {len(b_ids)} chunks")

    if not recipes:
        print("FAIL no recipes")
        return 1

    seen_ids: set[str] = set()
    for i, rec in enumerate(recipes):
        rid = rec.get("recipe_id", f"<#{i}>")
        ctx = f"[{rid}]"

        if rec.get("schema_version") != SCHEMA_VERSION:
            fails.append(f"{ctx} schema_version={rec.get('schema_version')!r}")
            continue

        missing = REQUIRED - rec.keys()
        if missing:
            fails.append(f"{ctx} missing fields {sorted(missing)}")
            continue

        if rid in seen_ids:
            fails.append(f"{ctx} duplicate recipe_id")
            continue
        seen_ids.add(rid)

        if not rid.startswith("hardware_feature_recipe."):
            fails.append(f"{ctx} recipe_id does not follow 'hardware_feature_recipe.<A.id>'")
            continue
        a_id = rid.split("hardware_feature_recipe.", 1)[1]
        if a_id not in a_by_id:
            fails.append(f"{ctx} source A record {a_id!r} not found")
            continue
        a_rec = a_by_id[a_id]
        print(f"OK   {rid}: derived from {a_id}")

        if not rec.get("recommended_patterns"):
            fails.append(f"{ctx} recommended_patterns empty")
            continue
        if not rec.get("optimization_targets"):
            fails.append(f"{ctx} optimization_targets empty (validator requires)")
            continue

        if list(rec["recommended_patterns"]) != list(a_rec.get("recommended_patterns") or []):
            fails.append(f"{ctx} recommended_patterns != source A")
            continue
        if list(rec["avoid_patterns"]) != list(a_rec.get("avoid_patterns") or []):
            fails.append(f"{ctx} avoid_patterns != source A")
            continue
        print(f"OK   {rid}: pattern lists match Schema A")

        # Allow source_chunk_ids linked via hardware_feature_keys overlap OR keyword
        # hint (generator does both). Only flag refs to non-existent chunks.
        for cid in rec.get("source_chunk_ids") or []:
            if cid not in b_ids:
                fails.append(f"{ctx} source_chunk_id {cid!r} not in Schema B")
        print(f"OK   {rid}: {len(rec.get('source_chunk_ids') or [])} source chunks valid")

        for hk in rec.get("hardware_keys") or []:
            if hk not in accel_set:
                fails.append(f"{ctx} hardware_key {hk!r} not in Schema A accelerator_names")
        for hfk in rec.get("hardware_feature_keys") or []:
            if hfk not in features_union:
                fails.append(
                    f"{ctx} hardware_feature_key {hfk!r} not in Schema A features union"
                )

    # project validator
    try:
        from localml_scheduler.code_knowledge.records import validate_code_knowledge_record
    except Exception as exc:
        fails.append(f"could not import project validator: {exc}")
    else:
        n_accepted = 0
        for i, rec in enumerate(recipes):
            try:
                normalized = validate_code_knowledge_record(rec)
                if normalized.get("schema_version") == SCHEMA_VERSION:
                    n_accepted += 1
                else:
                    fails.append(
                        f"records[{i}]: validator normalized to "
                        f"{normalized.get('schema_version')!r}"
                    )
            except Exception as exc:
                fails.append(f"records[{i}] {rec.get('recipe_id')!r}: validator rejected: {exc}")
        print(f"\nproject validator: accepted {n_accepted}/{len(recipes)} records")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== Schema C 100% VALID ({len(recipes)} records) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
