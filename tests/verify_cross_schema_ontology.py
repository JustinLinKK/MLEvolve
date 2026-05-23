"""Enforce cross-schema ontology alignment.

Hard rules (any violation -> FAIL):
  1. NO HARDWARE-NAMESPACE tag inside any B/C/D `technology_keys`
     (these belong in `hardware_feature_keys`).
  2. `hardware_feature_keys` and `technology_keys` of the same record MUST be disjoint.
  3. Singular `tensor_core` not allowed anywhere - canonical is `tensor_cores`.
  4. All B/C/D `hardware_keys` MUST be ⊆ Schema A `accelerator_names` union.
  5. All B/C/D `hardware_feature_keys` MUST be ⊆ Schema A `features` union.
  6. All B/C/D `workload_types` MUST be ⊆ Schema A `workload_types` union.

Each rule prints OK or per-violation FAIL.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

HW_NAMESPACE = {
    "tensor_cores", "tensor_cores_3gen", "tensor_cores_4gen", "tensor_cores_5gen",
    "bf16", "fp16", "fp32", "fp64", "fp4", "fp8", "fp8_e4m3", "fp8_e5m2",
    "tf32", "int8",
    "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120",
    "pcie5_x16", "cuda_graphs", "amp",
}
FORBIDDEN_SINGULARS = {"tensor_core"}


def union(records: list[dict], field: str) -> set[str]:
    return {x for r in records for x in (r.get(field) or [])}


def main() -> int:
    fails: list[str] = []
    a = load_records("hardware_feature_records")
    b = load_records("code_doc_chunks")
    c = load_records("optimization_recipe_chunks")
    d = load_records("api_symbol_chunks")
    print(f"loaded A={len(a)} B={len(b)} C={len(c)} D={len(d)}")

    a_accels = union(a, "accelerator_names")
    a_feats = union(a, "features")
    a_workloads = union(a, "workload_types")

    # Rule 1+2: HW namespace must not appear in technology_keys; tech ∩ hfk = empty
    for tag, recs in (("B", b), ("C", c), ("D", d)):
        for rec in recs:
            tech = set(rec.get("technology_keys") or [])
            hfk = set(rec.get("hardware_feature_keys") or [])
            ident = (rec.get("chunk_id") or rec.get("recipe_id")
                     or rec.get("api_symbol_id") or "?")
            leak = tech & HW_NAMESPACE
            if leak:
                fails.append(f"[{tag}::{ident}] technology_keys leaks HW namespace: {sorted(leak)}")
            dupe = tech & hfk
            if dupe:
                fails.append(f"[{tag}::{ident}] technology_keys ∩ hardware_feature_keys: {sorted(dupe)}")

    # Rule 3: no singular forms
    for tag, recs in (("A", a), ("B", b), ("C", c), ("D", d)):
        for rec in recs:
            for field in ("features", "hardware_feature_keys", "technology_keys"):
                for v in rec.get(field) or []:
                    if v in FORBIDDEN_SINGULARS:
                        ident = (rec.get("record_id") or rec.get("chunk_id")
                                 or rec.get("recipe_id") or rec.get("api_symbol_id"))
                        fails.append(f"[{tag}::{ident}] {field} contains forbidden singular {v!r}")

    # Rule 4-6: subset checks
    for tag, recs in (("B", b), ("C", c), ("D", d)):
        hk = union(recs, "hardware_keys")
        hfk = union(recs, "hardware_feature_keys")
        wl = union(recs, "workload_types")
        bad_hk = hk - a_accels
        bad_hfk = hfk - a_feats
        bad_wl = wl - a_workloads
        if bad_hk:
            fails.append(f"[{tag}] hardware_keys orphans (not in A.accelerator_names): {sorted(bad_hk)}")
        if bad_hfk:
            fails.append(f"[{tag}] hardware_feature_keys orphans (not in A.features): {sorted(bad_hfk)}")
        if bad_wl:
            fails.append(f"[{tag}] workload_types orphans (not in A.workload_types): {sorted(bad_wl)}")

    # Summary print
    print(f"\nHW namespace size: {len(HW_NAMESPACE)}")
    print(f"A.accelerator_names: {len(a_accels)} | A.features: {len(a_feats)} | A.workload_types: {len(a_workloads)}")
    for tag, recs in (("B", b), ("C", c), ("D", d)):
        tech = union(recs, "technology_keys")
        print(f"  {tag}.technology_keys union: {len(tech)} tags")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("=== cross-schema ontology 100% ALIGNED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
