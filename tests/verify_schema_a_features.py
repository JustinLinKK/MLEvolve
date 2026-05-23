"""Verify each Schema A record's `features` tag list against NVIDIA's
architectural capability table.

A feature tag is correct only if every record that lists it has a
compute_capability inside the architecturally-supported range. Sources of
truth (NVIDIA cuda-gpus + Hopper/Ada/Blackwell whitepapers):

  Tensor cores
    tensor_cores         CC >= 7.0  (Volta+)
    tensor_cores_3gen    CC in {8.0, 8.6, 8.7}            (Ampere)
    tensor_cores_4gen    CC in {8.9, 9.0}                 (Ada Lovelace + Hopper)
    tensor_cores_5gen    CC in {10.0, 10.3, 12.0, 12.1}   (Blackwell)

  Precision
    fp16        CC >= 5.3
    bf16        CC >= 8.0   (Ampere+ native)
    tf32        CC >= 8.0   (Ampere+ tensor cores only)
    int8        CC >= 6.1   (DP4A); tensor-core int8 CC >= 7.5
    fp64        all NVIDIA  (rate differs; consumer 1:64, datacenter 1:2)
    fp8         CC >= 8.9   (Ada introduced E4M3/E5M2; Hopper + Blackwell)
    fp8_e4m3    CC >= 8.9
    fp8_e5m2    CC >= 8.9
    fp4         CC >= 10.0  (Blackwell only)
    amp         any         (framework concept, not hardware feature)

  SM marker tags
    sm_80       CC == 8.0
    sm_86       CC == 8.6
    sm_89       CC == 8.9
    sm_90       CC == 9.0
    sm_100      CC == 10.0
    sm_120      CC == 12.0

  Other (cross-reference with vendor whitepaper for accuracy)
    cuda_graphs CC >= 7.0   (any CUDA 10+, effectively any modern GPU)
    pcie5_x16   CC >= 9.0   (Hopper datacenter SXM/PCIe; Blackwell)

For each (record, feature_tag) pair we apply the rule; record any
violations. Reports per-feature aggregate at the end.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402


def cc_float(rec: dict) -> float | None:
    ccs = rec.get("compute_capabilities") or []
    if not ccs:
        return None
    try:
        return float(str(ccs[0]))
    except (TypeError, ValueError):
        return None


# rule: feature_tag -> callable(cc: float) -> bool ("is this feature consistent with cc?")
RULES: dict[str, callable] = {
    # tensor-core generations (set membership)
    "tensor_cores":         lambda cc: cc >= 7.0,
    "tensor_cores_3gen":    lambda cc: cc in (8.0, 8.6, 8.7),
    "tensor_cores_4gen":    lambda cc: cc in (8.9, 9.0),
    "tensor_cores_5gen":    lambda cc: cc in (10.0, 10.3, 12.0, 12.1),

    # precisions
    "fp16":      lambda cc: cc >= 5.3,
    "bf16":      lambda cc: cc >= 8.0,
    "tf32":      lambda cc: cc >= 8.0,
    "int8":      lambda cc: cc >= 6.1,
    "fp64":      lambda cc: True,
    "fp8":       lambda cc: cc >= 8.9,
    "fp8_e4m3":  lambda cc: cc >= 8.9,
    "fp8_e5m2":  lambda cc: cc >= 8.9,
    "fp4":       lambda cc: cc >= 10.0,

    # software / framework concepts
    "amp":         lambda cc: True,
    "cuda_graphs": lambda cc: cc >= 7.0,

    # connectivity (PCIe gen5: Hopper datacenter, Blackwell consumer)
    "pcie5_x16":   lambda cc: cc >= 9.0,

    # SM markers (exact)
    "sm_80":   lambda cc: cc == 8.0,
    "sm_86":   lambda cc: cc == 8.6,
    "sm_89":   lambda cc: cc == 8.9,
    "sm_90":   lambda cc: cc == 9.0,
    "sm_100":  lambda cc: cc == 10.0,
    "sm_120":  lambda cc: cc == 12.0,
}


def main() -> int:
    records = load_records("hardware_feature_records")
    print(f"loaded {len(records)} Schema A records")

    fails: list[str] = []
    unknown_tags: set[str] = set()
    per_feature_pass: dict[str, int] = defaultdict(int)
    per_feature_fail: dict[str, int] = defaultdict(int)

    for rec in records:
        rid = rec["record_id"]
        cc = cc_float(rec)
        if cc is None:
            fails.append(f"[{rid}] no compute_capabilities")
            continue
        for feat in rec.get("features") or []:
            rule = RULES.get(feat)
            if rule is None:
                unknown_tags.add(feat)
                continue
            if rule(cc):
                per_feature_pass[feat] += 1
            else:
                per_feature_fail[feat] += 1
                fails.append(f"[{rid}] cc={cc} declares {feat!r} but rule says incompatible")

    print()
    print("=== per-feature totals ===")
    all_feats = sorted(set(per_feature_pass) | set(per_feature_fail) | set(RULES))
    for feat in all_feats:
        p = per_feature_pass[feat]
        f = per_feature_fail[feat]
        status = "OK " if f == 0 else "FAIL"
        print(f"  {status} {feat:22s} pass={p:3d}  fail={f}")

    if unknown_tags:
        print(f"\n--- WARN: unknown feature tags (no rule) ---")
        for t in sorted(unknown_tags):
            print(f"  ? {t}")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails[:50]:
            print(f"  - {f}")
        if len(fails) > 50:
            print(f"  ... + {len(fails) - 50} more")
        return 1
    print(f"=== Schema A features 100% VALID "
          f"({sum(per_feature_pass.values())} (record, feature) pairs checked) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
