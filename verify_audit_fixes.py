"""Verify the 10 audit fixes are in place.

Each fix has a dedicated assertion. Run after _apply_audit_fixes.py + regen.
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records, SCHEMA_DIR  # noqa: E402


def check(cond: bool, msg: str, fails: list[str]) -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


def main() -> int:
    fails: list[str] = []

    # HIGH-1: ontology vocab matches data
    print("[HIGH-1] ontology vocabulary")
    onto = yaml.safe_load((SCHEMA_DIR / "ontology" / "hardware_feature_keys.yaml").read_text())
    onto_keys = set(onto.get("keys", {}).keys())
    a_features = set()
    for rec in load_records("hardware_feature_records"):
        a_features.update(rec.get("features") or [])
    sample_canonical = {"tensor_cores", "bf16", "fp16", "tf32", "fp8", "sm_80", "sm_90", "sm_120"}
    check(sample_canonical.issubset(onto_keys),
          f"ontology contains canonical tags {sample_canonical}", fails)
    check("tensor_core" not in onto_keys,
          "ontology no longer has stale 'tensor_core' (singular)", fails)
    check("cuda_capability_80" not in onto_keys,
          "ontology no longer has stale 'cuda_capability_80'", fails)
    orphan = a_features - onto_keys
    check(len(orphan) <= 2,  # tolerate 1-2 niche tags
          f"<=2 orphan data tags (got {sorted(orphan)})", fails)

    # HIGH-2: 3 Hopper HBM3 records at 1593 MHz
    print("\n[HIGH-2] Hopper HBM3 vram_clock_mhz")
    for rid in ["nvidia.hopper.h100_sxm5_80gb.spec",
                "nvidia.hopper.h100_nvl_94gb.spec",
                "nvidia.hopper.gh200_grace_hopper.spec"]:
        p = SCHEMA_DIR / "hardware_feature_records" / f"{rid}.yaml"
        rec = yaml.safe_load(p.read_text())
        check(rec.get("vram_clock_mhz") == 1593,
              f"{rid}: vram_clock_mhz=1593 (got {rec.get('vram_clock_mhz')})", fails)

    # MEDIUM-1: C coverage expanded from 2 -> 6 records
    print("\n[MEDIUM-1] Schema C coverage")
    c_records = load_records("optimization_recipe_chunks")
    check(len(c_records) >= 6, f"C has >=6 records (got {len(c_records)})", fails)
    expected_a_origins = {
        "nvidia.ampere.a100_sxm4_80gb.spec",
        "nvidia.hopper.h100_sxm5_80gb.spec",
        "nvidia.hopper.h200_sxm_141gb.spec",
        "nvidia.ada_lovelace.geforce_rtx_4090.spec",
        "nvidia.blackwell.geforce_rtx_5070_ti.spec",
        "nvidia.blackwell.geforce_rtx_5090.spec",
    }
    derived_from = {r["recipe_id"].split("hardware_feature_recipe.", 1)[-1]
                    for r in c_records}
    missing = expected_a_origins - derived_from
    check(not missing, f"all 6 expected flagships derived (missing: {missing})", fails)

    # MEDIUM-2: B100/B200/GB200 nulls filled
    print("\n[MEDIUM-2] Blackwell DC spec nulls filled")
    for rid in ["nvidia.blackwell.b100.spec",
                "nvidia.blackwell.b200.spec",
                "nvidia.blackwell.gb200_nvl2.spec"]:
        p = SCHEMA_DIR / "hardware_feature_records" / f"{rid}.yaml"
        rec = yaml.safe_load(p.read_text())
        check(rec.get("sm_count") is not None and rec["sm_count"] > 0,
              f"{rid}: sm_count populated ({rec.get('sm_count')})", fails)
        check(rec.get("vram_clock_mhz") is not None and rec["vram_clock_mhz"] > 0,
              f"{rid}: vram_clock_mhz populated ({rec.get('vram_clock_mhz')})", fails)

    # MEDIUM-3: fp8 granularity standardized on Blackwell
    print("\n[MEDIUM-3] fp8 tag granularity on Blackwell")
    bw_with_fp8 = [r for r in load_records("hardware_feature_records")
                    if (r.get("architectures") or ["?"])[0] == "blackwell"
                    and "fp8" in (r.get("features") or [])]
    for rec in bw_with_fp8:
        feats = set(rec.get("features") or [])
        check("fp8_e4m3" in feats and "fp8_e5m2" in feats,
              f"{rec['record_id']}: has fp8_e4m3 + fp8_e5m2", fails)

    # LOW-1: recipe_id uses dot, not colon
    print("\n[LOW-1] recipe_id dotted-snake convention")
    for rec in c_records:
        rid = rec.get("recipe_id", "")
        check(":" not in rid and rid.startswith("hardware_feature_recipe."),
              f"{rid}: dotted prefix", fails)
    # Filename: prefix uses dot, not underscore (was hardware_feature_recipe_X before fix)
    for p in (SCHEMA_DIR / "optimization_recipe_chunks").glob("*.yaml"):
        check(p.stem.startswith("hardware_feature_recipe.") and
              not p.stem.startswith("hardware_feature_recipe_"),
              f"{p.name}: prefix uses dot not underscore", fails)

    # LOW-2: vram_probe decoupled from SKUs
    print("\n[LOW-2] vram_probe decoupled")
    vp = yaml.safe_load((SCHEMA_DIR / "code_doc_chunks"
                         / "pytorch.vram_probe.budget.001.yaml").read_text())
    check(vp.get("hardware_keys") == [],
          f"vram_probe.hardware_keys empty (got {vp.get('hardware_keys')})", fails)

    # LOW-3: rtx_3050 8GB titled
    print("\n[LOW-3] rtx_3050 disambiguated")
    rtx3050 = yaml.safe_load((SCHEMA_DIR / "hardware_feature_records"
                               / "nvidia.ampere.geforce_rtx_3050.spec.yaml").read_text())
    check("8GB" in rtx3050.get("title", ""),
          f"rtx_3050 title contains '8GB' ({rtx3050.get('title')})", fails)

    # LOW-4: GradScaler deprecated
    print("\n[LOW-4] torch.cuda.amp.GradScaler deprecated")
    gs_files = list((SCHEMA_DIR / "api_symbol_chunks").glob("torch.cuda.amp.GradScaler*.yaml"))
    check(len(gs_files) >= 1, "GradScaler yaml exists", fails)
    for p in gs_files:
        rec = yaml.safe_load(p.read_text())
        check(rec.get("deprecated") is True,
              f"{p.name}: deprecated=True (got {rec.get('deprecated')})", fails)

    # LOW-5: CUDAGraph keep_graph in 2.8-2.12 record (existing yaml unchanged)
    print("\n[LOW-5] CUDAGraph keep_graph param range")
    cg = yaml.safe_load((SCHEMA_DIR / "api_symbol_chunks"
                         / "torch.cuda.CUDAGraph.v2.8-2.12.yaml").read_text())
    check("keep_graph" in cg.get("signature", ""),
          f"CUDAGraph 2.8-2.12 sig contains keep_graph", fails)

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== all 10 audit fixes verified ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
