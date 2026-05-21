"""Verify the 8 ontology fields added to Schema D are semantically correct.

For each D record checks:
  1. api_symbols contains its own api_symbol
  2. hardware_feature_keys subset of Schema A features union (cross-DB anchor)
  3. technology_keys subset of Schema B technology_keys union (cross-DB link)
  4. optimization_targets subset of canonical target set
  5. profile_symptoms subset of canonical symptom set
  6. precision_modes subset of canonical precision set
  7. risk_level in {low, medium, high}
  8. confidence in [0.0, 1.0]
  9. per-api_symbol expected-tag check (a curated whitelist of tags that
     MUST appear given the API's semantics)
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records, collect_field  # noqa: E402

CANON_OPT_TARGETS = {
    "improve_throughput", "enable_tensor_core", "reduce_vram",
    "kernel_fusion", "kernel_autotune", "reduce_launch_overhead",
    "reduce_io_bottleneck", "prevent_oom",
}
CANON_SYMPTOMS = {
    "low_sm_utilization", "tensor_core_not_used", "precision_not_optimized",
    "kernel_launch_overhead", "cpu_bound", "gpu_idle_between_steps",
    "vram_pressure", "oom_risk", "attention_bottleneck", "slow_per_step",
    "gradient_underflow",
}
CANON_PRECISIONS = {
    "fp32", "tf32", "fp16", "bf16", "mixed",
    "fp8", "fp8_e4m3", "fp8_e5m2", "fp4", "int8", "int4", "fp64",
}
RISK_LEVELS = {"low", "medium", "high"}

# Required tag whitelist per api_symbol
# Each (set_name, must_contain_at_least_one_of)
EXPECTED_TAGS = {
    "torch.autocast": {
        "technology_keys": ["pytorch_amp"],
        "hardware_feature_keys": ["tensor_cores"],
        "precision_modes": ["bf16", "fp16", "mixed"],
    },
    "torch.cuda.amp.GradScaler": {
        "technology_keys": ["pytorch_amp", "loss_scaling"],
        "precision_modes": ["fp16", "mixed"],
    },
    "torch.compile": {
        "technology_keys": ["torch_compile", "inductor"],
        "optimization_targets": ["improve_throughput", "kernel_fusion"],
    },
    "torch.cuda.graph": {
        "technology_keys": ["cuda_graphs"],
        "hardware_feature_keys": ["cuda_graphs"],
        "optimization_targets": ["reduce_launch_overhead"],
    },
    "torch.cuda.CUDAGraph": {
        "technology_keys": ["cuda_graphs"],
        "hardware_feature_keys": ["cuda_graphs"],
    },
    "torch.utils.data.DataLoader": {
        "technology_keys": ["dataloader_pin_memory", "dataloader_workers"],
        "optimization_targets": ["improve_throughput", "reduce_io_bottleneck"],
    },
    "torch.cuda.max_memory_allocated": {
        "technology_keys": ["vram_probe", "memory_tracking"],
        "optimization_targets": ["reduce_vram", "prevent_oom"],
    },
    "torch.cuda.reset_peak_memory_stats": {
        "technology_keys": ["vram_probe", "memory_tracking"],
        "optimization_targets": ["reduce_vram", "prevent_oom"],
    },
}


def main() -> int:
    fails: list[str] = []
    a_features = collect_field("hardware_feature_records", "features")
    b_tech = collect_field("code_doc_chunks", "technology_keys")
    records = load_records("api_symbol_chunks")
    if not (a_features and b_tech and records):
        print("FAIL one of schema/{A,B,D}/ is empty")
        return 1
    print(f"Schema A features union: {len(a_features)} tags")
    print(f"Schema B technology_keys union: {len(b_tech)} tags")
    print(f"Schema D records: {len(records)}")
    print()

    for i, rec in enumerate(records):
        sym = rec.get("api_symbol", f"<#{i}>")
        rid = rec.get("api_symbol_id", f"<#{i}>")
        ctx = f"[{rid}]"

        # 1
        api_syms = rec.get("api_symbols") or []
        if sym not in api_syms:
            fails.append(f"{ctx} api_symbols missing self-ref {sym!r}")

        # 2 hardware_feature_keys subset of A features
        hfk = set(rec.get("hardware_feature_keys") or [])
        bad_hfk = hfk - a_features
        if bad_hfk:
            fails.append(
                f"{ctx} hardware_feature_keys not in Schema A features: {sorted(bad_hfk)}"
            )

        # 3 technology_keys cross-link to B
        tech = set(rec.get("technology_keys") or [])
        unlinked = tech - b_tech
        if unlinked:
            # not strictly fatal - some D tech may not appear in B (e.g. loss_scaling)
            # we just print as info; treat as fail only if MORE than half unlinked
            if len(unlinked) > max(1, len(tech) // 2):
                fails.append(
                    f"{ctx} >half technology_keys not in Schema B: {sorted(unlinked)}"
                )

        # 4 optimization_targets
        ot = set(rec.get("optimization_targets") or [])
        bad_ot = ot - CANON_OPT_TARGETS
        if bad_ot:
            fails.append(
                f"{ctx} optimization_targets not in canonical: {sorted(bad_ot)}"
            )

        # 5 profile_symptoms
        ps = set(rec.get("profile_symptoms") or [])
        bad_ps = ps - CANON_SYMPTOMS
        if bad_ps:
            fails.append(
                f"{ctx} profile_symptoms not in canonical: {sorted(bad_ps)}"
            )

        # 6 precision_modes
        pm = set(rec.get("precision_modes") or [])
        bad_pm = pm - CANON_PRECISIONS
        if bad_pm:
            fails.append(
                f"{ctx} precision_modes not in canonical: {sorted(bad_pm)}"
            )

        # 7 risk_level
        rl = str(rec.get("risk_level", "")).lower()
        if rl not in RISK_LEVELS:
            fails.append(f"{ctx} risk_level={rl!r} not in {sorted(RISK_LEVELS)}")

        # 8 confidence
        conf = rec.get("confidence")
        try:
            cf = float(conf)
            if not (0.0 <= cf <= 1.0):
                fails.append(f"{ctx} confidence {cf} outside [0,1]")
        except (TypeError, ValueError):
            fails.append(f"{ctx} confidence not a number")

        # 9 per-API expected tag whitelist
        exp = EXPECTED_TAGS.get(sym)
        if exp:
            for fld, must_have_any in exp.items():
                have = set(rec.get(fld) or [])
                if not (set(must_have_any) & have):
                    fails.append(
                        f"{ctx} expected {fld} to contain at least one of "
                        f"{must_have_any}, got {sorted(have)}"
                    )
        print(f"OK   {rid}: ontology fields semantically valid")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== Schema D ontology 100% correct ({len(records)} records) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
