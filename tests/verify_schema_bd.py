"""Verify Schema B + D records are 100% correct vs the report's spec.

Checks both:
  - VALIDATOR-REQUIRED fields (must be present; validator rejects otherwise)
  - REPORT-RECOMMENDED fields (should be present per section 6 / 8 of
    schema/hardware_knowledge_vector_database_report.md)
Plus type sanity, project validator round-trip, and basic content quality.
"""
from __future__ import annotations
import hashlib
import re
import sys
from pathlib import Path

import yaml  # noqa: F401

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

# from report section 6
B_REQUIRED = {"schema_version", "chunk_id", "title", "text"}
B_RECOMMENDED = {
    "source_id", "source_type", "source_title", "source_url", "source_version",
    "framework", "framework_version",
    "technology_keys", "hardware_keys", "hardware_feature_keys",
    "model_keys", "model_families", "workload_types",
    "optimization_targets", "profile_symptoms", "api_symbols",
    "precision_modes", "risk_level", "confidence", "deprecated",
}

# from report section 8
D_REQUIRED = {"schema_version", "api_symbol_id", "api_symbol", "usage_summary"}
D_RECOMMENDED = {
    "signature", "parameters_json",
    "source_id", "source_type", "source_title", "source_url", "source_version",
    "framework", "framework_version",
    "technology_keys", "hardware_feature_keys",
    "optimization_targets", "profile_symptoms", "api_symbols",
    "precision_modes", "risk_level", "confidence", "deprecated",
}

LIST_FIELDS = {
    "technology_keys", "hardware_keys", "hardware_feature_keys",
    "model_keys", "model_families", "workload_types",
    "optimization_targets", "profile_symptoms", "api_symbols",
    "precision_modes",
}
RISK_LEVELS = {"low", "medium", "high"}
CHUNK_ID_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
MIN_TEXT_LEN = 50

sys.path.insert(0, str(REPO))


def check_record(rec: dict, schema: str, required: set, recommended: set,
                 id_field: str, content_field: str, fails: list[str]) -> None:
    rid = rec.get(id_field, "<missing-id>")
    ctx = f"[{schema}/{rid}]"

    if rec.get("schema_version") != schema:
        fails.append(f"{ctx} bad schema_version")
        return
    missing_req = required - rec.keys()
    if missing_req:
        fails.append(f"{ctx} missing REQUIRED {sorted(missing_req)}")
        return
    missing_rec = recommended - rec.keys()
    if missing_rec:
        fails.append(f"{ctx} missing recommended {sorted(missing_rec)}")

    # id pattern (B only)
    if id_field == "chunk_id" and not CHUNK_ID_PATTERN.match(str(rid)):
        fails.append(f"{ctx} id pattern violated")

    # title + content non-empty
    for fld in ("title", content_field):
        v = rec.get(fld)
        if not isinstance(v, str) or not v.strip():
            fails.append(f"{ctx} {fld} empty/not-string")
    text = str(rec.get(content_field, "")).strip()
    if len(text) < MIN_TEXT_LEN:
        fails.append(f"{ctx} {content_field} shorter than {MIN_TEXT_LEN} chars ({len(text)})")

    # source_* strings
    for sf in ("source_id", "source_type", "source_title", "source_url", "source_version"):
        if sf in rec and not isinstance(rec[sf], str):
            fails.append(f"{ctx} {sf} not string")

    # risk_level / confidence / deprecated types
    rl = str(rec.get("risk_level", "medium")).lower()
    if rl not in RISK_LEVELS:
        fails.append(f"{ctx} risk_level={rl!r} not in {sorted(RISK_LEVELS)}")
    conf = rec.get("confidence", 0.5)
    try:
        cf = float(conf)
        if not (0.0 <= cf <= 1.0):
            fails.append(f"{ctx} confidence {cf} outside [0,1]")
    except (TypeError, ValueError):
        fails.append(f"{ctx} confidence not a number")
    if "deprecated" in rec and not isinstance(rec["deprecated"], bool):
        fails.append(f"{ctx} deprecated not bool")

    # list fields are lists of non-empty strings
    for fld in LIST_FIELDS:
        if fld not in rec:
            continue
        v = rec[fld]
        if not isinstance(v, list):
            fails.append(f"{ctx} {fld} not a list")
            continue
        for i, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                fails.append(f"{ctx} {fld}[{i}] not non-empty string")

    # text_hash sanity (B only)
    if id_field == "chunk_id" and "text_hash" in rec:
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if rec["text_hash"] != expected:
            fails.append(f"{ctx} text_hash mismatch")


def main() -> int:
    fails: list[str] = []
    # ---- Schema B ----
    b_recs = load_records("code_doc_chunks")
    print(f"Schema B: {len(b_recs)} records")
    seen_b_ids: set[str] = set()
    for rec in b_recs:
        cid = rec.get("chunk_id", "")
        if cid in seen_b_ids:
            fails.append(f"[B/{cid}] duplicate chunk_id")
        else:
            seen_b_ids.add(cid)
        check_record(rec, "code_doc_chunk_v1", B_REQUIRED, B_RECOMMENDED,
                     "chunk_id", "text", fails)
    print(f"  unique chunk_ids: {len(seen_b_ids)}")

    # ---- Schema D ----
    d_recs = load_records("api_symbol_chunks")
    print(f"Schema D: {len(d_recs)} records")
    seen_d_ids: set[str] = set()
    for rec in d_recs:
        rid = rec.get("api_symbol_id", "")
        if rid in seen_d_ids:
            fails.append(f"[D/{rid}] duplicate api_symbol_id")
        else:
            seen_d_ids.add(rid)
        check_record(rec, "api_symbol_chunk_v1", D_REQUIRED, D_RECOMMENDED,
                     "api_symbol_id", "usage_summary", fails)
    print(f"  unique api_symbol_ids: {len(seen_d_ids)}")

    # project validator round-trip
    try:
        from localml_scheduler.code_knowledge.records import validate_code_knowledge_record
    except Exception as exc:
        fails.append(f"could not import project validator: {exc}")
    else:
        n_b = sum(1 for r in b_recs
                  if _safe_validate(validate_code_knowledge_record, r) == "code_doc_chunk_v1")
        n_d = sum(1 for r in d_recs
                  if _safe_validate(validate_code_knowledge_record, r) == "api_symbol_chunk_v1")
        print(f"\nproject validator: B {n_b}/{len(b_recs)}, D {n_d}/{len(d_recs)}")
        if n_b != len(b_recs):
            fails.append(f"validator rejected {len(b_recs)-n_b} B records")
        if n_d != len(d_recs):
            fails.append(f"validator rejected {len(d_recs)-n_d} D records")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== Schema B + D 100% report-compliant "
          f"({len(b_recs)} B + {len(d_recs)} D records) ===")
    return 0


def _safe_validate(fn, rec):
    try:
        return fn(rec).get("schema_version")
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
