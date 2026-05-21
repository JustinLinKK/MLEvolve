"""Verify schema/code_doc_chunks.yaml against Schema B (code_doc_chunk_v1).

Strict checks:
  1. yaml parses, top-level is dict with `records` list
  2. every record has schema_version == 'code_doc_chunk_v1'
  3. every record passes the project validator
     (localml_scheduler.code_knowledge.records.validate_code_knowledge_record)
  4. chunk_id is unique, non-empty, follows dotted-key convention
  5. risk_level in {low, medium, high}
  6. confidence in [0.0, 1.0]
  7. all list-typed fields are actually lists of non-empty strings
  8. text length >= 80 characters (chunks must be substantive)
  9. text_hash matches sha256(text) when provided
"""
from __future__ import annotations
import hashlib
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
YAML_PATH = REPO / "schema" / "code_doc_chunks.yaml"

SCHEMA_VERSION = "code_doc_chunk_v1"
REQUIRED_FIELDS = {"schema_version", "chunk_id", "title", "text"}
LIST_FIELDS = {
    "technology_keys", "hardware_keys", "hardware_feature_keys",
    "model_keys", "model_families", "workload_types",
    "optimization_targets", "profile_symptoms", "api_symbols",
    "precision_modes",
}
RISK_LEVELS = {"low", "medium", "high"}
CHUNK_ID_PATTERN = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
MIN_TEXT_LEN = 80

sys.path.insert(0, str(REPO))


def main() -> int:
    fails: list[str] = []

    if not YAML_PATH.exists():
        print(f"FAIL yaml not found at {YAML_PATH}")
        return 1
    data = yaml.safe_load(YAML_PATH.read_text())
    if not isinstance(data, dict) or "records" not in data:
        print("FAIL yaml top-level must be a dict with 'records' key")
        return 1
    records = data["records"]
    if not isinstance(records, list) or not records:
        print("FAIL 'records' must be a non-empty list")
        return 1
    print(f"OK   yaml parsed, {len(records)} records")

    chunk_ids: list[str] = []
    for i, rec in enumerate(records):
        ctx = f"records[{i}]"
        if not isinstance(rec, dict):
            fails.append(f"{ctx}: not a dict")
            continue
        cid = rec.get("chunk_id", "<missing>")
        ctx = f"records[{i}] chunk_id={cid!r}"

        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            fails.append(f"{ctx}: missing required fields {sorted(missing)}")
            continue

        if rec["schema_version"] != SCHEMA_VERSION:
            fails.append(
                f"{ctx}: schema_version={rec['schema_version']!r} != "
                f"{SCHEMA_VERSION!r}"
            )

        if not isinstance(cid, str) or not cid.strip():
            fails.append(f"{ctx}: chunk_id empty or not a string")
            continue
        if not CHUNK_ID_PATTERN.match(cid):
            fails.append(
                f"{ctx}: chunk_id does not match dotted-snake-case pattern"
            )
        chunk_ids.append(cid)

        title = rec.get("title")
        if not isinstance(title, str) or not title.strip():
            fails.append(f"{ctx}: title empty or not a string")

        text = rec.get("text")
        if not isinstance(text, str) or not text.strip():
            fails.append(f"{ctx}: text empty or not a string")
        elif len(text.strip()) < MIN_TEXT_LEN:
            fails.append(
                f"{ctx}: text shorter than {MIN_TEXT_LEN} chars "
                f"({len(text.strip())} chars)"
            )

        risk = str(rec.get("risk_level", "medium")).lower()
        if risk not in RISK_LEVELS:
            fails.append(f"{ctx}: risk_level={risk!r} not in {RISK_LEVELS}")

        conf = rec.get("confidence", 0.5)
        try:
            cf = float(conf)
            if not (0.0 <= cf <= 1.0):
                fails.append(f"{ctx}: confidence={cf} outside [0.0, 1.0]")
        except (TypeError, ValueError):
            fails.append(f"{ctx}: confidence={conf!r} not a number")

        for fld in LIST_FIELDS:
            v = rec.get(fld)
            if v is None:
                continue
            if not isinstance(v, list):
                fails.append(f"{ctx}: {fld} must be a list, got {type(v).__name__}")
                continue
            for j, item in enumerate(v):
                if not isinstance(item, str) or not item.strip():
                    fails.append(f"{ctx}: {fld}[{j}] not a non-empty string")

        provided_hash = rec.get("text_hash")
        if provided_hash:
            expected = hashlib.sha256(
                str(rec.get("text", "")).encode("utf-8")
            ).hexdigest()
            if provided_hash != expected:
                fails.append(
                    f"{ctx}: text_hash mismatch (got {provided_hash[:16]}..., "
                    f"expected {expected[:16]}...)"
                )

        dep = rec.get("deprecated", False)
        if not isinstance(dep, bool):
            fails.append(f"{ctx}: deprecated must be bool, got {type(dep).__name__}")

    dupes = [c for c in set(chunk_ids) if chunk_ids.count(c) > 1]
    if dupes:
        fails.append(f"duplicate chunk_id values: {sorted(set(dupes))}")
    else:
        print(f"OK   all {len(chunk_ids)} chunk_id values unique")

    # run through the project validator (canonical source of truth)
    try:
        from localml_scheduler.code_knowledge.records import (
            validate_code_knowledge_record,
        )
    except Exception as exc:
        fails.append(f"could not import project validator: {exc!r}")
    else:
        for i, rec in enumerate(records):
            try:
                normalized = validate_code_knowledge_record(rec)
            except Exception as exc:
                fails.append(
                    f"records[{i}] chunk_id={rec.get('chunk_id')!r}: "
                    f"project validator rejects: {exc}"
                )
                continue
            if normalized.get("schema_version") != SCHEMA_VERSION:
                fails.append(
                    f"records[{i}]: validator normalized to "
                    f"{normalized.get('schema_version')!r}"
                )
        print(f"OK   project validator accepted {len(records)} records")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== Schema B 100% VALID ({len(records)} records) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
