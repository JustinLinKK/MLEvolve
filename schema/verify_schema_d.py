"""Verify schema/api_symbol_chunks.yaml is 100% accurate against installed libs.

Strict checks:
  1. yaml parses, all records use schema_version=api_symbol_chunk_v1
  2. api_symbol resolves via importlib + getattr (no MISSING)
  3. signature stored in yaml == str(inspect.signature(obj)) AT VERIFY TIME
     (this is the accuracy guarantee: if torch upgrades the signature, this fails)
  4. parameters_json round-trips and matches re-introspected params (name + kind + default + annotation)
  5. example_code is syntactically valid Python (compiles cleanly)
  6. all required Schema D fields present
  7. project validator (validate_code_knowledge_record) accepts every record
"""
from __future__ import annotations
import importlib
import inspect
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
D_PATH = REPO / "schema" / "api_symbol_chunks.yaml"
SCHEMA_VERSION = "api_symbol_chunk_v1"
REQUIRED = {
    "schema_version", "api_symbol_id", "title", "api_symbol",
    "signature", "usage_summary", "parameters_json", "example_code",
    "text", "framework", "framework_version", "deprecated",
    "risk_level", "confidence",
}

sys.path.insert(0, str(REPO))


def resolve(dotted: str):
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for p in parts[1:]:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            obj = importlib.import_module(f"{obj.__name__}.{p}")
    return obj


def current_params(obj) -> list[dict]:
    sig = inspect.signature(obj)
    out = []
    for name, p in sig.parameters.items():
        out.append({
            "name": name,
            "kind": p.kind.name,
            "default": "<empty>" if p.default is inspect.Parameter.empty else repr(p.default),
            "annotation": "<empty>" if p.annotation is inspect.Parameter.empty else str(p.annotation),
        })
    return out


def main() -> int:
    fails: list[str] = []

    if not D_PATH.exists():
        print(f"FAIL {D_PATH} missing")
        return 1
    data = yaml.safe_load(D_PATH.read_text())
    records = data.get("records") or []
    print(f"loaded {len(records)} Schema D records from {D_PATH.name}")

    for i, rec in enumerate(records):
        sym = rec.get("api_symbol", f"<#{i}>")
        ctx = f"[{sym}]"

        # 1. schema_version
        if rec.get("schema_version") != SCHEMA_VERSION:
            fails.append(f"{ctx} schema_version={rec.get('schema_version')!r}")
            continue

        # 6. required fields
        missing = REQUIRED - rec.keys()
        if missing:
            fails.append(f"{ctx} missing fields {sorted(missing)}")
            continue

        # 2. resolve symbol
        try:
            obj = resolve(sym)
        except Exception as exc:
            fails.append(f"{ctx} cannot resolve: {exc.__class__.__name__}: {exc}")
            continue
        print(f"OK   {sym}: resolved")

        # 3. signature equality
        try:
            cur_sig = str(inspect.signature(obj))
        except Exception as exc:
            fails.append(f"{ctx} inspect.signature failed: {exc}")
            continue
        stored_sig = str(rec["signature"]).strip()
        if cur_sig != stored_sig:
            fails.append(
                f"{ctx} signature mismatch\n"
                f"     yaml: {stored_sig}\n"
                f"     live: {cur_sig}"
            )
            continue
        print(f"OK   {sym}: signature exact match")

        # 4. parameters_json round-trip + match
        try:
            stored_params = json.loads(rec["parameters_json"])
        except json.JSONDecodeError as exc:
            fails.append(f"{ctx} parameters_json invalid: {exc}")
            continue
        cur_params = current_params(obj)
        if stored_params != cur_params:
            fails.append(
                f"{ctx} parameters_json mismatch\n"
                f"     yaml: {stored_params}\n"
                f"     live: {cur_params}"
            )
            continue
        print(f"OK   {sym}: parameters_json matches {len(cur_params)} params")

        # 5. example_code compiles
        try:
            compile(rec["example_code"], f"<{sym}.example>", "exec")
        except SyntaxError as exc:
            fails.append(f"{ctx} example_code SyntaxError: {exc}")
            continue
        print(f"OK   {sym}: example_code syntactically valid")

    # 7. project validator
    try:
        from localml_scheduler.code_knowledge.records import validate_code_knowledge_record
    except Exception as exc:
        fails.append(f"could not import project validator: {exc}")
    else:
        n_accepted = 0
        for i, rec in enumerate(records):
            try:
                normalized = validate_code_knowledge_record(rec)
                if normalized.get("schema_version") == SCHEMA_VERSION:
                    n_accepted += 1
                else:
                    fails.append(
                        f"records[{i}] {rec.get('api_symbol')!r}: validator "
                        f"normalized to {normalized.get('schema_version')!r}"
                    )
            except Exception as exc:
                fails.append(
                    f"records[{i}] {rec.get('api_symbol')!r}: validator "
                    f"rejected: {exc}"
                )
        print(f"\nproject validator: accepted {n_accepted}/{len(records)} records")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== Schema D 100% VALID ({len(records)} records) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
