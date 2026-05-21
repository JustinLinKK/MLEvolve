"""Shared IO helpers for per-record schema folders.

Each schema lives in its own subdir under schema/:
  schema/hardware_feature_records/    (Schema hardware_feature_record_v1)
  schema/code_doc_chunks/             (Schema code_doc_chunk_v1)
  schema/optimization_recipe_chunks/  (Schema optimization_recipe_chunk_v1)
  schema/api_symbol_chunks/           (Schema api_symbol_chunk_v1)
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent
SCHEMA_DIR = REPO / "schema"


def safe_name(rid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]", "_", rid)


def load_records(folder: str) -> list[dict[str, Any]]:
    """Read all per-record yamls in schema/<folder>/*.yaml, sorted by filename."""
    d = SCHEMA_DIR / folder
    if not d.exists():
        return []
    records = []
    for p in sorted(d.glob("*.yaml")):
        rec = yaml.safe_load(p.read_text())
        if isinstance(rec, dict):
            records.append(rec)
    return records


def save_record(folder: str, rid: str, record: dict[str, Any]) -> Path:
    d = SCHEMA_DIR / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{safe_name(rid)}.yaml"
    p.write_text(yaml.safe_dump(record, sort_keys=False, width=4096,
                                allow_unicode=True))
    return p


def collect_field(folder: str, field: str) -> set[str]:
    """Union of list-field values across all records in a schema folder."""
    out: set[str] = set()
    for rec in load_records(folder):
        for v in rec.get(field) or []:
            if isinstance(v, str):
                out.add(v)
    return out
