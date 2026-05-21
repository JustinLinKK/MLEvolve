"""Verify each example query returns the expected row count and field values.

Exit 0 if all queries pass; non-zero otherwise.

Per-query parameter dicts (not a single global) because the example queries
target different models:
  Q1 / Q3 / Q4 — resnet50 evidence
  Q2          — vit_base batch-size probe
"""
from __future__ import annotations

import sys
from pathlib import Path

from mini_graph_demo._driver import connect, session
from mini_graph_demo.queries import _split_queries

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_FILE = REPO_ROOT / "schema" / "example_queries.cypher"

PARAMS_Q1 = {
    "model_key": "mgd:resnet50.timm",
    "hardware_key": "mgd:nvidia.rtx_5090.cu128",
}
PARAMS_Q2 = {
    "model_key": "mgd:vit_base_patch16_224.timm",
    "hardware_key": "mgd:nvidia.rtx_5090.cu128",
}
PARAMS_Q3 = {
    "model_family": "resnet",
    "hardware_key": "mgd:nvidia.rtx_5090.cu128",
}
PARAMS_Q4 = {
    "job_id": "mgd:job-001",
}


def _expectation_q1(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    r = rows[0]
    if r.get("j.resolved_batch_size") != 256:
        return f"expected resolved_batch_size=256, got {r.get('j.resolved_batch_size')}"
    if r.get("j.peak_vram_mb") is None:
        return "peak_vram_mb is None"
    return None


def _expectation_q2(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    mx = rows[0].get("j.max_safe_batch_size")
    if not (isinstance(mx, int) and mx > 0):
        return f"max_safe_batch_size not > 0, got {mx!r}"
    return None


def _expectation_q3(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    members = rows[0].get("members") or []
    # Q3 filters WHERE m.model_family = $model_family, so collect(m.model_key)
    # only gathers members whose model_family matches the filter param ("resnet").
    # The packed job has 1 resnet member and 1 vit member; only the resnet one
    # passes the filter, yielding 1 entry.  Expect >= 1.
    if len(members) < 1:
        return f"expected >= 1 member, got {len(members)}"
    return None


def _expectation_q4(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    r = rows[0]
    if not r.get("hardware_key"):
        return "hardware_key empty"
    tech = r.get("technology_keys") or []
    if not tech:
        return "no technology keys returned"
    return None


EXPECTATIONS = [_expectation_q1, _expectation_q2, _expectation_q3, _expectation_q4]
ALL_PARAMS = [PARAMS_Q1, PARAMS_Q2, PARAMS_Q3, PARAMS_Q4]


def main() -> int:
    queries = _split_queries(QUERIES_FILE.read_text(encoding="utf-8"))
    if len(queries) < 4:
        sys.stderr.write(
            f"FAIL: example_queries.cypher has {len(queries)} queries, "
            f"expected >= 4\n"
        )
        return 1

    driver = connect()
    failed = 0
    try:
        with session(driver) as s:
            for i in range(4):
                rows = [rec.data() for rec in s.run(queries[i], **ALL_PARAMS[i])]
                err = EXPECTATIONS[i](rows)
                if err:
                    print(f"Query {i+1}: FAIL: {err}")
                    failed += 1
                else:
                    print(f"Query {i+1}: PASS")
    finally:
        driver.close()

    if failed:
        return 1
    print("verify_queries: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
