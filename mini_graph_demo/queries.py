"""Run schema/example_queries.cypher with fixture parameter values.

Read-only. Pretty-prints each query's result table.
"""
from __future__ import annotations

import re
from pathlib import Path

from mini_graph_demo._driver import connect, session

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_FILE = REPO_ROOT / "schema" / "example_queries.cypher"

PARAMS = {
    "model_key": "mgd:resnet50.timm",
    "hardware_key": "mgd:nvidia.rtx_5090.cu128",
    "model_family": "resnet",
    "job_id": "mgd:job-001",
}


def _split_queries(text: str) -> list[str]:
    # Strip // comment lines and split on `;` outside strings (queries do not
    # use embedded semicolons in string literals).
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        cleaned.append(line)
    body = "\n".join(cleaned)
    return [q.strip() for q in body.split(";") if q.strip()]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + " | ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> int:
    queries = _split_queries(QUERIES_FILE.read_text(encoding="utf-8"))
    driver = connect()
    try:
        with session(driver) as s:
            for i, q in enumerate(queries, 1):
                print(f"\n=== Query {i} ===")
                print(re.sub(r"^", "  ", q, flags=re.MULTILINE))
                print("--- result ---")
                rows = [rec.data() for rec in s.run(q, **PARAMS)]
                _print_table(rows)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
