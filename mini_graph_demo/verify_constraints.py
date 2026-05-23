"""Verify uniqueness constraints from schema/neo4j_constraints.cypher are
present AND enforced. Exit 0 on success, non-zero on any failure.

Assumes seed.py has already run.
"""
from __future__ import annotations

import sys

from neo4j.exceptions import ConstraintError

from mini_graph_demo._driver import connect, session

REQUIRED_CONSTRAINTS = {
    ("Job", "job_id"),
    ("Hardware", "hardware_key"),
    ("Model", "model_key"),
    ("TrainingConfig", "config_key"),
    ("Technology", "technology_key"),
    ("PackedJobMember", "member_id"),
}


def _present_constraints(s) -> set[tuple[str, str]]:
    rows = list(s.run("SHOW CONSTRAINTS"))
    present: set[tuple[str, str]] = set()
    for row in rows:
        d = row.data()
        labels = d.get("labelsOrTypes") or []
        props = d.get("properties") or []
        if d.get("type", "").startswith("UNIQUENESS") and labels and props:
            present.add((labels[0], props[0]))
    return present


def _check_present(s) -> list[str]:
    missing = REQUIRED_CONSTRAINTS - _present_constraints(s)
    return [f"missing UNIQUENESS constraint on ({lbl}, {prop})" for lbl, prop in missing]


def _check_negative_test(s) -> str | None:
    """Try to CREATE a duplicate job_id; expect ConstraintError."""
    try:
        s.run(
            "CREATE (:Job:SingleJob {job_id: 'mgd:job-001', demo_run: true})"
        ).consume()
    except ConstraintError:
        return None
    return ("negative test failed: duplicate job_id 'mgd:job-001' was accepted "
            "(expected ConstraintError)")


def main() -> int:
    driver = connect()
    failures: list[str] = []
    try:
        with session(driver) as s:
            failures.extend(_check_present(s))
            err = _check_negative_test(s)
            if err:
                failures.append(err)
    finally:
        driver.close()

    if failures:
        for line in failures:
            sys.stderr.write(f"FAIL: {line}\n")
        return 1
    print("verify_constraints: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
