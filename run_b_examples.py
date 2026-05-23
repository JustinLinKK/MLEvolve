"""Run every B chunk's example_code in a fresh subprocess. Verify all succeed."""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")


def run_example(cid: str, code: str) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([PY, path], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return "PASS", r.stdout.strip()
        msg = (r.stdout + r.stderr).strip()
        # SystemExit('SKIP: ...') exits with code 1 but is intentional
        if "SKIP" in msg:
            return "SKIP", msg.splitlines()[-1]
        return "FAIL", msg.splitlines()[-1] if msg else "<no output>"
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    records = load_records("code_doc_chunks")
    results = []
    for rec in records:
        cid = rec["chunk_id"]
        code = rec.get("example_code") or ""
        if not code.strip():
            results.append((cid, "NO_CODE", "example_code field empty"))
            continue
        status, msg = run_example(cid, code)
        results.append((cid, status, msg))
        print(f"{status:7s} {cid}: {msg[:80]}")

    print()
    fails = [r for r in results if r[1] == "FAIL"]
    skips = [r for r in results if r[1] == "SKIP"]
    passes = [r for r in results if r[1] == "PASS"]
    print(f"=== {len(passes)} PASS, {len(skips)} SKIP, {len(fails)} FAIL ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
