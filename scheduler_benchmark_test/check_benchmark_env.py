"""Preflight check for scheduler benchmark Python dependencies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_support import assert_benchmark_python_deps, assert_cassava_root


def _trace_data_root(trace_path: str | None) -> str | None:
    if not trace_path:
        return None
    path = Path(trace_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            data_root = payload.get("data_root")
            if data_root:
                return str(data_root)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    assert_benchmark_python_deps()
    data_root = os.environ.get("CASSAVA_ROOT")
    if data_root:
        assert_cassava_root(data_root)
    else:
        trace_data_root = _trace_data_root(args.trace)
        if trace_data_root:
            assert_cassava_root(trace_data_root)
    print("benchmark environment OK")


if __name__ == "__main__":
    main()
