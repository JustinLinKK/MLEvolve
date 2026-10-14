"""CLI for extracting a scheduler timeline replay fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .timeline_fixture import DEFAULT_SOURCE_RUN, extract_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract scheduler timeline replay fixture files.")
    parser.add_argument(
        "--source-run",
        default=str(DEFAULT_SOURCE_RUN),
        help="Comparison run root or scheduler_runtime root to extract from.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where timeline.json, jobs.jsonl, baseline_summary.json, and scheduler settings are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = extract_fixture(args.source_run, args.output_dir)
    summary = json.loads(paths.baseline_summary.read_text(encoding="utf-8"))
    print(json.dumps({"fixture": str(Path(args.output_dir).resolve()), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

