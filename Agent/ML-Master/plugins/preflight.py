"""Optional bounded source preflight before ML-Master executes a candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import expose_components


class PreflightPlugin:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def check(self, code: str, node_id: str) -> dict[str, Any]:
        """Persist a checker report; return its admission decision and path."""
        expose_components()
        from Preflight_checker.preflight import check_source

        node_dir = self.workspace / "working" / "preflight" / str(node_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        report = check_source(
            code,
            filename=f"ml-master-node-{node_id}.py",
            workspace=self.workspace,
        )
        report_path = node_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return {**report, "report_path": str(report_path)}
