"""Opt-in prompt guidance from the pinned hardware knowledge graph."""

from __future__ import annotations

from typing import Any

from .paths import expose_components


class HardwareKnowledgePlugin:
    def guidance(
        self,
        stage: str,
        task_description: str,
        candidate_code: str = "",
        **stage_context: Any,
    ) -> str | None:
        expose_components()
        from HWKD.hwkg.injection import hardware_guidance

        return hardware_guidance(stage, task_description, candidate_code, **stage_context)

    def enrich_task(self, task_description: dict[str, Any] | str) -> dict[str, Any] | str:
        enriched = dict(task_description) if isinstance(task_description, dict) else task_description
        guidance = self.guidance("draft", str(task_description))
        if guidance:
            if isinstance(enriched, dict):
                enriched["Hardware knowledge (advisory)"] = guidance
            else:
                enriched += "\n\n# Hardware knowledge (advisory)\n" + guidance
        return enriched
