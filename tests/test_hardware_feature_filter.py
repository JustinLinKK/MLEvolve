from __future__ import annotations

import json
from pathlib import Path

from localml_scheduler.hardware_knowledge.feature_filter import query_hardware_node


def test_stage_pattern_filter_ignores_keywords_hidden_in_source_urls(tmp_path: Path) -> None:
    graph_path = tmp_path / "hardware_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "hw:test",
                        "label": "Hardware",
                        "properties": {
                            "name": "Test GPU",
                            "hardware_id": "test-gpu",
                            "recommended_patterns": [
                                "use compact memory plan for local LLM PEFT [https://images.example.com/architecture.pdf]",
                                "use GPU image decode for dataset pipelines",
                            ],
                            "avoid_patterns": [
                                "no memory-only policy [https://example.com/dataset-guide]",
                                "no dense full-resolution image tensor loading",
                            ],
                        },
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    result = query_hardware_node("Test GPU", "datatype", graph_path=graph_path)

    assert result["recommended_patterns"] == ["use GPU image decode for dataset pipelines"]
    assert result["avoid_patterns"] == ["no dense full-resolution image tensor loading"]
