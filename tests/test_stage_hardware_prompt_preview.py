from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.hardware_context import compact_optimization_context, format_hardware_prompt_section


STAGES = ("model_design", "datatype_precision", "training_evaluation")
STAGE_ALIASES = {
    "model_design": "model_design",
    "model-design": "model_design",
    "stage1": "model_design",
    "datatype_precision": "datatype_precision",
    "datatype": "datatype_precision",
    "datatype_quantization": "datatype_precision",
    "datatype-quantization": "datatype_precision",
    "quantization": "datatype_precision",
    "training_evaluation": "training_evaluation",
    "optimizer": "training_evaluation",
    "tuning": "training_evaluation",
    "training": "training_evaluation",
}


def _selected_stages(stage: str | None) -> tuple[str, ...]:
    if stage is None or not stage.strip() or stage.strip().lower() == "all":
        return STAGES

    normalized = stage.strip().lower().replace("-", "_")
    normalized = STAGE_ALIASES.get(normalized, normalized)
    if normalized not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)} or all")
    return (normalized,)


def _load_feature_filter() -> ModuleType:
    """Load the filter module directly to avoid package import side effects on Windows."""
    module_path = REPO_ROOT / "localml_scheduler" / "hardware_knowledge" / "feature_filter.py"
    spec = importlib.util.spec_from_file_location("feature_filter_direct", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load feature filter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_hardware_payload(hardware_name: str = "GeForce RTX 5090", stage: str | None = None) -> dict:
    filters = _load_feature_filter()
    selected_stages = _selected_stages(stage)
    stages = []
    linked_features = []
    hardware = {}

    for selected_stage in selected_stages:
        node = filters.query_hardware_node(hardware_name, selected_stage)
        features = filters.query_hardware_features(hardware_name, selected_stage)
        assert node["found"] is True
        assert features["found"] is True
        assert features["stage_filter"] == selected_stage

        if not hardware:
            hardware = {
                "node_id": node.get("node_id"),
                "gpu_name": node.get("gpu_name"),
                "architecture": node.get("architecture"),
                "vram_MB": node.get("vram_MB"),
                "compute_capability": node.get("compute_capability"),
            }

        stage_features = list(features.get("features") or [])
        linked_features.extend(
            {"feature_id": feature.get("feature_id")}
            for feature in stage_features
            if feature.get("feature_id")
        )
        stages.append(
            {
                "stage": selected_stage,
                "node": node,
                "features": stage_features,
                "feature_count": features.get("feature_count"),
            }
        )

    return {
        "hardware_context": {
            "found": True,
            "hardware": {
                "gpu_name": hardware.get("gpu_name"),
                "summary_text": (
                    f"{hardware.get('gpu_name')} "
                    f"({hardware.get('architecture')}, CC {hardware.get('compute_capability')}, "
                    f"{hardware.get('vram_MB')} MiB VRAM)"
                ),
                "total_vram_mb": hardware.get("vram_MB"),
                "compute_capability": hardware.get("compute_capability"),
            },
            "backend_capabilities": {
                "mode": "unit_test",
                "effective_mode": "stage_filter_preview",
                "enabled_backends": ["exclusive"],
            },
            "scheduler_limits": {"safe_vram_budget_mb": 10_807},
        },
        "stage_hardware_features": {
            "found": True,
            "stage_filter": list(selected_stages),
            "hardware": hardware,
            "stages": stages,
            "features": linked_features,
            "feature_count": len(linked_features),
            "source": "hardware_knowledge_graph.json",
        },
        "confidence": 0.5,
    }


def build_stage_hardware_prompt_preview(
    hardware_name: str = "GeForce RTX 5090", stage: str | None = None
) -> tuple[dict, str]:
    raw_context = _stage_hardware_payload(hardware_name, stage=stage)
    compact = compact_optimization_context(raw_context)
    prompt = format_hardware_prompt_section(compact, max_chars=8_000)
    return compact, prompt


def test_stage_hardware_database_query_is_pieced_into_filtered_prompt() -> None:
    compact, prompt = build_stage_hardware_prompt_preview()
    stage_context = compact["stage_hardware_features"]
    stage_by_name = {stage["stage"]: stage for stage in stage_context["stages"]}

    assert set(stage_by_name) == set(STAGES)
    assert "# Hardware/Profile Optimization Context" in prompt
    assert "- Stage-filtered hardware knowledge:" in prompt
    assert "  - model_design" in prompt
    assert "  - datatype_precision" in prompt
    assert "  - training_evaluation" in prompt
    assert "filter_audit" in prompt

    model_design_ids = {feature["feature_id"] for feature in stage_by_name["model_design"]["features"]}
    datatype_ids = {feature["feature_id"] for feature in stage_by_name["datatype_precision"]["features"]}
    training_ids = {feature["feature_id"] for feature in stage_by_name["training_evaluation"]["features"]}
    assert "dataset_decomposition" in prompt
    assert "tensor_cores" in model_design_ids
    assert "bf16" in datatype_ids
    assert "muon_optimizer" in training_ids
    assert "soap_optimizer" not in training_ids
    assert "ademamix_optimizer" not in training_ids
    assert "omitted_not_recommended=['soap_optimizer', 'ademamix_optimizer']" in prompt

    assert "ns_steps=5" in prompt
    assert "momentum=0.95" in prompt
    assert "nesterov=True" in prompt
    assert "match_rms_adamw" in prompt
    assert "experimental_recipes" not in prompt
    assert "not widely confirmed" not in prompt


def test_single_stage_preview_only_contains_requested_stage() -> None:
    compact, prompt = build_stage_hardware_prompt_preview(stage="training_evaluation")
    stages = compact["stage_hardware_features"]["stages"]
    stage_names = [stage["stage"] for stage in stages]

    assert stage_names == ["training_evaluation"]
    assert "  - training_evaluation" in prompt
    assert "  - model_design" not in prompt
    assert "  - datatype_precision" not in prompt
    assert "muon_optimizer" in prompt
    assert "bf16" not in prompt
    assert "omitted_not_recommended=['soap_optimizer', 'ademamix_optimizer']" in prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the stage-filtered hardware-aware prompt section generated from the hardware DB."
    )
    parser.add_argument("stage_assignment", nargs="?", help="Optional shorthand like stage=training_evaluation.")
    parser.add_argument("--hardware", default="GeForce RTX 5090", help="Hardware name to query in the knowledge graph.")
    parser.add_argument(
        "--stage",
        choices=(*STAGES, *STAGE_ALIASES.keys(), "all"),
        default="all",
        help="Single pipeline stage to preview, or all stages.",
    )
    args = parser.parse_args()
    args.stage = STAGE_ALIASES.get(str(args.stage).strip().lower().replace("-", "_"), args.stage)
    if args.stage_assignment:
        key, separator, value = args.stage_assignment.partition("=")
        if key != "stage" or separator != "=":
            parser.error("positional shorthand must use stage=<stage>, for example stage=training_evaluation")
        if args.stage != "all":
            parser.error("use either --stage <stage> or stage=<stage>, not both")
        value = STAGE_ALIASES.get(value.strip().lower().replace("-", "_"), value)
        if value not in (*STAGES, "all"):
            parser.error(f"stage must be one of {', '.join((*STAGES, 'all'))}")
        args.stage = value
    return args


if __name__ == "__main__":
    args = _parse_args()
    requested_stage = None if args.stage == "all" else args.stage
    compact_context, prompt_section = build_stage_hardware_prompt_preview(args.hardware, stage=requested_stage)
    print(f"Hardware: {args.hardware}")
    print(f"Stage filter: {args.stage}")
    print("Stage hardware feature counts:")
    for stage_item in compact_context["stage_hardware_features"]["stages"]:
        feature_ids = [feature.get("feature_id") for feature in stage_item.get("features", [])]
        omitted = stage_item.get("omitted_not_recommended", [])
        print(
            f"- {stage_item['stage']}: shown={stage_item.get('shown_feature_count', 0)} "
            f"of {stage_item.get('feature_count', 0)}; features={feature_ids}; omitted={omitted}"
        )
    print()
    print(prompt_section)
