"""Apply the curated native-training precision policy to the hardware graph.

This update is intentionally deterministic so the JSON source and generated
Cypher mirror can be reproduced together.  It does not infer training support
from low-level PTX types: only formats with a documented end-to-end MLEvolve
training path are made visible to datatype optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "hardware_knowledge_graph.json"

MIXED_PRECISION_URL = (
    "https://docs.nvidia.com/deeplearning/performance/"
    "mixed-precision-training/index.html"
)
AMPERE_URL = "https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/"
ADA_URL = "https://docs.nvidia.com/cuda/archive/12.9.2/ada-tuning-guide/index.html"
FP8_URL = (
    "https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/"
    "low_precision_training/fp8_current_scaling/fp8_current_scaling.html"
)
MXFP8_URL = (
    "https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/"
    "low_precision_training/mxfp8/mxfp8.html"
)
NVFP4_URL = (
    "https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/"
    "user-guide/features/low_precision_training/nvfp4/nvfp4.html"
)


FEATURE_UPDATES: dict[str, dict[str, Any]] = {
    "amp": {
        "min_compute_capability": "7.0",
        "native_tensor_core_evidence": MIXED_PRECISION_URL,
        "training_backend": "torch.amp",
        "supported_training_backends": ["torch.amp", "NVIDIA CUDA Tensor Cores"],
        "optimization_modes": ["normal", "aggressive"],
        "prompt_visibility": "recommendation",
        "model_shape_limitations": "The autocast dtype must itself be allowed for the target architecture.",
    },
    "fp16": {
        "description": (
            "Hardware-native FP16 Tensor Core mixed-precision training. Use AMP/autocast for "
            "forward and backward computation, framework-managed FP32 optimizer state, and "
            "loss scaling when gradients would otherwise underflow."
        ),
        "min_compute_capability": "7.0",
        "native_tensor_core_evidence": MIXED_PRECISION_URL,
        "training_backend": "torch.amp",
        "supported_training_backends": ["torch.amp", "NVIDIA CUDA Tensor Cores"],
        "optimization_modes": ["normal", "aggressive"],
        "prompt_visibility": "recommendation",
        "model_shape_limitations": "Tensor Core-friendly matrix dimensions are required for best speed; retain FP32 islands for fragile operations.",
        "source_url": MIXED_PRECISION_URL,
    },
    "bf16": {
        "description": (
            "Hardware-native BF16 Tensor Core mixed-precision training on Ampere and later. "
            "Use AMP/autocast for forward and backward computation; BF16 normally does not "
            "need loss scaling and retains FP32-like exponent range."
        ),
        "min_compute_capability": "8.0",
        "native_tensor_core_evidence": AMPERE_URL,
        "training_backend": "torch.amp",
        "supported_training_backends": ["torch.amp", "NVIDIA CUDA Tensor Cores"],
        "optimization_modes": ["normal", "aggressive"],
        "prompt_visibility": "recommendation",
        "model_shape_limitations": "Tensor Core-friendly matrix dimensions are required for best speed; unsupported operators may run at higher precision.",
        "source_url": AMPERE_URL,
    },
    "tf32": {
        "description": (
            "Hardware-native TF32 Tensor Core math for FP32 matrix and convolution training on "
            "Ampere and later. It preserves FP32 storage and range while accelerating eligible operations."
        ),
        "min_compute_capability": "8.0",
        "native_tensor_core_evidence": AMPERE_URL,
        "training_backend": "pytorch_cuda_tf32",
        "supported_training_backends": ["PyTorch CUDA matmul", "cuDNN"],
        "optimization_modes": ["normal", "aggressive"],
        "prompt_visibility": "recommendation",
        "model_shape_limitations": "Only eligible FP32 matrix multiplication and convolution operations use TF32 Tensor Cores.",
        "source_url": AMPERE_URL,
    },
    "fp8": {
        "description": (
            "Hardware-native FP8 Tensor Core training on Ada and later through NVIDIA "
            "Transformer Engine. It is an aggressive-mode option that requires compatible "
            "modules, tensor shapes, packages, accuracy validation, and a BF16/FP16 fallback."
        ),
        "min_compute_capability": "8.9",
        "native_tensor_core_evidence": ADA_URL,
        "training_backend": "transformer_engine",
        "supported_training_backends": ["NVIDIA Transformer Engine"],
        "optimization_modes": ["aggressive"],
        "prompt_visibility": "aggressive_opt_in",
        "model_shape_limitations": "Requires Transformer Engine-compatible modules and GEMM shapes; benchmark end-to-end speed and accuracy against BF16/FP16.",
        "source_url": FP8_URL,
    },
    "fp8_e4m3": {
        "description": (
            "FP8 E4M3 training format used by supported Transformer Engine FP8 recipes. "
            "It may be selected as E4M3 or as the forward component of the HYBRID recipe; "
            "it is not a generic framework dtype policy."
        ),
        "min_compute_capability": "8.9",
        "native_tensor_core_evidence": ADA_URL,
        "training_backend": "transformer_engine",
        "supported_training_backends": ["NVIDIA Transformer Engine"],
        "optimization_modes": ["aggressive"],
        "prompt_visibility": "aggressive_opt_in",
        "model_shape_limitations": "Requires Transformer Engine-compatible modules, supported FP8 tensor shapes, and scaling metadata.",
        "source_url": FP8_URL,
    },
    "fp8_e5m2": {
        "description": (
            "FP8 E5M2 is retained only as the backward-gradient component of Transformer "
            "Engine's documented HYBRID recipe. Pure E5M2 is not an independent MLEvolve training policy."
        ),
        "min_compute_capability": "8.9",
        "native_tensor_core_evidence": ADA_URL,
        "training_backend": "transformer_engine_hybrid_backward",
        "supported_training_backends": ["NVIDIA Transformer Engine HYBRID recipe"],
        "optimization_modes": ["aggressive"],
        "prompt_visibility": "hybrid_component_only",
        "model_shape_limitations": "Use only within the documented HYBRID recipe; never select pure E5M2 as the end-to-end training policy.",
        "source_url": FP8_URL,
    },
    "fp8_rowwise_scaling": {
        "description": (
            "Specialized FP8 row-wise scaling training recipe. It remains an aggressive-mode "
            "opt-in and must be validated on the selected backend, model, and tensor shapes."
        ),
        "min_compute_capability": "8.9",
        "native_tensor_core_evidence": ADA_URL,
        "training_backend": "validated_fp8_training_stack",
        "supported_training_backends": ["torchao float8 training"],
        "optimization_modes": ["aggressive"],
        "prompt_visibility": "aggressive_opt_in",
        "model_shape_limitations": "Linear-heavy workloads only; conversion, compilation, and row scaling overhead can outweigh Tensor Core speedup.",
    },
    "mxfp8": {
        "feature_id": "mxfp8",
        "name": "mxfp8",
        "category": "precision",
        "description": (
            "Blackwell-native MXFP8 block-scaled training through NVIDIA Transformer Engine. "
            "It is an aggressive-mode opt-in, not an automatic recommendation."
        ),
        "example_code": (
            "from transformer_engine.common.recipe import MXFP8BlockScaling\n"
            "recipe = MXFP8BlockScaling()\n"
            "with te.fp8_autocast(enabled=True, fp8_recipe=recipe):\n"
            "    loss = model(inputs).loss"
        ),
        "api_symbols": ["transformer_engine.pytorch.fp8_autocast", "MXFP8BlockScaling"],
        "min_compute_capability": "10.0",
        "native_tensor_core_evidence": MXFP8_URL,
        "training_backend": "transformer_engine",
        "supported_training_backends": ["NVIDIA Transformer Engine"],
        "optimization_modes": ["aggressive"],
        "prompt_visibility": "aggressive_opt_in",
        "model_shape_limitations": "Requires Blackwell, compatible Transformer Engine modules, and supported block-scaled tensor layouts and shapes.",
        "source_url": MXFP8_URL,
        "recommended_patterns": [
            "Benchmark an explicit MXFP8BlockScaling Transformer Engine recipe against BF16 before retaining it."
        ],
        "avoid_patterns": [
            "Do not infer MXFP8 training from a generic float8 dtype or from low-level instruction availability."
        ],
    },
    "nvfp4": {
        "feature_id": "nvfp4",
        "name": "nvfp4",
        "category": "precision",
        "description": (
            "Blackwell-native NVFP4 block-scaled training through NVIDIA Transformer Engine. "
            "NVFP4 is distinct from generic FP4 and is an aggressive-mode opt-in only."
        ),
        "example_code": (
            "from transformer_engine.common.recipe import NVFP4BlockScaling\n"
            "recipe = NVFP4BlockScaling()\n"
            "with te.fp8_autocast(enabled=True, fp8_recipe=recipe):\n"
            "    loss = model(inputs).loss"
        ),
        "api_symbols": ["transformer_engine.pytorch.fp8_autocast", "NVFP4BlockScaling"],
        "min_compute_capability": "10.0",
        "native_tensor_core_evidence": NVFP4_URL,
        "training_backend": "transformer_engine",
        "supported_training_backends": ["NVIDIA Transformer Engine"],
        "optimization_modes": ["aggressive"],
        "prompt_visibility": "aggressive_opt_in",
        "model_shape_limitations": "Requires a supported Blackwell target, Transformer Engine-compatible modules, and NVFP4 block-scaling shape constraints.",
        "source_url": NVFP4_URL,
        "recommended_patterns": [
            "Require the explicit NVFP4BlockScaling Transformer Engine recipe and compare accuracy and wall-clock speed with BF16."
        ],
        "avoid_patterns": [
            "Never treat the text FP4, a generic FP4 dtype, MXFP4, or an inference quantizer as proof of NVFP4 training."
        ],
    },
    "fp4": {
        "description": (
            "Generic FP4/MXFP4 capability marker for storage, inference, or specialized workflows. "
            "It is not NVFP4 and is hidden from datatype training-speed optimization."
        ),
        "min_compute_capability": "",
        "native_tensor_core_evidence": "No accepted generic FP4 end-to-end training path.",
        "training_backend": "none",
        "supported_training_backends": [],
        "optimization_modes": [],
        "prompt_visibility": "hidden",
        "model_shape_limitations": "Inference/storage capability only; generic FP4 cannot satisfy the NVFP4 training policy.",
    },
    "fp64": {
        "description": (
            "FP64 hardware capability for scientific and reference workloads. It remains "
            "queryable as a hardware fact but is hidden from datatype training-speed optimization."
        ),
        "min_compute_capability": "",
        "native_tensor_core_evidence": "Not an MLEvolve datatype speed-optimization path.",
        "training_backend": "framework_fp64",
        "supported_training_backends": ["PyTorch CUDA"],
        "optimization_modes": [],
        "prompt_visibility": "hidden",
        "model_shape_limitations": "Use only when the workload requires double-precision numerical behavior.",
    },
    "int8": {
        "description": (
            "INT8 Tensor Core hardware capability indicator. It is not a general training-speed "
            "recommendation; integer inputs, class labels, inference quantization, and explicit QAT "
            "are separate from the floating-point mixed-precision training policy."
        ),
        "min_compute_capability": "7.5",
        "native_tensor_core_evidence": "https://docs.nvidia.com/cuda/turing-tuning-guide/index.html",
        "training_backend": "none_for_general_training",
        "supported_training_backends": [],
        "optimization_modes": [],
        "prompt_visibility": "capability_indicator",
        "model_shape_limitations": "Not a general training recommendation; retain only as a hardware capability fact.",
    },
    "int4": {
        "feature_id": "int4",
        "name": "int4",
        "category": "precision",
        "description": (
            "INT4 Tensor Core hardware capability indicator. It is not a general training-speed "
            "recommendation and must not replace floating-point mixed-precision training."
        ),
        "api_symbols": [],
        "min_compute_capability": "7.5",
        "native_tensor_core_evidence": "https://docs.nvidia.com/cuda/turing-tuning-guide/index.html",
        "training_backend": "none_for_general_training",
        "supported_training_backends": [],
        "optimization_modes": [],
        "prompt_visibility": "capability_indicator",
        "model_shape_limitations": "Not a general training recommendation; retain only as a hardware capability fact.",
        "source_url": "https://docs.nvidia.com/cuda/turing-tuning-guide/index.html",
        "recommended_patterns": [],
        "avoid_patterns": ["Do not use INT4 as a generic neural-network training precision."],
    },
}

FP64_HARDWARE_IDS = {
    "hw:nvidia.ampere.a100_pcie_40gb.spec",
    "hw:nvidia.ampere.a100_pcie_80gb.spec",
    "hw:nvidia.ampere.a100_sxm4_40gb.spec",
    "hw:nvidia.ampere.a100_sxm4_80gb.spec",
    "hw:nvidia.ampere.a30.spec",
    "hw:nvidia.blackwell.b100.spec",
    "hw:nvidia.blackwell.b200.spec",
    "hw:nvidia.blackwell.gb200_nvl2.spec",
    "hw:nvidia.blackwell.gb200_nvl4.spec",
    "hw:nvidia.blackwell.gb200_nvl36.spec",
    "hw:nvidia.blackwell.gb200_nvl72.spec",
    "hw:nvidia.hopper.gh200_grace_hopper.spec",
    "hw:nvidia.hopper.h100_nvl_94gb.spec",
    "hw:nvidia.hopper.h100_pcie_80gb.spec",
    "hw:nvidia.hopper.h100_sxm5_80gb.spec",
    "hw:nvidia.hopper.h200_sxm_141gb.spec",
    "hw:nvidia.volta.tesla_v100_32gb.spec",
}


def _feature_node(feature_id: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"label": "Feature", "id": f"feat:{feature_id}", "properties": properties}


def _hardware_node(hardware_id: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"label": "Hardware", "id": f"hw:{hardware_id}", "properties": properties}


def _edge(source: str, feature_id: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "HAS_FEATURE",
        "from": source,
        "to": f"feat:{feature_id}",
        "properties": properties,
    }


def _architecture(properties: dict[str, Any]) -> str:
    values = properties.get("architectures") or [properties.get("architecture")]
    text = " ".join(str(value or "").lower() for value in values)
    if "blackwell" in text:
        return "blackwell"
    if "hopper" in text:
        return "hopper"
    if "ada" in text:
        return "ada_lovelace"
    if "ampere" in text:
        return "ampere"
    if "turing" in text:
        return "turing"
    if "volta" in text:
        return "volta"
    return "unknown"


def _precision_edge_properties(feature_id: str, architecture: str, capability: str) -> dict[str, Any]:
    feature = FEATURE_UPDATES[feature_id]
    baseline = feature_id in {"amp", "fp16"} or (
        architecture in {"ampere", "ada_lovelace", "hopper", "blackwell"}
        and feature_id in {"bf16", "tf32"}
    )
    integer = feature_id in {"int8", "int4"}
    hidden = feature_id in {"fp4", "fp64"}
    aggressive = feature_id in {"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4"}
    if integer:
        scope = "hardware_capability_indicator_not_general_training_recommendation"
    elif hidden:
        scope = "hardware_fact_hidden_from_datatype_speed_optimization"
    elif aggressive:
        scope = "aggressive_mode_opt_in_requires_end_to_end_validation"
    else:
        scope = "native_mixed_precision_training"
    return {
        "support_level": "supported",
        "min_compute_capability": feature.get("min_compute_capability", ""),
        "device_compute_capability": capability,
        "verified": True,
        "recommended": bool(baseline and not integer and not hidden),
        "recommendation_scope": scope,
        "prompt_visibility": feature.get("prompt_visibility"),
        "native_tensor_core_evidence": feature.get("native_tensor_core_evidence"),
        "training_backend": feature.get("training_backend"),
        "optimization_modes": list(feature.get("optimization_modes") or []),
        "limitations": feature.get("model_shape_limitations"),
    }


def _upsert_edge(edges: list[dict[str, Any]], source: str, feature_id: str, properties: dict[str, Any]) -> None:
    target = f"feat:{feature_id}"
    for edge in edges:
        if edge.get("from") == source and edge.get("to") == target and edge.get("type") == "HAS_FEATURE":
            edge["properties"] = properties
            return
    edges.append(_edge(source, feature_id, properties))


def _pre_ampere_hardware_nodes() -> list[dict[str, Any]]:
    shared_recommendations = [
        "Use FP16 through AMP/autocast with loss scaling and retain an FP32 fallback.",
        "Keep batch size and input dimensions configurable and benchmark Tensor Core-friendly shapes.",
    ]
    shared_avoid = [
        "Do not select FP8, MXFP8, NVFP4, generic FP4, or FP6 for training on this architecture.",
        "Do not treat integer Tensor Core capability as a general training recommendation.",
    ]
    return [
        _hardware_node(
            "nvidia.volta.tesla_v100_32gb.spec",
            {
                "hardware_id": "nvidia.volta.tesla_v100_32gb.spec",
                "name": "NVIDIA Tesla V100 32GB",
                "vendor": "nvidia",
                "aliases": ["v100", "tesla_v100", "nvidia_v100", "v100_32gb"],
                "architectures": ["volta"],
                "compute_capabilities": ["7.0"],
                "datatypes": ["fp16", "fp64"],
                "software_features": ["amp"],
                "recipes": [],
                "vram_MB": 32768,
                "vram_type": "hbm2",
                "sm_count": 80,
                "workload_types": ["vision_training", "transformer_training", "matmul_heavy"],
                "recommended_patterns": list(shared_recommendations),
                "avoid_patterns": list(shared_avoid),
                "memory_bandwidth_GBps": 900,
                "source_url": "https://www.nvidia.com/en-us/data-center/v100/",
                "experimental_recipes": [],
            },
        ),
        _hardware_node(
            "nvidia.turing.tesla_t4.spec",
            {
                "hardware_id": "nvidia.turing.tesla_t4.spec",
                "name": "NVIDIA T4",
                "vendor": "nvidia",
                "aliases": ["t4", "tesla_t4", "nvidia_t4"],
                "architectures": ["turing"],
                "compute_capabilities": ["7.5"],
                "datatypes": ["fp16", "int8", "int4"],
                "software_features": ["amp"],
                "recipes": [],
                "vram_MB": 16384,
                "vram_type": "gddr6",
                "sm_count": 40,
                "workload_types": ["vision_training", "transformer_training", "transformer_inference", "vision_inference"],
                "recommended_patterns": list(shared_recommendations),
                "avoid_patterns": list(shared_avoid),
                "memory_bandwidth_GBps": 320,
                "source_url": "https://www.nvidia.com/en-us/data-center/tesla-t4/",
                "experimental_recipes": [],
            },
        ),
    ]


def apply_policy(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    node_index = {node.get("id"): node for node in nodes}

    for feature_id, updates in FEATURE_UPDATES.items():
        node_id = f"feat:{feature_id}"
        node = node_index.get(node_id)
        if node is None:
            node = _feature_node(
                feature_id,
                {"feature_id": feature_id, "name": feature_id, "category": "precision"},
            )
            nodes.append(node)
            node_index[node_id] = node
        properties = node.setdefault("properties", {})
        properties.setdefault("feature_id", feature_id)
        properties.setdefault("name", feature_id)
        properties.setdefault("category", "precision")
        properties.update(updates)

    for feature_id, name in (("sm_70", "CUDA compute capability 7.0"), ("sm_75", "CUDA compute capability 7.5")):
        node_id = f"feat:{feature_id}"
        if node_id not in node_index:
            node = _feature_node(
                feature_id,
                {
                    "feature_id": feature_id,
                    "name": name,
                    "category": "compute_capability",
                    "description": f"{name} kernel target and device capability marker.",
                    "source_url": "https://developer.nvidia.com/cuda-gpus",
                },
            )
            nodes.append(node)
            node_index[node_id] = node

    for hardware_node in _pre_ampere_hardware_nodes():
        existing = node_index.get(hardware_node["id"])
        if existing is None:
            nodes.append(hardware_node)
            node_index[hardware_node["id"]] = hardware_node
        else:
            existing["properties"].update(hardware_node["properties"])

    hardware_nodes = [node for node in nodes if node.get("label") == "Hardware"]
    hardware_architectures: dict[str, tuple[str, str]] = {}
    for node in hardware_nodes:
        props = node.get("properties") or {}
        architecture = _architecture(props)
        capability = str((props.get("compute_capabilities") or [""])[0])
        hardware_architectures[node["id"]] = (architecture, capability)
        datatypes = [
            value
            for value in list(props.get("datatypes") or [])
            if str(value).strip().lower() != "fp6"
        ]
        if node["id"] in FP64_HARDWARE_IDS and "fp64" not in datatypes:
            datatypes.append("fp64")
        if architecture == "blackwell":
            for value in ("mxfp8", "nvfp4"):
                if value not in datatypes:
                    datatypes.append(value)
        props["datatypes"] = datatypes

        if architecture in {"ada_lovelace", "hopper", "blackwell"}:
            recommendation = (
                "Aggressive mode may use an explicit Transformer Engine FP8 recipe only after "
                "model, shape, package, speed, and accuracy validation; keep BF16/FP16 fallback."
            )
            if recommendation not in props.get("recommended_patterns", []):
                props.setdefault("recommended_patterns", []).append(recommendation)
        if architecture == "blackwell":
            recommendation = (
                "Aggressive mode may use explicit MXFP8BlockScaling or NVFP4BlockScaling "
                "Transformer Engine recipes; generic FP4 never satisfies the NVFP4 policy."
            )
            if recommendation not in props.get("recommended_patterns", []):
                props.setdefault("recommended_patterns", []).append(recommendation)

    precision_ids = set(FEATURE_UPDATES)
    edges = [edge for edge in edges if str(edge.get("to") or "").strip().lower() != "feat:fp6"]
    for edge in edges:
        source = str(edge.get("from") or "")
        feature_id = str(edge.get("to") or "").removeprefix("feat:")
        if source not in hardware_architectures or feature_id not in precision_ids:
            continue
        architecture, capability = hardware_architectures[source]
        edge["properties"] = _precision_edge_properties(feature_id, architecture, capability)

    for node in hardware_nodes:
        source = node["id"]
        architecture, capability = hardware_architectures[source]
        if architecture == "blackwell":
            for feature_id in ("mxfp8", "nvfp4"):
                _upsert_edge(
                    edges,
                    source,
                    feature_id,
                    _precision_edge_properties(feature_id, architecture, capability),
                )
        if source in FP64_HARDWARE_IDS:
            _upsert_edge(
                edges,
                source,
                "fp64",
                _precision_edge_properties("fp64", architecture, capability),
            )

    pre_ampere_edges = {
        "hw:nvidia.volta.tesla_v100_32gb.spec": ("volta", "7.0", ("amp", "fp16", "fp64", "tensor_cores", "sm_70")),
        "hw:nvidia.turing.tesla_t4.spec": ("turing", "7.5", ("amp", "fp16", "int8", "int4", "tensor_cores", "sm_75")),
    }
    for source, (architecture, capability, feature_ids) in pre_ampere_edges.items():
        for feature_id in feature_ids:
            props = (
                _precision_edge_properties(feature_id, architecture, capability)
                if feature_id in precision_ids
                else {
                    "support_level": "supported",
                    "device_compute_capability": capability,
                    "verified": True,
                    "recommended": True,
                }
            )
            _upsert_edge(edges, source, feature_id, props)

    nodes = [node for node in nodes if str(node.get("id") or "").strip().lower() != "feat:fp6"]
    serialized = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False).lower()
    if '"feature_id": "fp6"' in serialized or '"to": "feat:fp6"' in serialized:
        raise RuntimeError("FP6 must not exist in the structured hardware policy")
    return {"nodes": nodes, "edges": edges}


def main() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    migrated = apply_policy(graph)
    GRAPH_PATH.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
