"""Build guidance description for agent from task/model JSON."""
import json
from pathlib import Path
from typing import Dict, List, Any
import re

from engine.script_introspection import detect_model_key

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_models_for_task(
    task_name: str, tasks: Dict, models: Dict
) -> List[Dict[str, str]]:
    """Match model list for task from knowledge by task name."""
    if task_name not in tasks:
        return []
    category = tasks[task_name]  # flat string: "General Image", "NLP", etc.
    if category not in models:
        return []
    matched = []
    for m_name, m_info in models[category].items():
        matched.append({
            "model_name": m_name,
            "description": m_info.get("Description", ""),
            "code_template": m_info.get("Code_template", ""),
            "contract": m_info.get("Contract", {}),
        })
    return matched


def _model_is_available(model: Dict[str, str], torch_hub_dir: str) -> bool:
    template = model.get("code_template", "")
    return "{TORCH_HUB_DIR}" not in template or bool(torch_hub_dir)


def _render_model_contract(contract: Dict[str, Any]) -> str:
    if not isinstance(contract, dict) or not contract:
        return ""

    model_id = str(contract.get("model_id") or "").strip()
    library = str(contract.get("library") or "").strip()
    library_version = str(contract.get("library_version") or "").strip()
    resolved_model_type = str(contract.get("resolved_model_type") or "").strip()
    loader = str(contract.get("loader") or "").strip()
    preprocessing = contract.get("preprocessing") if isinstance(contract.get("preprocessing"), dict) else {}
    feature_apis = contract.get("feature_apis") if isinstance(contract.get("feature_apis"), list) else []

    lines = ["Model API contract (authoritative for this repository runtime):"]
    if model_id:
        lines.append(f"- Exact model id: `{model_id}`")
    runtime_bits = [bit for bit in (library, library_version) if bit]
    if runtime_bits:
        runtime = " ".join(runtime_bits)
        if resolved_model_type:
            runtime += f"; checkpoint resolves to `{resolved_model_type}`"
        lines.append(f"- Runtime: {runtime}")
    if loader and model_id:
        lines.append(f"- Load with: `{loader}.from_pretrained(\"{model_id}\")`")

    processor = str(preprocessing.get("processor") or "").strip()
    image_size = preprocessing.get("fixed_image_size")
    image_mean = preprocessing.get("image_mean")
    image_std = preprocessing.get("image_std")
    if processor and model_id:
        lines.append(f"- Preferred preprocessing: `{processor}.from_pretrained(\"{model_id}\")`")
    if image_size is not None:
        lines.append(f"- Required fixed image size: `{image_size}x{image_size}`")
    if image_mean is not None and image_std is not None:
        lines.append(f"- Manual normalization fallback: mean={image_mean}, std={image_std}")

    for feature_api in feature_apis:
        if not isinstance(feature_api, dict):
            continue
        modality = str(feature_api.get("modality") or "feature").strip()
        call = str(feature_api.get("call") or "").strip()
        return_type = str(feature_api.get("return_type") or feature_api.get("return_kind") or "").strip()
        return_shape = feature_api.get("return_shape")
        if call:
            lines.append(f"- {modality.title()} feature call: `{call}`")
        return_details = return_type
        if return_shape:
            return_details += f" with shape `{return_shape}`"
        if return_details:
            lines.append(f"- Feature return: {return_details}; use this value directly.")
        invalid_attributes = [str(value) for value in feature_api.get("invalid_result_attributes", []) if value]
        if invalid_attributes:
            attrs = ", ".join(f"`.{value}`" for value in invalid_attributes)
            lines.append(f"- Do not access {attrs} on the feature-call result.")
        dimension_path = str(feature_api.get("dimension_config_path") or "").strip()
        if dimension_path:
            lines.append(f"- Feature dimension config path: `model.config.{dimension_path}`")

    invalid_config_paths = [str(value) for value in contract.get("invalid_config_paths", []) if value]
    if invalid_config_paths:
        invalid = ", ".join(f"`model.config.{value}`" for value in invalid_config_paths)
        lines.append(f"- Invalid config assumptions for this checkpoint: {invalid}")
    smoke_assertions = [str(value) for value in contract.get("smoke_assertions", []) if value]
    for assertion in smoke_assertions:
        lines.append(f"- Pre-training smoke assertion: `{assertion}`")
    return "\n".join(lines)


def _build_guidance_text(task_name: str, tasks: Dict, models: Dict, torch_hub_dir: str = "") -> str:
    """Build guidance text from task name and knowledge."""
    model_list = collect_models_for_task(task_name, tasks, models)
    model_list = [m for m in model_list if _model_is_available(m, torch_hub_dir)]
    if not model_list:
        return "None model"
    lines = []
    for i, m in enumerate(model_list):
        lines.append(f"\nModel{i+1}: {m['model_name']}\n")
        lines.append(f"Description:{m['description']}\n")
        contract_text = _render_model_contract(m.get("contract") or {})
        if contract_text:
            lines.append(contract_text + "\n")
        lines.append("Code template (MUST copy exactly — do NOT change model variant names or file paths):\n```python\n" + m["code_template"] + "\n```")
    return "\n".join(lines)


def _task_category(task_name: str, tasks: Dict) -> str | None:
    return tasks.get(task_name)


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:/-]+", "-", str(value or "").strip().lower())
    return re.sub(r"-{2,}", "-", text).strip("-") or "unknown"


def _modality_for_category(category: str | None) -> str:
    normalized = str(category or "").strip().lower()
    if any(token in normalized for token in ("image", "vision", "detection", "segmentation")):
        return "vision"
    if any(token in normalized for token in ("nlp", "text", "language")):
        return "text"
    if "audio" in normalized or "music" in normalized:
        return "audio"
    if "graph" in normalized:
        return "graph"
    if "tabular" in normalized:
        return "tabular"
    return "generic"


def _default_shape_hints(category: str | None) -> dict[str, Any]:
    modality = _modality_for_category(category)
    if modality == "vision":
        return {"modality": modality, "channels": 3, "input_resolution": 256}
    if modality == "text":
        return {"modality": modality, "sequence_length": 512}
    if modality == "audio":
        return {"modality": modality, "sample_rate": 24000, "duration_seconds": 10}
    return {"modality": modality}


def _shape_hints_from_contract(contract: Dict[str, Any], category: str | None) -> dict[str, Any]:
    shape_hints = _default_shape_hints(category)
    if not isinstance(contract, dict):
        return shape_hints
    preprocessing = contract.get("preprocessing")
    if isinstance(preprocessing, dict) and preprocessing.get("fixed_image_size") is not None:
        shape_hints["input_resolution"] = preprocessing["fixed_image_size"]
    feature_apis = contract.get("feature_apis")
    if isinstance(feature_apis, list):
        for feature_api in feature_apis:
            if not isinstance(feature_api, dict):
                continue
            if feature_api.get("dimension") is not None:
                shape_hints["feature_dimension"] = feature_api["dimension"]
                break
    return shape_hints


def collect_startpoint_model_specs(cfg: Any) -> List[Dict[str, Any]]:
    """Return ordered cold-start model specs suitable for scheduler probing."""
    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    task_id = str(getattr(cfg, "exp_id", "mlevolve"))
    category = _task_category(task_id, tasks)
    if category is None:
        return []
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    model_list = collect_models_for_task(task_id, tasks, models)
    model_list = [m for m in model_list if _model_is_available(m, torch_hub_dir)]
    specs: list[dict[str, Any]] = []
    for index, model in enumerate(model_list):
        code_template = str(model.get("code_template") or "")
        if torch_hub_dir:
            code_template = code_template.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))
        display_name = str(model.get("model_name") or f"Model{index + 1}")
        model_key = detect_model_key(code_template) or _slug(display_name)
        model_contract = model.get("contract") if isinstance(model.get("contract"), dict) else {}
        shape_hints = _shape_hints_from_contract(model_contract, category)
        specs.append(
            {
                "rank": index + 1,
                "task_id": task_id,
                "category": category,
                "modality": shape_hints.get("modality", "generic"),
                "model_key": model_key,
                "display_name": display_name,
                "description": model.get("description", ""),
                "code_template": code_template,
                "model_contract": model_contract,
                "shape_hints": shape_hints,
            }
        )
    return specs


def collect_model_contracts(cfg: Any) -> List[Dict[str, Any]]:
    """Return model API contracts selected by the cold-start task mapping."""
    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    task_id = str(getattr(cfg, "exp_id", "mlevolve"))
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    selected = collect_models_for_task(task_id, tasks, models)
    contracts: list[dict[str, Any]] = []
    for model in selected:
        if not _model_is_available(model, torch_hub_dir):
            continue
        contract = model.get("contract")
        if not isinstance(contract, dict) or not contract:
            continue
        normalized = dict(contract)
        normalized.setdefault("display_name", model.get("model_name", ""))
        contracts.append(normalized)
    return contracts


def get_init_solution_paths(exp_id: str) -> List[str]:
    """Load init solution paths for exp_id from engine/coldstart/init_solution_paths.json."""
    if not INIT_SOLUTION_JSON.exists():
        return []
    try:
        data = _load_json(str(INIT_SOLUTION_JSON))
        paths = data.get(exp_id)
        if isinstance(paths, list):
            return [str(p) for p in paths if p]
        return []
    except Exception:
        return []


def build_guidance_description(cfg: Any) -> str:

    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    text = _build_guidance_text(cfg.exp_id, tasks, models, torch_hub_dir=torch_hub_dir)
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))
    return text
