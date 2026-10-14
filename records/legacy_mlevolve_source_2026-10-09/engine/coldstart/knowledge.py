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
        })
    return matched


def _model_is_available(model: Dict[str, str], torch_hub_dir: str) -> bool:
    template = model.get("code_template", "")
    return "{TORCH_HUB_DIR}" not in template or bool(torch_hub_dir)


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
        shape_hints = _default_shape_hints(category)
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
                "shape_hints": shape_hints,
            }
        )
    return specs


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
