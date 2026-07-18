from __future__ import annotations

import json
from types import SimpleNamespace

from engine.coldstart.knowledge import (
    build_guidance_description,
    collect_model_contracts,
    collect_startpoint_model_specs,
)


def test_coldstart_guidance_skips_torch_hub_templates_without_configured_path(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    model_path = tmp_path / "models.json"
    task_path.write_text(json.dumps({"vision-task": "General Image"}), encoding="utf-8")
    model_path.write_text(
        json.dumps(
            {
                "General Image": {
                    "LocalOnly": {
                        "Description": "requires local torch hub weights",
                        "Code_template": 'torch.hub.load("{TORCH_HUB_DIR}/repo", "model")',
                    },
                    "RemoteOk": {
                        "Description": "regular import path",
                        "Code_template": 'model = AutoModel.from_pretrained("safe/model")',
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        exp_id="vision-task",
        torch_hub_dir="",
        coldstart=SimpleNamespace(task_json_path=str(task_path), model_json_path=str(model_path)),
    )

    guidance = build_guidance_description(cfg)

    assert "LocalOnly" not in guidance
    assert "{TORCH_HUB_DIR}" not in guidance
    assert "RemoteOk" in guidance


def test_coldstart_guidance_substitutes_configured_torch_hub_path(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    model_path = tmp_path / "models.json"
    task_path.write_text(json.dumps({"vision-task": "General Image"}), encoding="utf-8")
    model_path.write_text(
        json.dumps(
            {
                "General Image": {
                    "LocalOnly": {
                        "Description": "requires local torch hub weights",
                        "Code_template": 'torch.hub.load("{TORCH_HUB_DIR}/repo", "model")',
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        exp_id="vision-task",
        torch_hub_dir="/models/torchhub/",
        coldstart=SimpleNamespace(task_json_path=str(task_path), model_json_path=str(model_path)),
    )

    guidance = build_guidance_description(cfg)

    assert "{TORCH_HUB_DIR}" not in guidance
    assert 'torch.hub.load("/models/torchhub/repo", "model")' in guidance


def test_collect_startpoint_model_specs_returns_ordered_probe_specs(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    model_path = tmp_path / "models.json"
    task_path.write_text(json.dumps({"vision-task": "General Image"}), encoding="utf-8")
    model_path.write_text(
        json.dumps(
            {
                "General Image": {
                    "LocalOnly": {
                        "Description": "requires local torch hub weights",
                        "Code_template": 'model = torch.hub.load("{TORCH_HUB_DIR}/repo", "dinov3_vitl16")',
                    },
                    "RemoteOk": {
                        "Description": "regular import path",
                        "Code_template": 'model = AutoModel.from_pretrained("safe/model")',
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        exp_id="vision-task",
        torch_hub_dir="/models/torchhub/",
        coldstart=SimpleNamespace(task_json_path=str(task_path), model_json_path=str(model_path)),
    )

    specs = collect_startpoint_model_specs(cfg)

    assert [spec["rank"] for spec in specs] == [1, 2]
    assert specs[0]["model_key"] == "dinov3_vitl16"
    assert specs[0]["modality"] == "vision"
    assert specs[0]["shape_hints"]["input_resolution"] == 256
    assert "{TORCH_HUB_DIR}" not in specs[0]["code_template"]
    assert specs[1]["model_key"] == "safe/model"


def test_coldstart_model_contract_drives_guidance_and_probe_shape_hints(tmp_path) -> None:
    task_path = tmp_path / "tasks.json"
    model_path = tmp_path / "models.json"
    contract = {
        "schema_version": 1,
        "library": "future-transformers",
        "library_version": "9.1.0",
        "model_id": "vendor/future-vision",
        "resolved_model_type": "future_vision",
        "loader": "AutoModel",
        "preprocessing": {
            "processor": "AutoProcessor",
            "fixed_image_size": 384,
        },
        "feature_apis": [
            {
                "modality": "image",
                "method": "encode_images",
                "call": "features = model.encode_images(pixel_values=pixel_values)",
                "return_kind": "tensor",
                "return_type": "torch.Tensor",
                "return_shape": "[batch, 768]",
                "dimension": 768,
                "dimension_config_path": "vision_config.hidden_size",
                "invalid_result_attributes": ["pooler_output"],
            }
        ],
    }
    task_path.write_text(json.dumps({"vision-task": "General Image"}), encoding="utf-8")
    model_path.write_text(
        json.dumps(
            {
                "General Image": {
                    "FutureVision": {
                        "Description": "future model family",
                        "Code_template": (
                            'model = AutoModel.from_pretrained("vendor/future-vision")\n'
                            "features = model.encode_images(pixel_values=pixel_values)"
                        ),
                        "Contract": contract,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        exp_id="vision-task",
        torch_hub_dir="",
        coldstart=SimpleNamespace(task_json_path=str(task_path), model_json_path=str(model_path)),
    )

    guidance = build_guidance_description(cfg)
    specs = collect_startpoint_model_specs(cfg)
    contracts = collect_model_contracts(cfg)

    assert "Model API contract (authoritative for this repository runtime)" in guidance
    assert "torch.Tensor" in guidance
    assert "Do not access `.pooler_output`" in guidance
    assert "Required fixed image size: `384x384`" in guidance
    assert specs[0]["shape_hints"]["input_resolution"] == 384
    assert specs[0]["shape_hints"]["feature_dimension"] == 768
    assert specs[0]["model_contract"] == contract
    assert contracts == [{**contract, "display_name": "FutureVision"}]
