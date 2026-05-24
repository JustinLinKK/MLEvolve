from __future__ import annotations

import json
from types import SimpleNamespace

from engine.coldstart.knowledge import build_guidance_description


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
