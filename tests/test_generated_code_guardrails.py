from __future__ import annotations

from types import SimpleNamespace
import zipfile

from agents import code_review_agent
from agents.prompts.validation_template_prompts import get_code_review_prompt
from engine.executor import Interpreter
from engine.search_node import SearchNode
from utils.data_preview import generate


def test_data_preview_marks_zip_inputs_as_archives(tmp_path):
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True)
    with zipfile.ZipFile(input_dir / "train.zip", "w") as archive:
        archive.writestr("train/image_001.png", b"fake-image")
    (input_dir / "sample_submission.csv").write_text(
        "id,value\n1,0\n", encoding="utf-8"
    )

    preview = generate(workspace)

    assert "`./input/train.zip` is a zip archive, not a directory" in preview
    assert "Do not call `os.listdir('./input/train')`" in preview
    assert "extract it with Python `zipfile` into `./working/<name>/`" in preview
    assert "input/train.zip is a zip archive with 1 files" in preview
    assert "Top-level archive directories include: train" in preview


def test_data_preview_marks_flat_zip_contents(tmp_path):
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True)
    with zipfile.ZipFile(input_dir / "train.zip", "w") as archive:
        archive.writestr("cat.1.jpg", b"fake-image")
        archive.writestr("dog.1.jpg", b"fake-image")

    preview = generate(workspace)

    assert "input/train.zip is a zip archive with 2 files" in preview
    assert "2 file(s) are stored at the archive root" in preview
    assert "do not assume an extra nested split directory exists" in preview
    assert "flat or nested" in preview


def test_code_review_prompt_uses_data_preview_and_real_dependency_boundary():
    prompt = get_code_review_prompt(
        task_desc="Predict taxi fares from train.csv.",
        data_preview="`./input/train.csv` is a file, not a directory.",
        code="print('ok')",
    )
    guidelines = "\n".join(prompt["Instructions"]["Code review guidelines"])

    assert prompt["Data preview"] == "`./input/train.csv` is a file, not a directory."
    assert (
        "Task-specific children under `./input/` may be files, directories, or archives"
        in guidelines
    )
    assert "do not assume dynamic `pip install`" in guidelines.lower()
    assert "torch.cuda.get_device_capability()" in guidelines
    assert "XGBClassifier`/`XGBRegressor` construction" in guidelines
    assert "Low-precision export" in guidelines
    assert "tensor.detach().to(torch.float32).cpu().numpy()" in guidelines


def test_interpreter_rejects_invalid_syntax_before_subprocess(tmp_path):
    interpreter = Interpreter(tmp_path, max_parallel_run=1)

    result = interpreter.run("def broken(:\n    pass\n", id="bad")

    assert result.exc_type == "SyntaxError"
    assert result.exc_info["lineno"] == 1
    assert "failed Python syntax validation before execution" in "".join(
        result.term_out
    )
    assert not list(tmp_path.glob("runfile_*.py"))


def test_interpreter_rejects_empty_generated_script(tmp_path):
    interpreter = Interpreter(tmp_path, max_parallel_run=1)

    result = interpreter.run("", id="empty")

    assert result.exc_type == "ValueError"
    assert result.exc_info["message"] == "generated script is empty"
    assert "no Python code was produced" in "".join(result.term_out)


def test_interpreter_rejects_model_contract_violation_before_subprocess(tmp_path):
    task_path = tmp_path / "tasks.json"
    model_path = tmp_path / "models.json"
    task_path.write_text('{"vision-task": "General Image"}', encoding="utf-8")
    model_path.write_text(
        '''{
          "General Image": {
            "FutureVision": {
              "Code_template": "model = AutoModel.from_pretrained('vendor/future-vision')",
              "Contract": {
                "schema_version": 1,
                "model_id": "vendor/future-vision",
                "feature_apis": [{
                  "method": "encode_images",
                  "call": "features = model.encode_images(pixel_values=pixel_values)",
                  "return_kind": "tensor",
                  "invalid_result_attributes": ["pooler_output"]
                }]
              }
            }
          }
        }''',
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        exp_id="vision-task",
        torch_hub_dir="",
        coldstart=SimpleNamespace(
            use_coldstart=True,
            task_json_path=str(task_path),
            model_json_path=str(model_path),
        ),
        agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1)),
        scheduler=SimpleNamespace(enabled=False),
        start_cpu_id="0",
        cpu_number="1",
    )
    interpreter = Interpreter(tmp_path, max_parallel_run=1, cfg=cfg)
    code = '''
MODEL_ID = "vendor/future-vision"
model = AutoModel.from_pretrained(MODEL_ID)
features = model.encode_images(pixel_values=pixel_values)
pooled = features.pooler_output
'''

    result = interpreter.run(code, id="contract-bad")

    assert result.exc_type == "ModelAPIContractError"
    assert result.exc_info["issues"][0]["category"] == "model_feature_return_contract_violation"
    assert "failed the pretrained-model API contract" in "".join(result.term_out)
    assert not list(tmp_path.glob("runfile_*.py"))


def test_code_review_retries_when_critical_runtime_finding_is_approved(monkeypatch):
    calls = []

    monkeypatch.setattr(
        code_review_agent,
        "validate_generated_training_code",
        lambda code, stage, model_contracts=None: {
            "ok": False,
            "critical_count": 1,
            "warning_count": 0,
            "issues": [
                {
                    "severity": "critical",
                    "category": "invalid_torch_scheduler_argument",
                    "message": "CosineAnnealingLR received unsupported keyword argument(s): T_eta_min.",
                    "evidence": "CosineAnnealingLR(optimizer, T_eta_min=1e-6, T_max=3)",
                    "repair_hint": "Replace T_eta_min with eta_min for CosineAnnealingLR.",
                }
            ],
        },
    )
    monkeypatch.setattr(
        code_review_agent,
        "get_hardware_context_for_stage",
        lambda *args, **kwargs: SimpleNamespace(prompt_section=""),
    )
    monkeypatch.setattr(code_review_agent, "get_internet_clarification", lambda *_args, **_kwargs: [])

    def fake_query(system_message, user_message, func_spec, model, temperature, cfg):
        del user_message, func_spec, model, temperature, cfg
        calls.append(system_message)
        if len(calls) == 1:
            return {"needs_revision": False, "reasoning": "Looks fine.", "revised_code": None}
        return {
            "needs_revision": True,
            "reasoning": "Fixes the invalid scheduler argument.",
            "revised_code": "print('fixed')",
        }

    monkeypatch.setattr(code_review_agent, "query", fake_query)

    agent = SimpleNamespace(
        task_desc="Train a small PyTorch classifier.",
        data_preview="",
        cfg=SimpleNamespace(pretrain_model_dir=""),
        acfg=SimpleNamespace(use_diff_mode=False, code=SimpleNamespace(model="test", temp=0.0)),
    )
    node = SearchNode(code="print('buggy')", stage="draft")

    reviewed = code_review_agent.run(agent, node)

    assert reviewed == "print('fixed')"
    assert len(calls) == 2
    assert "Runtime Compatibility Findings" in calls[0]
    assert "Runtime compatibility retry 1" in calls[1]["Instructions"]


def test_code_review_returns_deterministically_repaired_precision_export(monkeypatch):
    monkeypatch.setattr(
        code_review_agent,
        "get_hardware_context_for_stage",
        lambda *args, **kwargs: SimpleNamespace(prompt_section=""),
    )
    monkeypatch.setattr(code_review_agent, "get_internet_clarification", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        code_review_agent,
        "query",
        lambda **_: {"needs_revision": False, "reasoning": "No additional issues.", "revised_code": None},
    )

    agent = SimpleNamespace(
        task_desc="Train a small PyTorch classifier.",
        data_preview="",
        cfg=SimpleNamespace(pretrain_model_dir=""),
        acfg=SimpleNamespace(use_diff_mode=False, code=SimpleNamespace(model="test", temp=0.0)),
    )
    node = SearchNode(
        code=(
            "import torch\n"
            "AMP_DTYPE = torch.bfloat16\n"
            "preds_np = preds.cpu().numpy()\n"
            "labels_np = labels.cpu().numpy()\n"
        ),
        stage="draft",
    )

    reviewed = code_review_agent.run(agent, node)

    assert "preds.detach().to(torch.float32).cpu().numpy()" in reviewed
    assert "labels.cpu().numpy()" in reviewed
