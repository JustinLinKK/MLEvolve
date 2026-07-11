from __future__ import annotations

import zipfile

from agents.prompts.validation_template_prompts import get_code_review_prompt
from engine.executor import Interpreter
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
