import utils.data_preview as data_preview
from types import SimpleNamespace


def test_clean_task_desc_rejects_model_prompt_echo(tmp_path, monkeypatch):
    """A reasoning/prompt echo must never replace the actual task description."""
    import llm

    raw_description = "Predict Pawpularity. Evaluate with RMSE."
    echoed_prompt = (
        "We need answer user's request. Need clean task description.\n"
        "Task Description to Clean:\n"
        + raw_description
    )
    monkeypatch.setattr(llm, "query", lambda **_kwargs: echoed_prompt)
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=SimpleNamespace(model="qwen-test")),
        workspace_dir=str(tmp_path),
    )

    assert data_preview.clean_task_desc(raw_description, cfg) == raw_description


def test_clean_task_desc_can_be_disabled_for_controlled_runs(tmp_path, monkeypatch):
    """Controlled scheduler comparisons must use the supplied task text verbatim."""
    import llm

    calls = []
    monkeypatch.setattr(llm, "query", lambda **kwargs: calls.append(kwargs) or "different")
    raw_description = "Predict Pawpularity. Evaluate with RMSE."
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            code=SimpleNamespace(model="qwen-test"), clean_task_description=False
        ),
        workspace_dir=str(tmp_path),
    )

    assert data_preview.clean_task_desc(raw_description, cfg) == raw_description
    assert calls == []


def test_original_mlevolve_profile_restores_frozen_submission_and_alignment_text(
    tmp_path, monkeypatch
):
    """The preflight ablation must preserve the frozen baseline task-description core."""
    import llm

    monkeypatch.setattr(llm, "query", lambda **_kwargs: "CLEANED TASK")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample_submission.csv").write_text(
        "Id,Pawpularity\nabc,42.0\n", encoding="utf-8"
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            code=SimpleNamespace(model="qwen-test"),
            clean_task_description=True,
            prompt_compatibility_profile="original_mlevolve_20260924",
        ),
        workspace_dir=str(tmp_path),
    )

    cleaned = data_preview.clean_task_desc("RAW TASK", cfg)

    assert "CLEANED TASK" in cleaned
    assert "including column names and row order" in cleaned
    assert "TASK AND METRIC ALIGNMENT REQUIREMENT" in cleaned
    assert "lazy task redefinition is forbidden" in cleaned


def test_missing_transient_log_is_reported_unavailable(tmp_path):
    """A deleted candidate log must not abort the advisory data preview."""
    missing_log = tmp_path / "download.log"

    assert data_preview.get_file_len_size(missing_log) == (0, "unavailable")


def test_preview_skips_log_deleted_after_directory_walk(tmp_path, monkeypatch):
    transient_log = tmp_path / "download.log"
    transient_log.write_text("temporary\n")
    monkeypatch.setattr(data_preview, "file_tree", lambda _path: "")

    def delete_after_measurement(path):
        path.unlink()
        return 1, "1 lines"

    monkeypatch.setattr(data_preview, "get_file_len_size", delete_after_measurement)

    assert data_preview.generate(tmp_path) == "```\n```"
