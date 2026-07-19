from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents import result_parse_agent
from engine.executor import ExecutionResult
from engine.search_node import NodeOutcome, SearchNode


def _agent(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            workspace_dir=tmp_path,
            use_grading_server=False,
            exp_id="real-competition-id",
            exp_name="unrelated_name_parts",
        ),
        acfg=SimpleNamespace(
            feedback=SimpleNamespace(model="test-model", temp=0.0),
            use_global_memory=False,
            check_data_leakage=False,
        ),
        metric_maximize=False,
        metric_maximize_reasoning="log loss is minimized",
        global_memory=None,
        pipeline_logger=None,
        scfg=SimpleNamespace(repeated_failure_limit=2),
    )


def test_validation_uses_exp_id_and_quarantines_outage(tmp_path: Path, monkeypatch) -> None:
    node = SearchNode(code="", plan="draft", stage="draft")
    node._term_out = []
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / f"submission_{node.id}.csv").write_text("id,label\n1,0.5\n", encoding="utf-8")
    captured = {}

    def unavailable(**kwargs):
        captured.update(kwargs)
        return False, {"error": "validator offline"}

    monkeypatch.setattr(result_parse_agent, "_validate_submission_with_retry", unavailable)

    result_parse_agent._validate_format_with_retry(_agent(tmp_path), node)

    assert captured["exp_id"] == "real-competition-id"
    assert node.outcome == NodeOutcome.VALIDATION_UNAVAILABLE.value
    assert node.is_buggy is False
    assert node.is_terminal is True


def test_zero_metric_is_valid_and_llm_cannot_override_it(tmp_path: Path, monkeypatch) -> None:
    agent = _agent(tmp_path)
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {"is_bug": True, "summary": "summary", "metric": 99, "lower_is_better": False},
    )
    monkeypatch.setattr(
        result_parse_agent,
        "_validate_format_with_retry",
        lambda _agent, validated_node: validated_node.apply_outcome(NodeOutcome.VALID),
    )
    node = SearchNode(code="", plan="draft", stage="draft")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / f"submission_{node.id}.csv").write_text("id,label\n1,0.5\n", encoding="utf-8")

    parsed = result_parse_agent.run(
        agent,
        node,
        ExecutionResult(
            term_out=['MLEVOLVE_METRIC {"log_loss": 0.0}\n'],
            exec_time=1.0,
            exc_type=None,
            exc_info=None,
            exc_stack=[],
        ),
    )

    assert parsed.outcome == NodeOutcome.VALID.value
    assert parsed.metric.value == 0.0
    assert parsed.metric.maximize is False
    assert parsed.search_eligible is True


def test_repeated_failure_fingerprint_quarantines_lineage(tmp_path: Path, monkeypatch) -> None:
    agent = _agent(tmp_path)
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {"is_bug": True, "summary": "same failure", "metric": None, "lower_is_better": True},
    )
    execution = ExecutionResult(
        term_out=["RuntimeError: fallback loop never executed\n"],
        exec_time=1.0,
        exc_type="RuntimeError",
        exc_info={"message": "fallback loop never executed"},
        exc_stack=[],
    )
    parent = result_parse_agent.run(agent, SearchNode(code="", plan="draft", stage="draft"), execution)
    child = result_parse_agent.run(
        agent,
        SearchNode(code="", plan="debug", stage="debug", parent=parent),
        execution,
    )

    assert parent.outcome == NodeOutcome.CANDIDATE_EXCEPTION.value
    assert child.outcome == NodeOutcome.REPEATED_FAILURE.value
    assert child.is_terminal is True
    assert child.debug_eligible is False


def test_parser_builds_bf16_validation_bug_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {
            "is_bug": True,
            "summary": "Validation failed with TypeError: Got unsupported ScalarType BFloat16.",
            "metric": None,
            "lower_is_better": True,
        },
    )
    node = SearchNode(
        code="""
import torch
AMP_DTYPE = torch.bfloat16
def validate(preds):
    return preds.cpu().numpy().flatten()
""",
        plan="draft",
        stage="draft",
    )
    exec_result = ExecutionResult(
        term_out=[
            "Traceback (most recent call last):\n"
            "  File \"run.py\", line 12, in validate\n"
            "    all_preds.extend(preds.cpu().numpy().flatten())\n"
            "TypeError: Got unsupported ScalarType BFloat16\n"
        ],
        exec_time=1.0,
        exc_type="TypeError",
        exc_info={"message": "Got unsupported ScalarType BFloat16"},
        exc_stack=[],
    )

    parsed = result_parse_agent.run(_agent(tmp_path), node, exec_result)

    assert parsed.is_buggy is True
    assert "failure_category: bf16_numpy_conversion" in parsed.bug_report
    assert "missing_submission: True" in parsed.bug_report
    assert "missing_submission_role: consequence" in parsed.bug_report
    assert "unsupported ScalarType BFloat16" in parsed.bug_report
    assert "float32" in parsed.fix_report
    assert ".cpu().numpy()" in parsed.fix_report


def test_parser_builds_invalid_scheduler_argument_bug_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {
            "is_bug": True,
            "summary": "CosineAnnealingLR.__init__() got an unexpected keyword argument 'T_eta_min'.",
            "metric": None,
            "lower_is_better": True,
        },
    )
    node = SearchNode(
        code="""
from torch import optim
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_eta_min=1e-6, T_max=3)
""",
        plan="draft",
        stage="draft",
    )
    exec_result = ExecutionResult(
        term_out=[
            "TypeError: CosineAnnealingLR.__init__() got an unexpected keyword argument 'T_eta_min'\n"
        ],
        exec_time=1.0,
        exc_type="TypeError",
        exc_info={"message": "unexpected keyword argument"},
        exc_stack=[],
    )

    parsed = result_parse_agent.run(_agent(tmp_path), node, exec_result)

    assert parsed.is_buggy is True
    assert "failure_category: invalid_torch_scheduler_argument" in parsed.bug_report
    assert "T_eta_min" in parsed.bug_report
    assert "eta_min" in parsed.fix_report


def test_parser_builds_generic_low_precision_export_bug_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {
            "is_bug": True,
            "summary": "Validation failed with unsupported dtype during NumPy export.",
            "metric": None,
            "lower_is_better": True,
        },
    )
    node = SearchNode(
        code="""
import transformer_engine.pytorch as te
PRECISION = "nvfp4"
def validate(logits):
    return logits.cpu().numpy()
""",
        plan="draft",
        stage="draft",
    )
    exec_result = ExecutionResult(
        term_out=[
            "Traceback (most recent call last):\n"
            "  File \"run.py\", line 12, in validate\n"
            "    return logits.cpu().numpy()\n"
            "TypeError: Got unsupported ScalarType Float8_e4m3fn\n"
        ],
        exec_time=1.0,
        exc_type="TypeError",
        exc_info={"message": "Got unsupported ScalarType Float8_e4m3fn"},
        exc_stack=[],
    )

    parsed = result_parse_agent.run(_agent(tmp_path), node, exec_result)

    assert parsed.is_buggy is True
    assert "failure_category: low_precision_numpy_export" in parsed.bug_report
    assert "missing_submission_role: consequence" in parsed.bug_report
    assert "Float8" in parsed.bug_report
    assert "torch.float32" in parsed.fix_report


def test_parser_builds_submission_validation_bug_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        result_parse_agent,
        "query",
        lambda **_: {
            "is_bug": False,
            "summary": "Execution completed and validation metric was computed.",
            "metric": 0.42,
            "lower_is_better": True,
        },
    )

    def fake_validate(_agent, validated_node):
        validated_node.is_valid = False
        validated_node.is_buggy = True
        validated_node._term_out.append(
            "\nValidationError: submission columns were ['id', 'score'] but expected ['id', 'label']"
        )
        validated_node.analysis = (
            "FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\n"
            "Details: expected ['id', 'label'] columns."
        )

    monkeypatch.setattr(result_parse_agent, "_validate_format_with_retry", fake_validate)

    node = SearchNode(
        code="print('writes malformed submission')",
        plan="draft",
        stage="draft",
    )
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / f"submission_{node.id}.csv").write_text("id,score\n1,0.1\n", encoding="utf-8")
    exec_result = ExecutionResult(
        term_out=['MLEVOLVE_METRIC {"score": 0.42}\n'],
        exec_time=1.0,
        exc_type=None,
        exc_info=None,
        exc_stack=[],
    )

    parsed = result_parse_agent.run(_agent(tmp_path), node, exec_result)

    assert parsed.is_buggy is True
    assert "failure_category: submission_format_validation" in parsed.bug_report
    assert "missing_submission: False" in parsed.bug_report
    assert "submission format validation failed" in parsed.bug_report
    assert "submission columns" in parsed.bug_report
    assert "sample-submission columns" in parsed.fix_report
