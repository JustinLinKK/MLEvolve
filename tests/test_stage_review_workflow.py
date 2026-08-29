from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from agents import code_review_agent
from agents import debug_agent
from agents import result_parse_agent
from agents.review_contracts import ReviewDecision, ReviewIssue, StageRepairResult
from agents.stage_repair import group_repair_issues, repair_selected_stages
from engine.agent_search import AgentSearch
from engine import evaluation
from engine.execution import validate_executed_node
from engine.executor import ExecutionResult
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue
from utils.experiment_metrics import build_comparison_metrics
from utils.serialize import dumps_json, loads_json


def _agent(*, mode: str = "hardware_aware", parallel: bool = True):
    review = SimpleNamespace(
        enabled=True,
        max_repair_rounds=2,
        classifier_retries=1,
        repair_retries=1,
        reject_unresolved_critical=True,
        fail_open_on_unavailable=True,
        parallel_training_repairs=parallel,
    )
    return SimpleNamespace(
        task_desc="Train a model and maximize accuracy.",
        cfg=SimpleNamespace(
            experiment=SimpleNamespace(mode=mode),
            pretrain_model_dir="",
        ),
        acfg=SimpleNamespace(
            code=SimpleNamespace(model="fake", temp=0),
            review=review,
            hardware_context_enabled=True,
            precision_optimization_mode="normal",
        ),
        scheduler_client=None,
        pipeline_logger=None,
    )


def _node(code: str = "model = 0\ndtype = 0\ntrain = 0\n") -> SearchNode:
    return SearchNode(code=code, plan="test", stage="draft")


def _issue(owner: str, category: str | None = None) -> ReviewIssue:
    return ReviewIssue(
        source="static_review",
        severity="critical",
        category=category or f"{owner}_bug",
        owner=owner,
        evidence=f"{owner} is wrong",
        repair_instruction=f"repair {owner}",
    )


def _stage_from_prompt(prompt: str) -> str:
    for stage in ("model_design", "datatype_precision", "training_evaluation", "integration"):
        if f'"role": "{stage} repair specialist"' in prompt:
            return stage
    raise AssertionError("repair stage missing from prompt")


def _patch_for(stage: str) -> str:
    variable = {
        "model_design": "model",
        "datatype_precision": "dtype",
        "training_evaluation": "train",
        "integration": "integration",
    }[stage]
    return f"<<<<<<< SEARCH\n{variable} = 0\n=======\n{variable} = 1\n>>>>>>> REPLACE"


@pytest.mark.parametrize(
    ("owner", "expected_stage"),
    [
        ("datatype_precision", "datatype_precision"),
        ("training_evaluation", "training_evaluation"),
    ],
)
def test_single_issue_calls_only_owning_stage(owner: str, expected_stage: str) -> None:
    calls: list[str] = []

    def generator(_agent, prompt: str) -> str:
        stage = _stage_from_prompt(prompt)
        calls.append(stage)
        return _patch_for(stage)

    code, results, stats = repair_selected_stages(
        _agent(), _node(), _node().code, [_issue(owner)], generator=generator
    )

    assert calls == [expected_stage]
    assert [result.stage for result in results] == [expected_stage]
    assert stats["stage_calls_skipped"] == 2
    assert f"{'dtype' if owner == 'datatype_precision' else 'train'} = 1" in code


def test_baseline_routes_precision_issue_to_training_stage() -> None:
    grouped = group_repair_issues(_agent(mode="baseline"), [_issue("datatype_precision")])
    assert list(grouped) == ["training_evaluation"]
    assert grouped["training_evaluation"][0].owner == "training_evaluation"


def test_optimizer_category_is_always_owned_by_training_stage() -> None:
    issue = ReviewIssue.from_mapping(
        {
            "source": "review",
            "severity": "critical",
            "category": "optimizer_configuration",
            "owner": "model_design",
            "evidence": "optimizer is invalid",
            "repair_instruction": "fix optimizer",
        }
    )
    assert issue.owner == "training_evaluation"


def test_stage_one_and_two_are_sequential() -> None:
    calls: list[str] = []

    def generator(_agent, prompt: str) -> str:
        stage = _stage_from_prompt(prompt)
        calls.append(stage)
        if stage == "datatype_precision":
            assert 'model = 1\\n' in prompt
        return _patch_for(stage)

    code, _, stats = repair_selected_stages(
        _agent(),
        _node(),
        _node().code,
        [_issue("model_design"), _issue("datatype_precision")],
        generator=generator,
    )

    assert calls == ["model_design", "datatype_precision"]
    assert stats["parallel_batches"] == 0
    assert "model = 1" in code and "dtype = 1" in code


@pytest.mark.parametrize("primary", ["model_design", "datatype_precision"])
def test_training_stage_runs_in_parallel_with_one_or_two(primary: str) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def generator(_agent, prompt: str) -> str:
        nonlocal active, max_active
        stage = _stage_from_prompt(prompt)
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return _patch_for(stage)

    code, _, stats = repair_selected_stages(
        _agent(),
        _node(),
        _node().code,
        [_issue(primary), _issue("training_evaluation")],
        generator=generator,
    )

    assert max_active == 2
    assert stats["parallel_batches"] == 1
    assert "train = 1" in code


def test_all_three_runs_one_and_three_then_two_on_merged_code() -> None:
    calls: list[str] = []

    def generator(_agent, prompt: str) -> str:
        stage = _stage_from_prompt(prompt)
        calls.append(stage)
        if stage == "datatype_precision":
            assert 'model = 1\\n' in prompt
            assert 'train = 1\\n' in prompt
        return _patch_for(stage)

    code, _, stats = repair_selected_stages(
        _agent(),
        _node(),
        _node().code,
        [_issue("model_design"), _issue("datatype_precision"), _issue("training_evaluation")],
        generator=generator,
    )

    assert set(calls[:2]) == {"model_design", "training_evaluation"}
    assert calls[2] == "datatype_precision"
    assert stats["parallel_batches"] == 1
    assert code == "model = 1\ndtype = 1\ntrain = 1\n"


def test_integration_repairs_run_last() -> None:
    calls: list[str] = []
    original = "model = 0\ndtype = 0\ntrain = 0\nintegration = 0\n"

    def generator(_agent, prompt: str) -> str:
        stage = _stage_from_prompt(prompt)
        calls.append(stage)
        return _patch_for(stage)

    code, _, _ = repair_selected_stages(
        _agent(),
        _node(original),
        original,
        [_issue("model_design"), _issue("training_evaluation"), _issue("integration")],
        generator=generator,
    )
    assert calls[-1] == "integration"
    assert "integration = 1" in code


def test_overlapping_parallel_patches_regenerate_training_sequentially() -> None:
    calls: list[str] = []

    def generator(_agent, prompt: str) -> str:
        stage = _stage_from_prompt(prompt)
        calls.append(stage)
        if stage == "model_design":
            return "<<<<<<< SEARCH\nshared = 0\n=======\nshared = 1\n>>>>>>> REPLACE"
        search = "shared = 1" if 'shared = 1\\n' in prompt else "shared = 0"
        return f"<<<<<<< SEARCH\n{search}\n=======\nshared = 2\n>>>>>>> REPLACE"

    code, results, stats = repair_selected_stages(
        _agent(),
        _node("shared = 0\n"),
        "shared = 0\n",
        [_issue("model_design"), _issue("training_evaluation")],
        generator=generator,
    )

    assert calls.count("training_evaluation") == 2
    assert stats["patch_conflicts"] == 1
    assert stats["stage_repair_calls"] == 3
    assert results[-1].sequential_retry is True
    assert code == "shared = 2\n"


@pytest.mark.parametrize(
    "patch",
    [
        "not a patch",
        (
            "<<<<<<< SEARCH\nmodel = 0\n=======\nmodel = 1\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nkeep = 'byte-for-byte'\n=======\nkeep = 'changed'"
        ),
        "<<<<<<< SEARCH\nmodel = 0\n=======\nmodel = 0\n>>>>>>> REPLACE",
        "<<<<<<< SEARCH\nmodel = 0\n=======\nmodel = (\n>>>>>>> REPLACE",
    ],
)
def test_malformed_unchanged_and_invalid_syntax_patches_roll_back(patch: str) -> None:
    original = "model = 0\nkeep = 'byte-for-byte'\n"
    code, results, _ = repair_selected_stages(
        _agent(),
        _node(original),
        original,
        [_issue("model_design")],
        generator=lambda _agent, _prompt: patch,
    )
    assert code == original
    assert results[0].applied is False
    assert results[0].failure_reason


def test_non_overlapping_patch_preserves_unaffected_bytes() -> None:
    original = "model = 0\nkeep = '  spaces  '  # unchanged\ntrain = 0\n"
    code, _, _ = repair_selected_stages(
        _agent(), _node(original), original, [_issue("model_design")],
        generator=lambda _agent, _prompt: _patch_for("model_design"),
    )
    assert code == "model = 1\nkeep = '  spaces  '  # unchanged\ntrain = 0\n"


def _decision(*issues: ReviewIssue) -> ReviewDecision:
    return ReviewDecision(
        approved=not any(issue.severity == "critical" for issue in issues),
        reasoning="Classified the complete script and assigned each issue to its owning stage.",
        issues=tuple(issues),
    )


def test_review_approval_without_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_review_agent, "classify_code", lambda *_args: (_decision(), {"latency_seconds": 0.1}))
    node = _node()
    outcome = code_review_agent.review_and_repair(_agent(), node)
    assert outcome.status == "approved"
    assert outcome.code == node.code
    assert len(outcome.history) == 1


def test_warning_is_recorded_without_triggering_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = ReviewIssue(
        source="static_review",
        severity="warning",
        category="maintainability",
        owner="model_design",
        evidence="A name could be clearer.",
        repair_instruction="Rename it in a future improvement.",
    )
    monkeypatch.setattr(
        code_review_agent,
        "classify_code",
        lambda *_args: (_decision(warning), {"latency_seconds": 0.1}),
    )
    monkeypatch.setattr(
        code_review_agent,
        "repair_selected_stages",
        lambda *_args, **_kwargs: pytest.fail("warnings must not trigger repair"),
    )
    node = _node()
    outcome = code_review_agent.review_and_repair(_agent(), node)
    assert outcome.status == "approved"
    assert node.review_issues == [warning.to_dict()]


def test_review_repairs_and_rereviews_complete_script(monkeypatch: pytest.MonkeyPatch) -> None:
    decisions = iter([_decision(_issue("datatype_precision")), _decision()])
    seen_codes: list[str] = []

    def classify(_agent, _node, code):
        seen_codes.append(code)
        return next(decisions), {"latency_seconds": 0.1}

    monkeypatch.setattr(code_review_agent, "classify_code", classify)
    monkeypatch.setattr(
        code_review_agent,
        "repair_selected_stages",
        lambda *_args, **_kwargs: (
            "model = 0\ndtype = 1\ntrain = 0\n",
            [StageRepairResult(stage="datatype_precision", applied=True, patch_count=1)],
            {"stage_calls_skipped": 2, "stage_repair_calls": 1, "parallel_batches": 0, "patch_conflicts": 0},
        ),
    )
    outcome = code_review_agent.review_and_repair(_agent(), _node())
    assert outcome.status == "repaired"
    assert seen_codes[-1] == outcome.code
    assert len([entry for entry in outcome.history if entry["event"] == "review_decision"]) == 2


def test_second_round_then_unresolved_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    critical = _decision(_issue("training_evaluation"))
    monkeypatch.setattr(code_review_agent, "classify_code", lambda *_args: (critical, {"latency_seconds": 0.1}))
    repair_calls = 0

    def repair(_agent, _node, code, _issues):
        nonlocal repair_calls
        repair_calls += 1
        return code + f"# repair {repair_calls}\n", [StageRepairResult(stage="training_evaluation", applied=True)], {
            "stage_calls_skipped": 2,
            "stage_repair_calls": 1,
            "parallel_batches": 0,
            "patch_conflicts": 0,
        }

    monkeypatch.setattr(code_review_agent, "repair_selected_stages", repair)
    node = _node()
    outcome = code_review_agent.review_and_repair(_agent(), node)
    assert repair_calls == 2
    assert outcome.status == "rejected"
    assert node.review_status == "rejected"


def test_second_repair_round_can_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    critical = _decision(_issue("training_evaluation"))
    decisions = iter([critical, critical, _decision()])
    monkeypatch.setattr(
        code_review_agent,
        "classify_code",
        lambda *_args: (next(decisions), {"latency_seconds": 0.1}),
    )
    repair_calls = 0

    def repair(_agent, _node, code, _issues):
        nonlocal repair_calls
        repair_calls += 1
        return code + f"# repair {repair_calls}\n", [StageRepairResult(stage="training_evaluation", applied=True)], {
            "stage_calls_skipped": 2,
            "stage_repair_calls": 1,
            "parallel_batches": 0,
            "patch_conflicts": 0,
        }

    monkeypatch.setattr(code_review_agent, "repair_selected_stages", repair)
    outcome = code_review_agent.review_and_repair(_agent(), _node())
    assert repair_calls == 2
    assert outcome.status == "repaired"


def test_reviewer_unavailable_fails_open_only_without_known_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        code_review_agent,
        "classify_code",
        lambda *_args: (None, {"latency_seconds": 0.2, "error": "provider unavailable"}),
    )
    outcome = code_review_agent.review_and_repair(_agent(), _node())
    assert outcome.status == "unavailable_fail_open"


def test_review_disabled_cannot_bypass_deterministic_precision_policy() -> None:
    agent = _agent()
    agent.acfg.review.enabled = False
    node = _node('PRECISION = "fp4"\n')

    outcome = code_review_agent.review_and_repair(agent, node)

    assert outcome.status == "rejected"
    assert outcome.unresolved_issues[0].severity == "critical"
    assert outcome.unresolved_issues[0].category == "datatype_precision"
    assert outcome.unresolved_issues[0].owner == "datatype_precision"


def test_pre_execution_guard_rejects_precision_violation_after_review_bypass() -> None:
    search = AgentSearch.__new__(AgentSearch)
    base = _agent()
    search.acfg = base.acfg
    search.cfg = base.cfg
    search.scheduler_client = None
    search.task_desc = base.task_desc
    node = _node('PRECISION = "fp4"\n')
    node.review_status = "unavailable_fail_open"

    allowed = search._validate_node_precision_before_execution(node)

    assert allowed is False
    assert node.review_status == "rejected"
    assert any(
        issue["category"] == "datatype_precision" and issue["severity"] == "critical"
        for issue in node.review_issues
    )
    assert node.review_history[-1]["event"] == "pre_execution_precision_policy_rejected"


@pytest.mark.parametrize("path", ["deferred", "scheduler_batch"])
def test_deferred_execution_rechecks_precision_immediately_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(
        code='PRECISION = "fp4"\n',
        plan="draft",
        parent=root,
        stage="draft",
        local_best_node=root,
        review_status="approved",
    )
    node.pending_execution = True
    base = _agent()
    search = AgentSearch.__new__(AgentSearch)
    search.acfg = base.acfg
    search.cfg = base.cfg
    search.scheduler_client = None
    search.task_desc = base.task_desc
    search.journal = Journal(nodes=[root])
    search.journal_lock = threading.Lock()
    search.pipeline_logger = None
    search.current_step = 0
    monkeypatch.setattr(evaluation, "check_improvement", lambda *_args, **_kwargs: False)

    calls = 0

    def forbidden_executor(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("precision-rejected code must not be dispatched")

    if path == "deferred":
        search.execute_deferred_node(node, forbidden_executor)
    else:
        search.execute_deferred_nodes([node], forbidden_executor)

    assert calls == 0
    assert node.review_status == "rejected"
    assert node.pending_execution is False


def test_reviewer_unavailable_after_known_critical_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    decisions = iter([
        (_decision(_issue("model_design")), {"latency_seconds": 0.1}),
        (None, {"latency_seconds": 0.1, "error": "provider unavailable"}),
    ])
    monkeypatch.setattr(code_review_agent, "classify_code", lambda *_args: next(decisions))
    monkeypatch.setattr(
        code_review_agent,
        "repair_selected_stages",
        lambda _agent, _node, code, _issues: (
            code + "# attempted repair\n",
            [StageRepairResult(stage="model_design", applied=True)],
            {"stage_calls_skipped": 2, "stage_repair_calls": 1, "parallel_batches": 0, "patch_conflicts": 0},
        ),
    )
    outcome = code_review_agent.review_and_repair(_agent(), _node())
    assert outcome.status == "rejected"
    assert len(outcome.unresolved_issues) == 1


@pytest.mark.parametrize("path", ["immediate", "deferred", "scheduler_batch"])
def test_rejected_nodes_never_invoke_executor(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(
        code="bad = True\n",
        plan="draft",
        parent=root,
        stage="draft",
        local_best_node=root,
        review_status="rejected",
        review_issues=[_issue("model_design").to_dict()],
    )
    node.pending_execution = True
    search = AgentSearch.__new__(AgentSearch)
    search.journal = Journal(nodes=[root])
    search.journal_lock = threading.Lock()
    search.pipeline_logger = None
    search.current_step = 0
    monkeypatch.setattr(evaluation, "check_improvement", lambda *_args, **_kwargs: False)

    calls = 0

    def forbidden_executor(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not be called for a rejected node")

    if path == "immediate":
        search._finalize_review_rejected_node(node)
    elif path == "deferred":
        search.execute_deferred_node(node, forbidden_executor)
    else:
        returned = search.execute_deferred_nodes([node], forbidden_executor)
        assert returned == [node]

    assert calls == 0
    assert node in search.journal.nodes
    assert node.is_buggy is True
    assert node.metric.is_worst
    assert node.pending_execution is False


def test_deterministic_runtime_validators_append_stage_owned_issues(tmp_path) -> None:
    agent = SimpleNamespace(
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        branch_successful_nodes={},
    )
    node = _node("train = 0\n")
    node.id = "missing"
    node.is_buggy = False
    node.is_valid = True
    node.metric = MetricValue(0.4, maximize=True)
    validate_executed_node(agent, node)
    assert node.is_buggy is True
    assert any(
        issue["category"] == "missing_submission" and issue["owner"] == "training_evaluation"
        for issue in node.review_issues
    )

    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    zero = _node("train = 0\n")
    zero.id = "zero"
    (submission_dir / "submission_zero.csv").write_text("id,pred\n1,0\n", encoding="utf-8")
    zero.is_buggy = False
    zero.is_valid = True
    zero.metric = MetricValue(0.0, maximize=True)
    validate_executed_node(agent, zero)
    assert any(issue["category"] == "zero_metric" for issue in zero.review_issues)


def test_metric_direction_and_leakage_validators_use_expected_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SimpleNamespace(
        metric_maximize=True,
        metric_maximize_reasoning="accuracy is maximized",
        acfg=SimpleNamespace(check_data_leakage=True),
    )
    metric_node = _node()
    metric_node.analysis = "parsed"
    result_parse_agent._validate_metric_direction(
        agent,
        metric_node,
        {"lower_is_better": True, "metric": 0.9},
    )
    assert any(
        issue["category"] == "metric_direction" and issue["owner"] == "training_evaluation"
        for issue in metric_node.review_issues
    )

    leakage_node = _node()
    leakage_node.metric = MetricValue(0.999, maximize=True)
    leakage_node.analysis = "suspicious score"
    monkeypatch.setattr(result_parse_agent, "should_check_data_leakage", lambda *_args: True)
    monkeypatch.setattr(
        result_parse_agent.data_leakage_agent,
        "run",
        lambda *_args: {"has_leakage": True, "confidence": "high", "reason": "scaler fit before split"},
    )
    result_parse_agent._check_data_leakage(agent, leakage_node, {"metric": 0.999})
    assert any(
        issue["category"] == "data_leakage" and issue["owner"] == "model_design"
        for issue in leakage_node.review_issues
    )


def test_debug_repairs_only_classified_issue_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _node("model = 0\n")
    parent.is_buggy = True
    parent.is_valid = False
    parent.local_best_node = parent
    parent.branch_id = 1
    parent.analysis = "model construction failed"
    parent._term_out = ["model construction failed"]
    parent.review_issues = [_issue("model_design").to_dict()]
    agent = _agent()
    agent.data_preview = "preview"
    agent.global_memory = None
    agent.scfg = SimpleNamespace(num_bugs=1, max_debug_depth=3)
    agent.acfg.use_diff_mode = True

    hardware_ctx = SimpleNamespace(prompt_section="")
    monkeypatch.setattr(debug_agent, "get_hardware_context_for_stage", lambda *_args, **_kwargs: hardware_ctx)
    monkeypatch.setattr(debug_agent, "build_pipeline_decision", lambda *_args, **_kwargs: {"decision": "keep"})
    monkeypatch.setattr(debug_agent, "apply_hardware_context_to_node", lambda *_args: None)
    monkeypatch.setattr(debug_agent, "apply_pipeline_decision_to_node", lambda *_args: None)
    monkeypatch.setattr(debug_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        debug_agent,
        "repair_selected_stages",
        lambda *_args, **_kwargs: (
            "model = 1\n",
            [StageRepairResult(stage="model_design", applied=True, patch_count=1)],
            {"selected_stages": ["model_design"], "stage_calls_skipped": 2},
        ),
    )
    child = debug_agent.run(agent, parent)
    assert child is not None
    assert child.code == "model = 1\n"
    assert child.stage == "debug"
    assert "model_design" in child.fix_report


def test_review_fields_round_trip_and_metrics() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(
        code="x = 1\n",
        plan="draft",
        parent=root,
        stage="draft",
        review_status="rejected",
        review_issues=[_issue("model_design").to_dict()],
        review_history=[
            {"event": "review_decision", "latency_seconds": 0.5},
            {
                "event": "repair_round",
                "stage_repair_calls": 2,
                "stage_calls_skipped": 1,
                "parallel_batches": 1,
                "patch_conflicts": 1,
                "repairs": [],
            },
        ],
    )
    node.is_buggy = True
    node.is_valid = False
    node.exec_time = 0.0
    node.metric = MetricValue(0.5, maximize=True)
    node.local_best_node = node
    journal = Journal(nodes=[root, node])
    restored_journal = loads_json(dumps_json(journal), Journal)
    restored = restored_journal.nodes[1]
    assert restored.review_status == "rejected"
    assert restored.review_issues == node.review_issues
    assert restored.review_history == node.review_history
    assert restored.parent is restored_journal.nodes[0]
    assert restored.local_best_node is restored
    assert restored.reached_child_limit(SimpleNamespace(num_drafts=2, num_bugs=2)) is False

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="hardware_aware"), exp_name="run", exp_id="task"
    )
    metrics = build_comparison_metrics(cfg, journal, started_at=1.0, finished_at=2.0)
    assert metrics["review_round_count"] == 1
    assert metrics["stage_repair_call_count"] == 2
    assert metrics["review_stage_calls_skipped_count"] == 1
    assert metrics["parallel_repair_batch_count"] == 1
    assert metrics["review_patch_conflict_count"] == 1
    assert metrics["review_rejection_count"] == 1
    assert metrics["gpu_executions_avoided"] == 1


def test_result_parser_keeps_completed_metric_when_llm_parser_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    """A feedback outage must not discard a scheduler-completed training result."""
    node = _node()
    workspace = tmp_path / "workspace"
    submission_dir = workspace / "submission"
    submission_dir.mkdir(parents=True)
    (submission_dir / f"submission_{node.id}.csv").write_text("Id,Pawpularity\n0,50\n")
    agent = SimpleNamespace(
        cfg=SimpleNamespace(
            workspace_dir=workspace,
            exp_name="20260828_154842_petfinder",
            experiment=SimpleNamespace(mode="hardware_aware"),
        ),
        acfg=SimpleNamespace(
            feedback=SimpleNamespace(model="fake", temp=0),
            use_global_memory=False,
            check_data_leakage=False,
        ),
        metric_maximize=False,
        metric_maximize_reasoning="Petfinder uses root mean squared error.",
        global_memory=None,
    )
    result = ExecutionResult(
        term_out=["MLEVOLVE_EPOCH_METRIC {\"epoch\": 1, \"metric\": 20.9}\nFinal Validation Score: 20.667\n"],
        exec_time=12.0,
        exc_type=None,
    )
    monkeypatch.setattr(result_parse_agent, "query", lambda **_: (_ for _ in ()).throw(RuntimeError("parser offline")))
    monkeypatch.setattr(result_parse_agent, "_validate_format_with_retry", lambda *_: None)

    parsed = result_parse_agent.run(agent, node, result)

    assert parsed.is_buggy is False
    assert parsed.metric.value == pytest.approx(20.667)
    assert parsed.metric.maximize is False
