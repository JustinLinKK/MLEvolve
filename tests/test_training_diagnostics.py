from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from engine.search_node import SearchNode
from utils.node_diagnostics import build_node_diagnostics, write_node_diagnostics
from utils.training_diagnostics import TrainingDiagnostics, TRAINING_DIAGNOSTICS_MARKER


def test_diagnostics_preserve_updates_and_observe_bf16(capsys):
    torch.manual_seed(12)
    model = torch.nn.Linear(4, 1)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    batch = torch.ones(2, 4)
    with TrainingDiagnostics(model, optimizer) as diagnostics:
        for _ in range(2):
            for current, opt in ((model, optimizer), (reference, reference_optimizer)):
                opt.zero_grad()
                # Two microbatches, one optimizer update.
                for _ in range(2):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        loss = current(batch).float().square().mean() / 2
                    loss.backward()
                opt.step()
            diagnostics.after_update()
        report = diagnostics.report(epoch=1)
    assert report["attempted_updates"] == report["completed_updates"] == 2
    assert report["skipped_updates"] == 0
    assert report["autocast_dtypes_observed"] == ["torch.bfloat16"]
    assert report["parameter_dtypes"] == report["optimizer_state_dtypes"] == ["torch.float32"]
    for observed, expected in zip(model.parameters(), reference.parameters()):
        assert torch.equal(observed, expected)
    assert len(capsys.readouterr().out.splitlines()) == 3


@pytest.mark.parametrize("fused", [False, True])
def test_real_grad_scaler_overflow_is_counted_and_resumes(fused, capsys):
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=fused)
    scaler = torch.amp.GradScaler("cpu")
    with TrainingDiagnostics(model, optimizer, scaler=scaler) as diagnostics:
        for overflow in (False, True, False):
            optimizer.zero_grad()
            loss = model(torch.ones(2, 4)).square().mean()
            scaler.scale(loss).backward()
            before = [p.detach().clone() for p in model.parameters()]
            if overflow:
                next(model.parameters()).grad.fill_(float("inf"))
            scaler.step(optimizer)
            scaler.update()
            diagnostics.after_update()
            if overflow:
                assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
        state = diagnostics.state_dict()
    assert state == {"attempted_updates": 3, "completed_updates": 2, "skipped_updates": 1}
    with TrainingDiagnostics(model, optimizer, scaler=scaler) as resumed:
        resumed.load_state_dict(state)
        optimizer.zero_grad()
        scaler.scale(model(torch.ones(2, 4)).square().mean()).backward()
        scaler.step(optimizer)
        scaler.update()
        resumed.after_update()
        assert resumed.state_dict() == {"attempted_updates": 4, "completed_updates": 3, "skipped_updates": 1}


def test_failure_is_reported_without_swallowing_exception(capsys):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="training failed"):
        with TrainingDiagnostics(model, optimizer):
            raise ValueError("training failed")
    last = json.loads(capsys.readouterr().out.splitlines()[-1].split(" ", 1)[1])
    assert last["status"] == "failed"
    assert last["exception_type"] == "ValueError"
    assert not optimizer._optimizer_step_post_hooks
    assert not model._forward_pre_hooks


def test_scheduler_pause_is_interruption_not_training_failure(capsys):
    from localml_scheduler.execution.control import PauseRequested

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(PauseRequested):
        with TrainingDiagnostics(model, optimizer):
            raise PauseRequested("checkpoint and requeue")
    last = json.loads(capsys.readouterr().out.splitlines()[-1].split(" ", 1)[1])
    assert last["status"] == "interrupted"


def test_node_record_separates_runtime_inference_and_missing_evidence(tmp_path):
    cfg = SimpleNamespace(experiment=SimpleNamespace(mode="hardware_aware"), log_dir=tmp_path)
    parent = SearchNode(code="", stage="draft")
    node = SearchNode(code='PRECISION = "bf16"', stage="improve", parent=parent)
    node.prompt_input = "# Hardware-Aware Datatype/Precision Context\nBF16 is supported"
    report = build_node_diagnostics(cfg, node)
    assert report["runtime"] is None
    assert report["runtime_observation_status"] == "missing"
    assert report["hardware_knowledge"]["injection_status"] == "present"
    sample = {"schema_version": 1, "autocast_dtypes_observed": ["torch.float16"], "skipped_updates": 2}
    node._term_out = [TRAINING_DIAGNOSTICS_MARKER + " malformed\n",
                      TRAINING_DIAGNOSTICS_MARKER + " " + json.dumps(sample) + "\n"]
    path = write_node_diagnostics(cfg, SimpleNamespace(nodes=[parent, node]))
    record = json.loads(path.read_text().splitlines()[1])
    assert record["parent_id"] == parent.id
    assert record["inferred_training_settings"]["precision_mode"] == "bf16"
    assert record["runtime"]["autocast_dtypes_observed"] == ["torch.float16"]
    assert record["runtime"]["skipped_updates"] == 2
    assert "optimizer_updates_skipped" in record["diagnostic_flags"]
    assert "runtime_precision_differs_from_static_inference" in record["diagnostic_flags"]
    assert record["invalid_runtime_samples"] == 1
    cfg.experiment.mode = "origin"
    assert build_node_diagnostics(cfg, node)["hardware_knowledge"]["injection_status"] == "disabled_for_mode"


def test_hardware_retrieval_does_not_prove_prompt_injection(tmp_path):
    cfg = SimpleNamespace(experiment=SimpleNamespace(mode="hardware_aware"), log_dir=tmp_path)
    node = SearchNode(code="", stage="draft", hardware_evidence_refs=["retrieved-only"])
    assert build_node_diagnostics(cfg, node)["hardware_knowledge"]["injection_status"] == "unknown_no_prompt"
    node.diagnostics["hardware_prompt_stages"] = [{"stage": "datatype_precision", "context_present": False}]
    assert build_node_diagnostics(cfg, node)["hardware_knowledge"]["injection_status"] == "absent"


@pytest.mark.parametrize("scheduled", [False, True])
def test_generated_script_reports_through_both_execution_paths(tmp_path, scheduled, monkeypatch):
    from engine.executor import Interpreter, ExecutionResult

    monkeypatch.delenv("PYTHONPATH", raising=False)
    code = '''
import torch
from utils.training_diagnostics import TrainingDiagnostics
torch.manual_seed(7)
model = torch.nn.Linear(4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with TrainingDiagnostics(model, optimizer, settings={"planned_epochs": 1}) as diagnostics:
    for step in range(3):
        optimizer.zero_grad()
        model(torch.ones(2, 4)).square().mean().backward()
        optimizer.step()
        diagnostics.after_update()
    diagnostics.report(epoch=1)
print("Final Validation Score: 0.25")
'''
    if scheduled:
        from localml_scheduler.tests.test_mlevolve_runner import _build_context
        from localml_scheduler.adapters.mlevolve_runner import run_mlevolve_script_job
        from localml_scheduler.config import SchedulerSettings
        from localml_scheduler.domain import TrainingJob

        script = tmp_path / "candidate.py"
        script.write_text(code)
        result_path = tmp_path / "result.json"
        settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
        job = TrainingJob.create(
            runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            baseline_model_id="diagnostic-test", baseline_model_path=str(script),
            runner_kwargs={"script_path": str(script), "working_dir": str(tmp_path),
                           "result_path": str(result_path), "timeout": 30},
        )
        run_mlevolve_script_job(_build_context(settings, job))
        payload = json.loads(result_path.read_text())
        result = ExecutionResult(**{key: payload[key] for key in (
            "term_out", "exec_time", "exc_type", "exc_info", "exc_stack",
        )})
    else:
        # An isolated subprocess, with no scheduler attached.
        result = Interpreter(tmp_path, timeout=30, max_parallel_run=1).run(code, id="diagnostic")
    assert result.exc_type is None, result.term_out
    node = SearchNode(code=code, stage="draft")
    node.absorb_exec_result(result)
    record = build_node_diagnostics(SimpleNamespace(), node)
    assert record["runtime"]["completed_updates"] == 3
    assert record["runtime"]["skipped_updates"] == 0
    assert record["runtime"]["status"] == "completed"
