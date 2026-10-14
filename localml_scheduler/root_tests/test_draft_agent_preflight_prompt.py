"""Regression coverage for generated-candidate preflight requirements."""

import inspect
from types import SimpleNamespace

import agents.draft_agent as draft_agent
from engine.agent_search import AgentSearch

from agents.draft_agent import (
    controlled_cuda_execution_instructions,
    model_preflight_generation_instructions,
)


def _agent(
    *,
    scheduler: bool,
    preflight: bool,
    prediction_mode: str = "branch_profile",
    preflight_prompt_contract: bool = True,
):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            scheduler=SimpleNamespace(
                enabled=scheduler,
                settings={"prediction": {"mode": prediction_mode}},
            ),
            preflight=SimpleNamespace(
                enabled=preflight,
                prompt_contract_enabled=preflight_prompt_contract,
            ),
        )
    )


def test_baseline_prompt_has_no_treatment_contracts() -> None:
    sections = draft_agent.feature_prompt_contracts(
        _agent(scheduler=False, preflight=False)
    )

    assert "Backend design contract" not in sections
    assert "CPU preflight adapter contract" not in sections


def test_preflight_only_prompt_has_adapter_but_no_scheduler_contract() -> None:
    sections = draft_agent.feature_prompt_contracts(
        _agent(scheduler=False, preflight=True)
    )

    assert "CPU preflight adapter contract" in sections
    assert "Backend design contract" not in sections


def test_post_generation_preflight_does_not_change_draft_prompt() -> None:
    """A matched ablation may run preflight after generation without prompt leakage."""
    sections = draft_agent.feature_prompt_contracts(
        _agent(
            scheduler=False,
            preflight=True,
            preflight_prompt_contract=False,
        )
    )

    assert "CPU preflight adapter contract" not in sections
    assert "Backend design contract" not in sections


def test_ml_predictor_prompt_keeps_adapter_and_scheduler_contracts() -> None:
    sections = draft_agent.feature_prompt_contracts(
        _agent(scheduler=True, preflight=False, prediction_mode="ml_predictor")
    )

    assert "CPU preflight adapter contract" in sections
    assert "Backend design contract" in sections


def test_generation_instructions_require_complete_candidate_adapter() -> None:
    """A draft must be told the exact interface required before GPU admission."""
    instructions = "\n".join(model_preflight_generation_instructions())

    assert "class CandidateAdapter" in instructions
    for method in (
        "build_model",
        "build_optimizer",
        "build_train_batch",
        "build_validation_batch",
        "training_step",
        "validation_step",
    ):
        assert method in instructions
    assert "if __name__ == '__main__'" in instructions
    assert "mapping" in instructions
    assert "not a comma-separated string" in instructions


def test_draft_prompt_treats_dataset_preview_as_completed_context() -> None:
    """The single-pass draft must generate a solution, not repeat data exploration."""
    agent = SimpleNamespace(
        acfg=SimpleNamespace(prompt_compatibility_profile="current"),
        data_preview="CURRENT_PREVIEW",
    )
    prefix = draft_agent.draft_assistant_prefix(agent)

    assert "dataset preview below has already been completed" in prefix
    assert "First, I'll examine the dataset" not in prefix


def test_original_mlevolve_profile_restores_frozen_baseline_assistant_prefix() -> None:
    """The checker-only ablation must not change the baseline turn prefix."""
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            prompt_compatibility_profile="original_mlevolve_20260924"
        ),
        data_preview="FROZEN_PREVIEW",
    )

    prefix = draft_agent.draft_assistant_prefix(agent)

    assert prefix == (
        "Let me approach this systematically.\n"
        "First, I'll examine the dataset:\nFROZEN_PREVIEW"
    )


def test_original_mlevolve_profile_normalizes_task_description_tail() -> None:
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            prompt_compatibility_profile="original_mlevolve_20260924"
        )
    )

    assert draft_agent.task_description_for_prompt(
        agent, "frozen task\n\n\n"
    ) == "frozen task"


def test_original_mlevolve_profile_restores_multimodal_recommendation() -> None:
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            prompt_compatibility_profile="original_mlevolve_20260924"
        )
    )

    strategy = draft_agent.robustness_strategy_for_prompt(agent)
    rendered = "\n".join(next(iter(strategy.values())))

    assert "Data Modality" in rendered


def test_original_mlevolve_profile_omits_empty_context_separators() -> None:
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            prompt_compatibility_profile="original_mlevolve_20260924"
        )
    )

    prompt = draft_agent.compose_draft_user_prompt(
        agent,
        task_description="task",
        memory_section="",
        hardware_section="",
        lesson_section="",
        cuda_docs_section="",
        pipeline_decision_section="",
        instructions="\n# Instructions\n",
    )

    assert prompt == "\n# Task description\ntask\n\n# Instructions\n"


def test_current_profile_keeps_completed_preview_assistant_prefix() -> None:
    agent = SimpleNamespace(
        acfg=SimpleNamespace(prompt_compatibility_profile="current"),
        data_preview="CURRENT_PREVIEW",
    )

    prefix = draft_agent.draft_assistant_prefix(agent)

    assert "dataset preview below has already been completed" in prefix
    assert prefix.endswith("Solution and code:\n")


def test_controlled_cuda_prompt_rejects_cpu_only_training() -> None:
    """The accelerator experiment must ask for real CUDA work, not CPU surrogates."""
    instructions = "\n".join(controlled_cuda_execution_instructions())

    assert "PyTorch training" in instructions
    assert "LightGBM" in instructions
    assert "XGBoost" in instructions
    assert "silently fall back to CPU" in instructions
    assert "Subject Focus" in instructions
    assert "to_numpy(dtype=np.float32)" in instructions


def test_controlled_cuda_prompt_requires_executable_petfinder_interfaces() -> None:
    """Generated multimodal candidates must self-check the failures seen on A10."""
    instructions = "\n".join(controlled_cuda_execution_instructions())

    assert "PIL.Image.open" in instructions
    assert "model.forward" in instructions
    assert "len(TABULAR_COLS)" in instructions
    assert "weights=None" in instructions


def test_controlled_cuda_prompt_requires_labeled_validation_and_scalar_predictions() -> None:
    """Validation must not reuse the unlabeled test contract or tuple outputs."""
    instructions = "\n".join(controlled_cuda_execution_instructions())

    assert "validation dataset must return (image, tabular, target)" in instructions
    assert "tuple/list model output" in instructions


def test_controlled_cuda_prompt_requires_validated_precision_fallback() -> None:
    """Unmeasured AMP must use the repository's FP32 decision, not fail at runtime."""
    instructions = "\n".join(controlled_cuda_execution_instructions())

    assert "select_validated_precision" in instructions
    assert "precision_quality" in instructions
    assert "FP32 fallback" in instructions


def test_controlled_cuda_prompt_forbids_try_download_then_fallback() -> None:
    """Offline candidates must not spend an epoch repeatedly timing out on downloads."""
    instructions = "\n".join(controlled_cuda_execution_instructions())

    assert "unconditionally" in instructions
    assert "Weights.DEFAULT" in instructions
    assert "once before the training loop" in instructions


def test_agent_search_keeps_hardware_knowledge_optional() -> None:
    """Latest run.py may call the hook even in a Scheduler-only experiment."""
    source = inspect.getsource(AgentSearch.attach_hardware_knowledge)

    assert "hardware knowledge disabled" in source
    assert "if not self._hardware_context_enabled()" in source


def test_scheduler_only_skips_hardware_policy_but_keeps_scheduler() -> None:
    """No-HWKD experiments cannot be rejected by hardware-context policy."""
    precision_source = inspect.getsource(AgentSearch._validate_node_precision_before_execution)
    review_source = inspect.getsource(AgentSearch._review_training_parameters_before_submission)
    round_source = inspect.getsource(AgentSearch.execute_deferred_nodes)

    assert "if not self._hardware_context_enabled():" in precision_source
    assert "if not self._hardware_context_enabled():" in review_source
    assert "if self._hardware_context_enabled():" in round_source
