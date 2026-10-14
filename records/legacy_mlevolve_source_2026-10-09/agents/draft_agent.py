"""Draft Agent: initial plan and code draft."""

import logging
import copy
import time
from pathlib import Path
from typing import Any, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.coder import plan_and_code_query, stepwise_plan_and_code_query
from agents.triggers import register_node
from agents.hardware_context import (
    apply_hardware_design_brief_to_node,
    apply_hardware_context_to_node,
    apply_stepwise_hardware_decisions_to_node,
    build_stepwise_hardware_stage_sections,
    get_hardware_design_brief,
    get_hardware_context_for_stage,
    hardware_context_instructions,
)
from agents.cuda_docs_context import get_cuda_docs_context, format_cuda_docs_prompt_section
from agents.lesson_context import (
    apply_lesson_context_to_node,
    apply_lesson_context_to_pipeline_decision,
    get_lesson_context_for_stage,
    lesson_context_instructions,
)
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    apply_pipeline_decision_to_node,
    build_pipeline_decision,
    format_pipeline_decision_prompt_section,
    prompt_leakage_prevention,
    pipeline_decision_instructions,
    prompt_resp_fmt,
    get_prompt_environment,
    get_impl_guideline_from_agent,
)
from agents.prompts.impl_guideline import prompt_feature_flags_from_agent
from agents.planner import build_chat_prompt_for_model

logger = logging.getLogger("MLEvolve")


def model_preflight_generation_instructions() -> list[str]:
    """Return the non-negotiable CPU-admission interface for generated code."""
    return [
        "- **CPU PREFLIGHT CONTRACT (MANDATORY)**: Define a no-argument "
        "`class CandidateAdapter` in the generated file. It must implement "
        "`build_model(context)`, `build_optimizer(model, context)`, "
        "`build_train_batch(scenario, device)`, "
        "`build_validation_batch(scenario, device)`, "
        "`training_step(model, batch, context)`, and "
        "`validation_step(model, batch, context)`.",
        "- The adapter must exercise the same real model, loss, inputs, and optimizer "
        "as the training pipeline; do not use mock tensors or a different toy model. "
        "Its batch builders must honor `scenario['batch_size']` and return all inputs "
        "required by the model plus a target.",
        "- `scenario['fixture']` is a mapping from input names to non-batch shapes, "
        "not a comma-separated string. When it is present, construct every named input "
        "from that mapping and pass them through the real model forward path.",
        "- Keep imports, constants, class/function definitions, and read-only device "
        "configuration import-safe. Put all training, validation, prediction, and "
        "submission side effects under `if __name__ == '__main__':`.",
    ]


def feature_prompt_contracts(agent) -> dict[str, list[str]]:
    """Return only prompt contracts required by enabled treatment features."""
    flags = prompt_feature_flags_from_agent(agent)
    sections: dict[str, list[str]] = {}
    if flags.scheduler_enabled:
        sections["Backend design contract"] = [
            "- In the plan, explicitly state the selected memory strategy and physical/effective batch elasticity strategy.",
            "- State operator and normalization considerations, the model dimensions that remain configurable, and the fallback behavior.",
            "- List scheduler-owned backend controls that the generated subprocess must not implement.",
            "- Do not assume code-level integration, shared framework state, or synchronized start time with another job.",
        ]
    if flags.adapter_required:
        sections["CPU preflight adapter contract"] = (
            model_preflight_generation_instructions()
        )
    return sections


def draft_assistant_prefix(agent) -> str:
    """Build the assistant-prefill text selected by the experiment profile."""
    profile = str(
        getattr(agent.acfg, "prompt_compatibility_profile", "current") or "current"
    ).strip()
    if profile == "original_mlevolve_20260924":
        return (
            "Let me approach this systematically.\n"
            f"First, I'll examine the dataset:\n{agent.data_preview}"
        )
    return (
        "The dataset preview below has already been completed and is evidence, "
        "not a request to run exploration. Using it, I will now provide the "
        "required concise solution sketch followed by one complete runnable "
        "Python code block that trains, validates, prints the final score, and "
        "writes the submission. I will not emit data-inspection-only code.\n\n"
        f"Dataset preview:\n{agent.data_preview}\n\n"
        "Solution and code:\n"
    )


def _uses_frozen_baseline_prompt(agent) -> bool:
    return (
        str(
            getattr(agent.acfg, "prompt_compatibility_profile", "current")
            or "current"
        ).strip()
        == "original_mlevolve_20260924"
    )


def task_description_for_prompt(agent, task_description: str) -> str:
    """Normalize only the frozen baseline's historically saved task boundary."""
    if _uses_frozen_baseline_prompt(agent):
        return str(task_description).rstrip()
    return str(task_description)


def robustness_strategy_for_prompt(agent) -> dict[str, list[str]]:
    """Return the prompt profile's historical robustness guidance."""
    strategy = copy.deepcopy(ROBUSTNESS_GENERALIZATION_STRATEGY)
    if not _uses_frozen_baseline_prompt(agent):
        return strategy
    lines = strategy["💡 Recommendation: Robustness & Generalization Strategy"]
    modality = (
        "✅ **Data Modality**: When the dataset provides multiple modalities, try "
        "to use all available modalities, especially those most relevant to the "
        "task target."
    )
    if modality not in lines:
        architecture_index = next(
            index for index, line in enumerate(lines) if "**Architecture**" in line
        )
        lines.insert(architecture_index + 1, modality)
    return strategy


def compose_draft_user_prompt(
    agent,
    *,
    task_description: str,
    memory_section: str,
    hardware_section: str,
    lesson_section: str,
    cuda_docs_section: str,
    pipeline_decision_section: str,
    instructions: str,
) -> str:
    """Compose the user turn without leaking disabled-section whitespace."""
    if _uses_frozen_baseline_prompt(agent):
        return (
            f"\n# Task description\n{task_description}"
            f"{memory_section}\n{instructions}"
        )
    return (
        f"\n# Task description\n{task_description}{memory_section}\n"
        f"{hardware_section}\n{lesson_section}\n{cuda_docs_section}\n"
        f"{pipeline_decision_section}\n{instructions}"
    )


def controlled_cuda_execution_instructions() -> list[str]:
    """Return the workload contract for accelerator-scheduler measurements."""
    return [
        "- **CONTROLLED CUDA WORKLOAD (MANDATORY)**: This candidate is measured "
        "on an allocated CUDA GPU. Implement real PyTorch training and inference "
        "on that device: create a CUDA device, move the model and every training/"
        "validation batch to it, and run the actual forward, loss, backward, and "
        "optimizer steps there.",
        "- Do not choose a CPU-only estimator (including LightGBM, XGBoost, or "
        "scikit-learn) as the training pipeline. CPU-only image-feature loops do "
        "not satisfy this experiment.",
        "- Use offline local data and weights only. The CPU-safe CandidateAdapter "
        "remains for import/admission compatibility, but the main pipeline must "
        "exercise CUDA and must not silently fall back to CPU when CUDA is available.",
        "- **CSV SCHEMA CONTRACT**: Treat the dataset-preview column names as exact. "
        "Do not invent aliases (for PetFinder the column is `Subject Focus`, not "
        "`Focus`). Before tensor construction, select only existing numeric feature "
        "columns and convert explicitly, for example "
        "`frame[feature_cols].to_numpy(dtype=np.float32)` or "
        "`pd.to_numeric(..., errors='coerce').fillna(0).astype(np.float32)`. "
        "Never pass a Pandas object-dtype Series/array directly to `torch.tensor`.",
        "- **PETFINDER RUNTIME CONTRACT**: Load each image through "
        "`PIL.Image.open(path).convert('RGB')`; do not pass raw bytes to torchvision "
        "or call unavailable helpers such as `transforms.functional.pil_to_image`. "
        "For a multimodal model, define and use one matching `model.forward(images, "
        "tabular)` signature; never pass a tabular argument to `nn.Sequential`. Define "
        "`TABULAR_COLS` from the exact 12 prepared columns and set every tabular input "
        "layer width from `len(TABULAR_COLS)`, not an image-backbone dimension.",
        "- **OFFLINE WEIGHT CONTRACT**: Generated code must not initiate a download. "
        "Use `weights=None` / `pretrained=False` unless the code is given an exact, "
        "verified local checkpoint path; do not call torch.hub, Hugging Face, or timm "
        "pretrained loaders as a fallback. For torchvision, call the constructor "
        "unconditionally with `weights=None`: never put `Weights.DEFAULT` or any "
        "pretrained constructor in a try/except fallback, because the attempted "
        "download is still an invalid network operation. Build the model once before "
        "the training loop, never inside an epoch or batch loop.",
        "- **VALIDATION/OUTPUT CONTRACT**: The validation dataset must return "
        "(image, tabular, target); only the test dataset may return (image, tabular) "
        "or identifiers. Normalize a tuple/list model output at every train, validation, "
        "and test call before loss, metric, or `.numpy()` use: select its scalar "
        "prediction tensor explicitly rather than calling tensor methods on a tuple/list model output.",
        "- **PRECISION CONTRACT**: Do not enable AMP merely because CUDA is available. "
        "Before selecting precision, import and call `select_validated_precision`; store "
        "its returned decision in `settings['precision_quality']`, and derive autocast "
        "and GradScaler enablement from its returned `precision`. This task has no supplied "
        "matched FP32/AMP comparison, so use the returned FP32 fallback; do not label an "
        "unmeasured AMP path as accepted.",
    ]


def _format_used_prompt_sections(sections: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for idx, section in enumerate(sections, 1):
        name = str(section.get("name") or f"prompt_{idx}")
        prompt = str(section.get("prompt") or "")
        if not prompt:
            continue
        parts.append(f"# Used Prompt {idx}: {name}\n\n{prompt}")
    return "\n\n".join(parts)


def run(agent, init_solution_path: Optional[str] = None) -> SearchNode | None:
    """Generate initial draft. If init_solution_path is provided and readable, use file content directly."""
    if init_solution_path:
        try:
            code = Path(init_solution_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read init_solution from {init_solution_path}: {e}, falling back to LLM generation")
            init_solution_path = None
        else:
            plan = "User-provided init solution."
            hardware_ctx = get_hardware_context_for_stage(agent, "draft", code=code)
            if not agent.virtual_root.add_expected_child_count(agent.scfg):
                logger.info("Draft limit reached before init solution could reserve a child slot.")
                return None
            new_node = SearchNode(
                plan=plan,
                code=code,
                parent=agent.virtual_root,
                stage="draft",
                local_best_node=agent.virtual_root,
            )
            apply_hardware_context_to_node(new_node, hardware_ctx)
            lesson_ctx = get_lesson_context_for_stage(agent, "draft", parent_node=new_node, code=code)
            apply_lesson_context_to_node(new_node, lesson_ctx)
            register_node(agent, new_node, "User-provided init solution (no LLM).", new_branch=True)
            logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id}) [init_solution]")
            return new_node

    professional_identity = (
        "🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.\n\n"
        "**Your Standards**:\n"
        "✓ Design complete ML pipelines (data → model → training → inference)\n"
        "✓ Implement real models that LEARN from data (not baseline scripts with constants)\n"
        "✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample\n"
        "✓ Compete for TOP performance, not trivial baselines\n\n"
        "Your solution will be evaluated on a real leaderboard. Treat this with professionalism.\n\n"
    )

    introduction = (
        professional_identity +
        "Now, let's begin the competition. "
        "You need to come up with an excellent and creative plan for a competitive solution "
        "and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. "
        "We will now provide a description of the task."
    )
    prompt: Any = {
        "Introduction": introduction,
        "Task description": task_description_for_prompt(agent, agent.task_desc),
        "Memory": agent.virtual_root.fetch_child_memory(),
        "Instructions": {},
    }
    hardware_design_brief = get_hardware_design_brief(agent)
    hardware_ctx = get_hardware_context_for_stage(agent, "draft")
    hardware_stage_sections = build_stepwise_hardware_stage_sections(
        design_context=hardware_design_brief,
        execution_context=hardware_ctx,
        max_chars=getattr(agent.acfg, "hardware_context_max_prompt_chars", 3500),
    )
    hardware_section = "\n".join(
        section for section in (hardware_design_brief.prompt_section, hardware_ctx.prompt_section) if section.strip()
    )
    cuda_docs_ctx = get_cuda_docs_context(
        agent, "draft", hardware_context=hardware_ctx
    )
    cuda_docs_section = format_cuda_docs_prompt_section(
        cuda_docs_ctx,
        service=getattr(agent, "cuda_docs_service", None),
        role="draft",
    )
    lesson_ctx = get_lesson_context_for_stage(agent, "draft", parent_node=agent.virtual_root)
    lesson_section = lesson_ctx.prompt_section
    if lesson_section:
        prompt["Family–Hardware Lesson Profile"] = lesson_section
    pipeline_decision = build_pipeline_decision(
        agent,
        stage="draft",
        data_preview=agent.data_preview,
        hardware_contexts=[hardware_design_brief, hardware_ctx],
    )
    apply_lesson_context_to_pipeline_decision(pipeline_decision, lesson_ctx)
    pipeline_decision_section = format_pipeline_decision_prompt_section(pipeline_decision)
    prompt["Pipeline Decision"] = pipeline_decision
    prompt["Pipeline Decision Contract"] = pipeline_decision_section
    prompt["Instructions"] |= prompt_resp_fmt()
    prompt["Instructions"] |= hardware_context_instructions(hardware_ctx)
    prompt["Instructions"] |= lesson_context_instructions(lesson_ctx)
    prompt["Instructions"] |= pipeline_decision_instructions(pipeline_decision)

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Design": [
            "",
            "Before designing your solution, you must answer three fundamental questions:",
            "",
            "1. **WHAT makes this task unique?**",
            "   - Not generic observations like 'it's a classification task'",
            "   - What SPECIFIC patterns, challenges, or domain characteristics?",

            "",
            "2. **WHY is your approach suitable for this task?**",
            "   - Not just 'this model is good' - explain the MATCH between approach and task",
            "   - What properties of your method address the task characteristics?",

            "",
            "3. **HOW will you validate your hypothesis?**",
            "   - What outcome would confirm your approach is right?",
            "   - What outcome would suggest you need to reconsider?",

            "",
            "---",
            "",
            "⚠️ This is not a template to fill - this is how scientists think.",
            "Blindly applying standard methods without understanding WHY is not acceptable.",
            "",
            "Your plan should naturally reflect this reasoning process.",
        ],
    }

    prompt["Instructions"] |= {
        "Solution sketch guideline": [
            "- This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.\n",
            "- 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:\n",
            "  • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.\n",
            "  • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.\n",
            "  • **Step 2 - Choose Strategy**:\n",
            "    → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.\n",
            "    → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.\n",
            "  • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).\n",
            "  • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.\n",
            "- Don't propose the same modelling solution but keep the evaluation the same.\n",
            "- Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.\n",
            "- Propose an evaluation metric that is reasonable for this task.\n",
            "- Don't suggest to do EDA.\n",
            "- The data is already prepared in `./input` directory. No need to unzip files.\n",
        ],
        "Coding & Execution Guidelines (CRITICAL)": [
            "- **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.",
            "- **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.",
            "- **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser."
        ],
    }
    prompt["Instructions"] |= feature_prompt_contracts(agent)
    if bool(getattr(agent.acfg, "require_cuda_execution", False)):
        prompt["Instructions"]["Controlled CUDA execution contract"] = (
            controlled_cuda_execution_instructions()
        )
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()

    if agent.use_coldstart and (agent.coldstart_description != "None model"):
        coldstart_guideline = [
            f"""
            **Pretrained Model Strategy**:

            • **Option A [RECOMMENDED]**: {agent.coldstart_description}
              → SOTA models with proven performance. Use for end-to-end fine-tuning OR as frozen feature extractors.

            • **Option B**: Alternative pretrained models if better suited to task characteristics.

            • **Option C**: Train from scratch / non-DL methods (only when pretraining provides no advantage).

            **CRITICAL: When using any recommended pretrained model (Option A), you MUST copy the Code template EXACTLY as provided — including model variant names, file paths, and checkpoint filenames. Only the listed weights are available locally; other variants will fail to load. Do NOT invent Kaggle/input paths, dummy checkpoints, or placeholder model files. If a local path is not explicitly shown in the template, choose Option B or C instead.**

            **Key Techniques**:
            1. **Feature Extractor Pattern**: If dataset is small or domain mismatch exists → Freeze backbone + train only final layers (or feed to XGBoost/SVM).

            2. **Mixed Precision (MANDATORY for pretrained models)**: Use `torch.cuda.amp` (autocast + GradScaler) to save memory. DO NOT manually convert to .half().

            3. **Avoid Timeouts**: #1 cause is slow data loading, NOT GPU model.
               • Use DataLoader with num_workers>=2, pin_memory=True (NOT raw for loops)
               • For large datasets + heavy backbones: Extract & cache features to disk (.npy/.h5)
            """
        ]
    else:
        coldstart_guideline = [""]

    prompt["Instructions"]["Implementation guideline"].extend(coldstart_guideline)
    prompt["Instructions"] |= get_prompt_environment(
        prompt_compatibility_profile=getattr(
            agent.acfg, "prompt_compatibility_profile", "current"
        )
    )
    prompt["Instructions"] |= robustness_strategy_for_prompt(agent)

    instructions = f"\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

    user_prompt = compose_draft_user_prompt(
        agent,
        task_description=prompt["Task description"],
        memory_section=memory_section,
        hardware_section=hardware_section,
        lesson_section=lesson_section,
        cuda_docs_section=cuda_docs_section,
        pipeline_decision_section=pipeline_decision_section,
        instructions=instructions,
    )
    assistant_prefix = draft_assistant_prefix(agent)
    prompt_complete = build_chat_prompt_for_model(
        agent.acfg.code.model, introduction, user_prompt, assistant_prefix
    )
    if not agent.virtual_root.add_expected_child_count(agent.scfg):
        logger.info("Draft limit reached before draft generation could reserve a child slot.")
        return None

    prompt_for_log = prompt_complete
    if agent.use_stepwise_generation:
        stepwise_context_payload = {
            "stage": "draft",
            "memory": prompt.get("Memory", ""),
            "hardware_prompt_section": hardware_section,
            "hardware_stage_sections": hardware_stage_sections,
            "hardware_candidate": hardware_ctx.candidate,
            "hardware_context": {
                "design_brief": hardware_design_brief.compact_context,
                "execution_context": hardware_ctx.compact_context,
            },
            "cuda_docs_prompt_section": cuda_docs_section,
            "lesson_profile_section": lesson_section,
            "lesson_profile_context": lesson_ctx.compact_context,
            "pipeline_decision": pipeline_decision,
            "pipeline_decision_section": pipeline_decision_section,
        }
        plan, code, stepwise_metadata = stepwise_plan_and_code_query(
            agent_instance=agent,
            prompt_base=prompt,
            data_preview=agent.data_preview,
            context=stepwise_context_payload,
            return_metadata=True,
        )
        prompt_for_log = (
            _format_used_prompt_sections(stepwise_context_payload.get("used_prompt_sections") or [])
            or prompt_complete
        )
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)
        stepwise_metadata = {}
    new_node = SearchNode(plan=plan, code=code, parent=agent.virtual_root, stage="draft",
                        local_best_node=agent.virtual_root)
    apply_hardware_context_to_node(new_node, hardware_ctx)
    apply_hardware_design_brief_to_node(new_node, hardware_design_brief)
    apply_pipeline_decision_to_node(new_node, pipeline_decision)
    apply_lesson_context_to_node(new_node, lesson_ctx)
    apply_stepwise_hardware_decisions_to_node(
        new_node,
        stepwise_metadata,
        design_context=hardware_design_brief,
        execution_context=hardware_ctx,
    )
    register_node(agent, new_node, prompt_for_log, new_branch=True)

    logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id})")
    return new_node
