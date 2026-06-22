"""Stepwise code generation mode.

Provides stepwise code generation using multi-agent collaboration where specialized
agents handle different stages of the ML pipeline:
  - data_processing_and_feature_engineering
  - model_design
  - training_evaluation

Main entry: stepwise_plan_and_code_query()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

from llm import generate, compile_prompt_to_md
from utils.response import extract_code, extract_text_up_to_code, wrap_code
from agents.hardware_context import format_hardware_design_brief, format_hardware_prompt_section
from agents.planner.base_planner import (
    PLANNING_ALLOWED_MODULES,
    PLANNING_JSON_FORMAT,
    PLANNING_JSON_SCHEMA,
    parse_planning_response,
)

logger = logging.getLogger("MLEvolve")

_STEP_HARDWARE_STAGES = {
    "data_processing_and_feature_engineering": {"datatype"},
    "model_design": {"model", "optimizer"},
    "training_evaluation": {"tuning"},
}


@dataclass
class StepwiseContext:
    stage: str = "draft"
    memory: str = ""
    previous_code: str = ""
    execution_output: str = ""
    hardware_brief: str = ""
    hardware_candidate: Dict[str, Any] = field(default_factory=dict)
    hardware_context: Dict[str, Any] = field(default_factory=dict)
    pipeline_decision: Dict[str, Any] = field(default_factory=dict)
    pipeline_decision_section: str = ""
    used_prompts: List[Dict[str, str]] = field(default_factory=list)


def _hardware_brief_for_step(context: StepwiseContext, step_name: str) -> str:
    allowed_stages = _STEP_HARDWARE_STAGES.get(step_name)
    hardware_context = context.hardware_context or {}
    execution_context = dict(hardware_context.get("execution_context") or {})
    design_brief = dict(hardware_context.get("design_brief") or {})
    if not allowed_stages or not execution_context:
        return context.hardware_brief

    sections: list[str] = []
    if step_name == "model_design" and design_brief:
        filtered_design = _filter_design_feature_index(design_brief, allowed_stages)
        design_section = format_hardware_design_brief(filtered_design, max_chars=1800)
        if design_section.strip():
            sections.append(design_section)

    filtered_execution = _filter_compact_stage_hardware(execution_context, allowed_stages)
    if step_name == "data_processing_and_feature_engineering":
        filtered_execution.pop("graph_evidence", None)
        filtered_execution.pop("vector_evidence", None)
        filtered_execution.pop("derived_diagnosis", None)
    elif step_name == "model_design":
        filtered_execution.pop("graph_evidence", None)

    execution_section = format_hardware_prompt_section(filtered_execution, max_chars=1800)
    if execution_section.strip():
        sections.append(execution_section)
    return "\n".join(section for section in sections if section.strip()) or context.hardware_brief


def _filter_compact_stage_hardware(compact: Dict[str, Any], allowed_stages: set[str]) -> Dict[str, Any]:
    filtered = dict(compact)
    stage_context = dict(filtered.get("stage_hardware_features") or {})
    stages = [
        item
        for item in list(stage_context.get("stages") or [])
        if str(item.get("stage") or "") in allowed_stages
    ]
    if stages:
        stage_context["stages"] = stages
        stage_context["stage_filter"] = [item.get("stage") for item in stages if item.get("stage")]
        feature_ids: list[str] = []
        for item in stages:
            for feature in item.get("features") or []:
                feature_id = str(feature.get("feature_id") or "").strip()
                if feature_id and feature_id not in feature_ids:
                    feature_ids.append(feature_id)
        stage_context["feature_ids"] = feature_ids
        stage_context["feature_count"] = sum(int(item.get("feature_count") or 0) for item in stages)
    else:
        stage_context = {}
    filtered["stage_hardware_features"] = stage_context
    return filtered


def _filter_design_feature_index(compact: Dict[str, Any], allowed_stages: set[str]) -> Dict[str, Any]:
    filtered = dict(compact)
    feature_index = dict(filtered.get("hardware_feature_index") or {})
    features = list(feature_index.get("features") or [])
    if features:
        feature_index["features"] = [
            feature
            for feature in features
            if (
                str(feature.get("pipeline_stage") or "") in allowed_stages
                or str(feature.get("category") or "") in {"compute_capability", "interconnect", "kernel_optimization", "tensor_core", "optimizer"}
            )
        ]
        feature_index["feature_count"] = len(feature_index["features"])
        filtered["hardware_feature_index"] = feature_index
    return filtered


@dataclass
class StepAgent:
    name: str
    introduction: str
    description: str
    guidelines: List[str]

    def generate(
        self,
        task_desc: str,
        data_preview: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
    ) -> Tuple[str, str]:
        prompt = self._build_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview,
            previous_steps=previous_steps,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
            improvement_mode=improvement_mode,
            previous_module_code=previous_module_code,
            improvement_strategy=improvement_strategy,
        )
        context.used_prompts.append({"name": self.name, "prompt": prompt})

        completion_text = None
        for _ in range(retries):
            completion_text = generate(
                prompt=prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                return nl_text, code

            logger.debug(f"Extraction retry for {self.name}...")
        logger.warning(f"Code extraction failed after retries for {self.name}")
        return "", completion_text  # type: ignore

    def _build_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
    ) -> str:
        base_intro = prompt_base.get("Introduction", "")

        if context.stage == "improve":
            if improvement_mode and previous_module_code:
                step_specific_intro = (
                    f"You are currently working on improving the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the improved code for this specific step, based on the previous module code and the improvement strategy provided below. "
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}"
                )
            else:
                step_specific_intro = (
                    f"You are currently working on the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the code for this specific step that aligns with the overall improvement strategy. "
                    f"Base your implementation on the previous solution and execution results provided below, ensuring it integrates well with the improved approach."
                )
        else:
            step_specific_intro = (
                f"You are currently focusing on the '{self.name}' step of the solution. "
                f"Your task is to write ONLY the code for this specific step, not the complete solution."
            )
        introduction = base_intro + "\n\n" + step_specific_intro

        prev_summary = ""
        if previous_steps:
            prev_parts = []
            for step in previous_steps:
                prev_parts.append(f"### {step['name']}\n**Plan:** {step['plan']}\n**Code:**\n{wrap_code(step['code'])}")
            prev_summary = "\n\n".join(prev_parts)
        else:
            prev_summary = "This is the first step, no previous steps."

        guidelines_to_use = self.guidelines.copy()

        use_pretrain = (
            hasattr(agent_instance, 'use_coldstart') and
            agent_instance.use_coldstart and
            hasattr(agent_instance, 'coldstart_description') and
            agent_instance.coldstart_description != "None model"
        )

        if use_pretrain and context.stage == "draft":
            if self.name == "model_design":
                pretrain_emphasis = [
                    "**CRITICAL: You MUST prioritize using the recommended pretrained models provided in the Implementation guideline section below.**",
                    "The pretrained models are STRONGLY RECOMMENDED and should be your default first choice.",
                    "Only use custom architectures if the pretrained models are clearly unsuitable for this specific task."
                ]
                guidelines_to_use = pretrain_emphasis + guidelines_to_use
            elif self.name == "data_processing_and_feature_engineering":
                pretrain_awareness = [
                    "**IMPORTANT: Be aware that pretrained models may be used in later steps. Consider the input requirements of common pretrained models (e.g., image size, normalization, data format) when preparing the data and engineering features.**",
                    "For image tasks, ensure data is prepared in a format compatible with standard pretrained models (e.g., PIL Image, numpy arrays, proper image sizes).",
                    "For text tasks, ensure text data is properly tokenized and formatted for potential transformer models.",
                ]
                guidelines_to_use = pretrain_awareness + guidelines_to_use

        guidelines_text = "\n".join([f"- {g}" for g in guidelines_to_use])

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = (
            "Your response should be:\n"
            "1. A brief plan (2-3 sentences) describing what you will do in this step\n"
            "2. A single markdown code block (wrapped in ```) containing ONLY the code for this step\n"
            "IMPORTANT: Do NOT write code for other steps. Only write code for the current step."
        )

        prompt_instructions[f"{self.name} guidelines"] = [guidelines_text]

        if "Implementation guideline" in prompt_instructions:
            base_impl_guideline = prompt_instructions["Implementation guideline"]
            step_specific_impl = [
                "The code for this step must be self-contained and can be integrated with other steps.",
                "Use clear variable names that are consistent with previous steps.",
                "Do not duplicate code from previous steps - assume those parts already exist.",
                "Make sure to handle edge cases appropriately.",
            ]
            if isinstance(base_impl_guideline, list):
                prompt_instructions["Implementation guideline"] = base_impl_guideline + step_specific_impl
            else:
                prompt_instructions["Implementation guideline"] = [base_impl_guideline] + step_specific_impl

        hardware_brief = _hardware_brief_for_step(context, self.name)
        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Data preview": data_preview_str,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Hardware/Profile Optimization Context": hardware_brief,
            "Pipeline Decision Contract": context.pipeline_decision_section,
            "Previous steps": prev_summary,
            "Current step": {
                "Name": self.name,
                "Description": self.description,
            },
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve":
            if improvement_mode and previous_module_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(previous_module_code),
                    "Note": f"This is the previous code for the '{self.name}' module. Improve it based on the improvement strategy provided above."
                }
            elif "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        if context.stage == "draft":
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            assistant_suffix = ""
        elif context.stage == "improve":
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            if improvement_mode and previous_module_code:
                previous_module_code_wrapped = wrap_code(previous_module_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously implemented the '{self.name}' module with the following code:\n{previous_module_code_wrapped}\n"
                    f"The execution of the full solution yielded the following results:\n{execution_output_wrapped}\n"
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}\n"
                    f"I need to improve this specific module according to the strategy above, ensuring it integrates well with the other modules."
                )
            elif context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            assistant_suffix = ""

        model_name = agent_instance.acfg.code.model.lower()

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

        hardware_section = prompt.get("Hardware/Profile Optimization Context", "")
        pipeline_decision_section = prompt.get("Pipeline Decision Contract", "")

        previous_solution_section = ""
        if context.stage == "improve" and "Previous solution" in prompt:
            previous_solution_section = f"\n# Previous solution\n{prompt['Previous solution']['Code']}\n"

        user_prompt = (
            f"\n# Task description\n{prompt['Task description']}\n\n"
            f"{memory_section}\n"
            f"{hardware_section}\n"
            f"{pipeline_decision_section}\n"
            f"{previous_solution_section}"
            f"# Previous steps\n{prompt['Previous steps']}\n\n"
            f"# Current step: {prompt['Current step']['Name']}\n{prompt['Current step']['Description']}\n\n"
            f"{instructions}"
        )
        return f"{introduction}\n\n{user_prompt}\n\n{okay_text}\n{prompt['Data preview']}{assistant_suffix}"



@dataclass
class MetaAgent:
    def merge(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
    ) -> Tuple[str, str]:
        prompt = self._build_merge_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview_str,
            step_results=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
        )
        context.used_prompts.append({"name": "merge", "prompt": prompt})

        completion_text = None
        for _ in range(retries):
            completion_text = generate(
                prompt=prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                return nl_text, code

            logger.debug("Extraction retry for MetaAgent merge...")
        logger.warning("Code extraction failed after retries for MetaAgent merge")
        return "", completion_text 

    def _build_merge_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        ) -> Tuple[str, str]:
        introduction = (
            "You are a Kaggle grandmaster attending a competition, an expert in writing clean, efficient, and competition-winning Python code for ML tasks. "
            "You have received code snippets from a team of specialized agents, each focusing on a specific part of the ML pipeline. "
            "Your critical task is to intelligently merge these partial scripts into a single, cohesive, and fully runnable Python script."
        )

        steps_summary = []
        for i, result in enumerate(step_results, 1):
            steps_summary.append(f"""
        ### Step {i}: {result['name']}
        **Plan:** {result['plan']}
        **Code:**
        {wrap_code(result['code'])}
        """)

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = (
            "Your response should be a brief summary (2-3 sentences) of how you merged the steps, "
            "followed by a single markdown code block (wrapped in ```) containing the complete merged code. "
            "There should be no additional headings or text in your response."
        )

        prompt_instructions["Merge guidelines"] = [
            "- Combine all code sections into a single, runnable Python script",
            "- CRITICAL: You are a MERGER, not a designer. Faithfully integrate the code from all steps. Do NOT introduce new models, algorithms, or approaches that were not in the original steps.",
            "- Ensure variable names are consistent across steps",
            "- Remove duplicate imports and definitions",
            "- Resolve conflicts between steps by following the earlier step's design (e.g., model_design defines the model, training_evaluation trains it)",
            "- Ensure the execution flow is logical: data processing & feature engineering -> model design -> training & evaluation",
            "- Make sure the final code prints validation metric (must match task's Evaluation section) and saves submission.csv",
            "- The code should be a single-file Python program that can be executed as-is",
            "- Assume previous steps have NOT been executed; do not skip execution steps and only read files or outputs.",
            "- All parts must work together seamlessly",
        ]

        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Hardware/Profile Optimization Context": context.hardware_brief,
            "Pipeline Decision Contract": context.pipeline_decision_section,
            "Data preview": data_preview_str,
            "Step results": "".join(steps_summary),
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve":
            if "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"
        hardware_section = prompt.get("Hardware/Profile Optimization Context", "")
        pipeline_decision_section = prompt.get("Pipeline Decision Contract", "")

        okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"

        if context.stage == "improve":
            if context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            memory_section = f"# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}"
            okay_text = "Let me approach this systematically.\nFirst, I'll examine the dataset:"
            assistant_suffix = ""

        user_prompt = (
            f"\n# Task description\n{prompt['Task description']}\n\n"
            f"{memory_section}\n\n"
            f"{hardware_section}\n"
            f"{pipeline_decision_section}\n"
            f"# Step results\n{prompt['Step results']}\n\n"
            f"{instructions}"
        )
        return f"{introduction}\n\n{user_prompt}\n\n{okay_text}\n{prompt['Data preview']}{assistant_suffix}"


    def _simple_concat(self, step_results: List[Dict[str, str]]) -> str:
        code_parts = []
        for result in step_results:
            code_parts.append(f"# Step: {result['name']}\n{result['code']}\n")
        return "\n".join(code_parts)


def _hardware_reasoning_enabled(agent_instance) -> bool:
    cfg = getattr(agent_instance, "cfg", None)
    experiment = getattr(cfg, "experiment", None)
    mode = str(getattr(experiment, "mode", "") or "").strip().lower().replace("-", "_")
    if mode in {"origin", "baseline"}:
        return False
    acfg = getattr(agent_instance, "acfg", None)
    return bool(getattr(acfg, "hardware_context_enabled", True))


def create_default_step_agents(
    *,
    hardware_aware: bool = True,
    pipeline_decision_aware: bool = True,
) -> List[StepAgent]:
    data_guidelines = [
        "Your responsibility: Load data from `./input`, clean, create features (preprocessing, encoding, augmentation), and split dataset into train/validation/test.",
        "CRITICAL: This step MUST include BOTH data loading AND feature engineering. Do NOT only load the raw data. You must actively create, transform, and enhance features to improve model performance.",
        "IMPORTANT: Apply feature engineering techniques such as feature scaling, encoding, transformation, and data augmentation methods appropriate for the task. Explore and implement feature engineering strategies that can enhance the model's ability to learn from the data.",
        "CRITICAL: Do NOT build models, write training code, or perform evaluation. Focus ONLY on data preparation and feature engineering.",
    ]
    model_guidelines = [
        "Your responsibility: Design the model architecture or choose reference pretrained model, loss function, and optimizer based on the task and the features from previous steps.",
        "CRITICAL: Do NOT write the training loop, data processing, or feature engineering code. Only define the model, criterion, and optimizer objects.",
        "IMPORTANT: Consider the task's evaluation metric (from the task description's Evaluation section) when designing the model. The model output format should be compatible with the required evaluation metric calculation.",
        "IMPORTANT: When designing custom model architectures, include appropriate regularization components (e.g., Dropout layers) to prevent overfitting.",
    ]
    training_guidelines = [
        "Your responsibility: Write the training loop that uses the data, features, model, loss function, and optimizer from previous steps. Include validation, metric tracking, save the best model. Then load the best model, calculate validation metric (must match task's Evaluation section), perform test inference, and save `submission.csv` to `./submission/` directory.",
        "CRITICAL: Assume that all previous code steps have already been executed. You should start directly from the training step. Do NOT redefine or reload the data, features, model, loss function, or optimizer. These components are already defined and available from the previous steps.",
        "CRITICAL: You MUST use the variables and objects defined in previous steps AS-IS. Do NOT replace, redesign, or substitute them with different approaches. Your ONLY job is to write the training/evaluation code for what was already defined — not to introduce new models or pipelines.",
        "IMPORTANT: Your code should assume the data preprocessing, feature engineering, and model design steps have been completed. Simply use the existing variables without copying them.",
        "CRITICAL: Validation metric computation must use the same prediction method as test inference, using training data only as reference, to avoid data leakage and ensure the metric reflects true generalization performance.",
        "CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "CRITICAL: You MUST actively prevent overfitting. Do NOT only focus on validation set metrics, as this can easily cause the model to overfit. You can consider to use standard anti-overfitting techniques as default modeling strategies, including:",
        "  - Data augmentation (when applicable to the task)",
        "  - Early stopping (monitor validation metric and stop when it stops improving)",
        "  - Regularization (weight decay, L1/L2 regularization)",
        "  - Dropout (if using neural networks)",
        "  - Other appropriate regularization techniques for the specific model type",
        "CRITICAL: You MUST implement the exact evaluation metric as specified in the task description's 'Evaluation' section. Read the Evaluation section carefully and implement it precisely according to the exact formula, calculation steps, and aggregation method described.",
        "CRITICAL: You MUST NOT use dummy, simplified, or approximate metrics. The validation metric must be a REAL and COMPLETE implementation of the task's evaluation metric as described in the Evaluation section, not an approximation, placeholder, or simplified version.",
        "CRITICAL: If the Evaluation section specifies multiple thresholds, components, or aggregation steps, you MUST implement ALL of them. Do not skip any required calculation steps or use shortcuts.",
        "CRITICAL: The metric calculation must match the Evaluation section exactly - use the same matching criteria, the same formula, the same thresholds (if any), and the same aggregation method as specified.",
        "CRITICAL: The final line must be: `print(f'Final Validation Score: {{score}}')`. This is required for the score parser.",
    ]
    if pipeline_decision_aware:
        data_guidelines.extend(
            [
                "Pipeline decision: consume `datatype.modality`, `datatype.target_type`, and `datatype.shape_constraints` before choosing preprocessing or feature engineering.",
                "Pipeline decision: use `tuning.dataloader_policy` only for data loading mechanics; do not choose the model family or optimizer in this step.",
            ]
        )
        model_guidelines.extend(
            [
                "Pipeline decision: consume `datatype`, `model`, and `optimizer.loss`/`optimizer.optimizer` to define only the model, criterion, and optimizer.",
                "Pipeline decision: do not use unavailable weights, packages, or hardware-only tricks to justify a model family.",
            ]
        )
        training_guidelines.extend(
            [
                "Pipeline decision: consume `optimizer`, `tuning`, and `evidence` for training, validation, checkpointing, submission generation, and runtime logging.",
                "Pipeline decision: when evidence is missing, implement the recorded safe fallbacks rather than inventing hardware claims.",
            ]
        )
    if hardware_aware:
        data_guidelines.append(
            "Hardware-aware data pipeline: when using GPU training, keep input resolution/sequence length configurable, use DataLoader settings compatible with the hardware brief, and prefer pin_memory/non-blocking transfers when tensors move to CUDA."
        )
        model_guidelines.extend(
            [
                "Hardware-aware model design: use the Hardware-Aware Model Design Brief to compare model families before choosing an architecture. Optimize for the task metric first, while treating lower training time and higher GPU utilization as persistent objectives.",
                "This model_design step is the combined hardware-aware design stage: use the brief's compact hardware node, available feature keys, and selected feature details when they are relevant to architecture, precision, or layer choices.",
                "Prefer architectures and layers that can use documented hardware fast paths from the brief, such as tensor-core-friendly dimensions, AMP/bf16/fp16, transformer engine, torch.compile, or efficient convolution kernels, only when the evidence applies to this task and installed environment.",
                "Do not invent pretrained checkpoint paths, Kaggle input directories, torch hub directories, or dummy model weights to satisfy hardware advice. If a model source is not explicitly available, choose an available baseline-compatible model family.",
            ]
        )
        training_guidelines.extend(
            [
                "Hardware-aware training: optimize runtime at fixed modeling intent. Do NOT increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization unless the user explicitly asks for score improvement.",
                "Allowed hardware optimizations: physical batch size, gradient accumulation while preserving effective batch size, AMP/TF32/bf16, dataloader workers, pin_memory, persistent_workers, channels_last, safe torch.compile, checkpointing, and runtime logging.",
                "Scheduler-aware training: adapt to the scheduler backend config in the Hardware/Profile Optimization Context. Do not hardcode CUDA process, CUDA stream, MPS, or backend selection in the script; keep code backend-compatible and configurable.",
                "Hardware-aware training: use the hardware/profile context to choose physical batch size, AMP dtype, gradient accumulation, checkpoint cadence, and dataloader settings. If choosing a riskier setting for score reasons, include an explicit fallback path for OOM/timeout such as smaller batch size, accumulation, lower resolution, fewer epochs, or checkpoint resume.",
                "When feasible, log resolved batch size, selected precision, elapsed time, throughput, and peak CUDA memory so later scheduler graph evidence can learn from this run.",
            ]
        )
    return [
        StepAgent(
            name="data_processing_and_feature_engineering",
            introduction="You are a Data Preparation Specialist responsible for data loading, cleaning, and feature engineering.",
            description="Load data from `./input` directory, perform cleaning, feature engineering, and create train/validation/test splits.",
            guidelines=data_guidelines,
        ),
        StepAgent(
            name="model_design",
            introduction="You are a Model Architect responsible for designing the model architecture, loss function, and optimizer.",
            description="Design the model architecture (including pretrained models), and define the loss function and optimizer.",
            guidelines=model_guidelines,
        ),
        StepAgent(
            name="training_evaluation",
            introduction="You are a Training and Evaluation Expert responsible for implementing training, validation, and submission generation.",
            description="Implement the training loop, validation, metric tracking, model saving, and generate submission file.",
            guidelines=training_guidelines,
        ),
    ]


def stepwise_plan_and_code_query(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    ) -> Tuple[str, str]:
    logger.info("Using stepwise generation route.")

    stepwise_context = StepwiseContext(
        stage=context.get("stage", "draft"),
        memory=context.get("memory", ""),
        previous_code=context.get("previous_code", ""),
        execution_output=context.get("execution_output", ""),
        hardware_brief=context.get("hardware_prompt_section", ""),
        hardware_candidate=context.get("hardware_candidate", {}) or {},
        hardware_context=context.get("hardware_context", {}) or {},
        pipeline_decision=context.get("pipeline_decision", {}) or {},
        pipeline_decision_section=context.get("pipeline_decision_section", ""),
    )

    step_agents = create_default_step_agents(
        hardware_aware=_hardware_reasoning_enabled(agent_instance),
        pipeline_decision_aware=bool(stepwise_context.pipeline_decision_section),
    )
    meta_agent = MetaAgent()

    step_results: List[Dict[str, str]] = []
    for idx, agent in enumerate(step_agents, 1):
        logger.info(f"Step {idx}/{len(step_agents)}: {agent.name}")

        plan, code = agent.generate(
            task_desc=prompt_base["Task description"],
            data_preview=data_preview,
            previous_steps=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=stepwise_context,
        )

        step_results.append({
            "name": agent.name,
            "plan": plan,
            "code": code,
        })

    logger.info("Merging all steps...")
    final_plan, final_code = meta_agent.merge(
        task_desc=prompt_base["Task description"],
        data_preview_str=data_preview,
        step_results=step_results,
        prompt_base=prompt_base,
        agent_instance=agent_instance,
        context=stepwise_context,
    )

    logger.info("Stepwise generation completed.")

    context["used_prompt_sections"] = list(stepwise_context.used_prompts)
    return final_plan, final_code
