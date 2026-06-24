"""Stepwise code generation mode.

Provides stepwise code generation using multi-agent collaboration where specialized
agents handle different stages of the ML pipeline:
  - data_processing_and_feature_engineering
  - model_design
  - datatype_precision (hardware-aware mode only)
  - training_evaluation

Main entry: stepwise_plan_and_code_query()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

from llm import generate, compile_prompt_to_md
from utils.response import extract_code, extract_text_up_to_code, wrap_code
from agents.planner.base_planner import (
    PLANNING_ALLOWED_MODULES,
    PLANNING_JSON_FORMAT,
    PLANNING_JSON_SCHEMA,
    parse_planning_response,
)

logger = logging.getLogger("MLEvolve")

STEP_LOGICAL_STAGE: Dict[str, str] = {
    "data_processing_and_feature_engineering": "stage1_candidate_construction",
    "model_design": "stage1_candidate_construction",
    "datatype_precision": "stage2_datatype_precision",
    "training_evaluation": "stage3_training_evaluation",
}

STEP_LOGICAL_STAGE_LABEL: Dict[str, str] = {
    "stage1_candidate_construction": "Stage 1 candidate construction",
    "stage2_datatype_precision": "Stage 2 datatype/precision",
    "stage3_training_evaluation": "Stage 3 training/evaluation",
}


@dataclass
class StepwiseContext:
    stage: str = "draft"
    memory: str = ""
    previous_code: str = ""
    execution_output: str = ""
    hardware_brief: str = ""
    hardware_stage_sections: Dict[str, str] = field(default_factory=dict)
    hardware_candidate: Dict[str, Any] = field(default_factory=dict)
    hardware_context: Dict[str, Any] = field(default_factory=dict)
    pipeline_decision: Dict[str, Any] = field(default_factory=dict)
    pipeline_decision_section: str = ""
    stage_note_board: List[Dict[str, Any]] = field(default_factory=list)
    used_prompts: List[Dict[str, str]] = field(default_factory=list)

    def hardware_section_for(self, step_name: str) -> str:
        return self.hardware_stage_sections.get(step_name) or self.hardware_brief

    def logical_stage_for(self, step_name: str) -> str:
        return STEP_LOGICAL_STAGE.get(step_name, step_name)

    def note_board_section(self, step_name: str) -> str:
        lines = [
            "# Cross-Stage Note Board",
            "- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.",
        ]
        if self.stage_note_board:
            lines.append("- Notes so far:")
            for item in self.stage_note_board[-8:]:
                stage = item.get("stage") or "stage"
                stage_group = STEP_LOGICAL_STAGE_LABEL.get(str(item.get("stage_group") or ""), item.get("stage_group") or "")
                baseline_change = item.get("baseline_change") or "not recorded"
                purpose = item.get("purpose") or "not recorded"
                hardware_keys = item.get("hardware_keys") or []
                key_text = ", ".join(str(key) for key in hardware_keys[:8]) if hardware_keys else "none"
                prefix = f"{stage_group} / {stage}" if stage_group else str(stage)
                lines.append(
                    f"  - {prefix}: baseline_change={baseline_change}; purpose={purpose}; hardware_keys={key_text}"
                )
        else:
            current_group = STEP_LOGICAL_STAGE_LABEL.get(self.logical_stage_for(step_name), step_name)
            lines.append(f"- No prior notes yet. This step is part of {current_group}; write concise notes for later stages.")
        return "\n".join(lines) + "\n"

    def hardware_section_for_merge(self) -> str:
        sections: list[str] = []
        for step_name in (
            "model_design",
            "datatype_precision",
            "training_evaluation",
        ):
            section = self.hardware_stage_sections.get(step_name, "")
            if section.strip() and section not in sections:
                sections.append(section)
        return "\n".join(sections) or self.hardware_brief


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
            "2. A `Note board:` block with 1-3 bullets. Each bullet must use this exact shape: "
            "`- baseline_change: <what changed from the baseline or prior step> | purpose: <why this supports the candidate target> | hardware_keys: <comma-separated local feature keys or none>`\n"
            "3. A single markdown code block (wrapped in ```) containing ONLY the code for this step\n"
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

        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Data preview": data_preview_str,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Hardware/Profile Optimization Context": context.hardware_section_for(self.name),
            "Pipeline Decision Contract": context.pipeline_decision_section,
            "Cross-Stage Note Board": context.note_board_section(self.name),
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
        note_board_section = prompt.get("Cross-Stage Note Board", "")

        previous_solution_section = ""
        if context.stage == "improve" and "Previous solution" in prompt:
            previous_solution_section = f"\n# Previous solution\n{prompt['Previous solution']['Code']}\n"

        user_prompt = (
            f"\n# Task description\n{prompt['Task description']}\n\n"
            f"{memory_section}\n"
            f"{hardware_section}\n"
            f"{pipeline_decision_section}\n"
            f"{note_board_section}\n"
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

        has_datatype_step = any(result.get("name") == "datatype_precision" for result in step_results)
        execution_flow = (
            "data processing & feature engineering -> model design -> datatype/precision policy -> training & evaluation"
            if has_datatype_step
            else "data processing & feature engineering -> model design -> training & evaluation"
        )
        conflict_rule = (
            "- Resolve conflicts between steps by following the earlier step's design: model_design defines the model/loss/interface, datatype_precision defines precision policy, and training_evaluation consumes both."
            if has_datatype_step
            else "- Resolve conflicts between steps by following the earlier step's design (e.g., model_design defines the model, training_evaluation trains it)"
        )

        prompt_instructions["Merge guidelines"] = [
            "- Combine all code sections into a single, runnable Python script",
            "- CRITICAL: You are a MERGER, not a designer. Faithfully integrate the code from all steps. Do NOT introduce new models, algorithms, or approaches that were not in the original steps.",
            "- Ensure variable names are consistent across steps",
            "- Remove duplicate imports and definitions",
            conflict_rule,
            f"- Ensure the execution flow is logical: {execution_flow}",
            "- Make sure the final code prints validation metric (must match task's Evaluation section) and saves submission.csv",
            "- The code should be a single-file Python program that can be executed as-is",
            "- Assume previous steps have NOT been executed; do not skip execution steps and only read files or outputs.",
            "- All parts must work together seamlessly",
            "- Use hardware context only to preserve compatible precision, batch, dataloader, and runtime choices. Do not redesign the selected model or optimizer during merge.",
            "- Never emit merge-conflict markers such as <<<<<<<, =======, or >>>>>>>. Resolve conflicting snippets into ordinary Python before returning the final code.",
        ]

        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Hardware/Profile Optimization Context": context.hardware_section_for_merge(),
            "Pipeline Decision Contract": context.pipeline_decision_section,
            "Cross-Stage Note Board": context.note_board_section("merge"),
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
        note_board_section = prompt.get("Cross-Stage Note Board", "")

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
            f"{note_board_section}\n"
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
        "Your responsibility: Stage 1 candidate construction. Load data from `./input`, clean, create features (preprocessing, encoding, augmentation), and split dataset into train/validation/test.",
        "Stage 1 flow: hardware context lookup, data processing, and model design together define this round's candidate before datatype and training stages refine execution details.",
        "CRITICAL: This step MUST include BOTH data loading AND feature engineering. Do NOT only load the raw data. You must actively create, transform, and enhance features to improve model performance.",
        "IMPORTANT: Apply feature engineering techniques such as feature scaling, encoding, transformation, and data augmentation methods appropriate for the task. Explore and implement feature engineering strategies that can enhance the model's ability to learn from the data.",
        "CRITICAL: Do NOT build models, write training code, choose optimizer, choose batch size, choose AMP/dtype policy, or perform evaluation. Focus ONLY on data preparation and feature engineering.",
        "Note board: record what changed from the baseline data path and why, so model_design can complete the Stage 1 candidate target.",
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
    datatype_guidelines: List[str] = []
    if hardware_aware:
        data_guidelines.append(
            "Hardware-aware data pipeline: when using GPU training, keep input resolution/sequence length configurable, use DataLoader settings compatible with the hardware brief, and prefer pin_memory/non-blocking transfers when tensors move to CUDA."
        )
        hardware_node_rule = (
            "Hardware graph contract: the stage-specific hardware node response is already attached in the Hardware/Profile Optimization Context. "
            "Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper guidance is needed, and never fetch external URLs."
        )
        data_guidelines.append(hardware_node_rule)
        model_guidelines = [
            "Your responsibility: Stage 1 hardware-aware candidate construction. Complete the candidate target by choosing the model architecture or available pretrained model family, loss function, model output interface, and any model-local regularization needed for the task.",
            "Stage 1 flow: consume the Stage 1 hardware context plus the data_processing notes; this model step finalizes the candidate that Stage 2 and Stage 3 must preserve.",
            "CRITICAL: Do NOT write the training loop, data processing, feature engineering, optimizer, scheduler, batch size, epoch count, learning rate, dataloader worker settings, gradient accumulation, checkpoint cadence, or precision policy.",
            "CRITICAL: Do NOT choose AMP, bf16, fp16, fp32, TF32, GradScaler, autocast, or tensor dtype settings in this step. The datatype_precision step owns those decisions.",
            "IMPORTANT: Consider the task's evaluation metric when designing the model. The model output format should be compatible with the required evaluation metric calculation.",
            "Hardware-aware model design: use the Hardware-Aware Model Design Brief to compare model families before choosing an architecture. Optimize for the task metric first, while treating lower training time and higher GPU utilization as persistent objectives.",
            "Use the brief's compact hardware node, available feature keys, and selected feature details only when they are relevant to architecture, layer choice, tensor-core-friendly dimensions, or model-family feasibility.",
            hardware_node_rule,
            "Note board: Stage 1 notes are mandatory. Record what changed from the baseline model/data design and the purpose, because later stages must align precision and training choices to this target.",
            "Do not invent pretrained checkpoint paths, Kaggle input directories, torch hub directories, or dummy model weights to satisfy hardware advice. If a model source is not explicitly available, choose an available baseline-compatible model family.",
        ]
        datatype_guidelines = [
            "Your responsibility: Stage 2 hardware-aware datatype and tensor precision policy. Define reusable precision settings such as DEVICE, USE_AMP, AMP_DTYPE, USE_TF32, GradScaler behavior, autocast helper/context, and full-precision fallback behavior.",
            "Read the Cross-Stage Note Board first and preserve the Stage 1 candidate target; precision choices should support that target instead of changing the model/data design.",
            "CRITICAL: Do NOT change model architecture, loss function, data features, preprocessing, optimizer, scheduler, batch size, epochs, learning rate, dataloader workers, gradient accumulation, checkpoint cadence, validation metric, or submission logic.",
            "Use the Hardware/Profile Optimization Context to choose among fp32, tf32, fp16, bf16, or disabled AMP. Prefer bf16 only when hardware and framework evidence support it; use GradScaler for fp16 when needed; keep fp32 fallback for fragile losses or unsupported devices.",
            hardware_node_rule,
            "Note board: record how the precision policy supports the Stage 1 target and which feature keys drove the choice.",
            "Keep precision configurable and backend-compatible. Do not hardcode scheduler backend choices, CUDA process modes, CUDA streams, or MPS behavior.",
            "Expose simple variables/utilities that the training_evaluation step can consume directly, and include lightweight logging of selected precision without batch-level noise.",
        ]
        training_guidelines = [
            "Your responsibility: Stage 3 hardware-aware training hyperparameter optimization. Write the training loop using the data/features, model, loss/interface, and datatype_precision variables from previous steps. Define optimizer, learning rate, weight decay, scheduler, physical batch size, gradient accumulation, dataloader workers, checkpoint cadence, validation, test inference, and `submission.csv`.",
            "Read the Cross-Stage Note Board first and align optimizer, batch, dataloader, runtime, and fallback choices with the Stage 1 candidate target and Stage 2 precision policy.",
            "CRITICAL: Assume all previous code steps have already been executed. Do NOT redefine or reload data/features, redesign the model/loss, or replace the datatype_precision policy. Consume those variables and utilities AS-IS.",
            "CRITICAL: Own training hyperparameters in this step: batch size, effective batch size, accumulation steps, epochs, learning rate, weight decay, scheduler, early stopping, dataloader workers, pin_memory, persistent_workers, checkpointing, and runtime logging.",
            "CRITICAL: Use the datatype_precision variables/utilities for autocast, GradScaler, TF32, and fallback handling. Do NOT choose a different dtype policy unless the previous precision settings are impossible to use, and then keep a safe fallback.",
            hardware_node_rule,
            "Note board: record how training/runtime choices preserve the Stage 1 target and use the Stage 2 precision policy.",
            "CRITICAL: Validation metric computation must use the same prediction method as test inference, using training data only as reference, to avoid data leakage and ensure the metric reflects true generalization performance.",
            "CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled can cause large performance gaps between validation and test sets.",
            "CRITICAL: You MUST actively prevent overfitting. Use appropriate regularization, early stopping, and validation discipline without overfitting to the validation set.",
            "CRITICAL: You MUST implement the exact evaluation metric as specified in the task description's 'Evaluation' section. Do not use dummy, simplified, or approximate metrics.",
            "CRITICAL: The final line must be: `print(f'Final Validation Score: {score}')`. This is required for the score parser.",
        ]
        training_guidelines.extend(
            [
                "Hardware-aware training: optimize runtime at fixed modeling intent. Do NOT increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization unless the user explicitly asks for score improvement.",
                "Allowed hardware optimizations in this stage: physical batch size, gradient accumulation while preserving effective batch size, dataloader workers, pin_memory, persistent_workers, channels_last, safe torch.compile, checkpointing, and runtime logging. Precision choices must consume the datatype_precision policy.",
                "Scheduler-aware training: adapt to the scheduler backend config in the Hardware/Profile Optimization Context. Do not hardcode CUDA process, CUDA stream, MPS, or backend selection in the script; keep code backend-compatible and configurable.",
                "Hardware-aware training: use the hardware/profile context to choose physical batch size, accumulation, checkpoint cadence, and dataloader settings. If choosing a riskier setting for score reasons, include an explicit fallback path for OOM/timeout such as smaller batch size, accumulation, lower resolution, fewer epochs, or checkpoint resume.",
                "When feasible, log resolved batch size, selected precision, elapsed time, throughput, and peak CUDA memory so later scheduler graph evidence can learn from this run.",
            ]
        )
    if pipeline_decision_aware:
        data_guidelines.extend(
            [
                "Pipeline decision: consume `datatype.modality`, `datatype.target_type`, and `datatype.shape_constraints` before choosing preprocessing or feature engineering.",
                "Pipeline decision: use `tuning.dataloader_policy` only for data loading mechanics; do not choose the model family or optimizer in this step.",
            ]
        )
        if hardware_aware:
            model_guidelines.extend(
                [
                    "Pipeline decision: consume `datatype`, `model`, and `optimizer.loss` to define the model interface and criterion only; leave optimizer, scheduler, precision, and batch policy to later hardware-aware stages.",
                    "Pipeline decision: do not use unavailable weights, packages, or hardware-only tricks to justify a model family.",
                ]
            )
            datatype_guidelines.extend(
                [
                    "Pipeline decision: consume `tuning.precision_policy`, `evidence`, and the Hardware/Profile Optimization Context to define only dtype, AMP/TF32, GradScaler, autocast, and precision fallback behavior.",
                    "Pipeline decision: do not change model family, loss, optimizer, scheduler, dataloader policy, batch size, or validation/submission behavior in this step.",
                ]
            )
            training_guidelines.extend(
                [
                    "Pipeline decision: consume `optimizer.optimizer`, `optimizer.scheduler`, `tuning.batch_size_policy`, `tuning.dataloader_policy`, `tuning.fallbacks`, and `evidence` for training, validation, checkpointing, submission generation, and runtime logging.",
                    "Pipeline decision: when evidence is missing, implement the recorded safe fallbacks rather than inventing hardware claims.",
                ]
            )
        else:
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
    step_agents = [
        StepAgent(
            name="data_processing_and_feature_engineering",
            introduction="You are a Data Preparation Specialist responsible for data loading, cleaning, and feature engineering.",
            description="Load data from `./input` directory, perform cleaning, feature engineering, and create train/validation/test splits.",
            guidelines=data_guidelines,
        ),
        StepAgent(
            name="model_design",
            introduction="You are a Model Architect responsible for designing the model architecture, loss function, and output interface.",
            description="Design the model architecture (including pretrained models), loss function, and output interface.",
            guidelines=model_guidelines,
        ),
    ]
    if hardware_aware:
        step_agents.append(
            StepAgent(
                name="datatype_precision",
                introduction="You are a Hardware Precision Specialist responsible for tensor datatype and mixed-precision policy.",
                description="Define device, tensor dtype, AMP/autocast, TF32, GradScaler, fallback, and precision logging utilities.",
                guidelines=datatype_guidelines,
            )
        )
    step_agents.append(
        StepAgent(
            name="training_evaluation",
            introduction="You are a Training and Evaluation Expert responsible for implementing training, validation, and submission generation.",
            description="Implement the training loop, validation, metric tracking, model saving, and generate submission file.",
            guidelines=training_guidelines,
        ),
    )
    return step_agents


def _build_stepwise_metadata(
    *,
    step_results: List[Dict[str, str]],
    context: StepwiseContext,
    hardware_aware: bool,
) -> Dict[str, Any]:
    decisions = []
    for result in step_results:
        step_name = result.get("name", "")
        hardware_section = context.hardware_section_for(step_name)
        decisions.append(
            {
                "stage": step_name,
                "plan": result.get("plan", ""),
                "code_chars": len(result.get("code") or ""),
                "hardware_context_used": bool(hardware_section.strip()),
                "stage_group": STEP_LOGICAL_STAGE.get(step_name, step_name),
                "stage_notes": list(result.get("stage_notes") or []),
            }
        )
    return {
        "stage": context.stage,
        "hardware_aware": hardware_aware,
        "step_order": [result.get("name", "") for result in step_results],
        "decisions": decisions,
        "hardware_candidate": dict(context.hardware_candidate or {}),
        "hardware_context_keys": sorted((context.hardware_context or {}).keys()),
        "stage_note_board": list(context.stage_note_board or []),
    }


def _extract_stage_note_board(
    *,
    step_name: str,
    plan_text: str,
    code: str,
    context: StepwiseContext,
) -> List[Dict[str, Any]]:
    block_lines = _note_board_block_lines(plan_text)
    entries: List[Dict[str, Any]] = []
    for line in block_lines[:3]:
        parsed = _parse_note_board_line(line)
        if not parsed:
            continue
        parsed["stage"] = step_name
        parsed["stage_group"] = context.logical_stage_for(step_name)
        entries.append(parsed)
    if entries:
        return entries
    fallback = _fallback_stage_note(step_name=step_name, plan_text=plan_text, code=code, context=context)
    return [fallback] if fallback else []


def _note_board_block_lines(plan_text: str) -> List[str]:
    lines = str(plan_text or "").splitlines()
    collecting = False
    block: List[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:\d+\.\s*)?(?:#{1,6}\s*)?note board\s*:?\s*$", stripped, flags=re.IGNORECASE):
            collecting = True
            continue
        if not collecting:
            continue
        if re.match(r"^#{1,6}\s+\S+", stripped) and "note board" not in stripped.lower():
            break
        match = re.match(r"^[-*]\s+(.*)$", stripped)
        if match:
            block.append(match.group(1).strip())
        elif stripped and block:
            break
    return block


def _parse_note_board_line(line: str) -> Dict[str, Any]:
    parts: Dict[str, str] = {}
    for chunk in re.split(r"\s+\|\s+", line):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        parts[normalized_key] = value.strip()
    baseline_change = parts.get("baseline_change") or parts.get("change") or ""
    purpose = parts.get("purpose") or parts.get("why") or ""
    hardware_keys = _parse_note_hardware_keys(parts.get("hardware_keys") or parts.get("feature_keys") or "")
    if not baseline_change and not purpose:
        baseline_change = line.strip()
    return {
        "baseline_change": _compact_note_text(baseline_change),
        "purpose": _compact_note_text(purpose),
        "hardware_keys": hardware_keys,
    }


def _parse_note_hardware_keys(value: str) -> List[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "n/a", "na", "-"}:
        return []
    text = text.strip("[]()")
    keys: List[str] = []
    for item in re.split(r"[,;]", text):
        key = item.strip().strip("`'\"")
        if key and key.lower() not in {"none", "n/a", "na"} and key not in keys:
            keys.append(key)
    return keys[:8]


def _fallback_stage_note(
    *,
    step_name: str,
    plan_text: str,
    code: str,
    context: StepwiseContext,
) -> Dict[str, Any] | None:
    summary = _first_sentence(plan_text) or "Generated this stage's implementation."
    if not summary and not code:
        return None
    return {
        "stage": step_name,
        "stage_group": context.logical_stage_for(step_name),
        "baseline_change": _compact_note_text(summary),
        "purpose": _default_note_purpose(step_name),
        "hardware_keys": _infer_hardware_keys_from_text(plan_text)[:8],
    }


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"(?is)(?:\d+\.\s*)?(?:#{1,6}\s*)?note board\s*:.*$", "", str(text or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
    return (match.group(1) if match else cleaned[:180]).strip()


def _default_note_purpose(step_name: str) -> str:
    if step_name in {"data_processing_and_feature_engineering", "model_design"}:
        return "Define the Stage 1 candidate target for later precision and training stages."
    if step_name == "datatype_precision":
        return "Support the Stage 1 candidate target with a compatible precision policy."
    if step_name == "training_evaluation":
        return "Train and evaluate the Stage 1 candidate using the selected precision policy."
    return "Preserve cross-stage implementation intent."


def _infer_hardware_keys_from_text(text: str) -> List[str]:
    keys: List[str] = []
    for match in re.finditer(r"`([a-zA-Z0-9][a-zA-Z0-9_:-]{1,80})`", str(text or "")):
        key = match.group(1)
        if "_" in key and key not in keys:
            keys.append(key)
    return keys


def _compact_note_text(text: str, *, limit: int = 220) -> str:
    cleaned = re.sub(r"https?://\S+", "", str(text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def stepwise_plan_and_code_query(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    return_metadata: bool = False,
    ) -> Tuple[str, str] | Tuple[str, str, Dict[str, Any]]:
    logger.info("Using stepwise generation route.")

    stepwise_context = StepwiseContext(
        stage=context.get("stage", "draft"),
        memory=context.get("memory", ""),
        previous_code=context.get("previous_code", ""),
        execution_output=context.get("execution_output", ""),
        hardware_brief=context.get("hardware_prompt_section", ""),
        hardware_stage_sections=context.get("hardware_stage_sections", {}) or {},
        hardware_candidate=context.get("hardware_candidate", {}) or {},
        hardware_context=context.get("hardware_context", {}) or {},
        pipeline_decision=context.get("pipeline_decision", {}) or {},
        pipeline_decision_section=context.get("pipeline_decision_section", ""),
        stage_note_board=list(context.get("stage_note_board", []) or []),
    )

    hardware_aware = _hardware_reasoning_enabled(agent_instance)
    step_agents = create_default_step_agents(
        hardware_aware=hardware_aware,
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
        stage_notes = _extract_stage_note_board(
            step_name=agent.name,
            plan_text=plan,
            code=code,
            context=stepwise_context,
        )
        if stage_notes:
            stepwise_context.stage_note_board.extend(stage_notes)
            step_results[-1]["stage_notes"] = stage_notes

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
    if return_metadata:
        return final_plan, final_code, _build_stepwise_metadata(
            step_results=step_results,
            context=stepwise_context,
            hardware_aware=hardware_aware,
        )
    return final_plan, final_code
