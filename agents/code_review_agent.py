"""Code Review Agent: LLM-based code review and fix for node code."""

import logging
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from agents.hardware_context import (
    get_hardware_context_for_stage,
    hardware_context_instructions,
)
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import get_internet_clarification
from localml_scheduler.runtime_environment import validate_generated_training_code

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")


def _format_runtime_compatibility_findings(result: dict) -> str:
    issues = list((result or {}).get("issues") or [])
    if not issues:
        return ""
    lines = ["# Runtime Compatibility Findings", ""]
    for issue in issues:
        severity = str(issue.get("severity") or "warning").upper()
        lines.append(f"- {severity} [{issue.get('category', 'compatibility')}]: {issue.get('message', '')}")
        if issue.get("evidence"):
            lines.append(f"  Evidence: {issue['evidence']}")
        if issue.get("repair_hint"):
            lines.append(f"  Required fix: {issue['repair_hint']}")
    return "\n".join(lines)

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                "type": "string",
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")

    prompt = get_code_review_prompt(
        task_desc=agent.task_desc,
        code=node.code,
        data_preview=getattr(agent, "data_preview", "") or "",
    )
    compatibility_result = validate_generated_training_code(node.code, stage="code_review")
    critical_compatibility_count = int(compatibility_result.get("critical_count", 0) or 0)
    compatibility_section = _format_runtime_compatibility_findings(compatibility_result)
    if compatibility_section:
        prompt["Runtime Compatibility Findings"] = compatibility_section
        if "Instructions" not in prompt:
            prompt["Instructions"] = {}
        prompt["Instructions"]["Runtime compatibility fixes"] = [
            "The Runtime Compatibility Findings section is authoritative for this installed Python/PyTorch stack.",
            "Every CRITICAL finding must be fixed with a targeted SEARCH/REPLACE patch before execution.",
            "WARNING findings should be fixed when the change is local and does not alter the solution design.",
        ]
    hardware_ctx = get_hardware_context_for_stage(
        agent,
        "code_review",
        parent_node=getattr(node, "parent", None),
        code=node.code,
    )
    hardware_section = hardware_ctx.prompt_section
    if hardware_section:
        prompt_instructions = prompt.pop("Instructions")
        prompt["Hardware/Profile Optimization Context"] = hardware_section
        prompt["Instructions"] = prompt_instructions
        prompt["Instructions"] |= hardware_context_instructions(hardware_ctx)
        prompt["Instructions"]["Hardware-aware review guidance"] = [
            "Review concrete hardware-critical issues only when the provided profile evidence is strong.",
            "Examples: obvious OOM/timeout risk from an oversized fixed batch size, missing fallback around known risky settings, or ignoring a high-confidence precision/dataloader recommendation.",
            "Preserve the existing solution approach and model/backbone. Do not redesign the model to satisfy hardware guidance.",
        ]
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    if "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification

    use_diff_for_review = agent.acfg.use_diff_mode
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Code review retry attempt {attempt + 1}/{max_retries} for node {node.id}")
                time.sleep(5)

            review_response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=CODE_REVIEW_SPEC,
                    model=agent.acfg.code.model,
                    temperature=agent.acfg.code.temp,
                    cfg=agent.cfg
                ),
            )

            needs_revision = review_response.get("needs_revision", False)
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            if critical_compatibility_count and not needs_revision:
                if revised_code and revised_code.strip():
                    logger.warning(
                        "Code review returned revised code while needs_revision=False; "
                        "treating it as required because critical runtime compatibility findings exist."
                    )
                    needs_revision = True
                elif attempt < max_retries - 1:
                    logger.warning(
                        "Code review approved code despite critical runtime compatibility findings; "
                        "retrying with required-fix reminder."
                    )
                    prompt["Instructions"][f"Runtime compatibility retry {attempt + 1}"] = [
                        "Do not approve this code until every CRITICAL Runtime Compatibility Finding has a concrete SEARCH/REPLACE fix.",
                    ]
                    continue
                else:
                    logger.error(
                        "Code review approved code despite critical runtime compatibility findings after max retries; "
                        "returning original code because no usable patch was produced."
                    )
            logger.info(f"Code review for node {node.id}: needs_revision={needs_revision}")
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})
            try:
                from utils.pipeline_logging import log_pipeline_event, record_pipeline_node_action

                payload = {
                    "attempt": attempt + 1,
                    "needs_revision": bool(needs_revision),
                    "reasoning": reasoning,
                    "hardware_context_used": bool(hardware_section),
                    "runtime_compatibility_critical_count": compatibility_result.get("critical_count", 0),
                    "runtime_compatibility_warning_count": compatibility_result.get("warning_count", 0),
                    "revised_code_chars": len(revised_code or ""),
                }
                log_pipeline_event(agent, "code_review_completed", node=node, payload=payload)
                record_pipeline_node_action(agent, node, "code_review_completed", payload=payload)
            except Exception:
                pass

            if needs_revision:
                if revised_code and revised_code.strip():
                    if use_diff_for_review and (
                        "<<<<<<< SEARCH" in revised_code or "< SEARCH" in revised_code
                        ):
                        try:
                            logger.info("Code review returned diff format, applying patch")
                            patcher = SearchReplacePatcher()
                            patched_code, count = patcher.apply_patch(
                                revised_code, node.code, strict=False
                            )
                            if count > 0 and patched_code and patched_code != node.code:
                                logger.info(f"Successfully applied {count} review patch(es)")
                                return patched_code.strip()
                            logger.warning(
                                f"Diff patch failed (count={count}), keeping original code to avoid writing raw diff to runfile"
                            )
                            return node.code
                        except Exception as e:
                            logger.warning(
                                f"Failed to apply diff patch in code review: {e}, keeping original code to avoid writing raw diff to runfile"
                            )
                            return node.code
                    else:
                        # Full code revision (original behavior)
                        if use_diff_for_review:
                            return node.code
                        else:
                            logger.info("Using revised code from reviewer")
                            return revised_code.strip()

                if attempt < max_retries - 1:
                    logger.warning(f"Code review violation: needs_revision=True but revised_code is empty/None - Will retry ({attempt + 1}/{max_retries})")
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error(f"Code review violation: needs_revision=True but revised_code is empty/None - Max retries reached, returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                return node.code

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            return node.code

        except Exception as e:
            error_msg = f"Code review failed with exception: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"{error_msg} - Will retry (attempt {attempt + 1}/{max_retries})")
                continue
            logger.error(f"{error_msg} - Max retries reached, returning original code")
            return node.code

    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    return node.code
