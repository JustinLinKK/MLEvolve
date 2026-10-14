"""Base code generation mode (single-shot plan + code).

The simplest generation strategy: one LLM call produces a natural language
plan followed by a complete code block. Used as the default / fallback mode
when diff or stepwise generation is not enabled or fails.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from llm import generate
from utils.response import extract_code, extract_text_up_to_code

logger = logging.getLogger("MLEvolve")


def _retry_prompt_after_missing_code(prompt: Any, attempt: int) -> Any:
    """Make a deterministic retry request meaningfully different from its first try."""
    instruction = (
        "\n\nYour previous response did not contain a complete Python markdown code block. "
        f"Retry {attempt}: return a brief plan followed by one complete ```python fenced script, "
        "including its closing fence. Do not stop after explaining the dataset."
    )
    if isinstance(prompt, str):
        return f"{prompt}{instruction}"
    if isinstance(prompt, dict):
        retry_prompt = dict(prompt)
        for key in ("user", "system", "assistant"):
            if retry_prompt.get(key):
                retry_prompt[key] = f"{retry_prompt[key]}{instruction}"
                return retry_prompt
        retry_prompt["user"] = instruction.lstrip()
        return retry_prompt
    if isinstance(prompt, list):
        retry_prompt = [dict(message) if isinstance(message, dict) else message for message in prompt]
        retry_prompt.append({"role": "user", "content": instruction.lstrip()})
        return retry_prompt
    return prompt


# ============ Response format prompt (rewrite mode specific) ============

RESPONSE_FORMAT = {
    "Response format": (
        "Your response should be a brief outline/sketch of your proposed solution in natural language, "
        "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
        "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    )
}


def plan_and_code_query(
    agent_instance,
    prompt,
    retries: int = 3,
) -> Tuple[str, str]:
    """Generate plan + code in one LLM call; returns (nl_text, code). On failure returns ("", raw_completion_text)."""
    completion_text = None
    request_prompt = prompt
    for attempt in range(retries):
        completion_text = generate(
            prompt=request_prompt,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
            context_cache_stable_prefix=(
                prompt.get("system") if isinstance(prompt, dict) else None
            ),
        )
        code = extract_code(completion_text)
        nl_text = extract_text_up_to_code(completion_text)

        if code and nl_text:
            return nl_text, code

        logger.debug("Extraction retry...")
        request_prompt = _retry_prompt_after_missing_code(prompt, attempt + 1)

    logger.warning("Code extraction failed after retries")
    return "", completion_text  # type: ignore
