"""Hardware-only advice. AST evidence is a hint, never proof of execution."""

import ast
import hashlib
import re


SOURCES = {
    "precision": "https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html",
    "padding": "https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/data_collator",
    "transfer": "https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
}


def guidance_v4(stage, task, code, device, edges, max_chars=1600):
    try:
        tree = ast.parse(code)
        parsed = True
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
        parsed = False
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
             if isinstance(n, (ast.Name, ast.Attribute))}
    imports = {n.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.alias)}
    neural = bool(imports & {"torch", "transformers", "timm"} or
                  names & {"Module", "Sequential", "Trainer", "AutoModel", "create_model"} or
                  any(n.startswith("AutoModelFor") for n in names))
    unknown = not code or not parsed
    text = bool(re.search(r"\b(text|tweets?|sentiment|tokens?|nlp)\b", str(task), re.I) or
                names & {"AutoTokenizer", "DataCollatorWithPadding", "tokenizer"})
    verified = [key for key in ("feat:amp", "feat:bf16") if
                edges.get(key, {}).get("properties", {}).get("verified") is True and
                edges[key]["properties"].get("support_level") == "supported"]
    bf16 = set(verified) == {"feat:amp", "feat:bf16"}
    lines = [
        f"HWKG v4 | {device.name} | {device.total_memory // 1024**2} MiB total, not free.",
        "Hardware advice only; task/stage requirements take precedence. Keep the prescribed metric, "
        "split, seeds, threshold, truncation and training budget; do not add folds/ensembles for hardware use.",
        "Applicability is inferred from parent code/task, not execution. Recheck for your chosen model; "
        "CPU methods need no CUDA. No promised gain.",
    ]
    if stage == "debug":
        lines.append("Fix the reported failure only; defer unrelated performance changes.")
    cards = []
    if neural or unknown:
        precision = ("For CUDA training keep FP32 parameters. " +
                     ("BF16 autocast is optional if torch.cuda.is_bf16_supported(); otherwise FP32. "
                      if bf16 else "No verified BF16 recommendation here; use FP32. ") +
                     "Before full training check finite loss, gradients AND post-step parameters; "
                     "stop on failure. Verify the exact model/tokenizer loads first. "
                     "No blanket FP16 or silent model fallback.")
        cards.append(("precision", precision))
        if text and stage != "debug":
            cards.append(("padding", "If tokenizing text, cache unpadded tokens once; pad to each batch's "
                          "longest sequence with all masks/labels aligned in train/validation/test. "
                          "Preserve truncation and order; extra padding multiples require measurement."))
        if code and stage != "debug":
            cards.append(("transfer", "If moving CPU batches to CUDA, test pin_memory + non_blocking "
                          "within CPU quota; avoid per-step .item()/cpu() synchronization. "
                          "Existing flags alone do not prove transfer overlap."))
    measurement = (
        "Log HWKG_DECISION: advice IDs, applied/skipped and reasons. If training on CUDA, log "
        "HWKG_RUNTIME: actual dtype/batch and a warm, normal-training window with CUDA sync at BOTH "
        "boundaries, elapsed seconds, examples and non-padding tokens (if relevant) counted IN that "
        "window, peak allocated bytes; total time separately. Missing throughput=unknown. "
        "Keep the required final metric line last."
    )
    selected, omitted = [], []
    for key, advice in cards:
        line = f"[{key}] {advice}"
        if len("\n".join(lines + [line, measurement])) <= max_chars:
            lines.append(line)
            selected.append(key)
        else:
            omitted.append(key)
    lines.append(measurement)
    result = "\n".join(lines)
    if len(result) > max_chars:
        raise ValueError("HWKG mandatory context exceeds character budget")
    return result, {
        "policy": "hardware-v4", "advice_ids": selected, "omitted_advice_ids": omitted,
        "evidence_sources": {k: SOURCES[k] for k in selected},
        "verified_hardware_feature_ids": verified,
        "candidate_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "applicability_basis": "parent_ast_and_task_inference", "ast_parsed": parsed,
        "neural_candidate_or_unknown": neural or unknown,
        "precision_guard": "precision" in selected,
    }
