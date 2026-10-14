"""Opt-in stage policy; recommendations and source hints are not measurements."""
import ast
import hashlib
import re

from .policy import SOURCES, guidance_v4


def source_adoption(code):
    """Conservative syntax observations, never a claim about executed branches."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {'ast_parsed': False, 'runtime_verified': False}
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
             for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    return {'ast_parsed': True, 'runtime_verified': False,
            'finite_check_call_present': 'isfinite' in calls,
            'dtype_read_present': any(isinstance(n, ast.Attribute) and n.attr == 'dtype' for n in ast.walk(tree)),
            'autocast_call_present': 'autocast' in calls,
            'synchronize_call_present': 'synchronize' in calls,
            'padding_collator_call_present': 'DataCollatorWithPadding' in calls,
            'post_step_guard_verified': False}


def guidance_v5(stage, task, code, device, edges, max_chars=1600, *,
                phase='plan_and_code', proposed_plan='', failure_text=''):
    _, evidence = guidance_v4(stage, task, code, device, edges, max_chars)
    planned_neural = bool(re.search(r'\b(transformer|roberta|deberta|distilbert|torch|cuda|neural)\b', proposed_plan, re.I))
    neural = evidence['neural_candidate_or_unknown'] or planned_neural or phase == 'plan_and_code'
    bf16 = set(evidence['verified_hardware_feature_ids']) == {'feat:amp', 'feat:bf16'}
    lines = [f'HWKG v5 | {device.name} | {phase}/{stage}. Advice, not a measured gain.',
             'Task paths/limits are authoritative. Preserve metric, split, seed, threshold and budget. '
             'Do not replace the backbone/optimizer solely for hardware. CPU methods need no CUDA.']
    cards = []
    if phase == 'planner':
        cards.append(('plan', 'Choose at most one hardware change tied to observed cost; otherwise keep it. '
                      'Separate loading, API wait and training. Extra compute is not useful utilization. '
                      'A model change invalidates old applicability; recheck it before coding.'))
    elif phase == 'merge':
        cards.append(('merge', 'Preserve dtype casts, masks, numerical guards and aligned train/val/test '
                      'collation when merging. Retain actual measurement counters; do not substitute literal claims.'))
    if stage == 'debug' or re.search(r'\bnan\b|non-finite|nonfinite', failure_text, re.I):
        cards.append(('diagnose', 'Locate failure at loaded parameters, forward, gradients or update. '
                      'Do not assert a learning-rate/dtype cause without evidence; BF16 is not a NaN fix. '
                      'Reject corrupt parameters before inference; do not skip bad batches and print a score.'))
    if neural and phase != 'planner':
        cards.append(('precision', 'For CUDA, explicitly cast parameters to FP32 and log actual parameter dtype. '
                      'Check loaded parameters, then finite loss, gradients AND parameters after the first normal '
                      'optimizer step; raise on failure before continuing. '
                      + ('BF16 autocast is optional after a finite step check and torch.cuda.is_bf16_supported().' if bf16 else 'Use FP32; no verified BF16 advice.')))
        if re.search(r'\b(text|tweets?|nlp|tokens?)\b', str(task), re.I) and stage != 'debug':
            cards.append(('padding', 'Cache unpadded tokens once; batch-pad with aligned masks/labels and '
                          'identical val/test semantics. Preserve prescribed truncation.'))
    measurement = ('Log HWKG_DECISION IDs: applied/skipped + reason. For CUDA, measure a warm normal-training '
                   'window with sync at BOTH ends, actual completed examples/nonpadding tokens, elapsed and peak '
                   'allocated bytes; count skipped steps. Invalid training has no useful throughput. '
                   'Final required metric line stays last.')
    selected=[];omitted=[]
    # Numerical diagnosis/safety takes priority over optional performance cards.
    cards.sort(key=lambda item: {'diagnose':0,'precision':1}.get(item[0],2))
    for key, advice in cards:
        line=f'[{key}] {advice}'
        if len('\n'.join(lines+[line,measurement])) <= max_chars:
            lines.append(line);selected.append(key)
        else:omitted.append(key)
    lines.append(measurement)
    text='\n'.join(lines)
    if len(text)>max_chars:raise ValueError('HWKG mandatory stage context exceeds budget')
    evidence.update(policy='hardware-v5', phase=phase, advice_ids=selected,
                    omitted_advice_ids=omitted, precision_guard='precision' in selected,
                    evidence_sources={key: SOURCES[key] for key in selected if key in SOURCES},
                    local_policy_advice_ids=[key for key in selected if key not in SOURCES],
                    neural_candidate_or_unknown=neural, planned_neural_hint=planned_neural,
                    plan_sha256=hashlib.sha256(proposed_plan.encode()).hexdigest(),
                    failure_sha256=hashlib.sha256(failure_text.encode()).hexdigest(),
                    applicability_basis='parent_ast_and_unverified_plan_hint',
                    source_observations=source_adoption(code))
    return text,evidence
