"""Host-specific interventions selected using observed parent execution.

Source syntax, enabled flags and observed training remain distinct. A trusted
external runtime adapter is optional; without it measurements stay unknown.
"""
import ast
import hashlib
import json
import math
import re

from .host_policy import guidance_v6


def parent_feedback(code, text, device):
    records = []
    for line in text.splitlines():
        if line.startswith('MLEVOLVE_RUNTIME_JSON '):
            try:
                records.append(json.loads(line.removeprefix('MLEVOLVE_RUNTIME_JSON ')))
            except (ValueError, TypeError):
                pass
    if not records:
        return None, 'no shared runtime receipt; measurements unknown'
    row = records[-1]
    if row.get('origin') != 'shared-parent-execution-adapter' or row.get('schema') not in ('shared-runtime-v1', 'shared-runtime-v2'):
        return None, 'unrecognized receipt origin/schema'
    if row.get('code_sha256') != hashlib.sha256(code.encode()).hexdigest():
        return None, 'receipt source identity mismatch'
    gpu = row.get('gpu') or {}
    if gpu.get('name') != device.name or gpu.get('capability') != [device.major, device.minor]:
        return None, 'receipt GPU identity mismatch'
    if not row.get('execution_succeeded') or row.get('exception') or not row.get('first_step_finite'):
        return None, 'parent correctness gate failed; defer performance tuning'
    contract = row.get('input_contract') or {}
    if contract.get('status') == 'failed' or contract.get('violations'):
        return None, 'parent correctness gate failed; runtime input contract violated'
    return row, 'matched parent source, successful execution, GPU and first finite update'


def source_configured(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    return {'bf16': 'bfloat16' in attrs and 'autocast' in attrs,
            'channels_last': 'channels_last' in attrs,
            'tf32': 'allow_tf32' in attrs or 'set_float32_matmul_precision' in attrs}


def guidance_v7(stage, task, code, device, hardware, feature_nodes, edges, runtime, max_chars=1600,
                *, phase='plan_and_code', proposed_plan='', failure_text='', optimization_mode='normal'):
    base, audit = guidance_v6(stage, task, code, device, hardware, feature_nodes, edges, runtime, 10000,
                              phase=phase, proposed_plan=proposed_plan, failure_text=failure_text,
                              optimization_mode=optimization_mode)
    receipt, basis = parent_feedback(code, failure_text, device)
    configured = source_configured(code)
    audit.update(policy='hardware-v7', policy_revision='v7.3', source_configured=configured,
                 source_configuration_is_runtime_proof=False, parent_feedback_basis=basis,
                 matched_parent_receipt=receipt is not None, suppressed_advice_ids=[])
    # A successful numeric receipt does not certify output quality or task
    # correctness. Debug is invoked for those failures too; do not turn them
    # into an unsupported NaN/BF16 hardware diagnosis.
    if stage == 'debug' and receipt is not None:
        audit.update(advice_ids=[], decision_provenance=[], selected_feature_ids=[],
                     verified_hardware_feature_ids=[],
                     skip_reason='parent execution/numerics passed; task/output debug has no evidenced hardware repair')
        return None, audit
    if base is None or (stage == 'debug' and receipt is None) or basis.startswith('parent correctness gate failed'):
        audit.update(advice_ids=[], decision_provenance=[], selected_feature_ids=[],
                     verified_hardware_feature_ids=[],
                     skip_reason=basis if stage == 'debug' or basis.startswith('parent correctness gate failed')
                     else audit.get('skip_reason'))
        return None, audit
    cards = {m.group(1): m.group(2) for line in base.splitlines()
             if (m := re.match(r'^\[([^]]+)\] (.*)$', line))}
    provenance = {p['id']: p for p in audit['decision_provenance']}
    observed = (receipt or {}).get('training_observations') or []
    training = observed[0] if observed else {}
    windows = (receipt or {}).get('optimizer_windows') or []
    window = windows[0] if len(windows) == 1 and receipt.get('window_validity') == 'single_optimizer_observed' else None
    # Legacy windows did not observe freeze/unfreeze throughout the interval.
    # Their raw timing is preserved, but cannot drive measured performance cards.
    if window and not (receipt.get('schema') == 'shared-runtime-v2'
                       and window.get('eligible_for_comparison') is True
                       and not window.get('ineligible_reasons')
                       and window.get('configuration_count') == 1
                       and window.get('backward_calls_per_update') == [1] * 25
                       and (window.get('training') or {}).get('training_scope') == 'full_model'
                       and (receipt.get('input_contract') or {}).get('status') == 'observed_pass'
                       and window.get('single_forward_per_update') and window.get('successful_updates') == 25
                       and window.get('gradient_updates_observed') == 25
                       and isinstance(window.get('examples_per_second'), (int, float))
                       and math.isfinite(window['examples_per_second']) and window['examples_per_second'] > 0
                       and isinstance(window.get('wall_seconds'), (int, float))
                       and math.isfinite(window['wall_seconds']) and window['wall_seconds'] > 0):
        window = None
    if window:
        training = window.get('training') or training
    audit.update(runtime_parent_training=training or None, runtime_parent_window=window)
    # Executed configuration takes precedence over syntax in dead/unused paths.
    precision_done = (training.get('autocast_dtype') == 'torch.bfloat16'
                      if training else configured.get('bf16'))
    layout_done = (bool(training.get('input_channels_last') and training.get('model_channels_last'))
                   if training else configured.get('channels_last'))
    selected = []
    for key in ('precision', 'vision_layout'):
        done = precision_done if key == 'precision' else layout_done
        if done:
            cards.pop(key, None)
            audit['suppressed_advice_ids'].append(key)
    # A capacity reminder is useful on a fresh draft. For existing code, memory
    # tuning requires observed allocation, instead of repeating the reminder.
    if code:
        cards.pop('memory', None)
        audit['suppressed_advice_ids'].append('memory')
    props = hardware['properties']
    patterns = props.get('recommended_patterns', [])
    target = f'sm_{device.major}{device.minor}'
    peak = (receipt or {}).get('peak_allocated_bytes')
    def add(key, text, indices, fields):
        cards[key] = text
        provenance[key] = {'id': key, 'feature_ids': [],
                           'hardware_fields': fields, 'pattern_indices': indices,
                           'graph_patterns': [patterns[i] for i in indices],
                           'graph_feature_sources': {},
                           'reason': 'matched parent runtime observation and selected host graph pattern',
                           'runtime_measured': True}
    if window:
        reads = window.get('cuda_scalar_reads') or {}
        purposes = window.get('cuda_scalar_read_purposes') or {}
        logging_reads = purposes.get('loss_scalar', 0)
        audit['scalar_read_attribution'] = {'purposes': purposes, 'total_reads': reads,
            'basis': 'observed call site in exact executed source; loss values are not guaranteed dispensable'}
        indices = [i for i, p in enumerate(patterns) if 'avoid per-step CPU-GPU synchronization' in p]
        if indices and logging_reads >= window['successful_updates']:
            add('measured_sync', f'{target} parent window: {logging_reads} source-attributed CUDA loss scalar reads over 25 updates, '
                f'{window["wall_seconds"]:.3f}s. Trial tensor-aggregated loss/logging with sparse host reads; '
                'retain correctness checks. Counts show reads, not stall duration; verify complete-step speed and metric.',
                indices, ['compute_capabilities'])
        # Only recommend a memory-capacity experiment when there is actual
        # evidence of headroom and a host graph memory/batching pattern.
        indices = [i for i, p in enumerate(patterns) if 'physical batch' in p or 'activation memory' in p]
        if indices and isinstance(peak, int) and 0 < peak < .45 * device.total_memory:
            add('measured_batch', f'{target} / {device.total_memory // 1024**2} MiB: parent observed peak '
                f'{peak // 1024**2} MiB, physical batch {window["training"]["batch"]}, '
                f'{window["examples_per_second"]:.0f} observed examples/s. A larger physical batch is a trial '
                'only if it preserves the intended effective batch/update schedule; otherwise keep it. '
                'Recheck live free memory and quality, not merely GPU busy%.', indices, ['vram_MB'])
    # One performance intervention per generation call; architecture-specific
    # custom-kernel and numerical guard cards retain higher priority.
    for key in ('diagnose', 'kernel_target', 'measured_sync', 'precision', 'vision_layout', 'measured_batch',
                'attention', 'batching', 'memory', 'nvfp4', 'mxfp8', 'fp8'):
        if key in cards:
            selected.append(key)
            if key not in ('diagnose', 'kernel_target'):
                break
    if not selected:
        audit.update(advice_ids=[], decision_provenance=[], selected_feature_ids=[], verified_hardware_feature_ids=[],
                     skip_reason='no new applicable host intervention; spend no injection tokens')
        return None, audit
    lines = [f'HWKG v7 {phase}/{stage} | {device.name} {target}; {device.total_memory // 1024**2} MiB.',
             'Preserve task metric/split/seed/threshold/budget. Choose one applicable host intervention.']
    if training:
        lines.append(f'Observed parent: {training.get("autocast_dtype")}, input/model channels_last '
                     f'{training.get("input_channels_last")}/{training.get("model_channels_last")}; '
                     'enabled flags do not prove executed TF32 kernels.')
    else:
        lines.append('Parent syntax is configuration evidence; runtime measurements remain unknown.')
    kept = []
    for key in selected:
        text = f'[{key}] {cards[key]}'
        if len('\n'.join(lines + [text])) + 120 <= max_chars:
            lines.append(text);kept.append(key)
    lines.append('Use actual runtime counters to accept/reject this trial; never replace missing measurements with claims.')
    if not kept:
        return None, {**audit, 'advice_ids': [], 'decision_provenance': [], 'skip_reason': 'context budget'}
    audit.update(advice_ids=kept, omitted_advice_ids=sorted(set(cards)-set(kept)),
                 decision_provenance=[provenance[k] for k in kept],
                 selected_feature_ids=sorted({fid for k in kept for fid in provenance[k].get('feature_ids', [])}),
                 runtime_parent_training=training or None, runtime_parent_window=window,
                 suppressed_advice_ids=sorted(set(audit['suppressed_advice_ids'])))
    audit['precision_guard'] = 'precision' in kept
    audit['local_policy_advice_ids'] = []
    audit['evidence_sources'] = {k: audit.get('evidence_sources', {}).get(k, props.get('source_url')) for k in kept}
    audit['verified_hardware_feature_ids'] = audit['selected_feature_ids']
    return '\n'.join(lines), audit
