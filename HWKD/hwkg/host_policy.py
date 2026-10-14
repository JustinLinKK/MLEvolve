"""Graph-selected host decisions, not a GPU name pasted onto generic advice."""

import ast
import hashlib
import importlib.util
import re

from .staged_policy import source_adoption
from .device_support import architecture, backend as device_backend


SOURCES = {
    'precision': 'https://docs.pytorch.org/docs/stable/amp.html',
    'tf32': 'https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html',
    'attention': 'https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html',
    'kernel_target': 'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
}


def runtime_facts():
    """Read capabilities only; do not benchmark, install, or change backend flags."""
    import torch
    import inspect
    try: fused_api = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    except (TypeError,ValueError): fused_api = False
    try: fused_kernel = torch._C._dispatch_has_kernel_for_dispatch_key('aten::_fused_adamw_', 'CUDA')
    except (AttributeError,RuntimeError): fused_kernel = False
    try: amp_api = torch.amp.autocast_mode.is_autocast_available('cuda')
    except AttributeError: amp_api = hasattr(torch, 'autocast')
    try: bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError: bf16 = False
    free, total = torch.cuda.mem_get_info(0)
    return {
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'hip': getattr(torch.version,'hip',None),
        'backend': 'rocm' if getattr(torch.version,'hip',None) else 'cuda',
        'arch_list': torch.cuda.get_arch_list(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'bf16_supported': bf16, 'amp_api': amp_api,
        'adamw_fused_api': fused_api,
        'adamw_fused_kernel': fused_kernel,
        'grad_scaler_api': hasattr(torch.amp, 'GradScaler'),
        'free_mib_snapshot': free // 1024**2, 'total_mib': total // 1024**2,
        'sdpa_api': hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
        'flash_sdp_built': torch.backends.cuda.is_flash_attention_available(),
        'transformer_engine_package': importlib.util.find_spec('transformer_engine') is not None,
        'torchao_package': importlib.util.find_spec('torchao') is not None,
    }


def workload(task, code, plan):
    try:
        tree = ast.parse(code)
        names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree)
                 if isinstance(n, (ast.Name, ast.Attribute))}
        imports = {n.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.alias)}
        strings = ' '.join(n.value for n in ast.walk(tree)
                           if isinstance(n, ast.Constant) and isinstance(n.value, str))
        parsed = True
    except SyntaxError:
        names, imports, strings, parsed = set(), set(), '', False
    hint = str(task) + ' ' + plan + ' ' + strings
    transformer = bool(re.search(r'\b(transformer|roberta|deberta|distilbert|bert|language model)\b', hint, re.I)
                       or names & {'AutoModel', 'AutoModelForSequenceClassification', 'TransformerEncoderLayer'})
    neural = bool(imports & {'torch', 'transformers', 'timm'} or
                  names & {'Module', 'Trainer', 'Sequential', 'Linear', 'Conv2d'} or
                  transformer or re.search(r'\b(neural|cuda|torch|cnn|unet)\b', plan, re.I))
    return {'neural': neural, 'unknown': not code or not parsed, 'transformer': transformer,
            'text': bool(re.search(r'\b(text|tweets?|nlp|tokens?|sentiment)\b', hint, re.I)),
            'vision': bool(re.search(r'\b(images?|vision|cnn|unet)\b', hint, re.I) or 'Conv2d' in names),
            'long_lm': bool(re.search(r'\b(language model|autoregressive|next.token)\b', hint, re.I)
                            and re.search(r'\b(pretrain\w*|long|many steps|billion|million|corpus)\b', hint, re.I)),
            'custom_kernel': bool(imports & {'triton'} or names & {'load_inline', 'CUDAExtension'}),
            'ast_parsed': parsed}


def guidance_v6(stage, task, code, device, hardware, feature_nodes, edges, runtime, max_chars=1600,
                *, phase='plan_and_code', proposed_plan='', failure_text='', optimization_mode='normal'):
    if optimization_mode not in {'normal', 'aggressive'}:
        raise ValueError('Unknown HWKG optimization mode')
    props = hardware['properties']; hid = hardware['id']
    target = architecture(device)
    rocm = device_backend(device,runtime)=='rocm'
    w = workload(task, code, proposed_plan)
    rejected = {}
    all_features = sorted(edges)
    patterns = props.get('recommended_patterns', [])

    def feature(fid):
        e = edges.get(fid, {}).get('properties', {})
        return e.get('support_level') == 'supported' and e.get('verified') is True

    def pattern(*terms):
        return [i for i, p in enumerate(patterns) if all(t.casefold() in p.casefold() for t in terms)]

    def card(key, text, *, features=(), fields=(), pattern_ids=(), reason='', priority=50):
        return {'id': key, 'text': text, 'feature_ids': list(features), 'hardware_fields': list(fields),
                'pattern_indices': list(pattern_ids), 'reason': reason, 'priority': priority}

    # A concrete CPU implementation has no applicable CUDA implementation advice.
    # Planning can still use the host budget when selecting a subsequent model.
    active = w['neural'] or w['unknown'] or phase == 'planner'
    audit = {'policy': 'hardware-v6', 'phase': phase, 'hardware_node': hid,
             'graph_feature_inventory': all_features, 'runtime_facts': runtime, 'workload': w,
             'source_observations': source_adoption(code),
             'plan_sha256': hashlib.sha256(proposed_plan.encode()).hexdigest(),
             'failure_sha256': hashlib.sha256(failure_text.encode()).hexdigest(),
             'candidate_code_sha256': hashlib.sha256(code.encode()).hexdigest(),
             'applicability_basis': 'host_graph_runtime_parent_ast_and_unverified_plan_hint',
             'optimization_mode': optimization_mode}
    if not active:
        audit.update(advice_ids=[], omitted_advice_ids=[], selected_feature_ids=[],
                     verified_hardware_feature_ids=[], decision_provenance=[], rejected_features={
                         fid: 'concrete CPU candidate; no CUDA implementation advice' for fid in all_features},
                     skip_reason='no applicable hardware implementation intervention', precision_guard=False)
        return None, audit

    arch = '/'.join(props.get('architectures', []))
    bw = props.get('memory_bandwidth_GBps')
    free = runtime.get('free_mib_snapshot')
    free_text = f'; free snapshot {free} MiB, recheck at execution' if free is not None else ''
    lines = [f'HWKG v6 {phase}/{stage} | {device.name} {arch} {target}; '
             f'{device.total_memory // 1024**2} MiB total{free_text}.',
             'Preserve task metric/split/seed/threshold/budget. Choose host-fit implementations; '
             'capability is not a measured gain.']
    cards = []
    if stage == 'debug' or re.search(r'\bnan\b|non.?finite', failure_text, re.I):
        cards.append(card('diagnose', 'Locate first bad parameters/loss/grads/update; stop before inference. '
                          'BF16 is not a NaN repair. No untested LR/dtype cause.', priority=110))
    if phase == 'merge':
        cards.append(card('merge', 'Retain selected host kernels, dtype guards, masks and counters.', priority=105))
    if phase == 'planner':
        cards.append(card('plan', 'Use the host options below to choose an executable model/batch/kernel '
                          'within budget; distinguish API wait from GPU cost.', priority=100))
    if w['custom_kernel']:
        limits = (' 99 KiB/block shared-memory limit, 48 resident warps/SM; no sm_100a assumptions.'
                  if target == 'sm_120' and 'feat:sm_120' in edges else '')
        cards.append(card('kernel_target', f'Custom {"HIP/ROCm" if rocm else "CUDA"}: build for {target}; verify the package supports '
                          f'this target before using it.{limits}', features=['feat:'+target] if 'feat:'+target in edges else [],
                          fields=['compute_capabilities'], pattern_ids=pattern('custom CUDA'),
                          reason='candidate contains custom-kernel code on this architecture', priority=100))

    precision_features = []
    choices = []
    if feature('feat:amp') and feature('feat:bf16') and runtime.get('bf16_supported') is True:
        choices.append('prefer a BF16-autocast trial for eligible Tensor Core ops')
        precision_features += ['feat:amp', 'feat:bf16']
    else:
        rejected['feat:bf16'] = 'verified host AMP/BF16 edge or runtime BF16 support absent'
    if not rocm and feature('feat:tf32'):
        choices.append("FP32 path: test torch.set_float32_matmul_precision('high') (TF32-capable math)")
        precision_features.append('feat:tf32')
    if choices:
        cards.append(card('precision', f'{target}: '+ '; '.join(choices)+
                          '. Keep parameters explicitly FP32; check loaded params and finite loss/grads/first post-step params; '
                          'keep strict FP32 fallback.', features=precision_features,
                          fields=['compute_capabilities','datatypes'], reason='verified precision edges plus runtime support', priority=95))
    else:
        rejected['precision'] = 'no eligible verified host precision intervention'

    if w['transformer'] and runtime.get('sdpa_api') and 'feat:'+target in edges:
        cards.append(card('attention', f'{target}: try framework SDPA for compatible attention; verify '
                          'shape/backend eligibility. Preserve mask/bias semantics, eval dropout=0; '
                          'retain original attention if not equivalent.', features=['feat:'+target],
                          fields=['compute_capabilities'], reason='host target node plus installed framework API; shape eligibility unverified', priority=85))
    if pattern('activation checkpointing') or pattern('gradient accumulation'):
        cards.append(card('memory', f'{device.total_memory // 1024**2} MiB host: budget parameters+grads+optimizer'
                          '+activations; size physical batch from measured peak and live free memory. '
                          'Checkpoint/accumulate only if memory-limited.', fields=['vram_MB'],
                          pattern_ids=pattern('activation checkpointing')+pattern('gradient accumulation'),
                          reason='this GPU memory capacity and matching graph patterns', priority=80))
    if w['text'] and (pattern('variable-length') or pattern('sequence')):
        cards.append(card('batching', f'{props.get("vram_type", "GPU memory")}, {bw} GB/s graph bandwidth: '
                          'cache tokens; batch-pad aligned masks/labels, preserve truncation. '
                          'Use the largest measured-safe physical batch within task limits; avoid extra passes.',
                          fields=['vram_type','memory_bandwidth_GBps','vram_MB'],
                          pattern_ids=pattern('variable-length')+pattern('sequence'),
                          reason='variable-length text and host memory/bandwidth profile', priority=75))
    if w['vision'] and pattern('channels_last'):
        cards.append(card('vision_layout', f'{target} CNN path: trial channels_last model+inputs; '
                          'benchmark complete steps against current layout within memory budget.',
                          fields=['compute_capabilities'], pattern_ids=pattern('channels_last'),
                          reason='CNN/image workload matches the host graph layout pattern', priority=85))

    low_precision = ['feat:fp8', 'feat:mxfp8', 'feat:nvfp4', 'feat:fp8_rowwise_scaling']
    for fid in low_precision:
        if fid not in edges:
            continue
        backend = 'torchao_package' if fid == 'feat:fp8_rowwise_scaling' else 'transformer_engine_package'
        if not feature(fid):reason='host feature edge not verified/supported'
        elif not runtime.get(backend):reason='required backend not installed'
        elif not w['long_lm']:reason='long transformer/LM workload not established; conversion cost unamortized'
        elif optimization_mode != 'aggressive':reason='graph recipe requires explicit aggressive mode'
        else:
            recipe = {'feat:fp8':'DelayedScaling/HYBRID', 'feat:mxfp8':'MXFP8BlockScaling',
                      'feat:nvfp4':'NVFP4BlockScaling', 'feat:fp8_rowwise_scaling':'torchao ROWWISE'}[fid]
            cards.append(card(fid.removeprefix('feat:'), f'{target} conditional trial: {recipe}; '
                              'verify exact backend target/module/shape support and finite optimizer step; '
                              'compare end-to-end cost/quality to BF16 before retaining.', features=[fid],
                              fields=['compute_capabilities','datatypes'], reason='host feature, installed backend and long LM; actual recipe unvalidated', priority=90))
            continue
        rejected[fid]=reason
    for fid in all_features:
        if fid not in rejected and not any(fid in c['feature_ids'] for c in cards):
            rejected[fid]='no applicable single-host intervention selected for this workload/stage'

    measurement = ('Log HWKG_DECISION IDs+reason, actual dtype/backend. HWKG_RUNTIME: sync both ends, '
                   'completed examples/nonpadding tokens/time/peak bytes/skips in same normal window. '
                   'Final metric last.')
    # Failed numerical debugging should not consume tokens on unrelated tuning.
    if stage == 'debug':
        removed=[c for c in cards if c['id'] not in {'diagnose','precision','kernel_target','merge','plan'}]
        cards=[c for c in cards if c not in removed]
        stage_omitted=[c['id'] for c in removed]
    else:stage_omitted=[]
    selected=[]; omitted=[]
    for c in sorted(cards, key=lambda c: -c['priority']):
        line=f"[{c['id']}] {c['text']}"
        if len('\n'.join(lines+[line,measurement])) <= max_chars:
            lines.append(line);selected.append(c)
        else:omitted.append(c['id'])
    if not selected:
        audit.update(advice_ids=[],omitted_advice_ids=omitted,selected_feature_ids=[],
                     verified_hardware_feature_ids=[],decision_provenance=[],rejected_features=rejected,
                     skip_reason='no applicable host card fits',precision_guard=False)
        return None,audit
    lines.append(measurement)
    text='\n'.join(lines)
    if len(text)>max_chars:raise ValueError('HWKG host context exceeds budget')
    selected_features=sorted({fid for c in selected for fid in c['feature_ids']})
    audit.update(advice_ids=[c['id'] for c in selected],omitted_advice_ids=omitted,
                 stage_omitted_advice_ids=stage_omitted,selected_feature_ids=selected_features,
                 verified_hardware_feature_ids=[fid for fid in selected_features if feature(fid)],
                 decision_provenance=[{**{k:v for k,v in c.items() if k not in {'text','priority'}},
                     'graph_patterns':[patterns[i] for i in c['pattern_indices']],
                     'graph_feature_sources':{fid:feature_nodes.get(fid,{}).get('properties',{}).get('source_url')
                                              for fid in c['feature_ids']}}
                                     for c in selected],
                 rejected_features=rejected,precision_guard=any(c['id']=='precision' for c in selected),
                 local_policy_advice_ids=[c['id'] for c in selected if not c['feature_ids'] and not c['hardware_fields']],
                 evidence_sources={c['id']: SOURCES.get(c['id'], props.get('source_url')) for c in selected
                                   if c['feature_ids'] or c['hardware_fields']})
    return text,audit
