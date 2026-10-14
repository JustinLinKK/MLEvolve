"""Capability-guided candidates for this host, distinct from measured projections.

Advice only: no candidate/source/reward/API modifications or extra benchmark work.
Exact calibration remains optional evidence for claiming a measured projection.
"""
import ast
import re
from .adaptive_policy import source_configured
from .source_analysis import precision_scope
from .device_support import options, architecture, capability, runtime_label


def optimizer_option(code):
    """One literal CUDA AdamW call; groups do not borrow a timing record."""
    try: tree=ast.parse(code)
    except SyntaxError: return None
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and ast.unparse(n.func)=='torch.optim.AdamW']
    if len(calls)!=1: return None
    if any(isinstance(n,ast.Call) and getattr(n.func,'id',getattr(n.func,'attr','')) in
           {'SAM','first_step','second_step'} for n in ast.walk(tree)): return None
    options={'fused':None,'foreach':None,'differentiable':False,'capturable':False}
    try:
        for k in calls[0].keywords:
            if k.arg in options: options[k.arg]=ast.literal_eval(k.value)
    except (ValueError,TypeError): return None
    if (any(type(v) is not bool and v is not None for v in options.values())
            or options['foreach'] is True or options['differentiable'] is True
            or options['capturable'] is True): return None
    return options


def guidance(stage, code, device, edges, runtime, *, phase, parent_execution_succeeded,
             blocked_profiles=(), identity_already_present=False, max_chars=750, contract_note=''):
    if phase=='merge' or stage=='debug': return None
    host=options(device,edges,runtime)
    if host['target']=='unknown' or host['backend'] not in ('cuda','rocm'): return None
    bf16,fp16,fused=host['bf16'],host['fp16'],host['fused']
    target=architecture(device)
    if not code:
        if stage!='draft': return None
        facts={'compute_capability':capability(device),'backend':host['backend']}
        if identity_already_present:
            header=f'HWKG v8.19 {target} options:'
        else:
            facts.update(gpu_name=device.name,cuda_runtime_version=runtime.get('cuda'),
                         hip_runtime_version=runtime.get('hip'),
                         torch_version=runtime.get('torch'),total_vram_mib=device.total_memory//1024**2)
            header=(f'HWKG v8.19 {device.name} {target}; {runtime_label(device,runtime)}; '
                    f'Torch{runtime.get("torch","unknown")}; VRAMtotal{facts["total_vram_mib"]}MiB.')
        if bf16:
            precision='TRAIN BF16/FP32 params; FP16 needs GradScaler; keep eval dtype.'
        elif fp16:
            precision='TRAIN FP16 autocast/GradScaler, FP32 params; keep eval dtype.'
        else:
            precision='Use FP32; AMP eligibility not established; keep eval dtype.'
        lines=[header,'[precision_choice] '+precision]
        ids=['precision_choice'];features=[host['amp_feature']] if host['amp'] else []
        if bf16 and host['backend']=='cuda': features.append('feat:bf16')
        if fused and bf16:
            lines.append('[optimizer_choice] GPU AdamW: try fused=True (FP32/BF16).')
            ids.append('optimizer_choice');features.append(host['fused_feature'])
        lines.append('Unmeasured speed/quality; same budget/outputs; no extra passes.')
        if contract_note:
            lines[-1]='Unmeasured; same budget/outputs; no extra passes.'
            lines.append(contract_note)
        text='\n'.join(lines)
        # Long catalogue names must not suppress all advice. Compact only the
        # display name; retain the complete verified identity in the audit.
        if len(text)>min(max_chars,340) and not identity_already_present:
            lines[0]=f'HWKG v8.19 {target}; {runtime_label(device,runtime)}; VRAM{facts["total_vram_mib"]}MiB.'
            header=lines[0];text='\n'.join(lines)
        if len(text)>min(max_chars,340): return None
        return text, {'advice_ids':ids,'selected_feature_ids':features,
                      'draft_recipe_forced':False,'draft_identity_facts_deduplicated':identity_already_present,
                      'hardware_fact_values':facts,'hardware_fact_source':'visible CUDA device and installed runtime; not task preset',
                      'host_header_chars':len(header)+1,'advice_body_chars':len('\n'.join(lines[1:])),
                      'host_facts_and_advice_counted_in_total':True,
                      'stage_role':phase,'context_budget':min(max_chars,340),
                      'proposal_kind':'capability_guided_unmeasured','calibration_trial':None,
                      'measurements_verified':False,'end_to_end_gain_verified':False,
                      'declared_or_measured_cost_gate_applied':False,'added_provider_calls':0}
    if parent_execution_succeeded is not True: return None
    # Do not make hardware suggestions for an observed CPU-only implementation.
    if not re.search(r'\btorch\b',code) or not re.search(r'cuda|\.to\(.*DEV|\.to\(.*device',code,re.I): return None
    scope=precision_scope(code);types=scope.get('training_dtypes',[])
    configured=source_configured(code);opt=optimizer_option(code)
    ids=[];features=[];body=None;profile=None
    if (bf16 and fused and opt and opt['fused'] in (None,False) and types==['bfloat16']
            and not scope.get('training_scaler') and 'optimizer_fused_adamw' not in blocked_profiles
            and not re.search(r'\.half\s*\(|set_default_dtype',code)):
        ids=['optimizer_fused_adamw'];features=[host['fused_feature']];profile='optimizer_fused_adamw'
        body='Try only fused=True on existing GPU AdamW with FP32 parameters/BF16; retain groups/LR/WD. Check finite ordinary updates. Speed/quality unmeasured.'
    elif (fp16 and types==['bfloat16'] and opt and opt['fused'] in (None,False)
            and 'fp16_'+('channels_last' if configured.get('channels_last') else 'contiguous') not in blocked_profiles):
        ids=['precision'];features=[host['amp_feature']];profile='fp16_'+('channels_last' if configured.get('channels_last') else 'contiguous')
        body='Trial TRAIN-only torch.float16 autocast with GradScaler on FP32 parameters; unscale before existing clipping. Keep eval dtype separate and AdamW unfused. Track successful finite updates; retain BF16 if unstable. Speed/quality unmeasured.'
    elif (bf16 and types==['float16'] and opt
            and 'bf16_'+('channels_last' if configured.get('channels_last') else 'contiguous') not in blocked_profiles
            and not re.search(r'\.half\s*\(|set_default_dtype',code)):
        ids=['precision'];features=[host['amp_feature']]+(['feat:bf16'] if host['backend']=='cuda' else []);profile='bf16_'+('channels_last' if configured.get('channels_last') else 'contiguous')
        body='Trial TRAIN-only torch.bfloat16 autocast with FP32 parameters and ordinary backward/step; disable TRAIN GradScaler. Preserve shared eval precision and optimizer options. Check finite ordinary updates. Candidate speed/quality unverified.'
    elif (host['amp'] and not types and not re.search(r'\b(autocast|half|bfloat16|set_default_dtype)\b',code)
            and opt and opt['fused'] in (None,False)):
        dtype='bf16' if bf16 else 'fp16' if fp16 else None
        profile=(dtype+'_'+('channels_last' if configured.get('channels_last') else 'contiguous')) if dtype else None
        if profile and profile not in blocked_profiles:
            ids=['precision'];features=[host['amp_feature']]+(['feat:bf16'] if bf16 and host['backend']=='cuda' else [])
            body=('Trial TRAIN-only torch.bfloat16 autocast, FP32 parameters and ordinary backward/step. '
                  if bf16 else 'Trial TRAIN-only torch.float16 autocast with GradScaler and FP32 parameters. ')
            body+='Keep eval precision, optimizer and finite ordinary updates unchanged. Speed/quality unmeasured.'
    if not body: return None
    text=(f'HWKG v8.19 {target} {phase}: capability-guided hypothesis, no gain projection.\n'
          'Preserve batch/data order/epochs/recipe/eval/checkpoint and original response format. '
          'Use ordinary budgeted training; no extra benchmark or BN pass.\n'
          + (contract_note+'\n' if contract_note else '')+
          f'[{ids[0]}] {body}')
    if len(text)>min(max_chars,750): return None
    return text,{'advice_ids':ids,'selected_feature_ids':features,'stage_role':phase,
                 'proposal_kind':'capability_guided_unmeasured','calibration_trial':None,
                 'exploration_trial':{'profile':profile,'change':'optimizer_implementation' if ids[0]=='optimizer_fused_adamw' else 'training_precision',
                                      'quality_verified':False,'end_to_end_verified':False},
                 'context_budget':min(max_chars,750),'measurements_verified':False,
                 'end_to_end_gain_verified':False,'declared_or_measured_cost_gate_applied':False,
                 'added_provider_calls':0}
