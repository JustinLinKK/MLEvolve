"""Component-only phase advice; no executor/code repair or extra API calls."""
import re
import os
from .adaptive_policy import source_configured
from .source_analysis import precision_scope
from .native_evidence import number
from .host_policy import guidance_v6
from .native_evidence import native_feedback
from .calibration import select_record, choose_trial, calibration_path
from .contracts import scope_reminder
from .device_support import architecture, capability as device_capability, runtime_label


def guidance_v8(stage, task, code, device, hardware, feature_nodes, edges, runtime, max_chars=1600,
                *, phase='plan_and_code', proposed_plan='', failure_text='', optimization_mode='normal',
                parent_id=None, parent_execution_succeeded=None, blocked_profiles=(), provider_prompt=None):
    limit=min(max_chars,750)
    contract_note=scope_reminder(task,provider_prompt)
    evidence=native_feedback(code,failure_text,device,parent_id=parent_id,
                             execution_succeeded=parent_execution_succeeded)
    base,eligible=guidance_v6(stage,task,code,device,hardware,feature_nodes,edges,runtime,10000,
        phase=phase,proposed_plan=proposed_plan,failure_text=failure_text,optimization_mode=optimization_mode)
    audit={**eligible}
    audit.update(policy='hardware-v8',policy_revision='v8.19',native_parent_evidence=evidence,
        measurements_verified=False,end_to_end_gain_verified=False,calibration_basis=None,
        calibration_trial=None,added_provider_calls=0,blocked_profiles=list(blocked_profiles),
        shared_contract_reminder=contract_note,contract_runtime_enforced=False)
    def skip(reason):
        audit.update(advice_ids=[],decision_provenance=[],selected_feature_ids=[],
                     verified_hardware_feature_ids=[],precision_guard=False,skip_reason=reason)
        return None,audit
    def explore(identity_already_present=False):
        from .exploration_policy import guidance
        proposal=guidance(stage,code,device,edges,runtime,phase=phase,
            parent_execution_succeeded=parent_execution_succeeded,
            blocked_profiles=blocked_profiles,identity_already_present=identity_already_present,
            max_chars=limit,contract_note=contract_note)
        if not proposal:return None
        text,info=proposal
        features=info['selected_feature_ids']
        info.update(verified_hardware_feature_ids=features,
            decision_provenance=[{'id':key,'feature_ids':features,
                'reason':'matched host graph and installed backend/API; candidate benefit unverified'} for key in info['advice_ids']],
            exploration_basis='hardware capability and optional software API, not calibrated candidate cost',
            precision_guard='precision' in info['advice_ids'],skip_reason=None)
        return text,{**audit,**info}
    from .optimizer_policy import guidance as optimizer_guidance
    optimizer_trial = optimizer_guidance(stage, task, code, device, edges, runtime,
        phase=phase, parent_execution_succeeded=parent_execution_succeeded,
        blocked_profiles=blocked_profiles, provider_prompt=provider_prompt, max_chars=limit,
        contract_note=contract_note)
    if optimizer_trial:
        text, measured = optimizer_trial
        return text,{**audit, **measured, 'native_parent_evidence':evidence}
    if phase=='merge':return skip('merge preserves upstream output; no new hardware trial')
    if stage=='debug':
        if not re.search(r'CUDA out of memory|no kernel image|CUBLAS_STATUS_|CUDNN_STATUS_',failure_text,re.I):
            return skip('debug has no hardware-specific failure evidence')
    if parent_execution_succeeded is False and stage!='debug':return skip('failed parent: defer performance tuning')
    if not code and stage!='debug':
        # The upstream cleaner may remove host facts from agent.task_desc while
        # data preview puts them back into the actual provider prompt. Inspect
        # delivery, not that intermediate representation. Never log its text.
        capability = device_capability(device)
        cuda_build = (('rocm'+str(runtime.get('hip','')).replace('.', '')) if runtime.get('hip')
                      else 'cu'+str(runtime.get('cuda','')).replace('.', ''))
        if isinstance(provider_prompt, str):
            identity_text = provider_prompt
        elif isinstance(provider_prompt, dict):
            identity_text = '\n'.join(provider_prompt[k] for k in ('system','user','assistant')
                                      if isinstance(provider_prompt.get(k), str))
        else:
            identity_text = str(task) if provider_prompt is None else ''
        audit['draft_identity_check_basis'] = 'actual provider prompt' if provider_prompt is not None else 'direct policy task input'
        identity_present=(device.name in identity_text and 'capability '+capability in identity_text
                and cuda_build != 'cu' and cuda_build in identity_text)
        proposal=explore(identity_present)
        if proposal:return proposal
        if identity_present:
            return skip('actual provider prompt already supplies this host identity; defer hardware trial until a usable parent')
        # A parentless draft has no measured workload or recipe to preserve.
        # Supply verified host facts, not a compulsory BF16 experiment plus a
        # second timing/evaluation protocol the task already specifies.
        modes=[];features=[]
        amp=edges.get('feat:amp',{}).get('properties',{})
        bf16=edges.get('feat:bf16',{}).get('properties',{})
        if amp.get('verified') is True and amp.get('support_level')=='supported':
            modes.append('FP16');features.append('feat:amp')
        if bf16.get('verified') is True and bf16.get('support_level')=='supported' and runtime.get('bf16_supported'):
            modes.append('BF16');features.append('feat:bf16')
        if not modes:return skip('no verified host AMP facts for a parentless draft')
        layout=' CUDA CNNs can use model/input channels_last.' if 'vision_layout' in audit.get('advice_ids',[]) else ''
        text=(f'HWKG v8.19: {device.name}, {architecture(device)}, {runtime_label(device,runtime)}.\n'
              f'[hardware_target] Native AMP supports {"/".join(modes)}.{layout} '
              'No dtype winner is established for this draft. Keep the task and original response format.')
        if len(text)>min(limit,340):return skip('compact draft context budget')
        audit.update(advice_ids=['hardware_target'],decision_provenance=[{'id':'hardware_target','feature_ids':features,
            'reason':'verified host/runtime capability facts; no source-matched trial'}],selected_feature_ids=features,
            verified_hardware_feature_ids=features,precision_guard=False,stage_role=phase,context_budget=min(limit,340),
            draft_recipe_forced=False,calibration_basis='parentless draft: workload match unavailable')
        return text,audit
    configured=source_configured(code);spec=evidence['source_spec']
    audit.update(source_configured=configured,source_configuration_is_runtime_proof=False)
    record,basis=select_record(calibration_path(device),device,runtime,spec)
    if evidence['window'] and evidence['reasons']:
        record=None;basis='native work scope cannot support matched full-model calibration'
    audit['calibration_basis']=basis
    audit['calibration_file_sha256']=(record or {}).get('calibration_file_sha256')
    precision=precision_scope(code)
    types=precision.get('training_dtypes', [])
    dtype={'float16':'fp16','bfloat16':'bf16','float32':'fp32'}.get(types[0]) if len(types)==1 else None
    if not code: dtype='fp32'
    elif not types and not re.search(r'\b(autocast|half|bfloat16)\b',code): dtype='fp32'
    current=(dtype+'_'+('channels_last' if configured.get('channels_last') else 'contiguous')) if dtype else None
    profile_basis='source syntax only; conditional projection'
    if evidence['window']:
        win=evidence['window']; profile_basis='native self-report; not runtime-certified'
        dtype={'bfloat16':'bf16','torch.bfloat16':'bf16','float16':'fp16','torch.float16':'fp16',
               'float32':'fp32','torch.float32':'fp32','disabled':'fp32'}[win['autocast_dtype']]
        current=(dtype+'_'+('channels_last' if win['model_layout']=='channels_last' else 'contiguous')) if win['model_layout']==win['input_layout'] else None
    rows=re.search(r'train\.csv\s*\(\s*(\d+)\s*rows',str(task),re.I)
    updates=int(rows[1])//spec['batch']*spec['epochs'] if rows and spec['epochs'] and spec['batch'] else None
    try: reserve=float(os.environ.get('MLEVOLVE_HWKG_COST_RESERVE_SECONDS','5'))
    except ValueError: return skip('invalid declared hardware context cost reserve')
    if not number(reserve): return skip('invalid declared hardware context cost reserve')
    trial=choose_trial(record,current,updates,reserve,blocked_profiles)
    audit.update(calibration_trial=trial,projected_updates=updates,
        current_profile=current,current_profile_basis=profile_basis,
        declared_context_cost_reserve_seconds=reserve,
        generation_overhead_basis='declared reserve, not a measured marginal cost; token/reasoning latency unknown')
    order=('diagnose','kernel_target','precision','vision_layout','attention','batching','memory','nvfp4','mxfp8','fp8')
    ids=[k for k in order if k in (audit.get('advice_ids') or [])]
    ids=[k for k in ids if not (k=='precision' and configured.get('bf16'))
         and not (k=='vision_layout' and configured.get('channels_last')) and not (k=='memory' and code)]
    cards={m[1]:m[2] for line in (base or '').splitlines() if (m:=re.match(r'^\[([^]]+)\] (.*)$',line))}
    provenance={p['id']:p for p in audit.get('decision_provenance',[])}
    ids=[k for k in ids if k not in ('measured_sync','measured_batch')]
    eligible_cards={p['id']:p for p in eligible['decision_provenance']}
    if stage!='debug' and evidence['window'] and not evidence['reasons']:
        # A checked self-report can contradict unused source flags. It permits a
        # conditional implementation trial, never a measured-gain assertion.
        win=evidence['window']
        done={'precision':win['autocast_dtype'] in ('bfloat16','torch.bfloat16'),
              'vision_layout':win['model_layout']==win['input_layout']=='channels_last'}
        ids=[k for k in ids if not done.get(k,False)]
        if not any(k in ids for k in ('diagnose','kernel_target')):
            for key in ('precision','vision_layout'):
                if not done[key] and key in eligible_cards:
                    ids=[key];provenance[key]=eligible_cards[key];break
    if trial:
        target='vision_layout' if trial['change']=='layout' else 'precision'
        match=eligible_cards.get(target)
        if match:ids=[target];provenance[target]=match
        else:return skip('calibrated trial has no applicable host-graph implementation')
    elif record and updates is not None and current:
        proposal=explore()
        if proposal:return proposal
        return skip('matched calibration predicts no profitable single trial')
    # A known low-precision parent is not grounds for an unmeasured dtype swap.
    if not trial and current and current.startswith(('fp16_', 'bf16_')):
        ids=[k for k in ids if k!='precision']
    fallback_target='bf16_'+('channels_last' if configured.get('channels_last') else 'contiguous')
    if not trial and fallback_target in blocked_profiles: ids=[k for k in ids if k!='precision']
    if not ids:
        proposal=explore()
        if proposal:return proposal
        return skip('no unadopted applicable host intervention; no eligible capability-guided hypothesis')
    key=ids[0]
    target_dtype=(trial['profile'].split('_',1)[0] if trial else 'bf16')
    if key=='precision':
        support=edges.get('feat:bf16',{}).get('properties',{})
        if target_dtype=='fp32': return skip('no measured higher-precision intervention implemented')
        amp=edges.get('feat:amp',{}).get('properties',{})
        if target_dtype=='fp16' and not (amp.get('verified') is True and amp.get('support_level')=='supported'):
            return skip('no verified FP16 AMP implementation for this visible host/runtime')
        if target_dtype=='bf16' and not (runtime.get('bf16_supported') and support.get('verified') is True
                and support.get('support_level')=='supported'):
            return skip('no verified BF16 implementation for this visible host/runtime')
        if key in provenance:
            provenance[key]={**provenance[key], 'feature_ids':[f for f in provenance[key].get('feature_ids',[])
                if f in (('feat:bf16','feat:amp') if target_dtype=='bf16' else ('feat:amp',))]}
    lines=[f'HWKG v8.19 {phase}: {device.name}, {architecture(device)}.',
        'Preserve recipe/RNG/sample order/epochs, eval/test precision and original response format. '
        'This hardware trial changes only the selected item; keep EMA/checkpoint selection unchanged. '
        'No extra training/benchmark/BN-stat passes.']
    if contract_note:lines.append(contract_note)
    if trial:
        lines.append(f'Matched microcalibration: {trial["step_speedup"]:.2f}x steps; projected net '
            f'{trial["estimated_net_seconds"]:.1f}s after setup/{reserve:g}s reserve. Quality/full-pipeline gain unverified.')
    elif evidence['window']:
        lines.append('Native self-report; counter math checked, runtime/gain unverified.')
    else:lines.append('Source syntax only; gain unknown.')
    if phase=='planner':
        body={'precision':f'Plan one {target_dtype.upper()} TRAIN-only trial; retain FP32 parameters.',
              'vision_layout':'Plan one CNN model+input channels_last trial; retain batch/recipe.'}.get(key,cards.get(key,''))
        body+=' Estimate setup + full candidate cost.'
    elif key=='vision_layout':
        body=('model.to(memory_format=torch.channels_last); use the same format on existing batch transfer. '
              'Inspect is_contiguous(memory_format=torch.channels_last), never a .memory_format attribute.')
    elif key=='precision':
        dtype_name='float16' if target_dtype=='fp16' else 'bfloat16'
        body=(f"FP32 parameters; TRAIN forward/loss only: torch.autocast('cuda', dtype=torch.{dtype_name}). "
              + ('Use GradScaler; unscale before existing gradient clipping. ' if target_dtype=='fp16' else 'Use ordinary backward/step; disable TRAIN GradScaler. ')
              + 'Keep shared eval autocast helper unchanged. '
              'Layout: is_contiguous(memory_format=torch.channels_last), never a .memory_format attribute.')
    else:body=cards.get(key,'')
    lines.append(f'[{key}] {body}')
    text='\n'.join(lines)
    if not body or len(text)>limit:return skip('stage context budget; do not truncate output requirements')
    audit.update(advice_ids=[key],decision_provenance=[provenance[key]] if key in provenance else [],
        selected_feature_ids=provenance.get(key,{}).get('feature_ids',[]),
        verified_hardware_feature_ids=provenance.get(key,{}).get('feature_ids',[]),
        precision_guard=key=='precision',stage_role=phase,context_budget=limit)
    return text,audit
