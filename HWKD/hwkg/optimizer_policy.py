"""One source-scoped, host-matched optimizer implementation trial.

Microtiming is conditional evidence, never quality/full-pipeline gain proof.
Records must match their own host/runtime/BF16 workload; bundled R50 evidence
stays RTX5080-only. No source, provider request, native node or reward is modified.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from .adaptive_policy import source_configured
from .calibration import workload_key
from .native_evidence import source_spec, number
from .source_analysis import precision_scope
from .window_scope import training_window_scope
from .device_support import architecture, backend, options

BUNDLED_SHA256 = 'cf1d3ef1ab1b973036ae2ad304d965f4e84cba20a64ab088f9d9557e6e5ad875'
FEATURE = 'feat:optimizer_fused_adamw'


def load_record():
    configured=os.environ.get('MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH')
    if configured=='': return None
    path = Path(configured) if configured is not None else Path(__file__).with_name('calibrations')/'rtx5080-v814-r50-fused.json'
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size > 65536:
                return None
            raw = f.read(65537)
        digest=hashlib.sha256(raw).hexdigest()
        if configured is None and digest != BUNDLED_SHA256: return None
        record = json.loads(raw)
    except (OSError, ValueError, TypeError): return None
    if (not isinstance(record,dict) or record.get('schema') != 'hwkg-optimizer-calibration-v1'
            or record.get('quality_verified') is not False
            or record.get('full_pipeline_gain_verified') is not False): return None
    if (not isinstance(record.get('host'),dict) or not isinstance(record.get('workload'),dict)
            or not isinstance(record.get('runtime_flags'),dict)
            or not isinstance(record.get('profiles'),dict)
            or set(record['profiles'])!={'default','fused'}): return None
    for p in record['profiles'].values():
        if (not isinstance(p,dict) or not all(k in p for k in ('step_seconds','step_seconds_min','step_seconds_max',
                'setup_seconds_max','setup_seconds_min','repeats','normal_updates_per_trial','finite_assertions_completed'))
                or not all(number(p[k], True) for k in ('step_seconds','step_seconds_min','step_seconds_max'))
                or not p['step_seconds_min'] <= p['step_seconds'] <= p['step_seconds_max']
                or not all(number(p[k]) for k in ('setup_seconds_max','setup_seconds_min'))
                or p['setup_seconds_min'] > p['setup_seconds_max']
                or type(p['repeats']) is not int or type(p['normal_updates_per_trial']) is not int
                or p['repeats'] < 5 or p['normal_updates_per_trial'] < 200
                or p['finite_assertions_completed'] is not True): return None
    return {**record,'calibration_file_sha256':digest}


def guidance(stage, task, code, device, edges, runtime, *, phase='coder',
            parent_execution_succeeded=None, blocked_profiles=(), provider_prompt=None, max_chars=750,
            contract_note=''):
    if (not code or parent_execution_succeeded is not True or stage == 'debug'
            or phase == 'merge' or 'optimizer_fused_adamw' in blocked_profiles): return None
    if runtime.get('hardware_profile_identity_verified') is False: return None
    host_options=options(device,edges,runtime)
    if not host_options['bf16'] or not host_options['fused']: return None
    record = load_record()
    if not record: return None
    host = record['host']
    if (device.name != host['gpu_name'] or [device.major,device.minor] != host['capability']
            or any(runtime.get(k) != host[k] for k in ('torch','cuda'))): return None
    if backend(device,runtime)=='rocm' and (host.get('hip')!=runtime.get('hip')
            or host.get('gpu_architecture')!=architecture(device)): return None
    spec = source_spec(code)
    if spec['training_scope'] != 'full_model_source_screen': return None
    opts = spec.get('optimizer_options')
    if not isinstance(opts, dict) or not (opts.get('fused') is None or opts.get('fused') is False): return None
    spec = {**spec, 'optimizer_options':{**opts, 'fused':None}}
    key = workload_key(device, runtime, spec)
    if key['runtime_flags'] != record['runtime_flags']: return None
    if any(key.get(k) != v for k,v in record['workload'].items()): return None
    precision = precision_scope(code)
    if precision.get('training_dtypes') != ['bfloat16'] or precision.get('bf16_with_active_scaler'): return None
    if not source_configured(code).get('channels_last'): return None
    if re.search(r'\b(?:set_default_dtype|set_default_device|half|compile|triton|load_inline|CUDAGraph)\b',code): return None
    if training_window_scope(code)['status'] == 'standalone_gradient_window_before_epoch_iterator_reset_source_screen': return None
    text = str(task)
    if isinstance(provider_prompt, str): text += '\n'+provider_prompt
    elif isinstance(provider_prompt, dict):
        text += '\n'+'\n'.join(v for k,v in provider_prompt.items()
                              if k in ('system','user','assistant') and isinstance(v,str))
    row_counts = {int(n) for n in re.findall(r'train\.csv\s*\(\s*(\d+)\s*rows',text,re.I)}
    if row_counts != {40000} or type(spec.get('epochs')) is not int or not 1 <= spec['epochs'] <= 6: return None
    try: reserve = float(os.environ.get('MLEVOLVE_HWKG_COST_RESERVE_SECONDS','5'))
    except ValueError: return None
    if not number(reserve): return None
    default,fused = record['profiles']['default'],record['profiles']['fused']
    updates = 40000//spec['batch']*spec['epochs']
    setup = max(0.,fused['setup_seconds_max']-default['setup_seconds_min'])
    gross = updates*(default['step_seconds']-fused['step_seconds'])
    net = gross-setup-reserve
    spread = updates*(default['step_seconds_min']-fused['step_seconds_max'])-setup-reserve
    if spread <= 0 or fused['step_seconds'] > .95*default['step_seconds']: return None
    peak = fused['peak_allocated_bytes_max']
    free = runtime.get('free_mib_snapshot')
    if not number(peak,True) or peak > .5*device.total_memory: return None
    if free is not None and (not number(free,True) or peak > .6*free*1024**2): return None
    action = ('Plan one fused=True change to existing GPU AdamW.' if phase == 'planner'
              else 'Set fused=True on the existing GPU AdamW; keep FP32 parameters.')
    advice=(f'HWKG v8.19: {device.name}, {architecture(device)}; matched BF16 AdamW recipe.\n'
            f'[optimizer_fused_adamw] {action} Preserve AMP, batch, LR/WD, sample order, epochs, '
            'eval precision and EMA/checkpoint selection. Recheck live memory. '
            f'Local steps {default["step_seconds"]/fused["step_seconds"]:.2f}x; conditional {updates} updates net {net:.2f}s '
            f'after setup/{reserve:g}s declared reserve. Earlier short-repeat range conflicted; '
            'quality/full-pipeline/LLM cost unverified. No extra training or benchmark; original response format.')
    if contract_note:advice+='\n'+contract_note
    if len(advice)>min(max_chars,750): return None
    feature=host_options['fused_feature']
    audit={'policy':'hardware-v8','policy_revision':'v8.19','advice_ids':['optimizer_fused_adamw'],
           'selected_feature_ids':[feature,host_options['amp_feature']], 'verified_hardware_feature_ids':[feature,host_options['amp_feature']],
           'decision_provenance':[{'id':'optimizer_fused_adamw','feature_ids':[feature,host_options['amp_feature']],
               'reason':'exact measured host/runtime/primitive recipe; full dataflow and quality unknown'}],
           'calibration_file_sha256':record['calibration_file_sha256'],'calibration_basis':'scoped optimizer primitive microtiming; not full-pipeline equivalence',
           'calibration_trial':{'profile':'optimizer_fused_adamw','change':'optimizer_implementation','step_speedup':default['step_seconds']/fused['step_seconds'],
               'estimated_net_seconds':net,'repeat_spread_net_seconds':spread,'quality_verified':False,'end_to_end_verified':False},
           'projected_updates':updates,'source_spec':spec,'training_dtype':'bfloat16','stage_role':phase,'precision_guard':False,
           'declared_context_cost_reserve_seconds':reserve,'generation_overhead_basis':'declared reserve; marginal LLM cost unknown',
           'earlier_short_repeat_spread_conflict_retained':True,'repeat_spread_is_CI':False,'measurements_verified':False,
           'end_to_end_gain_verified':False,'added_provider_calls':0,'context_budget':min(max_chars,750)}
    return advice,audit


def only_fused_flag_changed(before, after):
    """Exact whole-source AST screen excluding one literal AdamW flag, not semantics."""
    a,b = source_spec(before),source_spec(after)
    if a.get('optimizer')!='AdamW' or b.get('optimizer')!='AdamW': return False
    aa,bb = a.get('optimizer_options'),b.get('optimizer_options')
    if not aa or not bb or not (aa.get('fused') is None or aa.get('fused') is False) or bb.get('fused') is not True: return False
    if {k:v for k,v in aa.items() if k!='fused'} != {k:v for k,v in bb.items() if k!='fused'}: return False
    def normalized(code):
        try: tree=ast.parse(code)
        except SyntaxError: return None
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and getattr(n.func,'attr',getattr(n.func,'id',''))=='AdamW']
        if len(calls)!=1: return None
        for k in calls[0].keywords:
            if k.arg=='fused' and not (isinstance(k.value,ast.Constant) and (k.value.value is None or type(k.value.value) is bool)): return None
        calls[0].keywords=[k for k in calls[0].keywords if k.arg!='fused']
        return ast.dump(tree,include_attributes=False)
    x,y=normalized(before),normalized(after)
    return x is not None and x==y
