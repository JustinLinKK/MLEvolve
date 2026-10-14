"""Run-local advice suppression; never alter a node, reward or execution.

Node references are read only after native execution has populated its fields.
Negative outcomes are warning evidence, not causal performance estimates.
"""
import hashlib
import json
import os
import threading
from .native_evidence import number, recipe_changes, source_spec
from .source_analysis import precision_scope
from .window_scope import training_window_scope

_lock = threading.Lock()
_pending = {}
_blocked = {}


def partition(task):
    return (os.environ.get('MLEVOLVE_HWKG_AUDIT_DIR', ''),
            os.environ.get('MLEVOLVE_HWKG_GRAPH_SHA256', ''), hashlib.sha256(task.encode()).hexdigest())


def recipe_key(code):
    spec = source_spec(code)
    return json.dumps({k: spec[k] for k in ('model', 'optimizer', 'optimizer_options', 'weights',
        'classifier_classes', 'loss', 'batch', 'input_shape', 'epochs', 'forwards_per_update')}, sort_keys=True)


def metric(node):
    m = getattr(node, 'metric', None)
    value = getattr(m, 'value', None)
    return (value, getattr(m, 'maximize', None))


def remember(task, parent, node, intervention_ids, target_profile=None):
    if not parent or not intervention_ids: return
    with _lock:
        _pending[(partition(task), getattr(node, 'id', id(node)))] = {
            'node': node, 'code_sha256': hashlib.sha256(node.code.encode()).hexdigest(),
            'before': parent.code, 'before_time': getattr(parent, 'exec_time', None),
            'before_metric': metric(parent), 'ids': set(intervention_ids), 'target': target_profile}


def observe(task, code):
    events = []
    run = partition(task)
    with _lock:
        for key, row in list(_pending.items()):
            if key[0] != run: continue
            node = row['node']
            if getattr(node, 'is_buggy', None) not in (True, False): continue
            # Some upstream nodes initialize is_buggy=False before execution.
            if not number(getattr(node, 'exec_time', None), True) and not getattr(node, 'exc_type', None): continue
            del _pending[key]
            if hashlib.sha256(node.code.encode()).hexdigest() != row['code_sha256']: continue
            changes = recipe_changes(row['before'], node.code)
            reasons = []
            if changes['evaluation_precision_changed']: reasons.append('evaluation precision changed outside hardware trial')
            if 'precision' in row['ids'] and changes['bf16_with_active_scaler']: reasons.append('BF16 trial retained active GradScaler')
            if getattr(node, 'is_buggy', None) is True: reasons.append('native candidate failed; attribution unknown')
            same_recipe = (changes['recipe_syntax_changed'] is False
                           and changes['evaluation_precision_changed'] is False)
            optimizer_flag_only = None
            if row['ids'] == {'optimizer_fused_adamw'} and changes['evaluation_precision_changed'] is False:
                from .optimizer_policy import only_fused_flag_changed
                optimizer_flag_only = only_fused_flag_changed(row['before'], node.code)
                same_recipe = same_recipe or optimizer_flag_only
            window_scopes = {'before': training_window_scope(row['before']),
                             'after': training_window_scope(node.code)}
            work_exclusions = [f'{side}: standalone gradient timing window before epoch iterator reset'
                               for side, scope in window_scopes.items() if scope['status'] ==
                               'standalone_gradient_window_before_epoch_iterator_reset_source_screen']
            # A selected recipe signature does not cover standalone gradient
            # passes. Exclude their time/quality comparison without blaming a
            # hardware profile. Failure/precision warnings remain independent.
            comparable = same_recipe and not work_exclusions
            confounded = not comparable
            before_time, after_time = row['before_time'], getattr(node, 'exec_time', None)
            if comparable and number(before_time, True) and number(after_time, True) and after_time > 1.05 * before_time:
                reasons.append('native candidate wall regressed over 5%; confounded observation')
            before_metric, direction = row['before_metric']; after_metric, after_direction = metric(node)
            if comparable and number(before_metric) and number(after_metric) and direction is after_direction and type(direction) is bool:
                if (after_metric < before_metric if direction else after_metric > before_metric):
                    reasons.append('native public metric regressed; confounded observation')
            target = row['target']
            if not target and row['ids'] & {'precision','vision_layout'}:
                dtypes = precision_scope(node.code).get('training_dtypes', [])
                dtype = {'bfloat16':'bf16','float16':'fp16','float32':'fp32'}.get(dtypes[0]) if len(dtypes) == 1 else None
                if dtype: target = dtype+'_'+('channels_last' if 'channels_last' in node.code else 'contiguous')
            if reasons and target:
                _blocked.setdefault((run, recipe_key(row['before'])), set()).add(target)
                _blocked.setdefault((run, recipe_key(node.code)), set()).add(target)
            events.append({'event':'hardware_trial_observed','candidate_id':getattr(node,'id',None),
                'code_sha256':row['code_sha256'],'target_profile':target,'suppression_reasons':reasons,
                'recipe_scope':changes,'native_before_seconds':before_time,'native_after_seconds':after_time,
                'training_window_scope_source_screen':window_scopes,'work_scope_comparison_exclusions':work_exclusions,
                'optimizer_flag_only_change_source_screen':optimizer_flag_only,
                'recipe_confounded':confounded,'time_or_quality_comparison_eligible_source_screen':comparable,
                'causal_gain_verified':False,'node_modified':False})
        blocked = sorted(_blocked.get((run, recipe_key(code)), set()))
    return blocked, events
