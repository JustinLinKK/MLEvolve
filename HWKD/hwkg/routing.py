"""Keep HWKG scoped to a candidate, across planners, coders and fallbacks.

Dispatch records mean the provider was called with context, not that the model
obeyed it. Provider-internal HTTP retries remain the upstream backend's policy.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .injection import DEFAULT_POLICY, hardware_guidance


def _staged_policy():
    return os.environ.get('MLEVOLVE_HWKG_EVIDENCE_POLICY', DEFAULT_POLICY) in {'v5', 'v6', 'v7', 'v8'}

def _feedback_policy():
    return os.environ.get('MLEVOLVE_HWKG_EVIDENCE_POLICY', DEFAULT_POLICY) == 'v8'


@dataclass
class CandidateContext:
    stage: str
    task: str
    parent_code: str
    parent_id: str | None
    trace_id: str
    advice: str | None = None
    proposed_plan: str = ''
    failure_text: str = ''
    injected_calls: int = 0
    parent_execution_succeeded: bool | None = None
    decisions: dict = field(default_factory=dict)
    intervention_ids: set = field(default_factory=set)
    blocked_profiles: list = field(default_factory=list)
    target_profile: str | None = None
    suppress_after_empty: bool = False


_context = ContextVar("hwkg_candidate", default=None)
_phase = ContextVar("hwkg_phase", default="plan_and_code")
_audit_lock = threading.Lock()


def hardware_context(stage):
    def decorate(func):
        signature = inspect.signature(func)
        @wraps(func)
        def scoped(agent, *args, **kwargs):
            if not os.environ.get("MLEVOLVE_HWKG_GRAPH_PATH"):
                return func(agent, *args, **kwargs)
            parent = signature.bind(agent, *args, **kwargs).arguments.get("parent_node")
            context = CandidateContext(stage, agent.task_desc,
                                       getattr(parent, "code", "") or "",
                                       getattr(parent, "id", None), uuid.uuid4().hex)
            output = getattr(parent, '_term_out', None)
            if isinstance(output, list):
                context.failure_text = ''.join(str(x) for x in output)[-131072 if _feedback_policy() else -6000:]
            if parent is not None and _feedback_policy():
                context.parent_execution_succeeded = (getattr(parent, 'exc_type', None) is None
                    and getattr(parent, 'is_buggy', None) is False)
            if _feedback_policy():
                from .trial_memory import observe
                context.blocked_profiles, events = observe(context.task, context.parent_code)
                for event in events: _audit(event)
            token = _context.set(context)
            try:
                result = func(agent, *args, **kwargs)
                if context.injected_calls and isinstance(getattr(result, "code", None), str):
                    observations = {}
                    if _staged_policy():
                        from .staged_policy import source_adoption
                        observations['source_observations'] = source_adoption(result.code)
                    if _feedback_policy():
                        from .native_evidence import recipe_changes
                        from .adaptive_policy import source_configured
                        configured=source_configured(result.code)
                        observations.update(recipe_scope=recipe_changes(context.parent_code,result.code),
                            intervention_observations={k:{'source_configured':bool(configured.get(
                                'channels_last' if k=='vision_layout' else 'bf16')),
                                'runtime_verified':False} for k in context.intervention_ids if k in ('precision','vision_layout')})
                        if context.intervention_ids & {'optimizer_fused_adamw','optimizer_choice'}:
                            from .exploration_policy import optimizer_option
                            observations['optimizer_implementation_observation'] = {
                                'fused_source_option':(optimizer_option(result.code) or {}).get('fused'),
                                'runtime_verified':False, 'gain_verified':False}
                        if 'precision_choice' in context.intervention_ids:
                            from .source_analysis import precision_scope
                            observations['precision_choice_observation'] = {
                                'training_dtypes_source':precision_scope(result.code).get('training_dtypes',[]),
                                'runtime_verified':False,'gain_verified':False}
                        from .trial_memory import remember
                        remember(context.task, parent, result, context.intervention_ids, context.target_profile)
                    _audit({"event": "candidate_return", "trace_id": context.trace_id,
                            "stage": stage, "candidate_id": result.id,
                            "code_sha256": hashlib.sha256(result.code.encode()).hexdigest(), **observations})
                return result
            finally:
                _context.reset(token)
        return scoped
    return decorate


def hardware_phase(phase):
    """Mark planner/coder/merge boundaries for staged policies; otherwise a no-op."""
    def decorate(func):
        @wraps(func)
        def scoped(*args, **kwargs):
            context = _context.get()
            if context is None or not _staged_policy():
                return func(*args, **kwargs)
            token = _phase.set(phase)
            try:
                result = func(*args, **kwargs)
                if phase == 'planner' and isinstance(result, (str, dict)):
                    context.proposed_plan = (result if isinstance(result, str) else json.dumps(result, sort_keys=True))[:6000]
                return result
            finally:
                _phase.reset(token)
        return scoped
    return decorate


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _audit(record):
    directory = os.environ.get("MLEVOLVE_HWKG_AUDIT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with _audit_lock, (path / "dispatch-audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_unix": time.time(), **record}, sort_keys=True) + "\n")


def _append_advice(prompt, advice):
    # Copy, never mutate shared prompt dictionaries or accumulated retry prompts.
    suffix = "\n\n# Hardware knowledge (advisory)\n" + advice
    if isinstance(prompt, str):
        return prompt + suffix
    if isinstance(prompt, dict) and any(k in prompt for k in ("system", "user", "assistant")):
        # Append to user instructions, never impersonate an assistant response.
        return {**prompt, "user": str(prompt.get("user") or "") + suffix}
    raise TypeError("Scoped HWKG generation requires an upstream text or chat-dict prompt")


def _append_feedback_advice(prompt, advice):
    import re
    block='\n# Hardware knowledge (advisory)\n'+advice+'\n\n'
    def insert(text):
        # Stateless provider retries still need the advice. Cache its decision,
        # and make delivery idempotent if a caller reuses our delivered prompt.
        if text.count(block)==1:return text
        if block in text:text=text.replace(block,'')
        matches=list(re.finditer(r'(?im)^#{1,6}\s*(?:response format|output format|output requirements|format requirements)\b',text))
        index=matches[-1].start() if matches else 0
        return text[:index]+block+text[index:]
    if isinstance(prompt,str):return insert(prompt)
    if isinstance(prompt,dict) and any(k in prompt for k in ('system','user','assistant')):
        return {**prompt,'user':insert(str(prompt.get('user') or ''))}
    return None


def _remove_own_feedback_blocks(prompt, decisions):
    """Remove only exact blocks inserted by this candidate's component."""
    blocks=['\n# Hardware knowledge (advisory)\n'+a+'\n\n' for a in decisions.values() if isinstance(a,str)]
    def clean(text):
        for block in blocks:text=text.replace(block,'')
        return text
    if isinstance(prompt,str):return clean(prompt)
    if isinstance(prompt,dict) and isinstance(prompt.get('user'),str):return {**prompt,'user':clean(prompt['user'])}
    return prompt


def with_hardware_prompt(func):
    @wraps(func)
    def dispatch(prompt, cfg, *args, **kwargs):
        context = _context.get()
        if context is None:
            return func(prompt, cfg, *args, **kwargs)
        if _feedback_policy() and context.suppress_after_empty:
            # Preserve the native retry count, parameters, response and parser.
            # Only the optional component content is withdrawn for this candidate.
            clean=_remove_own_feedback_blocks(prompt,context.decisions)
            _audit({'event':'advice_suppressed_after_empty','trace_id':context.trace_id,
                    'phase':_phase.get(),'native_retry_unchanged':True,'added_provider_calls':0,
                    'base_prompt_sha256':_digest(clean)})
            return func(clean,cfg,*args,**kwargs)
        # Lazy: loading an initial solution without an LLM call must not probe CUDA.
        if _feedback_policy():
            identity_prompt = (_remove_own_feedback_blocks(prompt, context.decisions)
                               if context.injected_calls else prompt)
            signature=(_phase.get(),_digest(context.proposed_plan),os.environ.get('MLEVOLVE_HWKG_GRAPH_SHA256'),
                os.environ.get('MLEVOLVE_HWKG_HARDWARE_ID'),os.environ.get('MLEVOLVE_HWKG_CALIBRATION_PATH'),
                _digest(identity_prompt) if not context.parent_code else None)
            if signature in context.decisions:
                _audit({'event':'context_cache_hit','trace_id':context.trace_id,'phase':_phase.get()})
                context.advice=context.decisions[signature]
            else:
                context.advice=hardware_guidance(context.stage,context.task,context.parent_code,
                    phase=_phase.get(),proposed_plan=context.proposed_plan,failure_text=context.failure_text,
                    parent_id=context.parent_id,parent_execution_succeeded=context.parent_execution_succeeded,
                    blocked_profiles=context.blocked_profiles,provider_prompt=identity_prompt)
                context.decisions[signature]=context.advice
        elif _staged_policy():
            context.advice = hardware_guidance(context.stage, context.task, context.parent_code,
                phase=_phase.get(), proposed_plan=context.proposed_plan, failure_text=context.failure_text)
        elif context.advice is None:
            context.advice = hardware_guidance(context.stage, context.task, context.parent_code)
        if context.advice is None:
            # Withdraw only exact content this component previously delivered.
            # The native request itself, response, retry and parser are retained.
            unchanged = identity_prompt if _feedback_policy() else prompt
            return func(unchanged, cfg, *args, **kwargs)
        delivered = _append_feedback_advice(identity_prompt,context.advice) if _feedback_policy() else _append_advice(prompt, context.advice)
        if delivered is None:
            _audit({'event':'unsupported_prompt_skipped','trace_id':context.trace_id});return func(prompt,cfg,*args,**kwargs)
        if _feedback_policy():
            import re
            context.intervention_ids.update(re.findall(r'^\[([^]]+)\]',context.advice,re.M))
            if '[optimizer_fused_adamw]' in context.advice:
                context.target_profile = 'optimizer_fused_adamw'
            if '[precision]' in context.advice:
                dtype=('bf16' if 'torch.bfloat16' in context.advice else
                       'fp16' if re.search(r'\bFP16\b|torch\.float16',context.advice) else 'bf16')
                context.target_profile=dtype+'_'+('channels_last' if 'channels_last' in context.parent_code else 'contiguous')
            elif '[vision_layout]' in context.advice:
                from .source_analysis import precision_scope
                dtypes=precision_scope(context.parent_code).get('training_dtypes', [])
                dtype={'float16':'fp16','bfloat16':'bf16','float32':'fp32'}.get(dtypes[0]) if len(dtypes)==1 else None
                if dtype: context.target_profile=dtype+'_channels_last'
        context.injected_calls += 1
        call_id = uuid.uuid4().hex
        common = {"trace_id": context.trace_id, "call_id": call_id,
                  "stage": context.stage, "parent_id": context.parent_id,
                  "parent_code_sha256": hashlib.sha256(context.parent_code.encode()).hexdigest()}
        if _staged_policy():common['phase'] = _phase.get()
        _audit({**common, "event": "provider_dispatch", "model": cfg.agent.code.model,
                "base_prompt_sha256": _digest(prompt), "delivered_prompt_sha256": _digest(delivered),
                "context_sha256": hashlib.sha256(context.advice.encode()).hexdigest(),
                "context_text": context.advice})
        try:
            started=time.perf_counter()
            response = func(delivered, cfg, *args, **kwargs)
        except BaseException as exc:
            _audit({**common, "event": "provider_failure", "error_type": type(exc).__name__})
            raise
        _audit({**common, "event": "provider_return", "response_sha256": _digest(response),
                'provider_wall_seconds':time.perf_counter()-started,
                'marginal_injection_cost_verified':False})
        if _feedback_policy() and isinstance(response,str) and not response.strip():
            context.suppress_after_empty=True
            _audit({**common,'event':'empty_response_advice_withdrawn','response_modified':False,
                    'cause_of_empty_response':'unknown; optional hardware context withdrawn on native retry'})
        return response
    return dispatch
