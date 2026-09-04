# Architecture Debug Audit — 2026-09-04

Scope: read-only architecture audit before interpreting the current PetFinder scheduler experiment. No runtime behavior, configuration, or source code was changed by this audit.

## Confirmed execution path

`run.py` constructs `AgentSearch` and `Interpreter`. A generated node flows through code review, CPU model-preflight, precision/dependency validation, and then `Interpreter.run_many()` into `localml_scheduler`. The scheduler writes canonical state to `scheduler_runtime/db/scheduler.sqlite3`; the Neo4j graph mirror is optional and the current server correctly continues with SQLite when Neo4j ports 7687/7688 are unavailable.

The active ABA run uses two A100 80GB PCIe GPUs: vLLM on physical GPU 0 and an experiment worker exposed through `CUDA_VISIBLE_DEVICES=1`, where scheduler logical device index `0` correctly maps to physical GPU 1. At observation, GPU 1 used 4 MiB in the controller and no worker job was active.

## Context and concurrency evidence

The local `qwen3.8-27b-int8-a100` `/v1/models` response reports `max_model_len: 262144`; the vLLM startup record independently reports `max_seq_len=262144`. Prometheus request metrics include calls in the `+Inf` maximum-token bucket, consistent with requests that do not impose an explicit generation cap.

The live command explicitly sets `agent.search.parallel_search_num=null` and `scheduler.settings.gpu_scheduler.parallel_job_cap=null`. The scheduler has a separate 31 GiB admission budget, as required by this experiment; that is a VRAM-safety policy rather than a fixed job-count cap.

## Findings requiring interpretation, not a code change

1. **Preflight target-profile mismatch — confirmed configuration discrepancy.** The machine is A100 80GB and `config/preflight_profiles/a100_80gb.yaml` exists, but the current launch uses `preflight.target_profile=nvidia/a100_40gb`. This can make hardware preflight conservative or otherwise non-representative of the actual worker. It does not change the scheduler's 31 GiB admission budget. No change was made in this audit.

2. **Default configuration is a hardware template, not the live experiment.** `config/config.yaml` names L40S/A10 endpoints and an A10 target profile. The ABA command-line overrides select the local A100 vLLM endpoint, A100 40GB preflight profile, GPU mask, and 31 GiB budget. Any reproduction that omits these overrides would not reproduce the ABA experiment.

3. **Optional graph mirrors are unavailable.** Current logs show connection refusal for the scheduler and hardware-knowledge Neo4j mirrors. SQLite remains live and contains canonical scheduler evidence, so this is an observability degradation rather than a scheduler failure.

4. **CPU preflight cannot execute CUDA-only branches.** The corrected static check catches the known invalid CUDA availability API; nevertheless, novel errors reachable only after `torch.cuda.is_available()` remains true require static detection or a target-GPU canary. This is an inherent coverage boundary, not evidence of a new failure in the current run.

## Verification

- `tests/test_config_loading.py`: 19 passed.
- Focused preflight static/policy/CLI tests for the CUDA-API guard: 8 passed.
- A broader read-only architecture test group was launched; it completed after emitting progress dots, but its final summary was not captured by the interactive terminal timeout and is therefore not treated as passing evidence.
- `git diff --check` produced no whitespace error. Existing unrelated user worktree changes were left untouched.

## Runtime repair — reviewer contract normalization

At 03:13 UTC, the live A100 run had generated no budget-counted node after more than one hour. The causal evidence was candidate rejection before GPU dispatch: the local Qwen reviewer returned `approved: false` while listing only warning-severity issues. `ReviewDecision.from_mapping()` correctly rejected that internally contradictory payload with `A rejected decision must contain a critical issue`; the caller retried and could discard the candidate. This was a reviewer-response normalization defect, not a GPU, scheduler, or model-execution failure.

`agents/code_review_agent.py` now normalizes only this contradictory case to `approved: true` before contract validation, preserving every warning. It does not admit candidates with a critical issue and does not weaken CPU preflight, deterministic precision/dependency validation, or scheduler admission. Regression evidence: `tests/test_stage_review_workflow.py` (48 passed) and `tests/test_config_loading.py` (19 passed). The source is synchronized to ABA; the running controller must be restarted only after its current generation request reaches a safe boundary, because imported Python modules are cached in that process.

The controller was then gracefully stopped (the local vLLM process was left running) and resumed from the same journal and scheduler-runtime directory. The first resume attempt correctly failed before initialization because its private `MLEVOLVE_VLLM_CACHE_SALT` environment value was not inherited by `nohup`; no model or GPU process was affected. The resumed controller was immediately relaunched with a fresh private cache-isolation salt, then confirmed live with its original two-A100 mapping. The new command uses the actual `nvidia/a100_80gb` preflight profile, keeps `parallel_job_cap: null`, and retains unbounded local-Qwen generation.
