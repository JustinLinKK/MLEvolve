# PetFinder A100-Agent / A10-Execution Comparison Plan

## Objective

Compare original MLEvolve with the scheduler plus Hardware Knowledge Database
variant on the same single NVIDIA A10. Both phases use the same text-only
Qwen3.8-27B INT8 agent served by vLLM on one NVIDIA A100-SXM4-80GB. A run is
not comparable if any phase falls back to a 40GB A100 or another A100 SKU.

## Controlled settings

- Task: PetFinder Pawpularity Score.
- Search seed: 42.
- Retained search nodes: 50 per phase.
- Search time budget: 43,200 seconds (12 hours) per phase, preventing the
  slower local Agent from producing negative remaining-time guidance.
- Agent: `qwen3.8-27b-int8-a100` at the cluster-local A100 service.
- Agent hardware: exactly `NVIDIA-A100-SXM4-80GB`, enforced with required node
  affinity and verified again with `nvidia-smi` before either phase starts.
- Execution device: exactly one NVIDIA A10.
- Baseline: the previously validated original-MLEvolve source snapshot, with
  scheduler disabled and non-stepwise generation (`use_stepwise_generation=false`).
- Modified method: hardware-aware mode, scheduler and Hardware Knowledge
  Database enabled, profile-based runtime prediction, and CPU preflight enabled.
- Neither phase configures a fixed maximum parallel-job count. The baseline uses
  original MLEvolve worker admission; the modified method uses branch profiles
  and live GPU telemetry with `parallel_job_cap=null`.
- Candidates rejected before GPU execution by stage review or preflight are
  discarded and do not count toward the 50 retained nodes.
- Failed executions detected in less than 60 seconds remain in the journal for
  debugging lineage but do not consume the 50-node experiment budget. The
  final comparison uses the first 50 budget-counted nodes from each phase.

## Execution and evidence

1. Terminate only the owned A100 filler and verify GPU memory release.
2. Start and health-check the A100 vLLM service.
3. Record post-warmup Time to First Token (TTFT) and generation tokens/second.
4. Preserve superseded or mixed-hardware runs as diagnostics only, then run
   baseline followed by the modified method from zero on the same A10 and the
   same A100-SXM4-80GB agent allocation.
5. Persist journal, generated code, scheduler events, hardware evidence,
   process logs, and GPU telemetry for recovery.
6. Produce one comparison PNG with Gantt charts above and metric-versus-node
   graphs below, then write the final experiment record and push the result.
