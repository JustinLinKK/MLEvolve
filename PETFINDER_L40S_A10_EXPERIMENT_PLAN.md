# PetFinder L40S Agent and Same-A10 Scheduler Comparison Plan

## Objective

Serve the text-only Qwen3.8-27B INT8 agent with vLLM on one L40S GPU. On the
same physical A10 GPU 2, run these 50-node PetFinder Pawpularity searches
sequentially:

1. Original MLEvolve baseline with the scheduler disabled.
2. Modified MLEvolve with the scheduler and Hardware Knowledge Database.

The modified run uses branch-profile runtime prediction because the learned
predictor is not ready.

## Invariants

- Do not configure `parallel_job_cap` or another fixed scheduler maximum.
- Let branch profiles and live VRAM telemetry determine scheduler admission.
- Use the same task description, dataset, 50-node budget, validation metric,
  L40S agent endpoint, and physical A10 for both phases.
- PetFinder is Pawpularity regression and RMSE is lower-is-better.
- Start the scheduler phase only after the baseline journal contains 50
  completed nodes.
- Preserve superseded or failed runs as diagnostic evidence, but exclude them
  from the final comparison.

## Completion Evidence

- Clean process exit and 50-node journal for both A10 phases.
- Per-node metrics and timestamps for both phases; scheduler decisions,
  positive runtime estimates, actual durations, and VRAM evidence for the
  modified phase.
- Valid `submission/submission.csv` artifacts for scored nodes.
- One PNG with both Gantt charts above and metric-versus-node curves below.
- Exact settings, paths, failures, results, and conclusions in `records/`.
- Regression tests covering nullable parallelism, positive branch-profile
  estimates, and search liveness after unusable repairs.
- Verified local recovery copies followed by commit and push.
