# PetFinder L40S Agent and Scheduler Comparison Plan

> Superseded on 2026-08-31 by `PETFINDER_L40S_A10_EXPERIMENT_PLAN.md` after
> both comparison phases were reassigned to the same A10 GPU.

## Objective

Use the text-only Qwen3.8-27B INT8 agent served by vLLM on one L40S GPU, then run the PetFinder Pawpularity task in these configurations:

1. Original MLEvolve baseline on one A100 GPU.
2. Scheduler plus Hardware Knowledge Database on the same A100 GPU after the baseline finishes.
3. Scheduler plus Hardware Knowledge Database concurrently on one A10 GPU.

The modified runs use branch-profile runtime prediction because the learned predictor is not ready.

## Invariants

- Do not configure a fixed `parallel_job_cap` or fixed scheduler maximum parallel-job count.
- Let live video random-access memory telemetry and branch profiles control admission.
- Use the same task description, data, 50-node search budget, validation metric, and agent endpoint for comparable runs.
- PetFinder is Pawpularity regression; root mean squared error is lower-is-better.
- Preserve failed runs as diagnostic evidence, but exclude them from the result comparison.

## Completion Evidence

- Process exit and final run logs for all three runs.
- Per-node records with metric, start/end timestamps, runtime estimate, actual duration, memory, and scheduler decision where applicable.
- Valid `submission/submission.csv` artifacts for scored nodes.
- One combined PNG with Gantt charts above and metric-versus-node graphs below.
- A Markdown record in `records/` containing exact settings, paths, failures, results, and conclusions.
- Tests for nullable parallelism and non-null branch-profile estimates.
- Commit and push after the experiment artifacts are verified.
