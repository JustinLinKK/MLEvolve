# Hardware-Aware KG and Branch-Profile Scheduler Progress

As of 2026-07-21.

## Summary

MLEvolve now has a usable hardware-aware evidence path and a conservative
single-GPU scheduler path. The knowledge layer can provide stage-specific
hardware guidance to agents; the scheduler can execute hardware-aware jobs with
batch probing, prediction traces, guardrails, and replayable logs. The main gap
is turning static/precomputed guidance into consistently active calibrated
runtime decisions.

## Hardware-Aware Knowledge Graph

Done:

- Built a static Neo4j hardware capability graph with ingest/query code, CLI
  commands, and pre-integration scripts.
- Current corpus: 105 nodes, 1,158 edges, 73 hardware nodes, and 32 feature
  nodes.
- Covered precision, tensor cores, compute capability, data pipeline, kernel
  optimization, optimizer, parallelism, and interconnect features.
- Added stage filters for `model_structure`, `datatype`, and
  `training_parameters`.
- Added Qdrant code-knowledge collections:
  `code_doc_chunks`, `optimization_recipe_chunks`, and `api_symbol_chunks`.
- Wired MCP/client context into draft, improve, debug, evolution, fusion,
  aggregation, planner, code-review, and stepwise-coder prompts.

Gaps:

- The vector corpus still needs more curated, source-backed records.
- Recent comparisons attached hardware context but did not use profile evidence,
  so graph-to-agent feedback is not yet consistently exercised.
- Production loading, freshness, and versioning for the static graph still need
  to be operationalized.

## Branch-Profile Scheduler

Done:

- Implemented single-GPU scheduling with queue policy, batch probing, runtime
  profiling, conservative packing, guardrails, checkpoint-aware control,
  telemetry, replay, and comparison tooling.
- Added unified prediction contracts and a branch adapter that consumes
  precomputed `branch_prediction` job metadata.
- Replaced the experimental router variants with two strict modes:
  `branch_profile` and `ml_predictor`.
- Replaced legacy serial/parallel scheduler variants with the event-driven
  `adaptive` planner, immutable authored batches, v3 profile curves, and
  transactional checkpoint/restart repacking.
- Refactored resource estimation to use live same-job correction, router
  predictions, explicit estimates, and same-job probe evidence instead of
  general cross-job profile reuse.
- Planner traces include prediction traces when available.
- PerfSeer integration remains available behind `ml_predictor`; unavailable
  predictions fail closed unless explicit exclusive fallback is configured.

Observed:

- Dogs-vs-cats comparison on 2026-07-13: scheduler-on reduced wall time from
  1,241.8s to 953.8s and candidate execution from 129.4s to 62.4s, but had no
  packed dispatches and introduced two timeout failures.
- Histopathology comparison on 2026-07-04/05: scheduler-on kept total wall time
  roughly unchanged, reduced candidate execution from 6,171.3s to 5,611.4s, and
  produced one packed-pair dispatch with one fallback. Probe and queue overheads
  were high: 894.3s probing and 2,263.6s queue wait.

## Next Steps

1. Ensure upstream jobs consistently attach usable branch prediction metadata.
2. Calibrate or replace the branch predictor with the PerfSeer adapter.
3. Load the static hardware graph in the independent hardware knowledge runtime.
4. Improve profile-evidence use in prompts and reduce scheduler probe/queue
   overhead before claiming utilization gains.



## Resource Predictor ML model

> Input: Pytorch File (With 1 `nn.Module` class) -> Converter -> Graph -> Predictor Model -> Resources_Prediction
Input pytorch files are variants generated from top Kaggle competitions. ground truth is then generated on 
Nautilus RTX 6000 (Blackwell) to create dataset for training Predictor

- Resources_Prediction (6 predictions): SM/VRAM Usage, 1 epoch time for Training/inference

Resources predictions serve as evidence for schduler to decide which model to be deployed first

## Batch Probe Profile

## Embedding Query

# Expected Novelty

> Scheduler decreases overall training time

> Hardware Based Knowledge Greph decreases inference time due to model efficiency improvement

## Questions
1. What kind of core idea would be for the paper? Traget is given a single GPU, we reduce the total training time in a given job timeline series. Then in agent side, the agent could use hardware knowledge to optimize model design based on given training/inference hardware, which reduce resource usage/faster inference while maintianing similar accuracy.
2. How to design experiment? No KG No scheduler (baseline); KG no scheduler; No KG Scheduler; KG Scheduler. Metrics? Accuracy, total job time, inference speed, resource uitilities? In what scale and pick what kind of task, necessary to do full SWE-bench?
3. How to balance the SM and VRAM, intend to priorize VRAM since SM always 100% during training (since work on batch). But how to estimate the drawdown of parallelize jobs?
