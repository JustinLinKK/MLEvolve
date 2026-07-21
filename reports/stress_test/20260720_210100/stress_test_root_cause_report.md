# MLEvolve Stress Test Root-Cause Report

## Executive Summary

The 20-node full stress run produced 9 framework-marked buggy nodes, but independent adjudication finds only 1 genuine defect. The other 8 framework-buggy nodes are budget-censored training-progress runs: they entered real training or inference work, then were killed by the intentional 120 second candidate execution budget.

The dominant root cause is a reporting/classification mismatch: `ExecutionTimeout` is converted into a debuggable buggy node even when raw evidence shows normal training progress. The genuine defect observed in the primary run is a candidate CUDA out-of-memory failure. Scheduler malfunction is not the dominant cause in the observed runs: all primary scheduler jobs were placed on the exclusive backend, batch probes completed optimizer steps, and no probe timeout, scheduler wait timeout, worker crash, or unresolved queue state was observed.

## Repository Commit And Environment

- Branch: `hardware-awared`
- Commit: `6974d974b0262dd67893249f0cca275e60342bdc`
- Python/Torch/GPU details: `evidence/environment.md`
- Codex CLI: `codex-cli 0.144.5` at `/home/vscode/.local/bin/codex`
- MPS: unavailable in this container (`which nvidia-cuda-mps-control` returned nonzero)

The MLEvolve runs were invoked with `agent.code.provider=codex` and `agent.feedback.provider=codex`, using `/home/vscode/.local/bin/codex`, `gpt-5.5`, low reasoning, ephemeral isolated homes, and empty API-key/base-url overrides.

Relevant code paths inspected:

- `engine/search_node.py:165-177`: `EXECUTION_TIMEOUT` is included in the debuggable outcomes and sets `is_buggy=True`.
- `agents/result_parse_agent.py:317-326`: timeout results call `apply_outcome(NodeOutcome.EXECUTION_TIMEOUT)`.
- `localml_scheduler/adapters/mlevolve_runner.py:545-580`: scheduler records metric samples and heartbeats from raw candidate output.
- `localml_scheduler/adapters/mlevolve_runner.py:900-1119`: scheduler runner enforces candidate timeout, writes phase timings, failure diagnostics, and result JSON.
- `localml_scheduler/profiling/batch_probe.py:646-875`: batch-probe preflight records selected batch size and warnings.
- `localml_scheduler/scheduler/service.py:108-140`: auto backend probe computes effective backend availability/priority.

## Stress Procedure And Matrix

Primary evidence is the saved 20-step stress run:

- Path: `runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass`
- Task: `dogs-vs-cats-redux-kernels-edition`
- Seed: `5220`
- Steps: 20
- Initial drafts: 3
- Candidate execution timeout: 120 seconds
- Scheduler: enabled, auto mode; actual job placement was exclusive for all 20 jobs
- Hardware knowledge: enabled; live graph disabled in this run

Fresh bounded retry evidence:

- `matrix_kg_off_scheduler_off`: 2 nodes, classifications {'budget_censored_training_progress': 2}, backends {'direct': 2}
- `matrix_kg_off_scheduler_on_exclusive`: 2 nodes, classifications {'budget_censored_training_progress': 2}, backends {'exclusive': 2}

An initial matrix invocation failed before MLEvolve execution because `agent.code.base_url=` was parsed as `None`; the corrected retry used quoted empty-string overrides. That failed invocation is retained under `matrix/kg_off_exclusive` as command hygiene evidence, not as an experimental result.

Historical KG-on context:

- Available: True
- Path: `runs/bug2_codex_stress20/20260718_125523_bug2_codex_stress20`
- Raw old-journal summary: {'available': True, 'path': 'runs/bug2_codex_stress20/20260718_125523_bug2_codex_stress20', 'total_nodes': 20, 'buggy_nodes': 16, 'metric_nodes': 4, 'exc_type_counts': {'RuntimeError': 15, 'TimeoutError': 1, 'None': 4}, 'note': 'Historical graph-on run only; not an identical-code replay and old journal lacks modern outcome fields.'}
- Limitation: this is not an identical-code replay and cannot by itself prove KG causality.

## Timeout Adjudication Rules

Timeouts were not accepted as bugs merely because `is_buggy=True`, `NodeOutcome.EXECUTION_TIMEOUT`, missing metric, or missing submission was present. A timeout was classified as `budget_censored_training_progress` only when:

- the terminating condition was the intentional 120 second candidate execution budget;
- raw evidence showed real training progress, such as scheduler metric samples, epoch logs, nonzero training phase timings, or optimizer/backward evidence;
- no prior OOM, CUDA error, data-loading error, import error, API error, worker failure, or scheduler failure preceded the timeout;
- missing metric/submission was downstream of forced cutoff.

## Counts

Primary 20-node run:

- Total executed nodes: 20
- Raw framework-marked buggy nodes: 9
- Genuine defects: 1
- Budget-censored training-progress nodes: 8
- Recovered probe timeouts: 0
- Infrastructure/inconclusive nodes: 0
- Valid completions: 11
- Raw framework bug rate: 9/20 = 45.0%
- Adjudicated genuine-defect rate: 1/12 = 8.3%

The adjudicated denominator excludes budget-censored nodes and inconclusive/infrastructure failures: `20 - 8 - 0 = 12`.

## Failure Taxonomy And Fingerprints

- `budget_censored_training_progress`: 8 primary nodes, fingerprint usually `25035b4a6e362a9c35e0`. These nodes logged epoch/metric progress or nonzero training phase time before timeout.
- `candidate_exception`: 1 primary node, fingerprint `093c672a12ab1211df97`. Node `890c89f51f354cd8aed66d22bc5fcdfc` failed with CUDA out of memory after a CUBLAS warning.
- `valid_completion`: 11 primary nodes completed normally and produced metrics.
- `probe_timeout_recovered`: 0 observed.
- `scheduler_wait_timeout`: 0 observed.

## Evidence Traces

Timeout cluster examples:

- Node `13fcac766cae492480e5292d94ce021b`: logged two epochs with validation log loss at batch 192, had 113.891 seconds instrumented training time, then hit the 120 second execution budget.
- Node `5b405bd0ec404585a2a557a70a099c4d`: logged epochs 1 through 4 of 6, then timed out during training/backward work.
- Node `57dc39df29d943039cb40dc6d83fd265`: logged epoch 1/1 validation progress, then timed out during tail work after training progress.
- Node `9a58d700ce8c4cd39679929a478695b3`: logged epoch 1/1 and entered inference/export before the forced cutoff.

Genuine defect example:

- Node `890c89f51f354cd8aed66d22bc5fcdfc`: failed in 42.718 seconds with `RuntimeError`; term output includes CUBLAS internal warning followed by CUDA out-of-memory termination. This is generated candidate/resource behavior, not a scheduler timeout.

Scheduler/probe evidence:

- Primary scheduler events: {'scheduler_auto_backend_probe': 1, 'scheduler_session_started': 1, 'job_ready': 20, 'planner_decision_trace': 4763, 'worker_launched': 20, 'job_dispatched': 20, 'job_started': 20, 'batch_probe_cache_miss': 6, 'batch_probe_started': 6, 'batch_probe_trial': 6, 'batch_probe_selected': 6, 'batch_probe_result': 6, 'batch_probe_warning': 6, 'job_candidate_failed': 9, 'worker_finished': 20, 'runtime_probe_profiled': 11, 'job_completed': 11}
- Fresh scheduler-on events: {'scheduler_auto_backend_probe': 1, 'scheduler_session_started': 1, 'job_ready': 2, 'planner_decision_trace': 2, 'worker_launched': 2, 'job_dispatched': 2, 'job_started': 2, 'batch_probe_cache_miss': 2, 'batch_probe_started': 2, 'batch_probe_trial': 3, 'batch_probe_selected': 2, 'batch_probe_result': 2, 'batch_probe_warning': 2, 'job_candidate_failed': 2, 'worker_finished': 2}
- All primary job placement records reported `exclusive`.
- Batch probe warnings were `max_batch_size_cap`, meaning the configured probe cap was reached before VRAM saturation, not a probe failure.

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| KG causes invalid model designs | Inconclusive for KG-on, refuted for the dominant primary timeout cluster | Primary full stress had live graph disabled, yet most framework bugs were timeouts. Historical KG-on run had more runtime errors, but it is not an identical-code replay. |
| Scheduler malfunction | Refuted as primary cause; partially supported for observability/enforcement gaps | All primary jobs ran exclusive; probes succeeded; no scheduler wait timeout or worker crash. However scheduler selected batch sizes are not always reflected by runtime-logged generated batch sizes. |
| MLEvolve-scheduler integration failure | Partially supported | Node/job/result mapping exists and results were returned. The integration still promotes candidate execution timeouts into buggy/debuggable nodes and needs clearer timeout categories. |
| MPS/CUDA process/CUDA stream compatibility | Inconclusive/refuted as primary for observed failures | MPS binary unavailable. Auto mode reported stream/cuda_process available, but observed jobs were singleton exclusive placements. No packed-backend failure was established. |

## Direct Vs Scheduler And Exclusive Vs Packed

Fresh KG-off retry:

- Direct scheduler-off: 2/2 nodes timed out after nonzero training phase evidence.
- Scheduler-on exclusive placement: 2/2 nodes timed out after successful batch probes; actual placement was exclusive.

This is a replay proxy, not an identical-script counterfactual. It supports that the timeout cluster appears under both direct and scheduler-managed execution with the same 120 second budget. Strict identical-code replay across direct/exclusive/cuda_process/stream was not completed because the Codex-generated stress cells are expensive and slow; use the saved runfiles and commands in `evidence/minimal_reproducers/README.md` for that next step.

Packed backend isolation:

- `exclusive`: exercised in primary and fresh scheduler-on runs.
- `cuda_process`: reported available by auto probe, but no actual candidate in the analyzed runs was placed there.
- `stream`: reported available by auto probe, but no actual candidate in the analyzed runs was placed there.
- `mps` / `stream_mps`: skipped because `nvidia-cuda-mps-control` is unavailable.

## Ranked Root Causes

1. Timeout false positives in result classification. Frequency: 8/20 primary nodes. Confidence: high. Effect: raw bug rate is inflated from 5.0% genuine defects over all nodes to 45.0% framework-marked buggy.
2. Stress budget too short for valid generated training plans. Frequency: 8/20 primary nodes, plus 4/4 fresh matrix nodes. Confidence: high. Effect: many viable candidates are censored before metric/submission.
3. One genuine generated-code/resource defect: CUDA OOM. Frequency: 1/20 primary nodes. Confidence: high. Effect: real candidate failure that should remain debuggable.
4. Batch-size observability/enforcement mismatch. Frequency: visible in several scheduler jobs where scheduler probe selected 32 but runtime logs still show larger generated batch sizes. Confidence: medium. Effect: can reduce trust in probe outcomes and may contribute to memory/time risk.

## False-Positive Timeout List

- step 1 node `13fcac766cae492480e5292d94ce021b`: exec=124.187s, training=113.891s, last_epoch=2, framework_outcome=execution_timeout, fingerprint=25035b4a6e362a9c35e0
- step 2 node `8def670eec354414bc020cf25fe26467`: exec=121.617s, training=115.112s, last_epoch=2, framework_outcome=execution_timeout, fingerprint=25035b4a6e362a9c35e0
- step 3 node `5b405bd0ec404585a2a557a70a099c4d`: exec=124.125s, training=87.3s, last_epoch=4, framework_outcome=execution_timeout, fingerprint=25035b4a6e362a9c35e0
- step 11 node `d0eae56ed3d7464e9391e58afaeb7117`: exec=124.128s, training=86.915s, last_epoch=3, framework_outcome=repeated_failure, fingerprint=25035b4a6e362a9c35e0
- step 13 node `19665f40a78f42c1a21b950389fb3ecc`: exec=124.067s, training=69.923s, last_epoch=1, framework_outcome=execution_timeout, fingerprint=25035b4a6e362a9c35e0
- step 14 node `57dc39df29d943039cb40dc6d83fd265`: exec=121.61s, training=116.754s, last_epoch=1, framework_outcome=repeated_failure, fingerprint=25035b4a6e362a9c35e0
- step 18 node `ef305fd0aee9460f9de1fb83b7a24027`: exec=124.187s, training=118.315s, last_epoch=6, framework_outcome=execution_timeout, fingerprint=25035b4a6e362a9c35e0
- step 19 node `9a58d700ce8c4cd39679929a478695b3`: exec=124.169s, training=62.778s, last_epoch=1, framework_outcome=repeated_failure, fingerprint=25035b4a6e362a9c35e0

## Prioritized Fix Recommendations

1. Add a first-class `budget_censored_training_progress` or `execution_budget_censored` outcome and keep it out of `debug_eligible`/genuine bug counts when timeout evidence meets the rule above.
2. Persist timeout provenance in `ExecutionResult`: candidate execution budget vs scheduler wait vs probe startup/step timeout vs cancellation.
3. Surface scheduler metric samples, last epoch/global step, phase timings, and timeout provenance directly in `SearchNode` and reports.
4. Make batch override status explicit: record whether the AST rewrite found a safe training batch-size knob, what was overridden, and whether runtime logs agree with the resolved batch size.
5. Keep CUDA OOM as a real candidate defect, but group by earliest causal evidence and avoid treating missing submission as the root cause after crashes.
6. Add a cheap saved-script replay harness for direct, scheduler-exclusive, cuda_process, and stream backends, reusing already generated runfiles.

## Remaining Uncertainties

- The historical KG-on run suggests more runtime exceptions, but because it used different generated code and older journal fields, KG causality remains unproven.
- Packed/concurrent backend attribution remains incomplete because analyzed jobs were exclusive singletons.
- Direct vs scheduler evidence is a fresh generated-code proxy, not strict identical-code replay.
- Neo4j/Qdrant localhost port exposure was not reachable from this workspace; direct container IP access worked during investigation.

## Artifact Index

- Machine-readable CSV: `stress_test_node_classification.csv`
- Machine-readable JSON: `stress_test_node_classification.json`
- Environment: `evidence/environment.md` and `evidence/environment.json`
- Sanitized configs and commands: `evidence/config_overlays/`
- Event-count excerpts: `evidence/log_excerpts/`
- Reproducer notes: `evidence/minimal_reproducers/README.md`
