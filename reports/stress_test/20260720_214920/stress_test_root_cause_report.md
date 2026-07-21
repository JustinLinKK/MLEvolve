# MLEvolve Stress Test Root-Cause Report

## Executive Summary

The fresh rerun was started from the current `hardware-awared` worktree with Codex CLI providers, but it could not complete the requested 20 nodes. It reached 2 completed SearchNodes and then entered a deterministic live-scheduler framework loop: `ValueError: max() iterable argument is empty` repeated 23,053 times before I stopped the run with Ctrl-C to preserve evidence and avoid burning more Codex/compute time.

The two completed nodes do not show generated-code defects. One was an intentional 120 second stress-budget cutoff after 111.148 seconds of instrumented training. The other executed successfully at the worker level, logged three epochs plus a final validation score, then was quarantined as `validation_unavailable` because the local submission validation server at `http://127.0.0.1:5005` was offline.

The fresh run therefore exposes one high-confidence run-level framework defect: `SearchNode.fetch_child_memory()` treats `is_buggy=False` as “successful”, but then assumes at least one such child has a non-null metric. A validation-unavailable non-buggy child violates that assumption. The live scheduler loop catches the exception and retries immediately while an outstanding job exists, producing a tight log-spam loop.

## Repository Commit And Environment

- Branch: `hardware-awared`
- Commit: `6974d974b0262dd67893249f0cca275e60342bdc`
- Worktree status: see `evidence/environment.json`
- Python/Torch/GPU: `evidence/environment.md`
- Codex CLI: `codex-cli 0.144.5` at `/home/vscode/.local/bin/codex`
- Services: Neo4j and Qdrant containers were up; port `5005` validation service was not reachable.
- MPS: unavailable (`nvidia-cuda-mps-control` not found)

The MLEvolve invocation used `agent.code.provider=codex` and `agent.feedback.provider=codex`, `/home/vscode/.local/bin/codex`, low reasoning, ephemeral isolated homes, ignore-user-config, and empty API-key/base-url overrides.

## Stress Procedure And Matrix

Primary fresh run:

- Output root: `reports/stress_test/20260720_214920`
- Run: `reports/stress_test/20260720_214920/full_stress/runs/20260720_214957_dogs-vs-cats-redux-kernels-edition_current_stress_20260720_214920`
- Task: `dogs-vs-cats-redux-kernels-edition`
- Seed: `5220`
- Requested steps: 20
- Completed SearchNodes: 2
- Candidate execution timeout: 120 seconds
- Scheduler wait timeout: 150 seconds
- Scheduler mode: `auto`; actual completed jobs used `exclusive`
- Hardware knowledge: enabled with profile evidence; graph lookup disabled
- Exact command: `evidence/config_overlays/current_full_stress_command.txt`
- Sanitized effective config: `evidence/config_overlays/current_full_stress_config.sanitized.yaml`

Pre-run focused test suite:

- Command/output: `evidence/targeted_test_results.txt`
- Result: `121 passed in 52.96s`

Requested matrix cells A-J from `Stress_test_report.md` were not run after this hard blocker, because the current code cannot safely continue the live-scheduler stress workflow once it has a non-buggy metricless child. Running more cells first would mostly spend more LLM budget on a known framework loop.

Backend status:

- `exclusive`: exercised by completed scheduler jobs.
- `cuda_process`: reported available by auto probe but not selected before the blocker.
- `stream`: reported available by auto probe, but stream execution was disabled in the effective config.
- `mps` / `stream_mps`: skipped because MPS control binary is unavailable.

## Timeout Adjudication Rules

Timeouts were adjudicated independently from `is_buggy`, `NodeOutcome.EXECUTION_TIMEOUT`, missing metric, or missing submission. A timeout was classified as `budget_censored_training_progress` only when the termination came from the short candidate execution budget, raw evidence showed real training progress, and no earlier OOM/CUDA/import/data/scheduler/probe failure preceded the cutoff.

## Counts

- Total executed nodes: 2
- Raw framework-marked buggy nodes: 1
- Genuine defects: 1 run-level framework defect, 0 per-node candidate defects
- Genuine per-node candidate defects: 0
- Genuine run-level framework defects: 1
- Budget-censored training-progress nodes: 1
- Recovered probe timeouts: 0
- Infrastructure/inconclusive nodes: 1
- Valid completions: 0
- Raw framework bug rate: 1/2 = 50.0%
- Adjudicated per-node genuine-defect rate: not defined; denominator is 0 after excluding budget-censored and infrastructure nodes.

## Failure Taxonomy And Fingerprints

- `budget_censored_training_progress`: 1 node. Fingerprint `25035b4a6e362a9c35e0`; 111.148 seconds instrumented training before the 120 second cutoff.
- `infrastructure_failure`: 1 node. Worker-level success, but validation server `127.0.0.1:5005` was offline, so MLEvolve quarantined the node as `validation_unavailable`.
- `mlevolve_search_loop_failure`: 1 run-level framework blocker. First observed at `MLEvolve.log` line 221; repeated 23,053 times.
- Pending/cancelled scheduler jobs after stop: 01698790-6f58-460e-977c-111c17e91468.

## Evidence Traces

Timeout false positive:

- Node `d852c55b52f34b3f91601af941c22f87` ran on job `00a5af5a-4229-4164-9fcd-1a8648b63bbc`.
- Framework outcome: `execution_timeout`, `is_buggy=True`.
- Independent classification: `budget_censored_training_progress`.
- Evidence: phase timings show `111.148` seconds of training and the failure diagnostic is `execution_timeout`, with no earlier CUDA OOM or candidate exception.

Validation-unavailable trigger:

- Node `bb39bfc4449b45f4a8761522429272fe` completed worker execution successfully.
- Term output logged epochs 1-3 and `Final Validation Score: 0.032823926211471975`.
- `agents/result_parse_agent.py:569-572` converted offline validation service into `NodeOutcome.VALIDATION_UNAVAILABLE`.
- `engine/search_node.py:174-177` leaves that node `is_buggy=False`, `search_eligible=False`, and metricless.

Run-level blocker:

- `agents/draft_agent.py:94` calls `agent.virtual_root.fetch_child_memory()`.
- `engine/search_node.py:336` puts metricless validation-unavailable children into `successful` because `is_buggy is False`.
- `engine/search_node.py:344` then calls `max(n.metric.value for n in successful if n.metric and n.metric.value is not None)`, which raises on an empty iterable.
- `run.py:431-434` catches and returns `None`; `run.py:476-493` immediately retries while an outstanding scheduler job exists, with no sleep/backoff on this path.

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| KG causes invalid model designs | Inconclusive, not supported by this rerun | Hardware knowledge was enabled but graph lookup was disabled. The completed failures are stress-budget cutoff and offline validation service, not invalid KG-induced code. |
| Scheduler malfunction | Partially supported for live-loop handling, not for completed worker execution | Scheduler jobs launched and reported terminal results for the two completed nodes. The run blocker is MLEvolve search/live-loop handling after a metricless non-buggy child, while a third scheduler job remained outstanding. |
| MLEvolve-scheduler integration failure | Supported | Completed worker success plus validation-unavailable state produced a metricless non-buggy node; the next draft generation crashed and the live scheduler loop retried without backoff. |
| MPS/CUDA process/CUDA stream compatibility | Inconclusive/refuted as primary in this run | Completed jobs used exclusive placement. MPS is unavailable; cuda_process/stream were not exercised before the blocker. |

## Direct Vs Scheduler And Exclusive Vs Packed

No direct-vs-scheduler or exclusive-vs-packed replay was run after the hard blocker. Completed jobs only establish that exclusive scheduler execution and batch probes work up to worker completion/result collection for two nodes. Packed backend attribution remains untested in this fresh rerun.

## Ranked Root Causes

1. Metricless non-buggy child crashes search memory summary. Frequency: 1/1 fresh run, 23,053 repeated exceptions. Confidence: high. Effect: blocks the stress workflow before 20 nodes.
2. Live scheduler generation failure retry loop lacks backoff/stop while outstanding jobs exist. Frequency: 1/1 fresh run after the first exception. Confidence: high. Effect: severe log spam and wasted CPU/LLM orchestration time.
3. External validation service offline. Frequency: 1/2 completed nodes. Confidence: high. Effect: converts a worker-level successful candidate into `validation_unavailable` with metric `None`.
4. Stress execution budget too short for some real training jobs. Frequency: 1/2 completed nodes. Confidence: high. Effect: false raw buggy node unless independently classified as budget-censored.

## False-Positive Timeout List

- step 1 node `d852c55b52f34b3f91601af941c22f87`: exec=124.105s, training=111.148s, framework_outcome=execution_timeout, fingerprint=25035b4a6e362a9c35e0

## Prioritized Fix Recommendations

1. Make `fetch_child_memory()` handle non-buggy children with no metric by reporting them separately or computing best metric only from metric-bearing children.
2. Add backoff and a terminal stop condition in the live scheduler generation loop when generation repeatedly returns `None` while jobs are outstanding.
3. Treat `validation_unavailable` as an infrastructure/quarantine state that cannot be summarized as “successful” unless it has a metric.
4. Start or explicitly disable the local validation service for stress runs, so worker-level success is not converted into metricless quarantine by accident.
5. Keep the `budget_censored_training_progress` distinction for 120 second training cutoffs and exclude those from genuine bug rates.
6. After those fixes, rerun the A-J matrix and identical-code replay requested by `Stress_test_report.md`.

## Remaining Uncertainties

- The full 20-node distribution is unknown because the current run blocked at 2 completed nodes.
- Direct execution, cuda_process, stream, and packed/concurrent behavior are untested in this fresh rerun.
- KG causality cannot be inferred from this run because the observed blocker occurs in validation/search summarization, not model-design generation.
- The pending third job was cancelled by the investigator after the framework loop was confirmed, so it is not classified as an executed node.

## Artifact Index

- CSV: `stress_test_node_classification.csv`
- JSON: `stress_test_node_classification.json`
- Environment: `evidence/environment.md` and `evidence/environment.json`
- Targeted tests: `evidence/targeted_test_results.txt`
- Command/config: `evidence/config_overlays/`
- Log excerpts: `evidence/log_excerpts/`
- Reproducer notes: `evidence/minimal_reproducers/README.md`
