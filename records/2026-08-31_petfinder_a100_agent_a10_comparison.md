# PetFinder: A100 Agent and Single-A10 Scheduler Comparison

## Status

Running. The A100 agent deployment and serving benchmark are complete. The
fresh same-agent comparison is in the original-MLEvolve baseline phase; the
profile-based scheduler plus Hardware Knowledge Database phase will start only
after the baseline completes 50 budget-counted nodes.

## A100 agent deployment

- GPU: one NVIDIA A100-SXM4-80GB.
- Model: text-only `Qwen3.8-27B-INT8-W8A16-MTP`.
- Runtime: vLLM 0.27.1, compressed-tensors INT8 weights, Marlin linear kernel,
  FlashAttention 2, 32,768-token maximum context.
- Served name: `qwen3.8-27b-int8-a100`.
- Kubernetes service: `mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000`.
- Health evidence: HTTP 200 from both the A100 Pod and the A10 experiment Pod.
- Resident GPU memory after initialization: approximately 73,473 MiB.
- The owned `a100_goal_filler` process was terminated before model startup;
  the A100 Pod and persistent storage were not deleted or recreated.

## Serving benchmark

The benchmark used one warm-up request followed by three sequential streaming
requests. Each measured response contained 39 completion tokens.

| Metric | Median | Individual measurements |
| --- | ---: | --- |
| Time to First Token | 0.19697 s | 0.19807, 0.19697, 0.19665 s |
| Generation throughput | 13.2088 tokens/s | 13.2214, 13.1844, 13.2088 tokens/s |
| End-to-end request time | 3.14923 s | 3.14784, 3.15501, 3.14923 s |

Raw benchmark JSON is retained in
`.cache/qwen38_a100_benchmark_20260831/result.json`. The required combined Gantt
and metric-node image is
`records/2026-08-31_qwen38_a100_vllm_benchmark.png`.

## Controlled PetFinder comparison

- Execution GPU: exactly one NVIDIA A10, exposed as `CUDA_VISIBLE_DEVICES=0`
  inside `gpu-dev-a10-experiment-746c959459-2xwdd`.
- Task: PetFinder Pawpularity Score.
- Both phases use the same A100 Agent endpoint, model, seed 42, and 50
  budget-counted nodes.
- Both phases use a 43,200-second search budget. The original 28,800-second
  value was too close to the projected wall time at the measured 13.21
  tokens/second and would eventually make the prompt's remaining-time value
  negative, even though it is not a process-level hard stop.
- Baseline phase uses the previously validated original-MLEvolve snapshot at
  `/root/downeyflyfan/mlevolve_a10_baseline_20260829`, with scheduler disabled
  and `use_stepwise_generation=false`. This is necessary because the modified
  source hard-codes stepwise generation and cannot represent the original
  baseline by configuration switches alone.
- Modified phase enables hardware-aware mode, profile-based runtime prediction,
  preflight, scheduler, and Hardware Knowledge Database.
- `agent.search.parallel_search_num=null` in both phases. The modified phase
  also sets `parallel_job_cap=null`; live admission owns concurrency.
- Stage-review or preflight rejections that avoid GPU execution are discarded
  and do not count toward the 50-node target.
- Failed executions with `exec_time < 60` seconds are retained as diagnostic
  attempts and parent nodes, but a shared accounting rule excludes them from
  the 50-node budget. Successful fast nodes and failures lasting at least 60
  seconds still count. The original baseline source remains unchanged; the
  experiment uses an isolated copy with only this shared accounting rule.
- The sequence controller validates the budget-counted journal total and
  refuses to advance after an early exit with fewer than 50 effective nodes.

Active comparison root:

`/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_comparison_20260831_193545`

The superseded L40S diagnostic run was stopped without deleting its journal or
artifacts. It is excluded from the final same-agent comparison.

The earlier A100-agent baseline diagnostic at
`a100_agent_a10_comparison_20260831_191248` demonstrated that the unmodified
counter incorrectly charged three 25--43 second dtype/metadata failures against
the node budget. It was stopped and preserved. The active run uses the isolated
budget-aware baseline copy and starts from an empty journal.

### Live baseline validation

At 2026-08-31 20:23 UTC, the active baseline had six retained attempts and two
budget-counted nodes. Four failed attempts with execution times of 7.82, 9.13,
37.83, and 35.66 seconds remained in the journal as diagnostics but did not
consume the search budget. The first long execution ran for 1,164.37 seconds
and failed after real training, so it correctly counted. The second effective
node completed successfully with PetFinder RMSE 18.1561. The controller then
reported `2/50 steps completed` and continued with three tasks in flight.
The optional MLEBench format service on port 5005 was not installed in the
runtime environment, so the successful submission was validated directly
against `prepared/public/sample_submission.csv`: both had 992 rows, identical
`Id` order, and columns `Id, Pawpularity`; the result had no nulls, 992 unique
predictions, and a prediction range of 19.165056--84.26641.

At 2026-08-31 21:25 UTC, the baseline reached 10/50 budget-counted nodes from
19 retained attempts. Nine quickly detected failures remained available for
diagnosis but were excluded from the node budget. Six of the ten effective
nodes completed successfully. The best RMSE remained 18.1561; the tenth
effective node completed with RMSE 18.5498, after which the controller
immediately replenished the three-task pipeline.

The now-unused L40S continues serving its existing model and is kept occupied
by the owned `keep_qwen_l40s_busy.sh` request loop. Its PID and command marker
are checked before reuse or termination, so the filler cannot be confused with
the A100 agent or A10 experiment process.

Before the delayed scheduler phase, the exact remote configuration was composed
through MLEvolve's own configuration loader. It resolved to hardware-aware
mode, 50 nodes, a 43,200-second budget, the A100 vLLM endpoint, profile-based
prediction, scheduler/HWKD/preflight enabled, and both
`parallel_search_num=null` and `parallel_job_cap=null`. Runtime-profile tests
then passed 81/81 both locally and inside the A10 Pod. These tests cover profile
persistence, epoch-based estimation when no runtime profile exists, zero-value
fallback, and per-job fallback from an unavailable predictor to branch profiles.

### Effective-node visualization validation

The comparison renderer now imports the same `node_counts_toward_budget`
helper used by the live runner and final phase verifier. Consequently, a buggy
node with less than 60 seconds of execution remains in the raw journal but is
absent from both the Gantt chart and completed-node count; a long failed run
still appears and consumes a node. The metric graph renumbers the filtered
nodes consecutively, so its horizontal axis is the budget-counted node number
rather than the raw attempt number. The plot title also records the actual
A100-hosted Qwen agent and one-A10 execution setup.

A preview made from the live baseline journal reported 10/50 completed nodes,
6 scored nodes, best RMSE 18.1561, and a 1.44-hour span. The preview was checked
visually at `.cache/petfinder_a100_a10_live/baseline_preview.png`. Focused
accounting and renderer regression tests passed 4/4.

At 2026-08-31 21:43 UTC, the baseline reached 12/50 budget-counted nodes from
21 attempts. The twelfth effective node failed with CUDA Out-of-Memory after
267.26 seconds: two concurrent candidates already held approximately 9.07 and
10.49 GiB, while the third candidate had approximately 2.47 GiB in use and
could not allocate another 50 MiB. This is not a quick static failure and
therefore correctly consumes a node. It is also direct baseline evidence for
the scheduler comparison: original MLEvolve admitted the third GPU workload
without profile or live-telemetry placement, while the modified phase is
expected to reject or delay the equivalent admission without a fixed job cap.
The two established training processes remained alive and the controller
continued after recording the failed node.

At 2026-08-31 22:15 UTC, the baseline reached 14/50 effective nodes and
produced a new best RMSE of 17.9916. The node executed for 2,974.21 seconds,
stopped after epoch 13 of 40, and replaced the previous 18.1561 best. Its
node-specific submission contains 992 rows with exactly the required
`Id, Pawpularity` columns and sample-submission ID order, no missing values,
992 distinct predictions, and a prediction range of 21.960424--100.0. The
other two candidate processes remained active, and A10 utilization stayed at
100 percent after the successful process released its memory.
