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

At 2026-08-31 22:20 UTC, the baseline reached 15/50 effective nodes. A second
long training completed successfully after 3,027.35 seconds with RMSE 18.4460.
Immediately before it, a separate overlapping candidate encountered CUDA
Out-of-Memory after only 36.49 seconds. That quick failure is retained as raw
attempt 24 but excluded from the effective-node total, directly validating the
requested short-failure accounting rule under a real memory-pressure event.

At 2026-08-31 23:54 UTC, the baseline reached 16/50 effective nodes. The
effective node ran for 7,207.66 seconds and hit its two-hour execution limit
after completing five-fold CNN validation but before full-data retraining, so
it correctly counts as a long attempted node. Two intervening attempts failed
after 38.96 seconds (invalid tensor dimensionality) and 28.24 seconds (CUDA
Out-of-Memory); both remain diagnostic journal entries but are excluded from
the effective-node budget. Two other candidate processes remained active.

At 2026-09-01 00:09 UTC, those concurrent candidates completed and advanced
the baseline from 16/50 to 19/50. Their RMSE values were 18.0570, 18.5776, and
18.7309 after 6,302.64, 6,164.66, and 547.19 seconds respectively. A separate
9.93-second dependency-version failure remained in the journal but was
excluded. The A10 briefly reached zero resident processes after all three
completions while the still-running controller requested the next candidates
from the A100 agent; no filler was inserted into that experiment transition.

At 2026-09-01 00:17 UTC, the baseline reached 20/50 effective nodes. The
twentieth node extracted image embeddings, fit five LightGBM folds, and
completed successfully in 315.94 seconds with held-out RMSE 18.7644. The
controller had already launched the next candidate, which restored A10 load
without any fixed parallel-job cap or external filler.

At 2026-09-01 01:53 UTC, the baseline reached its halfway point at 25/50
effective nodes. The twenty-fifth node completed successfully after 5,490.93
seconds with RMSE 18.3310; one intervening quick failure remained excluded.
The controller immediately continued with a new candidate using approximately
15.8 GiB at 100 percent A10 utilization. At this milestone there were 44 raw
attempts, 19 excluded quick failures, and 17 scored effective nodes; the best
RMSE remained 17.9916.

At 2026-09-01 02:33 UTC, the baseline had 27/50 effective nodes from 57 raw
attempts; 30 quickly detected failures were excluded and 17 effective nodes
had valid RMSE values. The best remained 17.9916. The active candidate used
19,253 MiB at 100 percent A10 utilization, while the A100 Qwen agent used
73,483 MiB and continued serving requests. Process identity inspection found
no A100 filler to terminate: the only GPU process was the real vLLM engine.

The long-run controller was hardened without interrupting the active candidate.
On restart it now discovers the newest phase journal and passes it through
`resume_journal`, preserving the existing workspace and node tree. The
persistent monitor now verifies the controller command identity and restarts
it only when the experiment is incomplete and that controller has actually
stopped. Two focused recovery tests passed, both shell scripts passed syntax
validation, the updated controller was synchronized to the A10, and the live
monitor verified that PID 55065 remained the original running controller.

A local recovery snapshot was then pulled to
`.cache/petfinder_a100_a10_live/baseline_27_nodes_20260901_0234`. It contains
the complete 334 MiB run directory, the exact isolated baseline source, and
the controller state. The copied journal parsed successfully with 58 raw
attempts and 27 budget-counted nodes. The PetFinder input links were restored
as symbolic links to the canonical server dataset paths, so transferring the
snapshot back to a machine with the same dataset layout preserves the resume
contract. The live remote run continued during this copy.

At 2026-09-01 03:26 UTC, the baseline reached 28/50 effective nodes. Node
`73176972ad114f43bba3edcb8210a2f0` trained EfficientNet-B0 with a gated
metadata-fusion head for 2,869.74 seconds and completed successfully with
validation RMSE 18.4674; early stopping fired at epoch 17 after the best epoch
9. It produced a 992-row submission and increased the scored-node count to 18,
while the overall best remained 17.9916. Five immediately preceding failures
ran for 7.07--15.88 seconds and remained excluded. The two other long-running
candidates continued on the A10, and the controller immediately used the A100
agent to generate a replacement without inserting a filler or fixed job cap.

At 2026-09-01 03:32 UTC, the baseline reached 29/50 effective nodes. Node
`a551cfd9bd3447479ded3ea740f50a4c` completed successfully after 1,851.67
seconds with an EfficientNet-B0 plus metadata MLP. Early stopping fired at
epoch 14, validation RMSE was 19.0494, and the generated submission contained
992 rows with predictions from 26.11 to 83.60. This node ran concurrently with
the still-active 19.2-GiB candidate, so the A10 remained saturated while the
A100 agent began generating the replacement task.

At 2026-09-01 03:50 UTC, the baseline reached 30/50 effective nodes. Node
`6bbb77cb72314ebfa309f926b57b5a99` ran for 7,207.07 seconds and reached the
two-hour execution timeout during epoch 17 of 35. Its best intermediate
validation RMSE was approximately 18.34 at epoch 12, but the timeout occurred
before final submission generation, so the node is recorded as a long failed
attempt and correctly consumes one effective-node slot. Four neighboring
18.75--44.79-second ViT failures remained excluded. After memory release, the
next candidate began on the same A10 and the A100 agent continued planning a
multi-branch fusion candidate.

At 2026-09-01 03:55 UTC, the baseline reached 31/50 effective nodes. Node
`eee741ef762d472392c2395e35b7f564` ran for 130.96 seconds before its
validation test-time-augmentation path failed because PyTorch does not support
the negative-step tensor slice used after `rot90`. The attempt therefore
crossed the 60-second threshold and correctly counts as a long failed node.
The two preceding NaN-validation failures each ran for roughly 43 seconds and
remained excluded. One separate candidate continued in its CPU/data stage
while the A100 agent generated the replacement.

At 2026-09-01 04:41 UTC, the baseline reached 32/50 effective nodes. Node
`a8061185e78b4b7781f27380cc8bd89e` completed successfully after 2,364.75
seconds with validation RMSE 18.3646. Its multi-scale CNN plus metadata model
stopped early at epoch 16, applied five-view test-time augmentation, and wrote
a 992-row submission. The journal then contained 74 non-root attempts: 32
budget-counted nodes, 42 excluded quick failures, and 20 scored effective
nodes; the best RMSE remained 17.9916. Two other long candidates continued on
the A10 after this process released its memory.

At 2026-09-01 05:25 UTC, the baseline reached 33/50 effective nodes. Node
`95cc279bf8e04816a903c76aa66262cf` completed successfully after 4,914.55
seconds with validation RMSE 18.4343. Its ConvNeXt-base model used gated
bilinear metadata fusion, trained for 21 epochs before patience-15 early
stopping, and produced a 992-row submission. The journal contained 76
non-root attempts, of which 43 quick failures remained excluded; 21 effective
nodes had valid scores and the best RMSE remained 17.9916. Two real candidates
continued on the A10 while the A100 agent generated a replacement.
