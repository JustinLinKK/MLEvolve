# PetFinder: L40S Agent, A100 Baseline/Scheduler, and A10 Scheduler

Status: running. This record must be updated with final metrics and artifacts
before the experiment is considered complete.

## Experiment contract

- Task: PetFinder Pawpularity regression; metric is validation root mean
  squared error (RMSE), lower is better.
- Agent: local text-only Qwen3.8-27B INT8 W8A16 served by vLLM on one L40S.
- A100 comparison: original MLEvolve baseline first, then the hardware-aware
  MLEvolve scheduler on the same A100.
- A10 comparison: hardware-aware MLEvolve scheduler on one physical A10.
- Predictor mode: `branch_profile`; the unfinished learned predictor is not
  used.
- Parallelism: `parallel_search_num: null` and `parallel_job_cap: null`.
  Incremental admission and live GPU telemetry determine safe concurrency.
- Hardware knowledge: enabled and injected into prompts from the checked-in
  hardware-knowledge records; the optional Neo4j mirrors are unavailable and
  are not treated as evidence sources for this run.

## L40S agent

- Pod: `gpu-dev-l40s-1gpu-6fcb7f9bfc-vtlpj`.
- Served model: `qwen3.8-27b-int8-l40s`.
- Model context: 32,768 tokens, validated with an 18,016-token prompt.
- Quantization/runtime: compressed-tensors Marlin W8A16 under vLLM 0.28.0.
- During concurrent MLEvolve generation the server sustained roughly 40--76
  aggregate output tokens/s and remained at 100% GPU utilization.
- The old Qwen client limited every response to 4,096 tokens. Seven of the
  first 38 requests ended with `finish_reason=length`, including a generated
  training script truncated inside its epoch loop. The active scheduler source
  now requests 8,192 tokens and derives a context-safe retry budget from the
  server-reported context and prompt lengths. Regression tests: 6 passed.

## A100 baseline

- Pod: `mlevolve-a100-1gpu-77d4fc9848-vx6sv`.
- Run root:
  `/root/downeyflyfan/mlevolve_a10_baseline_20260829/runs/a100_baseline_timm_20260830_220623`.
- PID: 11624.
- Scheduler is disabled and the baseline snapshot has no hardware-knowledge
  configuration.
- First valid node: `4d72df5f27cf451bba295cb173b49112`, validation RMSE
  17.7193 at step 6/50.
- At 23:42 UTC the baseline had reached 21/50 with three tasks active. The
  best RMSE was still 17.7193. GPU utilization varies between candidate
  generation and training; the process and its automatic handoff controller
  remained alive.
- A controller waits for PID 11624 and will start the A100 scheduler run with a
  new per-run scheduler database, the 8K output-budget fix, branch profiles,
  and no fixed parallel-job cap.

## A10 scheduler

- Pod: `gpu-dev-a10-755d877bcd-zbbd2`.
- Valid run root:
  `/root/downeyflyfan/mlevolve_a10_scheduler_20260829/runs/a10_scheduler_gpu2_valid_32k8k_20260830_225458`.
- PID: 114385.
- Parent process mask: `CUDA_VISIBLE_DEVICES=2`; scheduler logical device index:
  0. A live sampler check resolved logical 0 to physical 2 and read
  1,569/23,028 MiB, matching `nvidia-smi`.
- Scheduler runtime root is isolated under the valid run directory. At launch
  it contained zero jobs and zero runtime profiles.
- Journal inspection proves the Hardware Knowledge Database path is active,
  not merely configured: accepted nodes contain `hardware_context.found=true`
  with the NVIDIA A10 hardware key, 22,589 MiB VRAM, compute capability 8.6,
  CUDA 13.0, enabled process backends, the 19,200 MiB safety budget, and live
  counts of runtime/batch observations. Optional external graph/vector mirrors
  are empty and are not used as evidence.
- The first accepted scheduler job,
  `11f3c8fa-d893-4318-9fcf-094a1d755970`, was submitted at 23:12 UTC and ran
  on physical GPU 2 with the CUDA-process backend. Its first epoch took
  368.793 seconds. The resulting `epoch_1` branch profile recorded a positive
  total-runtime estimate of 5,531.898 seconds and a remaining-runtime estimate
  of 5,163.105 seconds at confidence 0.75. This is live evidence that the
  former `None`/zero runtime-estimation path is repaired.
- The job then early-stopped after 8/15 epochs. Its final validation RMSE was
  18.7129, best epoch was 4, actual execution time was 823.091 seconds, and
  peak VRAM was 2,858.5 MiB. Completion updated the branch profile to
  823.091 seconds at confidence 0.95 with source
  `mlevolve_completed_wall_clock`; MLEvolve accepted the node as non-buggy.
- At 23:43 UTC the A10 search had reached 5/50. The second executed job,
  `945a544e-0e83-4575-bee2-b11c3f5175e9`, completed 12/15 epochs in 709.768
  seconds, achieved RMSE 18.3922 at epoch 8, and produced a positive final
  runtime profile at confidence 0.95. Earlier drafts rejected before execution
  for violating the FP32 precision contract or unsafe OOM fallbacks remain in
  the node count but not in scheduler execution metrics.
- The third executed job changed model family from EfficientNet-B0 to
  EfficientNet-B3. The scheduler correctly did not reuse the incompatible B0
  profile; after epoch 1 it measured a new positive 1,697.888-second runtime
  estimate at confidence 0.75. GPU 2 reached 99--100% utilization during this
  training phase.
- That B3 job completed 10/20 epochs in 511.260 seconds, recorded peak VRAM
  5,524.7 MiB, and reached RMSE 18.3897 at epoch 5. Its final runtime profile
  has confidence 0.95 and the associated batch observation persisted both
  time and memory evidence.
- The second B0 job predated the worker-side VRAM persistence patch. Its exact
  GPU-2 execution window (23:31:04--23:42:58 UTC) contained a measured
  `nvidia-smi` peak of 5,172 MiB. Only that previously-null observation was
  backfilled with avg/peak 5,172 MiB, total 23,028 MiB, and an explicit
  telemetry provenance tag; the live SQLite integrity check remained `ok`.

## Incremental recovery snapshot

- A10-only snapshot after the B3 result:
  `.cache/petfinder_live_20260830/snapshots/20260831_0002` (84 MiB). It has
  seven journal nodes, four scheduler jobs including the newly active job,
  three completed runtime profiles, and a scheduler database whose integrity
  check passed.
- New snapshot root:
  `.cache/petfinder_live_20260830/snapshots/20260830_2348` (285 MiB,
  103 files). It contains the A100 journal through 22 nodes and the A10
  journal through 5 nodes. The copied A10 scheduler database passed
  `PRAGMA integrity_check` and contains both completed jobs and profiles.
- Local snapshot root:
  `.cache/petfinder_live_20260830/snapshots/20260830_2328`.
- A10 scheduler snapshot: 21 MiB; SQLite integrity check passed, with one
  completed job and one runtime profile. The journal contains the virtual root
  plus three search nodes.
- A100 baseline snapshot: 245 MiB; journal contains the virtual root plus 14
  search nodes, three valid scored nodes, eight submission CSV artifacts, and
  four model checkpoints. The best copied RMSE is 17.7193.
- Input dataset symlinks were intentionally not dereferenced by `kubectl cp`;
  the run outputs, code, databases, checkpoints, logs, and submissions were
  copied, while the shared PetFinder dataset remains at its documented remote
  data path.

### Excluded A10 attempts

The following runs are retained for diagnosis but excluded from all metrics:

1. `a10_scheduler_timm_20260830_220624`: generated under the former 16K
   context, then executed its first candidate on physical GPU 0 because the
   scheduler confused CUDA logical index 0 with `nvidia-smi` physical index 0.
   The candidate failed before epoch 1 with an OpenCV empty-image error.
2. `a10_scheduler_gpu2_32k8k_20260830_224324`: output budget was corrected, but
   the scheduler runtime root override targeted the nested settings field and
   therefore reused the old database.
3. `a10_scheduler_gpu2_clean_32k8k_20260830_224504`: database and scheduler
   device index were correct, but the unmasked embedding workers defaulted to
   physical GPU 0, so the run did not satisfy the one-A10 isolation contract.

## Code corrections made during live validation

- MLEvolve search parallelism accepts `null`; the example configuration no
  longer fixes it to 2.
- Runtime estimates reject zero-valued profiles and fall back to measured
  branch epoch time.
- A completed runtime profile can now be reused after a generated script
  changes signature only when workflow, branch, model family, hardware,
  backend, and batch size remain compatible. Cross-branch reuse is rejected.
  The focused and full time-aware scheduler tests passed 3/3 and 76/76. An
  in-memory clone of the live A10 database also exercised the continuity
  trigger with a changed signature: it produced an 887.210-second estimate at
  confidence 0.80 from the completed 709.768-second, 12-epoch profile, while
  preserving database integrity.
- Qwen output budget increased from 4,096 to 8,192 with a dynamic context-safe
  retry; fixed 2K retries no longer truncate long training scripts by default.
- Added logical-to-physical CUDA device mapping for scheduler worker launch,
  NVIDIA Management Library telemetry through `nvidia-smi`, and hardware
  driver detection. The combined configuration, device-mapping, process
  backend, Qwen output-budget, and time-aware scheduler regression suite
  plus the live-journal chart parser/render checks passed 105 tests.

The sibling comparison checkout was independently fast-forwarded to the
verified latest `origin/hardware-awared` commit `dcb0611`; the former
`420e989` target is not the current branch tip.

## Required completion evidence

- 50/50 nodes and final best RMSE for the original MLEvolve baseline on A10
  physical GPU 2.
- 50/50 nodes, runtime profiles, Hardware Knowledge Database evidence,
  placement trace, and final best RMSE for the modified scheduler on the same
  A10 physical GPU 2.
- One PNG with scheduling Gantt chart above and metric-versus-node curves below
  for all comparison runs.
- Final run directories, scheduler databases, logs, submissions, and plot
  copied locally under `.cache`/`records`, followed by tests and Git push.

The final plotting entry point now accepts repeated live-journal run
specifications instead of the obsolete two-V100 trace constants. It renders
one image with one column per run, Gantt windows above, and RMSE-versus-search
step plus running best below. Two parser/render tests passed, and a visually
inspected two-run preview was written to
`.cache/petfinder_live_20260830/preview_comparison_2348.png`; the records PNG
will be generated only after all three 50-node journals are final.

## 2026-08-31 00:14 UTC live checkpoint

- The A100 baseline remains live as PID 11624 at 24/50 completed search steps
  (25 journal nodes). Three autonomous baseline tasks are running, the A100 is
  at 100% utilization with 53,160 MiB allocated, and the best RMSE remains
  17.7193. Controller PID 17736 is still waiting and will start the isolated
  A100 scheduler run only after the baseline exits.
- The A10 scheduler remains live as PID 114385 at 7/50 completed search steps
  (8 nodes), with `parallel_search_num=null`. Its fourth scheduler job ran for
  675.038 seconds and failed before epoch 1 because the generated candidate
  projected a `1x1` tensor through a `1280x256` linear layer. This is a model
  candidate shape bug, not a scheduler failure; the worker persisted the
  failed completion and MLEvolve immediately submitted a debugging child.
- The L40S vLLM process remains healthy with 39,141 MiB resident and is serving
  the text-only Qwen3.8-27B INT8 agent used by both searches.
- After the failed candidate was journaled, the A10 search exposed a liveness
  bug: an unusable stage-owned debug repair returned `None` after reserving the
  parent's only debug-child slot, so `has_selectable_work()` became false and
  the run exited early at 7/50. A regression test first reproduced the leaked
  reservation. The latest-branch implementation now releases that reservation
  before returning; focused tests passed 47/47 locally and 4/4 in the A10 pod.
- The run was resumed from the existing `logs/journal.json`, not restarted.
  Resume PID 135338 restored seven completed nodes, five branches, the existing
  workspace, and the same scheduler runtime database, then began generating
  three new candidates with no fixed parallel-job cap. A consistent local
  recovery snapshot is at
  `.cache/petfinder_live_20260830/snapshots/20260831_0022_a10_resume`; it contains
  the eight-node journal, the failed-node result, resume log, scheduler events,
  and SQLite backups whose integrity checks both returned `ok`.
- The resumed process then reproduced the same malformed stage-repair condition
  on node `900561c006004403888adf5b49f2f357`. This time the live fix released the
  reservation: instead of stopping, the run logged `Submitted next task based
  on node None`, selected another node, and remained at three active generation
  tasks. This is end-to-end evidence for the liveness repair, beyond the unit
  regression test.

## 2026-08-31 00:30 UTC objective correction

- The comparison hardware was corrected from A100 to one A10: the final run
  order is now original MLEvolve baseline first, followed by the fresh
  profile-based scheduler plus Hardware Knowledge Database run on the same
  physical A10 GPU 2. L40S continues to host the text-only Qwen agent.
- The superseded A100 run was stopped cleanly after its partial search and its
  automatic scheduler controller was stopped before it could launch. A
  dedicated A100 filler PID 34262 now holds that otherwise-unused GPU at 100%
  utilization.
- The earlier A10 scheduler trial was also stopped cleanly and remains an
  excluded diagnostic run. Its liveness fix and snapshots are retained, but
  its metrics will not be mixed into the corrected comparison.
- The corrected A10 baseline started as PID 137047 at
  `/root/downeyflyfan/mlevolve_a10_baseline_20260829/runs/a10_baseline_gpu2_l40s_20260831_003039`.
  It uses `CUDA_VISIBLE_DEVICES=2`, scheduler disabled, and
  `agent.search.parallel_search_num=null`; the three-worker generation pool is
  the original MLEvolve interpreter capacity, not a configured scheduler job
  cap. Controller PID 137133 will start a fresh scheduler run only if the
  baseline journal reaches 50 completed nodes.
- The first baseline draft completed code generation and review at 00:39:21
  UTC as node `477b2d675e924d9ca407a54381b1f4f8` (branch 1). Original
  MLEvolve's three stepwise L40S generation calls took about 132, 139, and 192
  seconds before code review; draft 2 then began immediately. This latency is
  agent generation time, not an idle or failed A10 process.

## 2026-08-31 00:47 UTC A10 baseline checkpoint

- The corrected A10 baseline PID 137047 remains live. The L40S vLLM engine is
  actively serving one request at roughly 20 generated tokens/second with
  39,141 MiB resident and about 89% GPU utilization; its health request and
  candidate-generation requests returned HTTP 200.
- Baseline draft 2 completed code generation and review at 00:47:28 UTC as
  node `029e0ef3a91f4089a043b5d4360825da`. Draft 3 began immediately.
- A10 physical GPU 2 remains deliberately reserved and idle only during this
  initial code-generation phase. It will execute the three generated PetFinder
  candidates once draft 3 is ready. A10 physical GPUs 0, 1, and 3 remain at
  100% utilization.

## 2026-08-31 01:00 UTC first valid A10 baseline result

- The initial three candidates all executed on the intended baseline path.
  Their first failures were candidate defects rather than infrastructure
  defects: a missing optional LightGBM dependency, prose embedded in Python,
  and a fusion-layer shape mismatch. The third candidate briefly allocated
  about 11 GiB on physical A10 GPU 2 before its shape error.
- Original MLEvolve continued through its autonomous debug branches. Node
  `e52d2a02e6ca48b08eb168c7a68f4615` became the first valid result at search
  step 8 with RMSE 19.4535 after about 251 seconds of execution. Its submission
  passed format validation and was saved as the current top solution.
- Two other baseline candidates remained live concurrently on physical A10 GPU
  2 with `CUDA_VISIBLE_DEVICES=2`. They used about 16.8 GiB total while the GPU
  remained at 100% utilization; no out-of-memory event occurred.
- A recoverable local checkpoint containing the 9-node journal, filtered
  journal, current best submission, and runner log is stored at
  `.cache/petfinder_live_20260830/snapshots/20260831_0100_a10_baseline_first_valid`.
