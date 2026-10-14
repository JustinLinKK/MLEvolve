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

## 2026-08-31 01:27 UTC A10 baseline progress

- The original baseline reached 13/50 completed search steps and produced four
  valid submissions. Best RMSE improved through 19.4535, 19.0419, and 18.7125;
  the fourth valid candidate scored 19.4483 and did not replace the best.
- Node `3ab4ee1f66344a9e9e13fb8aa00d5f6f` is the current best at RMSE 18.7125.
  Its improvement over its parent was 0.7410 RMSE.
- Three baseline candidates continued concurrently on physical A10 GPU 2. The
  card remained at 100% utilization with about 18.2/23.0 GiB allocated and no
  out-of-memory event.

## 2026-08-31 02:33 UTC A10 baseline 20-node checkpoint

- The original baseline reached 20/50 completed search steps. The current best
  is node `d133988b37774feb9ba6973b425537e2` at RMSE 18.2042; later valid nodes
  scored 18.6373 and 18.8847 without replacing it.
- Step 20 failed quickly with an `AttributeError` in the generated candidate;
  original MLEvolve immediately submitted its debug child. This is counted as
  a baseline search step, consistent with the original algorithm.
- The live run remains on physical A10 GPU 2 with no fixed scheduler job cap.
  A local recovery snapshot of the journal, filtered journal, best submission,
  and runner log is at
  `.cache/petfinder_live_20260830/snapshots/20260831_0233_a10_baseline_20`.

## 2026-08-31 03:50 UTC A10 baseline 30-node checkpoint

- The original baseline reached 30/50 completed search steps. Node
  `24b425fcb02e4609aa111b0767da2952` set a new best RMSE of 18.0473 at step 26;
  subsequent candidates had not improved it by step 30.
- Concurrent baseline candidates reached about 21.6/23.0 GiB on physical A10
  GPU 2 without an out-of-memory failure. All three long-running candidates
  completed, released the GPU, and the search continued through result parsing
  and new code generation.
- A second live recovery checkpoint is stored at
  `.cache/petfinder_live_20260830/snapshots/20260831_0350_a10_baseline_30`.

## 2026-08-31 05:04 UTC A10 baseline 40-node checkpoint

- The original baseline reached 40/50 completed search steps. Node
  `902458b51e9a4da4a5ecd2341d5fc924` established the current best RMSE of
  17.680440943125806 at step 37, improving on the previous 18.0473 best.
- The baseline remained on physical A10 GPU 2 throughout, with the L40S agent
  producing the code and reviews. No fixed scheduler cap was configured and no
  out-of-memory event interrupted the search.
- The 40-node recovery snapshot is stored at
  `.cache/petfinder_live_20260830/snapshots/20260831_0504_a10_baseline_40`.

## 2026-08-31 A10-agent correction and runtime diagnosis

- The superseded L40S-agent baseline completed 50/50 search steps on physical
  A10 GPU 2. Its best result was RMSE 17.680440943125806. The complete local
  recovery copy is
  `.cache/petfinder_live_20260830/final_a10_baseline_gpu2_l40s_20260831_003039`.
  This result remains diagnostic and will not be reported as the final
  same-agent comparison.
- The requested final comparison was corrected to use the local A10-hosted
  Qwen3.8-27B INT8 agent. Three-way tensor parallelism is structurally invalid
  for this checkpoint because its 16 vision attention heads are not divisible
  by 3, even in vLLM text-only mode. The viable allocation is therefore two
  A10s for the agent and one A10 for the PetFinder experiment.
- vLLM 0.28.0 with PyTorch 2.13.0 and CUDA 12.9 loaded the compressed-tensors
  checkpoint on two A10s at about 15.1 GiB per GPU using the Marlin INT8
  kernel. Startup then exposed two independent runtime faults: the custom
  all-reduce path returned `invalid argument`, and the packaged PyTorch CUDA
  runtime produced illegal-memory-access errors in an isolated ordinary
  BF16 `torch.sort` test. PyTorch 2.11.0 with CUDA 13.0 reproduced the same
  isolated failure, so model weights and scheduler code are not the cause.
- A CUDA 12.8 PyTorch environment was downloaded for the compatibility test,
  but the Nautilus node became `ContainerStatusUnknown` before that test ran.
  Kubernetes created a replacement deployment request. Because three
  co-located A10s remained unschedulable, the allocation was split into a
  two-A10 local-agent Pod and a later one-A10 experiment Pod. The two-A10
  request remains Pending, so no A10-agent TTFT/TPS or corrected 50-node
  comparison is yet claimed.
- The live ConfigMap no longer auto-launches the obsolete four-A10 TP4
  comparison when the replacement Pod starts. Controller PID 2092640 waits
  for readiness and will automatically install the PyTorch 2.11 CUDA 12.8
  stack, the vLLM 0.26 CUDA 12.9 binary, and run an isolated BF16 sort check
  before any model server is started.
- A separate one-A10 `gpu-dev-a10-experiment` Deployment is also queued for
  PetFinder. The `qwen38-a10-agent` ClusterIP service exposes port 8000 from
  the two-A10 agent Pod to the experiment Pod without a remote Internet agent
  call. Both requests are Pending at this checkpoint.
- Focused regression coverage for positive runtime estimates, nullable search
  parallelism, and absent fixed scheduler cap passed 44/44 locally after the
  node loss.

## 2026-08-31 09:47 UTC restored L40S-agent/A10-experiment contract

- The active comparison is again the original contract: one L40S serves the
  text-only Qwen3.8-27B INT8 agent, while one A10 runs the PetFinder baseline
  and scheduler+Hardware Knowledge Database phases sequentially. The attempted
  A10-agent deployment is diagnostic only and has been scaled to zero; no
  persistent data was deleted.
- The completed L40S-agent baseline is accepted as the final baseline half of
  the comparison. Its journal contains the virtual root plus 50 search nodes,
  its log ends at `50/50 steps completed`, scheduler mode is disabled,
  `parallel_search_num` is null, and its best RMSE is 17.680440943125806.
- The fresh one-A10 experiment Deployment requests 4 CPU cores, 16 GiB RAM,
  and one NVIDIA A10, with a 64 GiB RAM limit. Reducing only the reservation
  removed the scheduler's `Insufficient memory` rejection while retaining the
  larger runtime safety limit. It is still Pending because available A10 nodes
  are tainted/reserved or have no schedulable GPU capacity.
- The one-L40S agent Deployment remains Pending. Its ClusterIP service and
  selector are correct, but there is no ready endpoint until the Pod is placed
  and the vLLM health check succeeds. No scheduler result is claimed yet.
- The A10 run tree has been prepared on the shared persistent volume at
  `/root/downeyflyfan/mlevolve_a10_scheduler_active_20260831`; its synchronized
  source is now commit `dce797b`. Its operational config was checked before launch: 50 nodes,
  L40S service endpoint, scheduler and Hardware Knowledge Database enabled,
  `prediction.mode=branch_profile`, `parallel_search_num=null`, and
  `parallel_job_cap=null`. Python compilation, required imports, task data, and
  task-description checks passed from the same environment the A10 will use.
- Detached tmux controller `mlevolve-a10-goal-controller` polls both requests
  every 30 seconds. Two consecutive ticks were verified. If one GPU arrives
  first, the controller keeps it compute-busy with a controller-owned filler;
  after both the L40S health endpoint and the A10 device check succeed, it
  stops only its own fillers and launches the scheduler phase on CUDA device 0.

## 2026-08-31 10:11 UTC pre-launch merge repair and A10 placement

- Inspection of the merged `run.py` found that commit `5a1b838` had added a
  second call to `prepare_run_context_cache` beside the implementation already
  present from commit `368c926`. This would prepare and freeze context packs
  twice for every run. A lifecycle regression test reproduced two calls before
  the interpreter was constructed; after removing only the merge-added block,
  the local and shared-volume suites both passed 85/85 tests.
- The corrected `run.py` and regression test were synchronized into the A10
  active run tree. The controller was also made idempotent: it will not launch
  a second `run.py`, will resume a stopped partial journal in place, and will
  keep polling until a 50-node journal and 50-node comparison metrics agree.
- The A10 Pod was assigned to `hcc-nrp-shor-c5825.unl.edu` at about 10:09 UTC.
  Its persistent volume attached successfully and the NVIDIA PyTorch image is
  being pulled. This is placement evidence, not yet a Ready GPU or experiment
  launch. The L40S request remains Pending.
- The A10 container became Ready at 10:12 UTC. Live detection reports NVIDIA
  A10, 22,589 MiB VRAM, compute capability 8.6, CUDA 13.2, driver 595.71.05,
  and PyTorch 2.12.0a0 from the NVIDIA 26.04 image. Isolated BF16 `torch.sort`
  and FP16 matrix multiplication both completed successfully, ruling out the
  illegal-memory-access fault seen on the lost older A10 runtime.
- After that self-test, the detached controller was restarted and verified to
  hold the A10 at 100% compute utilization with its owned filler while the L40S
  agent remains unscheduled. The filler uses 2,579 MiB and will be stopped only
  after the L40S API passes its health check.

## 2026-08-31 10:33 UTC A10 local-runtime validation

- A full config validation using the dependency environment on the shared Ceph
  volume stalled in `ceph_mdsc_wait_request` while reading an OpenAI client
  bytecode file. This was filesystem metadata latency, not scheduler code or a
  CUDA deadlock. A plain Python process remained healthy.
- The first node-local environment attempt used `uv pip`; its resolver ignored
  the visible system packages and installed PyTorch 2.13.0, the runtime family
  that previously failed on A10. That temporary `/tmp` environment and cache
  were deleted. The replacement was created with system-site-packages and pip,
  which retained the NVIDIA image's PyTorch
  `2.12.0a0+0291f960b6.nv26.04.48445190` and added only missing packages.
- Node-local imports now complete in 12 seconds. Full config construction on
  the physical A10 proves: PetFinder, 50 nodes, `hardware_aware` mode,
  Hardware Knowledge Database enabled, CUDA-process packing,
  `prediction.mode=branch_profile`, `parallel_search_num=null`, and
  `parallel_job_cap=null`. BF16 sort and FP16 matmul both pass from this exact
  runtime. The launch controller now selects the node-local interpreter with a
  persistent-volume fallback and has restored its A10 filler to 100% compute.

## 2026-08-31 18:15 UTC preflight repair and live scheduler launch

- Commit `85655dc` restores the preflight block that the branch merge had
  dropped from `config.example.yaml`. It also changes node accounting so a
  candidate rejected by stage review or CPU preflight is recorded in pipeline
  telemetry but discarded from the search journal; a seconds-fast rejection
  therefore cannot consume one of the 50 PetFinder nodes. The immediate,
  deferred, and scheduler-batch regression paths pass, and 94 directly related
  tests pass together.
- The exact A10 runtime now has the pinned `model-preflight` submodule installed.
  A real A10-profile smoke candidate completed all static, construction,
  abstract-forward, CPU-training, validation, and memory stages with
  `overall_status=PASS` and `gpu_submission_recommended=true`. The operational
  config uses `nvidia/a10_24gb` and `fail_open_on_internal_error=false`, so a
  broken preflight installation cannot be silently bypassed.
- The synchronized A10 source marker is `85655dc`. Live config construction
  proves `hardware_aware`, 50 retained nodes, `branch_profile`,
  `cuda_process`, `parallel_search_num=null`, `parallel_job_cap=null`, and
  preflight enabled only for hardware-aware mode.
- The L40S Pod became Ready on `hcc-nrp-pki-c1705.unl.edu`. The original
  bootstrap incorrectly resolved the model under `/root`; the actual 29.44-GiB
  checkpoint is under `/root/downeyflyfan`. Installing the vLLM runtime onto
  Ceph then stalled in `ceph_mdsc_wait_request`, so the runtime was moved to
  node-local `/tmp` while the model remained on the persistent volume. vLLM
  0.27.1 recognized `Qwen3_5ForConditionalGeneration`, compressed-tensors
  INT8, the Marlin WNA16 kernel, text-only mode, and a 32,768-token context.
  It reserved 8.59 GiB for 130,343 KV-cache tokens and exposes nearly four
  full-context concurrent requests.
- The live endpoint reached HTTP 200 at 18:10:39 UTC with 39.1/46.1 GiB VRAM.
  During the actual MLEvolve workload, the first ten requests averaged 5.57 s
  time to first token, including Triton just-in-time compilation. Three
  concurrent agent requests sustained 53-59 aggregate generated tokens/s,
  about 18-20 tokens/s per request, with no waiting requests and no errors.
- The profile-based scheduler phase launched on the physical A10 at 18:11:05
  UTC, PID 32631. Its resolved run directory is
  `/root/downeyflyfan/mlevolve_a10_scheduler_active_20260831/runs/a10_scheduler_profile_hkwd_l40s_20260831_181108/20260831_181136_petfinder_scheduler_profile_hkwd_a10_l40s`.
  Hardware sampling is active every two seconds. The controller was repaired
  in place to resolve this nested run directory for progress, completion, and
  resume checks; the live process was not interrupted.
- A broad local scheduler suite reported 384 passes and five execution-test
  failures. An isolated rerun proves the local `.venv` PyTorch lacks RTX 5070
  Ti `sm_120` support; this is a local test-runtime incompatibility, not the
  A10 runtime used by the experiment. The A10's NVIDIA image has independently
  passed its CUDA BF16-sort and FP16-matmul checks.
