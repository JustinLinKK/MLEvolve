# ABA A100 PetFinder controlled baseline/scheduler pair

## Purpose

Run a controlled PetFinder comparison on ABA. The baseline is original MLEvolve behavior. The treatment changes only scheduler enablement and adds the exact PerfSeer v3.2 predictor requested for scheduler estimates.

## Hardware allocation

- GPU 1: NVIDIA A100 80GB PCIe in Exclusive Process mode, local Qwen3.8-27B INT8 agent via vLLM.
- GPU 0: NVIDIA A100 80GB PCIe in Default compute mode, concurrent MLEvolve candidate execution.
- No fixed `Max Parallel Jobs` or `parallel_job_cap` is configured.

## Shared experiment settings

- Task: PetFinder Pawpularity regression.
- Dataset: identical local dataset for both phases under `.cache/data/petfinder-pawpularity-score`.
- Seed: 42 for Python, NumPy, PyTorch, generated code, and local Qwen requests.
- Agent model: `lued/Qwen3.8-27B-INT8-W8A16-MTP`, served as `qwen3.8-27b-int8-a100`.
- Generation: single-pass (`use_stepwise_generation=false`), temperature 0.7, three initial drafts.
- Hardware Knowledge Database (HWKD): disabled.
- Preflight checker: disabled.
- Context cache, lesson profiles, review, CUDA documentation retrieval, and hardware context: disabled.
- Baseline node budget: 50.
- Scheduler node budget: 20.

## Treatment-only settings

- Scheduler enabled with `parallel_time_aware`, `backend_aware`, CUDA-process packing.
- Predictor source: exact local `predictor_v3.2/source` package.
- Predictor checkpoint: `predictor_v3.2_transfer_A100.pt`.
- Predictor capture device: CUDA on the experiment process's visible device,
  physical GPU 0. GPU 1 remains dedicated to vLLM.
- Required estimates: training time per epoch and peak VRAM.

## Pre-run verification

- Predictor checkpoint SHA-256: `c89eedb6291f11eab264ac6110defec34a0ce3d26bfc16d778040fc30e803b54` on local and ABA.
- Exact predictor successfully ran through the scheduler adapter on ABA GPU 1:
  - predicted peak VRAM: 1329.730713 MiB
  - predicted training time per epoch: 0.355096 seconds
- vLLM: 0.27.1 CUDA 12.9 build, using Marlin W8A16, FlashAttention 2, MTP speculative decoding, prefix caching, CUDA graphs, and text-only mode.
- Context capacity: 262,144 tokens.
- Warm performance on ABA GPU 1, three deterministic 146-token requests:
  - TTFT: 0.1038, 0.1004, 0.0999 seconds (mean 0.1014 seconds)
  - decode throughput: 79.78, 79.76, 79.69 tokens/second (mean 79.74 tokens/second)
  - end-to-end throughput: 74.48, 74.61, 74.56 tokens/second (mean 74.55 tokens/second)
- The experiment GPU was idle before launch.
- Two simultaneous CUDA allocation processes were verified on GPU 0 before the corrected launch; both created independent CUDA contexts successfully, confirming that Default compute mode supports the experiment's concurrent candidates.
- After the role swap, a repeated warm request on GPU 1 measured 0.0996-second TTFT and 80.38 decode tokens/second.
- All six PerfSeer v3.2-specific scheduler tests passed. The remaining legacy predictor tests reference the old `perfseer_student` package, which is not present in the user-required PerfSeer submodule branch; they are outside this v3.2 experiment path.

## Persistence

The pair launcher writes the current pair path to `.cache/petfinder_aba_a100_current_pair`. Runtime work uses `/dev/shm`; every phase copies its logs, traces, generated programs, metrics, and scheduler runtime to the durable pair directory on exit.

## Aborted infrastructure-invalid attempt

The first launch placed candidate execution on GPU 1. Its first generated candidate had a genuine ResBlock channel bug. A concurrent candidate failed with `CUDA-capable device(s) is/are busy or unavailable`; inspection then showed GPU 1 was configured as `Exclusive Process` while GPU 0 was `Default`. The run was stopped immediately and preserved at `petfinder_aba_a100_20260924T220814Z`. It is not valid comparison data. The GPU roles above are the corrected assignment.

## Aborted source-invalid attempt

The second launch used GPU 0 successfully and demonstrated three simultaneous
CUDA contexts, but its first valid node had RMSE 21.8303 rather than the
historical original-MLEvolve starting region near RMSE 18.16. Root-cause
inspection showed that this run used the current repository with only
`scheduler.enabled=false`. That is not an original-MLEvolve control: current
generation, prompting, and execution changes remain active when the scheduler
is disabled. The run was stopped immediately and preserved at
`petfinder_aba_a100_20260924T221844Z`; it is excluded from the comparison.

## Corrected original control

- Control source: the previously validated original-MLEvolve recovery snapshot
  copied into `.cache/original_mlevolve_baseline_source` on ABA.
- Snapshot content hash after the seed-control patch:
  `0f71b6d12668304e9a46a3d697010a17634cc42ed86957ff0d0cf361ac7f9f4f`.
- The control snapshot receives only deterministic-control fixes shared by the
  treatment contract: seed 42 for local Qwen requests, controller PyTorch, and
  candidate-process Python/NumPy/PyTorch. It receives no scheduler, Predictor,
  HWKD, preflight, context-cache, or lesson-profile treatment.
- Formal original-control configuration parsing passed with 50 nodes,
  scheduler disabled, and `parallel_search_num=null`.
- Two identical Qwen requests with request seed 42 produced identical output.
- Corrected durable run root:
  `.cache/experiments/petfinder_aba_a100_20260924T223611Z`.

That corrected run was stopped before candidate execution after draft 2 exposed
one remaining agent-setting mismatch: the original snapshot limited Qwen output
to 4,096 tokens, while the current treatment allowed 16,384. The truncated
control response failed code extraction, so the run is excluded. The control
default and treatment launcher are now both explicitly 16,384 tokens before
the next launch.

The post-fix controlled pair started at
`.cache/experiments/petfinder_aba_a100_20260924T224146Z`.

That run found a scope error in the control-only subprocess seed patch:
`configured_seed` was defined in the unused scheduler branch rather than the
original subprocess branch. Candidate execution failed before user code ran,
so the run was stopped and excluded. The definition and prologue injection were
moved into `_run_subprocess`; a real two-process regression test then produced
identical Python, NumPy, and PyTorch random values with no exception.

The subprocess-seed-fixed controlled pair started at
`.cache/experiments/petfinder_aba_a100_20260924T225127Z`.

The subprocess bug was gone, but all three generated initial candidates were
quick code failures and produced no RMSE. This failed the historical-start
gate, so the run was stopped and excluded. Historical records then identified
two exact September-2 generation-contract details: Qwen used an 8,192-token
completion budget, and fixed-seed retries need a numbered format instruction
so they do not repeat an identical malformed response. The control snapshot
and treatment launcher now both use 8,192 tokens; the original control coder
received only the shared retry-safe prompt fix. Its regression test passed.

The parity-corrected controlled pair started at
`.cache/experiments/petfinder_aba_a100_20260924T230257Z`.

### Live validation snapshot

At 23:53 UTC the original-MLEvolve control reached 15/50 budget-counted nodes
with three tasks active. Its best RMSE improved from 19.0999 to 18.1502. Three
candidate jobs remained in the pipeline, and overlapping long-running candidate
processes were observed on physical GPU 0, while Qwen vLLM remained isolated on
physical GPU 1. GPU 0 was at 100% utilization and approximately 285 W during a
long-running training sample; fresh replacement candidates continued to enter
the third slot, ruling out a stalled controller.
Fast candidate-code failures were excluded from the node budget as intended; no
controller, CUDA, or vLLM infrastructure error was present.

While the control runs from its isolated snapshot, the current treatment
source was synchronized to `/home/yufan/MLEvolve` on ABA. A composite SHA-256
over `run.py`, the phase launcher, Scheduler service, executor, and local-Qwen
client matches locally and remotely:
`af8ca10c442112f3490d864f1c5b3da493102789b192800e76bba1de8304b843`.

## Final result

Both phases completed cleanly with process exit code 0. The original MLEvolve
control completed 50/50 budget-counted nodes in a 9.92-hour recorded span: 29
successful/scored executions, 3 true candidate runtime failures, and 18
external stops or non-runtime failures; its best validation RMSE was 18.1502
and peak execution concurrency was 3. The scheduler treatment completed 20/20
budget-counted nodes in a 2.20-hour recorded span: 9 successful/scored
executions, 10 true candidate runtime failures, and 1 external stop or
non-runtime failure; its best validation RMSE was 18.8516 and peak execution
concurrency was 5.

The counts above use the accounting rule active in each recorded run. True
candidate runtime failures are red, successful executions are blue, and
scheduler/watchdog terminations or other non-runtime failures are gray. Because
the two phases have different node budgets and generated different candidate
workloads, the 9.92-hour and 2.20-hour spans are reported directly and are not
presented as an equal-work speedup.

- Combined Gantt and metric-node figure:
  `records/2026-09-24_petfinder_aba_a100_baseline50_vs_scheduler20.png`
- Figure SHA-256:
  `e2559b112835bb72d285120920bb1c899951b0bde62f36665b904ad1064e0dfb`
- Durable local trace bundle:
  `records/2026-09-24_petfinder_aba_a100_controlled_pair_raw`
- Durable ABA source:
  `.cache/experiments/petfinder_aba_a100_20260924T230257Z`

## Post-run first-20 and gap diagnosis

Using the accounting rule active in each phase and ordering nodes by completion,
the first 20 budget-counted nodes took 10,981 seconds (3:03:01) for the original
MLEvolve control and 7,934 seconds (2:12:14) for the scheduler treatment. The
observed treatment throughput was therefore 1.384 times the control throughput,
or 27.75% less elapsed time. This is not an isolated scheduler speedup: the
control's first 20 contained 10 scored nodes with best RMSE 18.1502, while the
treatment contained 9 scored nodes with best RMSE 18.8516, and the generated
candidate programs were not matched.

The largest apparent treatment gap is between the 10th counted completion at
10:32:51 UTC and the 11th counted execution start at 10:57:46 UTC. The local
Qwen service was active throughout that 24:55 interval: the log contains 28
generation responses, 21 scheduler-round submissions, and 19 completed
execution attempts wholly inside the interval. All 19 attempts ran for less
than 30 seconds and therefore do not appear as budget-counted bars. Sixteen
were refused after Predictor conversion/estimation failures and three failed
with candidate TypeErrors. The trigger immediately after node 10 was
`Expanded root draft budget from 5 to 20 to satisfy the experiment budget`,
which led to serial draft generation followed by a fail-closed Predictor retry
storm. Thus the gap was not a vLLM outage; it exposes Predictor coverage and
generation/admission-loop problems in the treatment path, amplified visually by
the chart's budget-node filter.

Seed 42 was present in both saved configurations, but the phases did not feed
identical requests to Qwen. The control source root was
`.cache/original_mlevolve_baseline_source`; the treatment source root was the
current `/home/yufan/MLEvolve`. Their first prompt lengths were 22,605 versus
25,212 characters and their prompt SHA-256 prefixes were `dcd3ebc35a3fad7b`
versus `f2b35daa8747f7db`. Their first generated-code SHA-256 prefixes were
therefore also different: `7029ab351ee3bfeb` versus `d8a234b678464c5e`.
Identical random seeds reproduce identical sampling only for identical inputs
and code paths; they cannot align different prompts. In addition, scheduler
parallelism changes completion/feedback order, so a feedback-driven search can
diverge even from initially matched candidates. An exact metric-step comparison
requires either a fixed candidate replay or an identical generator/source with
candidate generation decoupled from completion order.

The prompt difference is not in the system role: both first system messages are
identical (714 characters, SHA-256 prefix `c2669b24c5156a00`). It is in the
composed user and assistant messages. The baseline first user message is 18,620
characters; the treatment's is 20,899. The current treatment source
unconditionally adds a `Backend design contract`, a `CPU preflight adapter
contract`, a `Scheduler Model Family Contract`, offline-package restrictions,
the 20-step budget instead of 50, and a PetFinder adapter fixture. It also uses
a different assistant prefix declaring that dataset preview is already
complete. The baseline source uses the older generic implementation guidance
and assistant prefix. This occurs even though `preflight.enabled=false`, because
`agents/draft_agent.py` and `agents/prompts/impl_guideline.py` add those sections
without gating them on that configuration. Consequently, the shared task/data
description and seed did not produce the same actual LLM request; this is a
control-confounding prompt-construction bug in the experiment setup.
