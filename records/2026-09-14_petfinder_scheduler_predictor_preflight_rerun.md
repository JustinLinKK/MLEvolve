# PetFinder rerun: Predictor v3.2, scheduler, and pre-flight checker

## Controlled treatment

This live treatment was launched at `2026-09-14 02:50:07 UTC` on one Nautilus
NVIDIA A10.  It uses seed 42, a 50-node budget, a 43,200-second search budget,
`CUDA_VISIBLE_DEVICES=0`, no fixed parallel-job cap, the local text-only
Qwen3.8-27B-INT8-W8A16-MTP vLLM service on one 80 GB A100, Predictor v3.2,
the pre-flight checker, and the telemetry-aware CUDA-process scheduler.
Hardware Knowledge Database use is intentionally off.

### Reproducibility correction for the next clean controller

The original live command sets `agent.seed=42`, which controls MLEvolve's
search-side Python and NumPy randomness.  During the active-trace audit, the
local vLLM request path was found not to forward that seed to Qwen sampling.
The request client now adds `seed=42` for local Qwen/vLLM calls by inheriting
the experiment seed; non-vLLM providers are unchanged.  The focused client,
context-cache, and A10 compatibility suite passed 27 tests, and the patched
client was synchronized to the A10 source.  This change affects a fresh
controller process only, so it does not alter the preserved in-flight trace.

The A100 service was verified before launch with a 262144-token context,
Marlin W8A16 kernels, FlashAttention 2, MTP, prefix caching, and compiled CUDA
graphs.  A post-warmup one-request benchmark measured 0.2642 s time to first
token and 70.69 generated tokens/s for 125 tokens.

## Pre-run gap mitigations

1. Scheduler submissions now default to `packing_eligible=true`.  This only
   permits consideration; safe admission still requires pre-flight approval,
   Predictor/telemetry evidence, backend compatibility, and live video-memory
   headroom.  There is no fixed concurrency cap.
2. Pre-flight repairs are limited to one and review repairs to two, matching
   the established baseline contract instead of repeatedly re-running repair
   loops.
3. The A100 server no longer uses eager execution and has the native 262144
   context.  The local code-generation service is therefore not a remote API
   latency source.
4. A watchdog records progress and sends `TERM` to the verified controller only
   after 3600 seconds with no increase in budget-counted nodes.  Its log and
   the controller log are retained under the run root.

## First live issue and repair

The first reviewed candidate completed generation at `02:59:23 UTC`, then
pre-flight failed before GPU admission with:

```
ModuleNotFoundError: No module named 'model_preflight'
```

The isolated remote source had the checker at
`nn-model-preflight-checker/src/model_preflight`, but the launch `PYTHONPATH`
contained only the MLEvolve root and Predictor source.  This was a deployment
path bug, not a candidate or GPU failure.

The failure was reproduced by importing `model_preflight` with the old active
path.  For the live controller, a source-root link now exposes the package;
the exact import of `check` and `write_json` passed.  The persistent launcher
now requires `MODEL_PREFLIGHT_SOURCE_ROOT` (defaulting to the in-tree checker
`src`) and includes it in `PYTHONPATH`, so a future process has no dependency
on the link.  The controller was not restarted; its next dynamic pre-flight
import can resolve the repaired package.

The first candidate remains a diagnostic non-submission and does not establish
any scheduler performance result.

## Second live issue: data-schema blind spot

The first candidate admitted by pre-flight and submitted to the scheduler was
`527cbd6dc85145a5a42eff0e8ad2c179`.  CUDA-process execution started at
`03:22:18 UTC` and failed after 24.69 s, before completing an epoch.  The
saved runtime result identifies a candidate-model shape defect, not a
scheduler failure:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x10 and 12x64)
```

The prepared public `train.csv` has 14 columns: `Id`, `Pawpularity`, and the
12 metadata columns `Subject Focus`, `Eyes`, `Face`, `Near`, `Action`,
`Accessory`, `Group`, `Collage`, `Human`, `Occlusion`, `Info`, and `Blur`.
The candidate instead requested historical names `Occupy` and `Off_Center`;
its feature filter silently dropped both unavailable names, yielding 10 values
while its model remained 12-wide.  The generic 12-feature fixture cannot
detect a *wrong column name* by shape alone, so it admitted this defective
candidate.  The node is correctly recorded as buggy and is excluded from
metrics.

The persistent checker now statically rejects a literal PetFinder
`TABULAR_COLS` list unless it exactly matches the 12 public metadata columns.
The generation guideline provides the same canonical list and requires model
width to be derived from `len(TABULAR_COLS)`.  The focused regression suite
passed (8 tests).  These source changes were synchronized to the remote source
for future launches.  The already-live controller was intentionally not
restarted, so its in-memory prompt/checker modules retain their launch-time
version; its ongoing candidate generation and scheduler loop continue
uninterrupted.

## Third live issue: Predictor v3.2 metadata alias

The first valid submitted candidate, `e65025145b864b2291365d07547f138c`,
exposed why the scheduler still had no time estimate.  Its payload recorded
`estimated_runtime_seconds: null`, and the predictor fallback error was:

```
TypeError: PawpularityModel.forward() missing 1 required positional argument: 'tabular'
```

The candidate defined its 12 metadata names as `TAB_COLS`.  The executor only
recognized `TABULAR_COLUMNS`, `TABULAR_COLS`, and related long aliases while
building the Predictor v3.2 model specification, so it supplied a single
image input despite the actual `forward(image, tabular)` signature.  Predictor
conversion therefore failed and the scheduler reverted to a no-time,
solo-anchor decision.

The executor now recognizes the common `TAB_COLS` alias and emits both image
and tabular shapes.  Three focused bridge tests pass, and the fix is synced to
the remote source for future launches.  As with the other persistent fixes,
the live controller is not restarted; this preserves the requested trace, but
the already-submitted first candidate retains its original missing estimate.

## Fourth live issue: post-CUDA DataLoader deadlock

Candidate `e65025145b864b2291365d07547f138c` was the first valid scheduler
execution.  It allocated about 1.9 GB on the A10 but produced no GPU work for
11.5 minutes.  Its two DataLoader workers were blocked in filesystem page
reads and then disappeared; the parent process remained blocked waiting for
them.  The shared Ceph-backed source volume had 113 GB free, and direct reads
of the current JPEGs succeeded, so this was not capacity exhaustion.

The stalled candidate process alone was terminated after this evidence was
captured.  The controller, scheduler, A100 service, and the next READY job
were preserved.  The runner now injects a `DataLoader` worker override of zero
for scheduler-executed generated scripts.  This avoids forked worker deadlock
after CUDA initialization, while preserving the candidate's batch size and
training code.  Three focused tests pass and the remote source is updated.

## First sustained live training after repairs

At `04:12 UTC`, candidate `2837fc7e649546f19d41f01afd8e2a77` was dispatched
as scheduler job `0829c188-42f7-46fa-9426-299bc0f446eb`. The fresh worker used
the zero-worker runner path and maintained 65--91% A10 utilization with about
4.3--4.5 GB allocated. It passed three completed epochs without the earlier
DataLoader deadlock. The scheduler's online runtime probe produced a batch-32
total-runtime estimate of 2,756.6 seconds at confidence 0.80; after epoch
three, its remaining-runtime estimate was 2,323.2 seconds.

This candidate still fell back from Predictor v3.2 because the already-live
controller had loaded the pre-fix executor and emitted a one-input
`perfseer_model` record for the `TAB_COLS` alias. That limitation is isolated
to this existing controller process; the synchronized source emits the correct
two-input record for future controller launches. The live trace is retained
without a restart as requested.

## Scheduler behavior with a measured active job

After the sustained job accumulated epoch evidence, three additional
pre-flight-admitted candidates became READY. All declared
`packing_eligible: true` with the CUDA-process backend, and the saved
scheduler configuration kept `parallel_job_cap: null`. At epoch seven, the
active job's remaining-runtime estimate was 1,797.8 seconds.

### Admission investigation correction

The prior interpretation that the scheduler had rejected colocation because
of an unfavorable amortized-gain calculation was too strong.  The service was
healthy and its memory admission gate was open: one live sample reported 4.5
GB of 23.0 GB A10 memory in use.  However, a read-only reproduction of the
planner's state returned an empty batch-option set for the active job on both
the CUDA-process and exclusive backends; consequently, the backend-aware
planner returned no dispatch plan.  This is a missing-estimate failure, not a
VRAM threshold or a fixed parallel-job limit.

The direct cause is consistent with the persisted active-job error:

```
TypeError: PawpularityModel.forward() missing 1 required positional argument: 'tabular'
```

The old controller generated a one-input Predictor specification for the
`TAB_COLS` model.  With no valid Predictor batch-time/VRAM option and no
completed compatible profile, the scheduler correctly declines to launch a
second process.  The `TAB_COLS` fix is synchronized for fresh controller
processes; the live controller is deliberately left untouched to preserve its
trace.  A bounded CPU-only diagnostic was removed after collection and did
not interrupt the active A10 worker, which progressed from epoch 11 to epoch
13 during the check.

## Fifth live issue: `NUM_TABULAR_FEATURES` alias

The sustained live candidate declares `NUM_TABULAR_FEATURES = 12`, rather than
`TAB_COLS`.  A regression test showed that this common name was still absent
from the static Predictor-metadata aliases, so a fresh controller could have
reintroduced the same one-input `forward(image)` conversion failure.  The
executor now recognizes `NUM_TABULAR_FEATURES` and emits the explicit second
input shape `[$batch, 12]`.  The new real-metadata regression test and the
script-introspection suite passed (30 tests), and the tested executor was
synchronized to the A10 source.  The existing controller is not restarted.

## Interpreting the current long final phase

The sustained candidate reached epoch 20/20 at `05:01 UTC` and then entered
its full public-test inference phase.  Its live process retained an open file
descriptor to successive JPEG files under the PetFinder test directory and
continued consuming CPU; this is evidence of forward progress, not an idle
Predictor or scheduler wait.  Its generated code evaluates the full test set
through the same `predict_dataset` path after restoring the best checkpoint.
The scheduler heartbeat remains at the last completed epoch until the program
prints its final metric and exits, so that phase must not be mislabelled as a
scheduler gap or killed before the no-progress threshold.

## Candidate-code failures after the first completion

After the sustained node completed, the scheduler attempted a real
CUDA-process co-location trial: it launched two low-memory candidates together
without a fixed parallel-job cap.  One candidate failed after 4.06 seconds
because its generated quantile split compared a 10-element bin-edge vector
directly to 8,920 labels, causing a broadcast `ValueError`.  A separate
candidate failed after 5.34 seconds because its split helper called
`.to_numpy()` on a value already returned as a NumPy array.  These are buggy
nodes by the experiment definition.  They are not OOMs, no scheduler process
was killed, and neither failure establishes a scheduler-performance result.

## Sixth live issue: one-step Predictor epoch-time scale

The first healthy node after the co-location trial exposed a separate time
calibration defect. Before it ran, the static Predictor supplied 0.3981 s per
epoch. Its measured first full epoch was 147.27 s (and 143.63 s after two
epochs), a 99.7% one-node absolute percentage error. This is not presented as
a global accuracy statistic.

The causal defect was deterministic: the v3.2 capture worker hard-coded
`steps_per_epoch = 1`, so the artifact's epoch-time target described one
optimizer step. The repair carries a labelled PetFinder `train.csv` row count
through `perfseer_model`, derives `ceil(dataset_size / candidate_batch_size)`
for every candidate batch option, includes it in the Predictor cache key, and
uses it during source capture. At the observed 8,920-row training split and
batch 32 this is 279 steps rather than one. It is a scale correction, not a
claim that the resulting prediction is exact: validation time and each
candidate's split policy still require online epoch calibration.

The tested executor and Predictor source files were prepared for the A10
clean-rerun source.  The focused suite passed 43 tests.  A later byte-for-byte
audit found that the preserved controller's remote `v32_source_worker.py`
still hard-codes one step per epoch.  It will be synchronized only after this
preserved trace is terminal, so its diagnostic behavior is not mixed with the
clean rerun.  This live controller therefore retains the old module imports
and worker behavior; a fresh controller is required to exercise the repair.

## Later buggy-node evidence

A 7-epoch node trained normally (best validation RMSE 20.9802 at epoch 3),
then failed only in final test inference because its test dataset returned a
`None` target and default PyTorch collation rejects `None`. A separately
generated node failed before its first epoch after requesting nonexistent
PetFinder columns `Paws` and `Legs`. These are candidate-code failures, not
GPU exhaustion or Scheduler failures; both are counted as buggy nodes.

## Clean 15-node continuation: one-pass generation and first measured result

The current clean continuation keeps the scheduler and pre-flight checker on,
uses Predictor v3.2 without the Hardware Knowledge Database, seed 42, and a
15 *budget-counted* node target.  It deliberately has no fixed parallel-job
cap.  Generation is now one-pass (`agent.use_stepwise_generation=false`): a
candidate is drafted and reviewed in one generation cycle rather than waiting
for a separate stepwise continuation.

The A100 Qwen endpoint is routed through the lightweight in-cluster vLLM
transport.  After warm-up its measured decoding throughput is 55.6 generated
tokens/s.  This is generation-service throughput, not end-to-end node time.

The first valid completed clean node was `439cdf3b6873457da862c9bcf67636ac`
(scheduler job `187ef237-4409-4d1d-8efa-f60ef7dc2c05`).  It ran five epochs
in 1,350.77 s and achieved PetFinder RMSE 20.10200336; validation marked it
non-buggy.  The 2 September baseline has 30 valid scored nodes, with best
RMSE 17.9433 and best-of-first-ten RMSE 17.9916.  Thus this first clean node
is currently worse in quality; it is not evidence of an improvement.  Its
execution time is also not yet an apples-to-apples speed result because the
generated candidate and training plan differ from the baseline candidate.

The static Predictor v3.2 estimate for that node was 0.44 s/epoch, while the
observed first epoch was 275.15 s.  This is an out-of-distribution cold-start
error, not a missing value.  The estimator now prioritizes a matching online
runtime profile over static Predictor time; the regression test and the
time-aware scheduler suite passed (78 tests).  The active controller is not
restarted, so this correction applies to the next controller while preserving
the current trace.

The next worker (`28aa70aad85d46699dadfb3620eb876d`) is active and confirms
that the DataLoader zero-worker workaround is no longer inherited: its
environment contains only the scheduler-selected effective batch controls
(`MLEVOLVE_BATCH_SIZE_OVERRIDE=8`,
`MLEVOLVE_GRADIENT_ACCUMULATION_OVERRIDE=4`) and no
`MLEVOLVE_FORCE_NUM_WORKERS`.  This keeps generated-code loader choices
intact while retaining safe effective batch sizing.

## Comparability repair: remove forced stepwise generation

## Diagnostic-only speed and quality check against the September 2 baseline

This section compares the preserved trace only to identify regressions; it is
not a Scheduler claim.  For each journal, non-buggy nodes with a recorded
metric were ordered by completion time and measured from the first generated
node.  At the common ten-scored-node checkpoint, the September 2 PetFinder
baseline reached the checkpoint in 264.95 minutes, while the preserved trace
reached it in 457.30 minutes (1.73 times slower).  The preserved trace has
only 11 scored non-buggy nodes so far, whereas the baseline has 30.

The apparent preserved-trace best RMSE of 0.13046 is invalid and must not be
treated as model quality.  Its admitted candidate
`2837fc7e649546f19d41f01afd8e2a77` implements
`np.clip(paw, eps, 1-eps) / 100`: it clips raw 0--100 labels before scaling,
thereby mapping nearly every target above one to approximately 0.01 and
collapsing the validation target domain.  Pre-flight admitted that candidate
with an `INCONCLUSIVE` result; this is a semantic target-transform defect, not
a valid low-RMSE model.  Excluding it, the best valid-looking preserved score
is 18.7952, versus 17.9916 in the first ten scored baseline nodes (0.8036
RMSE, or 4.47% worse).  The preserved trace also uses forced stepwise
generation and the pre-fix Predictor path, so neither observation is evidence
about the clean Scheduler/Predictor/pre-flight treatment.

The preserved live controller revealed a comparison confound unrelated to the
Scheduler: its modified source set `AgentSearch.use_stepwise_generation =
True` unconditionally.  The reference Pathfinder/PetFinder baseline used
single-pass generation (`agent.use_stepwise_generation=false`).  Stepwise
generation serializes model-design, training/evaluation, and merge calls; in
the live log one draft took roughly 5--6 minutes before code review and
pre-flight.  That time is agent-generation overhead, not a Scheduler gap.

The source now exposes `agent.use_stepwise_generation`, defaults it to false,
and makes `AgentSearch` honor it.  The fresh Predictor/pre-flight/Scheduler
launcher passes `agent.use_stepwise_generation=false` explicitly.  Thus the
next clean treatment matches the baseline's agent-generation mode while
retaining only the intended differences: Scheduler, Predictor v3.2, and the
pre-flight checker.  The regression test covers both boolean settings; the
configuration suite passed 23 tests and the launcher passed shell syntax
validation.  These files were synchronized to the A10 source without
restarting the preserved controller.

The preserved trace itself remains unsuitable for a controlled comparison:
it was started before the Predictor metadata and epoch-cardinality fixes, and
it used forced stepwise generation.  It is retained as diagnostic evidence
only, as requested.

## Effective-setting parity audit

The historical baseline's repository-level `config.yaml` is not its effective
run configuration: its sequence launcher overrode the model and endpoint to
the local A100 Qwen service.  The clean treatment launcher therefore makes
all shared execution controls explicit: one A10 (`CUDA_VISIBLE_DEVICES=0` and
`agent.search.num_gpus=1`), 7,200-second candidate timeout,
43,200-second search budget, three initial drafts, seed 42, data preview,
leakage checking, diff mode, no global memory, temperature 0.7, and no fixed
parallel-job cap (`null`).

The historical baseline budget is 50 nodes.  The clean treatment launcher now
uses the explicitly requested 15-node budget; this is the sole non-treatment
setting deviation and is not suitable for a wall-clock comparison against all
50 baseline nodes.  Any later comparison will use an equal 15-node prefix.

The clean launcher also uses the baseline's `openai-compatible` client setting
for both code and feedback.  This still targets the identical local A100 vLLM
endpoint and served Qwen model; aligning the client setting avoids treating a
different request-construction path as a Scheduler, Predictor, or pre-flight
effect.  Local-Qwen backend detection now recognizes that stable in-cluster
endpoint even through the baseline-compatible client, so both streamed and
non-streamed requests still forward the experiment seed (`42`).  A launcher
semantic comparison plus focused configuration, generation-mode, and seed
tests (`34 passed`) verified this correction before remote synchronization.

### Launcher continuation repair

The initial clean launcher placed two shell comments inside the backslash-
continued `run.py` command.  Bash treated the first following setting as a
separate command, producing `agent.code.provider=openai-compatible: command
not found`; the Qwen provider, model, and endpoint arguments would therefore
not reach `run.py`.  This was discovered before launch by an executable
launcher test with a controlled `run.py` stand-in.  The comments were moved
above the command, the test then verified that `seed=42`, one-pass generation,
the baseline-compatible provider/model settings, and the null parallel-job cap
all reach the Python entry point.  Shell syntax plus the focused suite passed
`35` tests, and the corrected launcher hash was verified after synchronization
to the isolated A10 staging tree.

Three differences are intentional treatment components, not hidden
confounders: Scheduler is enabled, Predictor v3.2 replaces no static
prediction, and the pre-flight checker uses the canonical
`petfinder-pawpularity-score` task identifier.  Retaining that identifier is
necessary: it selects the PetFinder metadata contract in pre-flight; copying
the old baseline's logging-oriented experiment identifier would suppress that
task-specific validation.  Hardware Knowledge Database features remain
disabled for this requested Predictor/Scheduler/Pre-flight isolation.

## Preserved-trace result after the audit

Scheduler job `76404ead-042f-4e74-92a7-27efc1b1f44d` completed naturally at
`06:30 UTC` after ten epochs.  Its best validation RMSE was `42.4843` at epoch
six, and it returned exit code zero with a valid submission contract.  This is
a valid, scored node rather than a buggy node; its weak score is a candidate
quality result, not an execution failure.  The trace then returned to
stepwise agent generation, which is expected to take several minutes and is
recorded as agent-generation overhead rather than Scheduler idle time.

The following admitted node `1aae2a6012744ad7a20e9f96f8a34b32` was dispatched
32 seconds after reaching READY while A10 admission was fully open.  It failed
after 5.08 seconds before its first epoch with
`AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'` in its
generated stratified-split helper.  It is therefore a buggy node, not an OOM
or scheduling failure.  This same generated-code idiom had appeared earlier;
the preserved controller continues without a restart as requested, while the
failure evidence is retained for future pre-flight/prompt coverage work.

## Predictor metadata compatibility repair

The subsequent running node `2e9138cfeabd4e459b5f0021e8e81f53` confirmed a
remaining coverage issue in the preserved controller.  Its candidate correctly
declared `NUM_TAB_FEATURES = 17`, but the then-imported metadata extractor did
not recognize that alias.  It supplied only the image input to Predictor v3.2,
which consequently reported `PawpularityModel.forward() missing ... 'tabular'`
and fell back to a branch runtime profile.  The node itself is continuing
normally and has completed two epochs; this is neither an Out Of Memory event
nor a Scheduler admission failure.

The clean rerun now recognizes `NUM_TAB_FEATURES` in addition to the existing
tabular aliases and emits both the image and 17-wide tabular tensors in
`perfseer_model.input_shapes`.  The focused metadata suite passed three tests,
including a regression reproducing this exact PetFinder declaration.  As with
the prior fixes, the already-running preserved controller remains untouched;
the fresh single-pass controller will exercise this repaired path.

The actual candidate uses the expression `NUM_TAB_FEATURES = len(FLAG_COLS) +
5`, rather than a literal `17`.  The original alias addition alone was thus
insufficient because it only accepted literal values.  The extractor now
safely resolves constant lists, names, `len(...)`, and integer addition—never
arbitrary candidate code.  The exact preserved candidate was reprocessed
locally and produced image `[B, 3, 224, 224]`, tabular `[B, 17]`, and the
correct 8,920-row dataset size.  The metadata suite now has four passing tests.

## Large-gap mitigation plan for the clean rerun

The observed long spaces between node executions are not a single Scheduler
quantity and must be measured separately.  The clean run therefore applies
the following evidence-driven changes:

1. **Remove serial agent overhead.**  It uses the baseline's single-pass
   generation route, rather than the preserved controller's 5--6 minute
   stepwise design/evaluation/merge sequence.
2. **Restore static Predictor coverage.**  It supplies the true multimodal
   input signature and the real number of steps per epoch, so Scheduler
   admission is based on v3.2 estimates rather than a branch-profile fallback.
3. **Retain online calibration.**  The first completed epoch writes measured
   time into the runtime profile; this refines rather than replaces the static
   estimate for subsequent admissions.
4. **Do not impose a job-count cap.**  Admission remains incremental and uses
   compatibility, predicted/observed runtime, and live device telemetry.  A
   queued job is not force-started merely because VRAM happens to be free.
5. **Persist gap attribution.**  The final trace will separately report agent
   generation, pre-flight, READY-to-dispatch wait, candidate training, and
   post-execution parsing.  Only READY-to-dispatch wait is Scheduler queue
   time; the other intervals must not be presented as Scheduler delay.

These are testable interventions, not throughput claims.  The clean run must
show successful static Predictor metadata and recorded admission decisions
before any Scheduler-performance conclusion is made.

## Clean-treatment launch and local-transport repair

The older pre-fix diagnostic controller (`177303`) and its verified worker
group were intentionally terminated before this clean treatment, after
confirming that the A100 Qwen service was outside both process groups.  The
A10 returned to zero MiB and the A100 health endpoint remained healthy.

The first clean controller began at `12:21 UTC` with the intended 15-node,
seed-42 command, but it stalled before first-node generation in uninterruptible
disk I/O while the official OpenAI Python client recursively imported its
schema tree from the shared virtual-environment filesystem.  The launch log
showed its first query, but no request completion or node artifact.  Its run
artifacts were retained, then its isolated controller group was terminated.

The repository already contained the lightweight persistent HTTP transport in
`llm/vllm.py`.  The provider router now selects that transport only when a
Qwen model uses the stable in-cluster `mlevolve-qwen-*.svc.cluster.local`
endpoint, even though the stage remains configured as
`openai-compatible` for baseline parity.  External OpenAI-compatible services
are unchanged.  The routing regression, existing vLLM dispatch tests,
lazy-provider tests, Qwen tests, and the launcher contract test passed
(`15 passed`).

The corrected clean controller started at `12:26 UTC` with run id
`petfinder_scheduler_v32_a10_20260914_122644`.  Its first local-Qwen request
completed in about three seconds; metric direction was determined, the
Scheduler service started, and one-pass initial-draft generation began.  This
is launch evidence only, not a quality or speed claim; the final trace will
provide the controlled comparison.

### Early live-run observation (not a quality result)

The first generated candidate (`d035869f6d0c4cceb725247dee30335a`) was
rejected by pre-flight before it reached the A10.  The rejection is a useful
correctness result, not a Scheduler slowdown: the candidate had an actual
`LayerNorm(2060)` versus 1,012-feature tensor mismatch, plus executable
import-time statements.  The recorded diagnostic classes are `SHP004`,
`AUT002`, `VAL001`, `MLE_IMPORT001`, and `GPU003`.  It was therefore not
admitted and must not contribute either a training-time measurement or a
quality score.  The controller continued to draft the remaining initial
candidates normally.

The second candidate (`05cadb1b7f1f41178b3ce76777f4c5c3`) was also rejected
before A10 execution, with `CHK001`, `CON001`, `GPU003`, and
`MLE_IMPORT001`. Its pre-flight pass completed normally; the apparent
multi-minute gap was active CPU-side validation, not a hung Scheduler or GPU
queue. Root selection to admission result took 319 s and 305 s for the first
two candidates. That end-to-end time includes pre-flight and is not one-pass
agent latency. The first candidate's two logged model requests lasted about
25.6 s and 5.5 s; its remaining time was construction/review and pre-flight.

### Execution-path utilization repair (applies to future workers)

The first admitted candidate was dispatched at `12:47 UTC`. Its candidate
script requested four data-loader workers, but runner instrumentation had
silently exported `MLEVOLVE_FORCE_NUM_WORKERS=0`; this serialized image
decode/augmentation on CPU and left the A10 underutilized. The scheduler
runner now preserves the candidate's loader-worker choice by default and
explicitly removes any inherited override; a caller can still request an
explicit override when a controlled probe needs one. The regression and the
full runner test file passed (`11 passed`). This source repair was synchronized
to the remote staging source but cannot alter the already-started worker, which
is intentionally left running to preserve the current trace.

The first admitted job (`187ef237-4409-4d1d-8efa-f60ef7dc2c05`) began on the
A10 at `12:47 UTC`. Its first epoch completed at `12:51 UTC` and wrote a live
CUDA-process runtime profile: resolved physical batch size 8, epoch-1 time
275.15 s, and an eight-epoch estimate of 2,201.18 s at 0.75 confidence. This
is direct evidence that online per-epoch timing is reaching the Scheduler;
it is not yet a completed-node quality result.

### Predictor-time precedence correction

For the admitted candidate, the v3.2 cold-start model emitted 0.443 s per
epoch while the first real epoch measured 275.15 s. The source capture carried
the correct 8,920-row dataset and 1,115 steps-per-epoch metadata, so this is
an out-of-distribution timing error from the static student rather than a
missing-estimate (`None`) bug. The Scheduler did receive the observed epoch
profile, but `ResourceEstimator.estimate_batch_options()` mistakenly preferred
the static ML time over that exact runtime profile when constructing later
placement options. It now gives a positive live runtime profile or measured
batch observation precedence over static ML time, while retaining ML-predicted
VRAM. A regression reproducing 0.44 s static versus 275 s observed timing was
red before the correction and passes after it; the full time-aware Scheduler
suite passed (`78 passed`). The repair is synchronized to remote staging.
The active controller retains its already-imported module and is deliberately
not restarted mid-job; its first job is preserved as the calibration trace.

## Live continuation: second budget-counted node

The second scheduler job, `ae412e6e-c909-4fa6-acb3-3983340116d8` (node
`28aa70aad85d46699dadfb3620eb876d`), ran all 12 training epochs in 310.14 s.
Its online profile refined the predicted total duration from 340.89 s after
the first observation to 304.60 s, which is close to the actual duration.
This demonstrates the live-profile path; it must not be conflated with the
static Predictor cold-start estimate.

The node is correctly classified buggy because final test inference reused a
dataset accessor that unconditionally requested the training-only
`Pawpularity` column.  The public test table lacks that label, producing
`KeyError: 'Pawpularity'` after training completed.  Its best validation RMSE
was 20.7057 at epoch 10, but no submission was produced, so it is not a valid
quality result.  This is candidate-code failure, not an A10, scheduler,
Predictor, or pre-flight service failure.  It counts as node 2 of 15.

## Baseline-parity audit

The recorded 2 September Pathfinder baseline configuration and the live
continuation match on all search and agent controls except the requested node
budget and the experimental treatment: both use seed 42, three initial
drafts, diff mode, one-pass generation, the same full search configuration,
one GPU, and Qwen3.8-27B-INT8 at the same in-cluster A100 endpoint with
temperature 0.7.  The baseline has 50 steps and no Scheduler/pre-flight
checker; the live treatment has 15 steps, Scheduler + Predictor v3.2 +
pre-flight enabled, CUDA-process backend, and Hardware Knowledge Database
disabled.  These are the intended treatment differences rather than an
accidental change in agent model, seed, or search policy.

## Continued pre-flight evidence

Candidate `b252e72178a74250a2ef0d485a5fa873` was initially rejected before
A10 admission for a last-batch BatchNorm failure (`AUT002`), import-time
side effects, and unsafe pandas feature conversion.  The single permitted
pre-flight repair round produced an admitted candidate; it did not consume a
budget node until the scheduler dispatched job
`d4c961f8-edea-43aa-a211-5ed51b77ca9c`.

On that job's first epoch, the online runtime probe recorded roughly 514--515
s total runtime for resolved physical batch 8, replacing the static 0.351
s/epoch cold-start placement number for this job's subsequent decisions.  The
job remains running when this record was updated; no quality outcome is
claimed here.

## External pod termination

At approximately 13:19 UTC, while job
`d4c961f8-edea-43aa-a211-5ed51b77ca9c` was running, Nautilus terminated the
`gpu-dev-a10-experiment-746c959459-brjtz` pod.  The controller did not issue a
stop, and no Scheduler job-kill event preceded it.  Kubernetes retained only
the kubelet event `Stopping container petfinder-experiment`; it provides no
cause such as OOM or a candidate failure.  The replacement pod
`gpu-dev-a10-experiment-746c959459-6hcqp` is Pending with no A10 assignment.

The runtime workspace was deliberately under `/dev/shm`, so the active
controller and its in-progress node state must be treated as lost with this
pod.  The source staging tree and this record were synchronized to persistent
storage before the termination.  No replacement controller is started while
the requested A10 is Pending; a later recovery must explicitly distinguish a
new trace from the preserved first three budget-counted outcomes.

## Recovery trace after A10 capacity returned

The replacement A10 pod became Running at 16:14 UTC.  The first recovery
launch correctly stopped before creating a controller because its transient
`/tmp/Predictor` source and Predictor v3.2 weights had disappeared with the
previous pod.  The Qwen A100 service, pre-flight checker, and A10 device were
healthy, so this was a dependency-persistence failure rather than a service
failure.

Predictor v3.2 source and weights were copied to persistent experiment
storage at `predictor_v3.2/`; the student-weight SHA-256 matches the local
artifact: `988fa6fec1df66349a40c1dd2585ccc5f25ef2a72d88ea38af8f5b97e45850e7`.
The new controller started at 16:48 UTC with seed 42, `agent.steps=15`,
one-pass generation disabled from the stepwise route, Scheduler and pre-flight
enabled, Predictor source/weights taken from this persistent path, Hardware
Knowledge Database and hardware context disabled, one A10, and no fixed
parallel-job cap.  This is a new trace; it must not be reported as a resumption
of the externally terminated `/dev/shm` run.

## Restart: serial pre-flight startup bottleneck

The recovery trace was archived before restart at
`archives/petfinder_scheduler_v32_a10_20260914_164841_preflight_serial_20260914_171914`.
It is not discarded: it remains diagnostic evidence only and is excluded from
quality or speed conclusions.

The root cause of the slow first admission was not conservative GPU admission.
For the three initial drafts, `run.py` requests deferred execution, but
`AgentSearch._run_single_step()` nevertheless ran the complete CPU pre-flight
gate (and a repair/recheck when needed) serially before the Scheduler could
receive any candidate.  Two drafts were rejected after that expensive work;
the surviving draft was submitted only after the serial gate completed.

The repair defers pre-flight only for deferred initial drafts.  The Scheduler
worker still runs the same pre-flight gate immediately before GPU submission,
so no candidate can bypass admission; generation and CPU validation can now
overlap.  A focused regression test verifies that immediate execution still
runs pre-flight on the generation path, whereas deferred execution leaves it
for the worker.  The restarted trace retains seed 42, one-pass generation,
Scheduler, Predictor v3.2, pre-flight checking, no fixed parallel-job cap, and
Hardware Knowledge Database disabled.

## Node-local source recovery

The first restart after this code repair did not reach application logging: its
Python process entered Linux `D` state in `folio_wait_bit_common` while reading
the Ceph-mounted `/root` source tree.  This is a storage page-fault stall, not
a vLLM request, GPU admission decision, or candidate failure.  The process was
terminated before it could submit any work.

The active replacement executes a synchronized, source-only copy from
`/dev/shm/mlevolve_exec_staging_20260914`, while retaining the required Python
environment, Predictor v3.2 artifact, and pre-flight checker from persistent
storage.  It reached Scheduler attachment and began the first Qwen code
generation request within nine seconds of `run.py` startup.  vLLM reported one
running request and zero waiting requests at that point.  This removes the
Ceph source-read bottleneck without changing the model, seed, Scheduler,
Predictor, pre-flight, Hardware Knowledge Database, or concurrency contract.

## Reproducibility and checker-version repair

The prior seed utility initialized Python and NumPy only.  It now also invokes
`torch.manual_seed(42)` and, when CUDA is available,
`torch.cuda.manual_seed_all(42)`.  A unit test replaces the torch module with a
recording implementation and verifies both calls.  Local Qwen vLLM requests
already receive `seed=42`; the launch command and the agent configuration also
set `agent.seed=42`.

The first node-local recovery trace exposed a separate pre-flight deployment
bug: its remote checker schema rejected the current PetFinder `fixture`
scenario field even though the MLEvolve manifest generator emits it.  The
trace was archived at
`archives/petfinder_scheduler_v32_a10_20260914_172609_checker_schema_mismatch_20260914_180706`
and excluded from all results.  The matching repository checker, whose schema
explicitly permits `scenarios.fixture`, was staged to
`/dev/shm/model_preflight_checker_20260914`.  Its PetFinder manifest regression
test passes locally.  The active clean trace uses this matching checker,
node-local MLEvolve source, A100-hosted local Qwen3.8-27B-INT8 agent, one A10
executor, seed 42 for Python/NumPy/PyTorch/vLLM, Scheduler + Predictor v3.2 +
pre-flight, no Hardware Knowledge Database, and no fixed parallel-job cap.

## Baseline-parity budget correction

The preceding repaired trace still used a 15-node budget inherited from an
earlier smoke-test launcher.  The 2 September Pathfinder baseline completed
50 budget-counted nodes, so that setting was not a fair control match.  The
15-node trace was archived at
`archives/petfinder_scheduler_v32_a10_20260914_180853_node_budget_15_*` before
it had dispatched an A10 job.  The active clean scheduler-only trace starts
with `agent.steps=50`, confirmed from its live controller command line.  It
retains all already-verified parity controls and changes only Scheduler,
Predictor v3.2, and pre-flight checking relative to the historical baseline.

## Strict agent-path parity correction

Inspection of the archived 9/2 launcher showed that its baseline path had
neither pipeline-decision prompting nor code review.  Those current-repository
features were therefore disabled for the treatment as non-baseline agent work,
without disabling pre-flight.  The intermediate 50-node trace was archived at
`archives/petfinder_scheduler_v32_a10_20260914_181418_review_pipeline_parity_*`
before any A10 execution.  The active strict-parity controller has confirmed
arguments: 50 nodes; seed 42; three initial drafts; data preview and diff mode
enabled; one-pass generation; global memory, hardware context,
pipeline-decision, and code review disabled; Scheduler enabled with Predictor
v3.2; pre-flight enabled; Hardware Knowledge Database disabled; one A10; and
an uncapped CUDA-process time-aware Scheduler.  The agent endpoint is the
local A100 Qwen3.8-27B-INT8 service with temperature 0.7 and the same
OpenAI-compatible transport as the historical baseline.

The node-local remote source was independently verified with a recording torch
module: `set_global_seed(42)` called both `torch.manual_seed(42)` and
`torch.cuda.manual_seed_all(42)`.  This complements the local unit test and
the live command-line evidence for `agent.seed=42`.

## Strict-parity rejection-gate repair

The strict-parity trace generated six valid-looking PetFinder candidates but
rejected every one before CPU pre-flight or A10 submission.  The pipeline
SQLite trace identifies the cause: three critical `training_contract` issues
were added by an unconditional static-review guard even though
`agent.review.enabled=false`.  These issues required quality-batch declarations
and an `MLEVOLVE_EPOCH_METRIC` logging marker; they are review advice, not a
runtime failure, and they are absent from the 2 September baseline contract.

The repair keeps the non-bypassable precision guard and dependency check in
both paths, and keeps CPU pre-flight mandatory before execution.  It limits the
training-contract rejection to runs where review is enabled.  Focused tests
verify that review-disabled runs admit ordinary neural-training code while
still rejecting FP4 precision violations and missing dependencies.  The
rejected trace will be archived as diagnostic-only before the clean restart;
it has zero budget-counted A10 executions and must not be used as a result.

The clean replacement started at 18:29 UTC with the same 50-node strict-parity
command and seed 42.  Its first three drafts all reached deferred scheduler
submission; the pipeline log contains `model_preflight_stale_hash` events for
each, which is the expected hand-off to mandatory CPU pre-flight.  In contrast
to the archived trace, it has emitted no `review_execution_avoided` event and
has not applied a fixed parallel-job cap.  The run remains active; this is a
launch-state observation, not an experiment result.

## Deferred pre-flight repair hand-off

The next strict-parity trace reached the checker correctly, and each generated
candidate declared `SEED = 42` with NumPy and PyTorch seeding (the first four
candidates were inspected directly).  It nevertheless produced zero
budget-counted A10 jobs: CPU pre-flight rejected the initial candidates for
real import-safety, adapter-contract, and shape diagnostics.  This was not a
Scheduler rejection and no GPU work was started.

The cause was an implementation mismatch.  Deferred generation was correctly
changed to postpone CPU pre-flight until admission, but the subsequent
`_ensure_node_preflight_before_execution()` forcibly set `allow_repair=False`.
Thus it silently removed the configured one targeted repair round.  The repair
changes that call to preserve the configured repair policy.  A regression test
asserts that a deferred candidate is admitted to the gate with
`allow_repair=True`; existing tests still cover deferred generation timing and
bounded repair rounds.  This trace will be archived before restart as
diagnostic-only, with zero executed jobs.

The replacement started at 18:40 UTC with `agent.steps=50`, `agent.seed=42`,
the A100 local Qwen3.8-27B-INT8 endpoint, one A10 executor, Predictor v3.2,
Scheduler enabled, CPU pre-flight enabled, no Hardware Knowledge Database, and
`parallel_job_cap=null`.  The first Qwen generation request is live.  This is
again launch-state evidence only; no A10 training job has yet been admitted.

## Active-trace first execution diagnostic

At 18:47 UTC, the EfficientNet-B3 draft `cf494acd` passed the repaired
pre-flight admission path after one targeted repair (its network-dependent
`torch.hub.load` backbone was replaced with the local, weights-free torchvision
constructor).  The Scheduler submitted it with Predictor v3.2 metadata,
`random_seed=42`, CUDA-process placement, and no fixed concurrency cap.

The A10 job then failed before its first training epoch.  The instrumented
candidate raised `NameError: val_img_dir is not defined` while constructing the
validation dataset.  The Scheduler recorded the job as `FAILED`; MLEvolve
parsed it as a buggy node with no metric and excluded it from the 50-node
budget.  This is an actual generated-script data-path defect, not a predictor
latency, Scheduler admission, or GPU-capacity failure.  The pre-flight checker
did not detect this undefined data-path variable, so the evidence is retained
for a narrowly scoped checker fix only if the continuing trace fails to produce
budget-counted nodes for the same reason.

At 18:54 UTC, the next repaired candidate `b52b822d` (ConvNeXt-Tiny) completed
pre-flight admission and the Scheduler dispatched it.  Its job state is
`RUNNING`; the A10 reports 99--100% GPU utilization, approximately 6.1 GB
allocated, and approximately 147 W draw.  Its instrumented parent process and
four data-loader workers are live.  This is the first confirmed live training
execution in the clean 50-node Scheduler-only trace.  It establishes that the
end-to-end A100-Qwen -> pre-flight -> Predictor v3.2 -> uncapped Scheduler ->
A10 path works after the recorded generated-code failure.  No comparison result
is inferred until completed budget-counted nodes are available.

By 19:04 UTC, the running ConvNeXt-Tiny job had emitted four standard epoch
observations to the Scheduler.  The online profile for its exact script
signature records a resolved physical batch size of 16, an epoch-one duration
of 155.896 seconds, confidence 0.9, and an updated expected total runtime of
2,338.441 seconds.  Thus the Predictor/Scheduler is using measured
epoch-level runtime evidence during the job; the observed batch size is an
incremental-admission decision, not a configured job-count cap.  The job is
still live, so this is an in-flight timing observation rather than a completed
quality result.

## Live data-loader storage observation

During the same job, the four PyTorch data-loader workers entered Linux `D`
state in Ceph metadata/page waits while the parent remained live. The active
workspace is in `/dev/shm`, but `workspace/input/train` is a symlink to
`/root/downeyflyfan/mle-bench-data/petfinder-pawpularity-score/prepared/public/train`
on the Ceph mount. This explains brief CPU-heavy, low-GPU intervals without
attributing them to the Scheduler or Predictor. The input location is not
changed during this trace: moving it mid-run would break the required 2
September baseline storage-control match. The event is retained for a future,
separately controlled local-data-cache ablation only.

## CephFS recovery and local-input guard

The CephFS condition did not clear: direct `stat` and first-file lookup of the
canonical PetFinder directory timed out, and the ConvNeXt-Tiny data-loader
workers remained in uninterruptible `ceph_mdsc_wait_request` / page-wait
state.  The stalled trace (25 MB of pipeline, scheduler SQLite state, logs,
pre-flight reports, and hardware samples) was copied to
`records/preserved_stalled_ceph_run_20260914_184008/` before its A10 Pod was
recreated.  It remains a diagnostic trace: it has no completed budget-counted
node and supplies no performance result.

The launcher now accepts `PETFINDER_DATA_DIR`, validates the canonical
PetFinder public files before invoking MLEvolve, and passes that directory as
both `data_dir` and the description source.  This makes a verified local copy
an explicit run input rather than an implicit CephFS symlink.  The change is
covered by the launcher test suite (two tests passed) and shell syntax check.
The A10 deployment was also recreated with an 8 GiB Kubernetes memory request
(64 GiB limit retained), which removed the prior memory-only scheduling
rejection; it has not changed GPU count, model, Predictor, Scheduler,
pre-flight settings, random seed, or the absence of a fixed parallel-job cap.
The fresh Pod is now pending only on an available A10 GPU.  No restarted
experiment will begin until a local PetFinder cache is verified and its source
manifest is recorded.

## Active local-input Scheduler trace and timing-evidence repair

The subsequent local-input trace `petfinder_scheduler_v32_a10_20260914_235626`
is running on one A10 with the local A100 Qwen3.8-27B-INT8 endpoint.  Its live
command line verifies `agent.seed=42`, `scheduler.enabled=true`, Predictor
v3.2 on CUDA, pre-flight enabled, `hardware_knowledge.enabled=false`,
`agent.hardware_context_enabled=false`, and `parallel_job_cap=null`.

At the time of this record, four budget-counted candidates have completed:
node `67e7f8a1aa2043eea2c0a538458030a5` reached RMSE 19.0053 after eight
epochs, node `81e25fac0714475ea02f3084e0e517ab` reached RMSE 20.7959 after
eleven epochs, and the precision-repair node
`3abe2dd45fd947038d3bce55882b1568` reached RMSE 20.7215 after ten epochs.
Node `546523f638704d119ba166d0e656f884` subsequently completed with RMSE
20.7857 after 23 epochs.
One separate ViT candidate failed before epoch one with the generated script
dtype error `mat1 and mat2 must have the same dtype, but got Double and Half`.
A second candidate completed all 15 training epochs (best validation RMSE
20.7632) but then failed during its generated test-export loop because its
collate function attempted to stack `None` test targets.  Both are buggy nodes
under the experiment definition and are not counted as valid results.  Two
later Scheduler-admitted jobs concurrently used the A10 at 100% GPU
utilization with approximately 7.75 GiB allocated, without a configured
concurrency cap.

The running trace exposed a Scheduler telemetry gap: the raw MLEvolve
subprocess runner wrote `last_completed_epoch` but omitted
`runtime_epoch_timing_history`.  Colocation evidence therefore had zero
fresh samples even while both jobs emitted epoch output, leading to an
unhelpful evidence-deadline pause request.  The source repair persists
per-epoch intervals from each MLEvolve stdout epoch marker, retaining the most
recent sixteen samples.  It is deliberately not injected into this already
running trace.  `localml_scheduler/tests/test_mlevolve_runner.py` (12 tests)
and `localml_scheduler/tests/test_time_aware_scheduler.py` (78 tests) passed
after the repair.  The next launch will use the repaired source; this live
trace remains valid evidence only for the exact source version that launched
it.

## Strict seed-parity finding

Inspection of the active generated child script found `SEED = 42` only for its
train/validation split. It did not call `random.seed`, `numpy.random.seed`,
`torch.manual_seed`, or `torch.cuda.manual_seed_all`. Because every candidate
runs in a fresh subprocess, the controller's global seed cannot seed those
libraries on its behalf. Consequently this trace must not be used as strictly
seed-controlled experimental evidence, although it remains useful for
Scheduler failure diagnostics.

The raw-script instrumentation now receives the scheduler job seed through
`MLEVOLVE_JOB_SEED` and seeds Python, NumPy, PyTorch, and all CUDA devices
before candidate code executes. `build_mlevolve_job()` now carries the
experiment seed into `TrainingJob.config.seed`; both Scheduler submission paths
pass `agent.seed`, while metadata records the declared source seed separately.
A regression test executes a fresh raw child process with seed 42 and verifies
the exact Python, NumPy, and PyTorch first random values. The focused runner,
global-seed, and launcher suites (17 tests) pass. This code is reserved for the
next clean strict-parity launch and is not applied under the active controller.

## Seed-fixed replacement staging

At 01:22 UTC on 15 September, the active controller was still live and was
therefore left untouched. Its state was five completed and two failed
Scheduler-submitted jobs. A separate remote source tree was prepared at
`/tmp/mlevolve_exec_staging_seed42_20260915`; it is a copy of the active
staging tree with only the seed-propagation source files replaced. Remote
compilation succeeded and the staged runner contains the `MLEVOLVE_JOB_SEED`
helper, Python/NumPy/PyTorch seeding, and CUDA all-device seeding. It has not
been launched, so it cannot mix source versions into the diagnostic trace.

The drafts rejected after the latest completion were rejected by pre-flight
for real generated-code faults, including `timm` reporting the requested
`efficientvit_s2_256` backbone as unknown. They are neither Scheduler
admission failures nor timing observations. The active trace remains
diagnostic-only; a new 50-node result can be called seed-controlled only after
the staged source launches and a generated instrumented child script is
verified to receive `MLEVOLVE_JOB_SEED=42`.

## Baseline completion-budget repair

The 2 September Pathfinder baseline source invoked Qwen with the normal
8,192-token completion budget. The active diagnostic launcher instead passed
`vllm_client.default_completion_tokens=null`, which suppressed the client
parameter and let the serving stack stop a long code response before its
closing Markdown fence. The raw candidate file therefore began with the
model's natural-language plan and failed static pre-flight; this is not a
Scheduler outcome. A launcher regression test first reproduced the missing
argument and then passed after restoring
`vllm_client.default_completion_tokens=8192`. The seed-fixed staging tree has
the same one-line restoration. This is a parity repair: it restores the 9/2
agent-generation contract rather than altering the Scheduler treatment.

## Seed-clean Scheduler-only launch

The unreproducible diagnostic controller was copied in full to its remote run
directory and then terminated with `SIGTERM`; no A10 worker was active at that
time. The replacement began at 01:29 UTC on 15 September from
`/tmp/mlevolve_exec_staging_seed42_20260915`. Its live command verifies a
50-node budget, `agent.seed=42`, Qwen3.8-27B-INT8 at the local A100 endpoint,
one A10 executor, Scheduler and Predictor v3.2 enabled, CPU pre-flight
enabled, Hardware Knowledge Database and hardware context disabled, and
`parallel_job_cap=null`. It also verifies the restored 8,192-token Qwen
completion budget.

The first draft completed code generation at 01:30:45 UTC without the prior
`Code extraction failed after retries` warning. This is launch and generation
evidence only. The run will become seed-controlled execution evidence only
after a Scheduler job is materialized, its instrumented child script is shown
to receive `MLEVOLVE_JOB_SEED=42`, and the A10 execution result is recorded.

## Fixed-seed retry repair

The first seed-clean replacement proved that a fixed LLM seed creates a second
determinism requirement: `plan_and_code_query()` retried a missing-code reply
with an identical prompt, so Qwen reproducibly returned the same malformed
reply. The fault was reproduced with a failing unit test. The retry now appends
a format-specific instruction and numbered retry marker while retaining the
same configured LLM seed of 42; the test then passed. This changes neither the
model, sampling seed, task, Scheduler, Predictor, nor hardware setting.

That zero-A10-job trace was preserved at
`runs/petfinder_scheduler_v32_a10_20260915_012822/` and exited cleanly. The
active replacement began at 01:36 UTC from the same seed-fixed staging tree,
with the restored 8,192-token completion budget and retry-safe prompt path.

## Seed-controlled execution evidence

The active replacement materialized Scheduler job
`4a5429d8-5671-4c0c-8f69-526b751d87f6` for node
`37563820a24a487baa505424274cadf9`.  The persisted Scheduler payload records
`random_seed=42`.  Direct inspection of the live parent and all four
DataLoader worker environments showed `MLEVOLVE_JOB_SEED=42`.  The generated
instrumented script applies this value to Python `random`, NumPy,
`torch.manual_seed`, and `torch.cuda.manual_seed_all` before importing the
candidate program.  The generated program also defines and uses `SEED = 42`.

At 01:54 UTC the first job had reached epoch 7 of its configured 20 epochs on
the A10; it was still live, not a completed result.  Two additional candidates
had passed the same pre-flight gate and were `READY` in the Scheduler queue.
The A100 endpoint returned the required local model identifier
`qwen3.8-27b-int8-a100`.  The active controller command independently confirms
`ml_predictor`, CUDA capture for Predictor v3.2, `parallel_time_aware`
Scheduler admission, `parallel_job_cap=null`, and both Hardware Knowledge
Database switches disabled.  This is valid seed-controlled execution evidence,
but it is not yet a completed 50-node experiment or a performance claim.

## Result-parser false-bug repair

The first active Scheduler job completed training without a subprocess
exception, printed `Final Validation Score: 20.60912977420808`, and saved the
required eight-row submission (the local canonical PetFinder test split has
eight rows).  Nevertheless, the local Qwen result parser returned
`is_bug=true`, no metric, and a critical `data_integration` issue after seeing
the non-fatal pretrained-weight warning.  It incorrectly turned that valid
execution into a buggy MLEvolve node.  This is a result-parser false positive,
not a Scheduler, Predictor, A10, or generated-program failure.

The repair makes the deterministic completion contract authoritative only when
all three conditions hold: no subprocess exception, a finite printed final
score, and the required submission file.  In that narrow conflict case it
preserves the final score and downgrades LLM-only critical findings to warnings;
real process errors, missing metrics, and missing/invalid submissions remain
blocking.  The regression test first failed under the old behavior and then
passed, and the full result-parser workflow suite passed 51 tests.  The active
trace is preserved but excluded from comparison because it loaded the old
parser.

## Parser-fixed seed-42 replacement

The invalid parser trace was allowed to run its launcher cleanup, then its
remaining controller and uniquely identified worker processes were terminated.
The A10 returned to 0 MiB allocation.  Its durable archive contains the
Scheduler SQLite database, journal, controller log, and raw per-node result
files; nothing was deleted.  A new source tree was copied rather than modified
in place at `/tmp/mlevolve_exec_staging_seed42_parserfix_20260915`, and its
`result_parse_agent.py` hash was verified against the local fix
(`1d0773caa45868846da324187cff96c07be87b70574f547da22e3885e12ed260`).

The clean replacement began at 02:03 UTC on 15 September, with run root
`/dev/shm/mlevolve_petfinder_scheduler_v32_a10_20260915_020255/experiment/20260915_020259_petfinder_scheduler_v32_a10_20260915_020255`.
At launch it passed the A100 local-Qwen health check, set metric direction to
minimize RMSE, attached the Scheduler, and began the three configured initial
drafts.  It retains the 50-node budget, all seed-42 settings, Predictor v3.2,
pre-flight checking, no fixed parallel cap, and Hardware Knowledge Database
disabled.

At 02:08:49 UTC all three initial candidates had been generated and submitted
to the Scheduler.  Live inspection at 02:10:54 UTC found the controller still
running and two active pre-flight worker processes; the Scheduler database had
not yet materialized a job, and the A10 remained idle.  This is the expected
pre-admission stage, not an execution failure or an imposed concurrency limit.
The next required verification point is a materialized job plus its live
`MLEVOLVE_JOB_SEED=42` environment before treating any result as valid
execution evidence.

## First live Scheduler admission

At 02:12:48 UTC, Scheduler admitted node `733e88fc16c745ca94eef1dca8200eec`
as job `d338c4fb-74dd-4271-8e13-3a908558b183` using the `cuda_process`
backend.  Its persisted decision reports Predictor v3.2 as the timing and
VRAM source, with a cold-start 32-sample estimate of 809.84 MiB and 0.4699 s
per epoch.  The initial `resource_requirements` fields remain `None` by design
because they are static submission defaults; they are not the Predictor output.
The scheduling decision is therefore evidence that the original
"estimation-time-is-None" path is not active for this job.

The live instrumented parent process had `MLEVOLVE_JOB_SEED=42`.  Its injected
prologue seeds Python `random`, NumPy, `torch.manual_seed`, and
`torch.cuda.manual_seed_all`; the generated candidate independently defines
and uses `SEED = 42`.  The local-Qwen request code passes the same agent seed
to vLLM for both regular and streamed requests.  Thus the actual job has the
specified Agent, Python, NumPy, and PyTorch/CUDA seed path.

The first empirical runtime probe is substantially slower than the cold-start
Predictor estimate: after two completed epochs it measured 68.97 s per epoch
and estimated 1,034.51 s for the planned remaining execution.  This is an
accuracy observation, not a missing-estimate failure; it must not be presented
as a Scheduler speed result.  The Scheduler then attempted a safe concurrent
trial for candidate `893e71872b3d47bfb7a1d31f871bfccc`; that candidate failed
before its first epoch with the genuine generated-code error
`NameError: val_img_dir is not defined`.  The original running job remains
active, and this candidate failure does not invalidate it.

## First valid result

At 02:18:14 UTC the first admitted job completed successfully after six epochs
with early stopping (15 epochs had been proposed).  Its best validation RMSE
was **18.045** at epoch 2; the full curve was 18.3804, 18.0450, 18.4302,
18.7528, 19.0113, and 19.0788.  The parser accepted this as a non-buggy node,
preserved the metric, and saved its top-1 submission.  This is the first valid
Scheduler+Predictor result in the clean seed-42 run.  It is not a completed
50-node comparison and must not yet be used for a performance claim.

At 02:22:16 UTC Scheduler admitted a second, repaired candidate
`82cbd534bae44a77b443c069566b0a63` as job
`06681e67-4a7b-4954-b720-ebb954dce769`, again using Predictor v3.2 without a
fallback error.  Direct inspection showed `MLEVOLVE_JOB_SEED=42` in the
instrumented parent and in all four live DataLoader worker processes.  This
confirms seed propagation through a second independent Scheduler dispatch.

At 02:24 UTC the second job briefly entered uninterruptible Linux I/O wait at
the Ceph metadata-service wait channel while its image DataLoader initialized.
No controller, Scheduler, Predictor, or generated-code error was present, so
the run was not restarted.  At 02:26 UTC it had completed two epochs; at
02:27 UTC the A10 was actively computing (48% utilization, 2,039 MiB in use).
This establishes that the wait was transient dataset-metadata I/O rather than
a Scheduler stall.  The experiment configuration and its running process were
left unchanged.

## Live-trace durability

Because the active controller uses `/dev/shm`, a lightweight mirror was started
at 02:31 UTC to persistent Ceph storage under
`/root/downeyflyfan/.cache/mlevolve_petfinder_v32_scheduler_rerun/live_trace_mirror_20260915_020255`.
While controller PID `39574` remains live, it takes a consistent SQLite
backup and copies the Scheduler event log and controller log every 90 seconds.
The first snapshot completed successfully (`scheduler.sqlite3`, 220 KiB;
`events.jsonl`, 15 KiB).  This monitor is observational only: it neither
changes candidate code nor interacts with GPU admission or execution.

At 02:33 UTC, a live endpoint query returned only
`qwen3.8-27b-int8-a100`.  The concurrently running Scheduler job persisted
`random_seed=42` and `vram_prediction_source=ml_predictor`; its A10 was
actively computing at 44% utilization with 2,039 MiB allocated.  This is
runtime evidence that the intended local A100 agent, A10 executor, seed, and
Predictor path remain in effect after launch.

## Second valid Scheduler result

At 02:40:06 UTC, job `06681e67-4a7b-4954-b720-ebb954dce769` for node
`82cbd534bae44a77b443c069566b0a63` completed successfully after 14 epochs.
Its final/best validation RMSE was **20.6323**. The result parser accepted it
as non-buggy, recorded it in the journal, and saved it as the second ranked
submission. An optional local service at `127.0.0.1:5005` was unavailable
during a separate stage; that stage is explicitly non-blocking and was skipped.
The required format and result validation both passed, so this is a second
valid budget-counted Scheduler+Predictor execution, not a failed node.

## Live local-agent liveness check

During a long repair-generation request at 02:55 UTC, the local A100 Qwen
endpoint processed 653 output tokens over 12 seconds (**54.4 tokens/s**).
The server had one running request and none queued. This measures the active
server's aggregate generation progress for this request; it is liveness and
throughput evidence, not an end-to-end experiment-time result.

## Third live Scheduler admission

At 03:00 UTC, Scheduler admitted node
`34bc77236d5e491a8c286395e3826b08` as job
`7e5c05d1-92dc-4358-8c9a-3f439318b36d`.  Its persisted job metadata has
`random_seed=42`, `vram_prediction_source=ml_predictor`, and no Predictor
fallback error.  Direct inspection found `MLEVOLVE_JOB_SEED=42` in the parent
and all four live DataLoader workers.  This is a third independently verified
seed-controlled Scheduler dispatch; it is still running, so no quality claim
is made yet.

The third dispatch terminated at 03:03 UTC as a genuine buggy node before its
first completed epoch. Its generated model passed a final mini-batch of one
sample into training-mode BatchNorm in `vision_proj`, causing
`ValueError: Expected more than 1 value per channel when training, got input
size torch.Size([1, 256])`. It therefore produced no finite metric or
submission and was correctly recorded as a failed Scheduler job. This is a
generated-candidate defect, not an A10, Predictor, seed, or Scheduler failure;
the configured run continues without modifying the candidate.

At 03:06 UTC, the targeted repair node
`84b11bce20914b18b18a2569175b500b` passed preflight and Scheduler admitted it
as job `cd59fede-f932-4869-84b0-85487c9ddd3e`. The job metadata again records
seed 42 and `ml_predictor`; its parent plus four live DataLoader workers all
have `MLEVOLVE_JOB_SEED=42`. This is the repair branch continuing through the
same verified Scheduler-only execution path.

At 03:24 UTC, that repair branch completed six validation epochs and reached
best validation RMSE **18.97667587579347** at epoch 2, but correctly ended as
a genuinely buggy node rather than a valid result.  Its inference DataLoader
used the training `collate_fn`, which calls `torch.stack` on `item['target']`;
PetFinder test examples have no target, producing `TypeError: expected Tensor
as element 0 in argument 0, but got NoneType`.  This is a post-training,
generated-candidate test-inference defect, not a Scheduler, Predictor, seed,
or A10 failure.  The controller preserved the failure record and immediately
admitted the next Predictor-backed Scheduler job
`f1d755e7-dbc3-4f4b-b87e-0d7f3dd11cb4` (node
`d7daa0d31ae340fe879b84ec60ef3c10`) without restarting the experiment.

## Predictor-coverage defect discovered during the live run

At 03:53 UTC, the third valid Scheduler result completed: job
`f1d755e7-dbc3-4f4b-b87e-0d7f3dd11cb4` (node
`d7daa0d31ae340fe879b84ec60ef3c10`) trained for 15 epochs and reached best
validation RMSE **19.639**.  The next queued job,
`7ffd9d22-8423-46a3-a2fc-53d9d1a14c6a`, exposed a Predictor-coverage defect:
the Scheduler event recorded `runtime estimate unavailable; packed-backend solo
anchor`, and its persisted source was `branch_profile`, with no Predictor
exception.  Its generated source defined a single no-required-argument
`DualPathModel(nn.Module)` but did not define either `build_model` or
`CandidateAdapter.build_model`; the adapter consequently omitted
`perfseer_model` metadata, so Predictor v3.2 had no source, shapes, or
steps-per-epoch contract to evaluate.

This contaminates the current trace for a strict Scheduler+Predictor-only
comparison; it is retained as diagnostic evidence and the active job is allowed
to finish without interruption.  The local adapter is corrected to append a
Predictor-only `build_model()` wrapper only when source contains exactly one
top-level model class with no required constructor arguments.  It instantiates
that exact generated class rather than substituting a proxy model.  The focused
metadata suite passes (5 tests), and a separate remote staging copy contains the
fix for the corrected rerun.  A candidate outside this narrow, safely
instantiable shape remains explicitly unsupported rather than silently being
misrepresented.

The corrected adapter was then evaluated against the exact live dual-path
source in a separate staging directory, without invoking the Predictor or
competing for the A10.  It produced `input_shapes = [["$batch", 3, 256, 256],
["$batch", 12]]`, `fp32` precision, and dataset size 9,912, and wrote the
wrapper source successfully.  Temporary validation sources were removed
afterward.  This verifies that the next clean run can supply v3.2 with the
actual generated architecture and inputs for this previously uncovered shape.

## Safe cutover to the corrected trial

At 04:08 UTC, the old controller received `SIGSTOP` after its active worker
was verified live.  This freezes dispatch of further candidates using the
obsolete adapter but does **not** terminate, pause, or modify the active A10
worker: it continues training its already-submitted job.  The old trace stays
diagnostic-only.  Once that worker has reached a terminal state, the old
controller will be retired and the isolated corrected staging tree will start
the clean Scheduler-plus-Predictor trial.

## Corrected clean trial launched

At 04:11 UTC, the old worker had reached a terminal state and no A10 scheduler
worker remained.  The old frozen controller was terminated, preserving its
diagnostic runtime tree.  A fresh **50-node** trial then started from the
isolated patched execution tree with the following decisive launch properties:

- `scheduler.enabled=true`, Predictor v3.2 mode `ml_predictor`, and preflight
  enabled;
- `hardware_knowledge.enabled=false`, `agent.hardware_context_enabled=false`,
  global memory disabled, pipeline decision disabled, and review disabled;
- `agent.seed=42`, local `qwen3.8-27b-int8-a100`, and one A10 executor;
- `parallel_job_cap=null` (incremental scheduler admission only).

The new process is `petfinder_scheduler_v32_a10_20260915_041134`.  Its first
admitted job must persist `vram_prediction_source=ml_predictor`; the run is not
accepted as a clean result until that live check passes.

## First clean Predictor-backed Scheduler admission

At 04:19 UTC, Scheduler admitted job
`fc2fa090-2e54-4546-ae73-5e0230a37eaf` for node
`33acf66538104bb1b8fbcd9dcdddd840`.  Its persisted metadata proves the
required cold-start contract: `random_seed=42`,
`vram_prediction_source=ml_predictor`, no Predictor error, a real
`build_model` source, image-and-tabular input shapes `[$batch,3,256,256]` and
`[$batch,12]`, and public-train dataset size 9,912.  It had completed epoch
one when inspected.

The active worker and each of its four DataLoader children have
`MLEVOLVE_JOB_SEED=42`.  The instrumented source applies that value to Python
random, NumPy, `torch.manual_seed`, and `torch.cuda.manual_seed_all` before
candidate imports.  This is live execution evidence—not merely launcher
configuration—that the clean Scheduler-plus-Predictor treatment has begun.

The Predictor also supplied a non-null cold-start time estimate: 0.419
seconds/epoch for the selected physical batch 16.  That static value is far
below the observed first two epochs (34.76 and 35.06 seconds), so it must not
be reported as accurate.  Crucially, it is not absent: the runtime probe
replaced it after epoch one with 34.91 seconds/epoch and a 418.94-second total
estimate at confidence 0.80.  The Scheduler therefore has a live time
estimate for subsequent placement decisions, resolving the former
"estimation-time is None" coverage failure while leaving Predictor calibration
as a separately recorded accuracy issue.

The first clean job completed successfully at 04:25 UTC after 11 epochs.  It
obtained validation RMSE **20.758827209472656**, passed format/content and
metric-direction validation, and produced a submission.  Its observed epoch
time had converged to 32.64 seconds.  It is the first valid budget-counted
node of the 50-node Scheduler-plus-Predictor run; it is not a baseline result
and no cross-run quality comparison is claimed here.

## Replacement-trial invalidation: adapter fixture contract

The second candidate in the replacement trial was not a Predictor regression.
Its generated `CandidateAdapter` incorrectly treated `scenario["fixture"]` as
a comma-separated string, although the preflight manifest supplies a mapping
from named inputs to non-batch shapes.  Consequently every real CPU check
failed at batch construction with `AttributeError: 'dict' object has no
attribute 'split'`, but the checker classified that as `INCONCLUSIVE` and the
balanced admission policy allowed it through.  The candidate then had a real
model error—ViT returned 1,000 classifier logits to a fusion head expecting a
768-dimensional embedding—and Predictor v3.2 correctly reported the same
forward error before falling back to `branch_profile`.

This makes the replacement trial unsuitable for the strict
Scheduler-plus-Predictor comparison.  After confirming no A10 worker was
active, its controller was frozen.  The repaired preflight integration now
treats deterministic adapter fixture failures (`FIX001` with a programming
exception) as a rejection and gives the stage repair loop the exact mapping
contract.  The generation prompt was made equally explicit.  The new staging
tree was checked against the precise recorded `FIX001` diagnostic and rejects
it with targeted guidance; no Predictor fallback is accepted in the next
fresh trial.

## Aborted replacement startup: completion-budget verification

The next clean startup was retired before any Scheduler job was submitted.
Its third initial code draft was still streaming after roughly six minutes, so
the controller was stopped only after confirming that no A10 worker existed.
This is not an experiment result and contributes zero nodes.  The apparent
stall was investigated rather than attributed to the Scheduler: the exact
runtime configuration resolves `vllm_client.default_completion_tokens` to
8,192, the live request construction contains `max_tokens=8192`, seed 42, and
`enable_thinking=false`, and direct streamed and non-streamed requests to the
same A100 vLLM server obey `max_tokens` (a forced long response ended at the
16-token limit).  Thus the client and server limit contract is intact; a long
single code response remains possible within the baseline-compatible 8,192
token allowance.  The following clean launch retains that allowance and will
not interrupt a live generation merely because it outlasts a short heuristic.

## Current Scheduler-only trial: configuration and seed audit

At 04:51 UTC, a new clean 50-node trial started as
`petfinder_scheduler_v32_a10_20260915_045052`, using the corrected isolated
execution tree. Its live controller command explicitly sets
`scheduler.enabled=true`, Predictor v3.2 `ml_predictor`, preflight enabled,
and `parallel_job_cap=null`. It also sets `hardware_knowledge.enabled=false`
and `agent.hardware_context_enabled=false`; this treatment tests Scheduler,
Predictor, and preflight only—not the Hardware Knowledge Database or a
fixed-concurrency policy.

The one NVIDIA A10 is the executor; both code and feedback generation target
the local A100 vLLM endpoint with `qwen3.8-27b-int8-a100`. It preserves the
September-2 Pathfinder comparison settings: 50 nodes, 43,200 seconds, three
initial drafts, data preview and leakage check enabled, diff mode enabled,
stepwise generation and global memory disabled, and 8,192 completion tokens.

Seed 42 is enforced at three layers: `run.py` seeds Python random, NumPy,
PyTorch, and all CUDA devices; the local vLLM client attaches seed 42 to each
completion; and the Scheduler runner injects `MLEVOLVE_JOB_SEED=42` then
instruments generated candidate scripts to seed Python random, NumPy, PyTorch,
and CUDA before candidate imports. At first observation, two of three initial
drafts existed, but no A10 job had yet been admitted; per-job Predictor
metadata will be reported only after that admission.

## First Predictor-backed Scheduler admission in the current trial

At 04:59 UTC, node `ecf0fe0b576a4befaae83c5b5de44de5` passed its one allowed
targeted preflight repair and Scheduler admitted job
`2c1e89bc-2fde-401f-befe-52d15dead7af`. The persisted job payload records
`random_seed=42`, `vram_prediction_source="ml_predictor"`, and no Predictor
error. Its PerfSeer v3.2 model describes a real `build_model` entry with image
shape `[$batch,3,256,256]`, tabular shape `[$batch,12]`, and dataset size
9,912.

The Scheduler's cold-start placement metadata contains a non-null estimate of
0.4431505126953125 seconds per epoch and 6.647257690429688 seconds remaining
for 15 planned epochs. These are cold-start estimates, not measured throughput;
the enabled epoch-one runtime probe will replace them with an observed value.
The job entered `RUNNING` at 04:59:56 UTC through a
`localml_scheduler.execution.worker_entry` A10 worker. This establishes that
the former `estimation time = None` failure is not present on the first current
admission. The other initial candidates remain in their independent preflight
repair/admission paths and are not yet counted as experimental execution
results.

## Live runtime-estimate verification

The first current Scheduler job remained actively training on the A10 and had
completed five epochs at the next inspection. Its initial Predictor estimate
(0.443 seconds/epoch) was badly optimistic, but it was not absent: the
epoch-one runtime probe persisted every observed epoch and updated the
Scheduler estimate online. The five observed durations were 70.13, 35.79,
94.74, 50.22, and 49.08 seconds. The current stored estimate is 59.99
seconds/epoch, 899.91 seconds total, with confidence 0.95.

Thus the required estimator path is live and measurable: Predictor v3.2
provides cold-start admission, while the Scheduler replaces an inaccurate
cold-start time with an observed profile before later placement decisions.
The cold-start calibration error must be reported separately from coverage; it
does not justify claiming a speed improvement.

## Predictor coverage guard: torchvision resolution syntax

One later repaired candidate used only `transforms.Resize((256, 256))` to
express its image shape. The previous script introspector recognized named
assignments such as `IMG_SIZE = 256`, but not that torchvision syntax. It
therefore omitted the Predictor input model and the Scheduler would have
fallen back from the requested Predictor-only treatment. The queued job was
paused before execution; the active first job was not changed.

`detect_input_resolution` now recognizes square `Resize((H, W))` transforms.
The new regression test fails on the prior implementation and passes after
the change. The focused introspection and Predictor metadata suite passed:
35 tests in 1.48 seconds. This parser repair will be synchronized into the
isolated execution tree only after the active valid A10 job reaches a terminal
state.

## Fresh trial after Predictor-coverage repair

The diagnostic job completed all 15 epochs with best validation RMSE 24.3443
at epoch 12. Its final persisted runtime profile was 53.69 seconds/epoch,
811.83 seconds total, confidence 0.95, with `random_seed=42`,
`vram_prediction_source="ml_predictor"`, and no Predictor error. It is valid
evidence that the Predictor and online timing paths function, but it is not
part of the clean 50-node comparison because a different queued candidate had
no Predictor image model and was held before execution.

After confirming the A10 worker was terminal and the GPU was idle, that mixed
controller was stopped. The isolated execution tree received the tested
torchvision `Resize((H, W))` parser repair (identical staged SHA-256:
`ee4ddcd324307daa7e10cb55db65cb7b0e620a141c15d94b284c9822369e11ca`). A
fresh clean 50-node Scheduler-only run started at 05:15 UTC as
`petfinder_scheduler_v32_a10_20260915_051510`. It retains the same seed-42,
local A100 Qwen 3.8-27B-int8, A10 executor, September-2 Pathfinder settings,
Predictor v3.2, preflight, disabled hardware knowledge/context, and null
parallel-job cap. Its evidence must remain separate from all earlier trials.

## Current clean-run admission and seed audit

The current clean run admitted job
`f3967ecd-8e8f-4b9a-aafb-c74d61f8f384` for candidate
`d6de08637a784e52a60667309c862b2a` after the one permitted preflight repair.
The Scheduler payload records `random_seed=42` and
`vram_prediction_source="ml_predictor"`; its dispatch event records an
initial Predictor-derived placement estimate of 0.4432 seconds per epoch.
The instrumented A10 child inherited `MLEVOLVE_JOB_SEED=42` and its executable
contains explicit `random.seed(42)`, `numpy.random.seed(42)`,
`torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)` calls.

The first-epoch runtime probe measured 73.3651 seconds per epoch and updated
the Scheduler profile to 1,100.48 seconds for 15 epochs (confidence 0.75).
That large calibration error is recorded as a Predictor-quality observation,
not a scheduler speed claim. The job remains live on the single NVIDIA A10;
the local endpoint's advertised model is exactly
`qwen3.8-27b-int8-a100`, with root path
`/root/downeyflyfan/qwen38-v100-int8/models/Qwen3.8-27B-INT8-W8A16-MTP` and
262,144-token maximum context. Hardware Knowledge Database and agent hardware
context remain disabled; the null parallel-job cap remains in force.

## First completed node in the clean 50-node Scheduler-only run

Job `f3967ecd-8e8f-4b9a-aafb-c74d61f8f384` completed normally at 05:37 UTC
for node `d6de08637a784e52a60667309c862b2a`: return code 0, 15 completed
epochs, best validation RMSE 24.3443, and a valid submission. Its recorded
execution duration was 833.01 seconds (the pipeline's end-to-end job interval
was 863.18 seconds), batch size 8, and final observed online estimate was
55.1020 seconds per epoch at confidence 0.95. The final Scheduler payload
still records seed 42, `vram_prediction_source="ml_predictor"`, and no
Predictor error.

The cold estimate of 0.4432 seconds per epoch was therefore substantially too
optimistic, whereas the online runtime probe converged to the measured run.
This is evidence of estimator availability and online correction, not evidence
of cold-start accuracy or a speedup. The subsequent candidate
`4d8f1d15d1d245e4864200757d88cabd` was correctly rejected by preflight after
one permitted repair for genuine import/shape/validation failures and never
used the A10; it is recorded as a buggy generated node, not a scheduler
execution result.

## Second completed Predictor-backed Scheduler node

Node `281a3794e47140d79e7cde0e567873da` was admitted only after its single
preflight repair and completed normally as job
`4ece58c5-4f77-4546-baf9-7f8589c59cee`. The Scheduler selected batch size 16
from its dynamic batch probe (not a fixed parallel-job cap). The candidate
returned code 0, trained for 15 epochs according to the parsed execution
report, wrote a valid submission, and achieved validation RMSE
19.2889739522. Scheduler execution time was 603.3661 seconds; its online
profile converged to 39.6726 seconds/epoch, 603.3661 seconds total, confidence
0.95. The persistent job payload records `random_seed=42`,
`vram_prediction_source="ml_predictor"`, and no Predictor failure.

The adapter's generic `metadata.completed_epochs` field remained zero for this
otherwise valid candidate even though the parsed execution report proves 15
epochs. This is a persistence/telemetry discrepancy, not a training or
Scheduler failure: the runtime-profile event and final tuning outcome both
persist the 15-epoch, 603.3661-second run. It must be fixed separately before
using that generic field for resume accounting, but the live experiment is not
interrupted because the requested Predictor admission and online timing paths
are intact.

## Buggy generated node after preflight admission

Job `7d253ea2-28d3-4514-ab71-7354d0af1a93` for node
`c0118438f9c84ae49eab930bc0ffe746` was Predictor-backed and seeded correctly
(`random_seed=42`, `vram_prediction_source="ml_predictor"`) but is a buggy
node by the execution-error definition. It failed in the first validation
phase with return code 1 after 46.75 seconds. The persisted error is precise:
the candidate initialized test-time-augmentation `preds_sum` with
`torch.zeros_like(images)`, shape `[B,3,288,288]`, then attempted to add the
model's scalar predictions, shape `[B]`. The resulting RuntimeError was
`The size of tensor a (288) must match the size of tensor b (64) at
non-singleton dimension 3`.

Preflight had validated construction/abstract paths and admitted the repaired
candidate as INCONCLUSIVE because only GPU and memory checks remained. It did
not exercise that real validation-time test-time-augmentation branch, so this
is a preflight coverage gap. The Scheduler correctly recorded
`job_failed`, preserved the no-Predictor-error metadata, and returned control
to search; no A10 worker was left running. The live controller is deliberately
not restarted or rewritten mid-experiment: this event is logged as one buggy
generated node, and subsequent candidates continue through the same requested
Scheduler-plus-Predictor treatment.

## Resume after Predictor-admission repair

The preceding statement was superseded on 2026-09-15 at 06:20 UTC after a
concrete scheduler correctness defect was established. Multi-stage generated
programs such as `NUM_EPOCHS_STAGE1 = 5` followed by
`NUM_EPOCHS_STAGE2 = 15` were parsed as having no epoch budget. Consequently,
the scheduler could not form an epoch-time estimate and used its legacy
unprofiled `branch_profile` solo-dispatch path. Two already-failed pre-repair
jobs (`787b168f-7f2b-4cce-adee-5c7f3d2ee8d4` and
`5116a619-93ce-4601-be22-da418090f8e5`) are therefore excluded from the clean
Predictor-only analysis.

`detect_epoch_count` now sums explicitly executed staged epoch loops, so the
example above resolves to 20. In addition, `require_ml_prediction=true` is an
explicit scheduler setting: before dispatch, any job whose VRAM prediction
source is not exactly `ml_predictor` is terminally rejected without starting
an A10 worker. Focused regression tests cover both guarantees.

The controller received SIGTERM only after the active scheduler worker set was
empty; it exited cleanly after saving its journal, scheduler database, and
pipeline database. It resumed from the same `logs/journal.json`, workspace,
and scheduler runtime at 06:23 UTC rather than reinitializing search. The
resumed command retains seed 42 for the local Qwen request configuration and
all instrumented Python randomness, the local A100
`qwen3.8-27b-int8-a100` endpoint, one NVIDIA A10 executor, disabled Hardware
Knowledge Database and agent hardware context, and `parallel_job_cap=null`.

The first post-resume candidates were either still in CPU preflight or
rejected there for genuine candidate-code defects (image rank mismatch,
feature-width mismatch, and BatchNorm batch-of-one failure). They never
created an A10 scheduler worker, so they do not constitute scheduler failures
or branch-profile fallback executions.

At 06:29 UTC the resumed controller admitted its first clean post-repair
jobs: `6b271d4a-09d5-4533-8b80-e1c4c5595eaa` for node
`b5aa8d91c81a40efa7e8c426bdae2a0d` and
`9defa5f7-222a-4ddd-bf90-7e74540cb88b` for node
`fc9ef85d77d04f6980d89552916149ef`. Their live payloads record
`vram_prediction_source="ml_predictor"`, no Predictor error, and
`random_seed=42`. Static epoch parsing produced 5 and 25 epochs respectively;
the scheduler selected batch sizes 8 and 16 dynamically. Both ran
concurrently on the single A10 without a configured parallel-job cap, using
only about 2.3 GiB and reaching 96% GPU utilization at the verification
instant. This is live confirmation that the resumed treatment is Predictor
backed and uses incremental scheduler admission rather than a fixed width.

## Live-trial checkpoint-safety repair

At 06:32 UTC the Scheduler correctly concluded that the measured co-location
gain for the second job was below the configured threshold and wrote a pause
command. The raw-script adapter continued executing because it can observe an
epoch printed by an opaque candidate subprocess but cannot serialize or restore
that subprocess's model and optimizer state. A synthetic scheduler checkpoint
would have made later resume silently restart training, which is not valid
experimental accounting.

Raw MLEvolve-script jobs now declare `supports_checkpoint_resume=false`.
Both the incremental time-aware planner and the backend-aware trial planner
refuse unknown live co-location trials unless every job can truly checkpoint
and resume. Known, measured safe co-location profiles remain eligible; the
guard applies only to trial placements that may need rejection and pausing.
Regression tests prove the raw-job capability declaration and that an unknown
incremental trial is not emitted for a non-resumable candidate. This patch is
staged on the A10 host but is intentionally not injected into the already
running worker processes; they continue uninterrupted. It will take effect at
the next safe controller resume after the active workers have finished.

## Alias-aware multi-stage epoch accounting

A later queued candidate exposed a second variant of the same accounting bug:
it used `total_epochs = NUM_EPOCHS_STAGE2` and then
`range(total_epochs)`. The former parser recognized the two literal stage
budgets but missed that local alias, reporting five rather than the actual
twenty sequential epochs. The worker was allowed to continue uninterrupted;
the controller was frozen so it cannot dispatch further work under the old
estimate. `detect_epoch_count` now resolves direct local aliases before it
totals explicitly executed stage loops. A focused regression test verifies
the exact 5 + 15 alias pattern and all six epoch-introspection tests pass.

The runner also previously treated `Stage2 Epoch 1/15` as a new epoch one,
overwriting its live progress after stage one. It now turns explicit stage
labels (and unlabeled counter resets) into a monotonic cumulative epoch
counter. Thus the observed transition from `Stage1 Epoch 5/5` to
`Stage2 Epoch 1/15` is persisted as epoch six of twenty. Five runner tests
cover live-profile and stage-transition accounting. This is a Scheduler
telemetry repair; it neither changes the generated candidate nor uses
Hardware Knowledge Database data.

## Preflight-package availability repair

The isolated A10 staging environment initially contained the
`nn-model-preflight-checker` submodule but did not expose its
`model_preflight` package to the server virtual environment. As a result,
preflight reported an internal import failure and fell back according to its
configured fail-open policy. The checker was installed from that local
submodule with no external dependency change and its import was verified.
`engine.preflight` now also has a repository-bundled fallback: if the package
is absent from site packages, it loads
`nn-model-preflight-checker/src` from the same staged checkout. The fallback
API test passes. A staged package link makes the already-running controller
able to import the checker immediately; no A10 worker was stopped.

The first restart process itself was aborted before any model request because
its non-interactive SSH environment did not contain
`MLEVOLVE_VLLM_CACHE_SALT`. It was removed only after confirming that it had
no scheduler worker and no generated node. The replacement received a fresh
in-process 256-bit salt, not persisted in any file or log. It restored the
same journal (seven completed nodes) and reached the local
`qwen3.8-27b-int8-a100` endpoint. Live A100 telemetry then showed the agent
model using its A100 while the Scheduler worker used the separate A10.

Candidates admitted while the package import was unavailable are retained as
trace evidence but excluded from the clean post-repair Scheduler analysis:
their admission metadata is `preflight_status=INTERNAL_ERROR`. The running
job from that interval remains Predictor-backed and seed-42, but is likewise
labelled separately rather than being presented as evidence for preflight
coverage. Jobs created after the repair must have a real checker result before
being included in the clean comparison.

Three queued jobs carrying exactly that stale internal-error status were placed
on a scheduler hold after their job identifiers, status, and diagnostic code
were verified. They were not deleted, cancelled, or rewritten; the hold merely
prevents their fail-open admission from consuming the A10 before a genuinely
checked candidate. This does not configure parallelism and does not alter any
completed node or active worker.

## Metricless buggy-node resume repair

At 07:13 UTC, an admitted generated candidate failed without producing a
metric or a local-best node.  The Scheduler worker had already finalized the
failure correctly, but the search-side evaluation path assumed both objects
existed and raised `AttributeError` before the next scheduling round.  This was
not a scheduling decision or a Predictor failure; it was an error-handling
defect that could stall the controller after an ordinary buggy candidate.

`engine.evaluation.check_improvement` now immediately backpropagates a
metricless or explicitly buggy node with its normal reward before any
metric-dependent comparison.  Its regression test first reproduced the
exception and then passed after the guard.  The controller was stopped only
after confirming that no A10 worker existed, and the same journal was resumed
at 07:19 UTC; no nodes, jobs, or search state were reset.  The resumed source
also verifies seed 42 in the local Qwen vLLM request, Python/NumPy/PyTorch/CUDA
controller initialization, and the instrumented A10 worker.  Focused seed,
Qwen-request, and evaluation tests passed (14 tests total).

## Isolated pre-flight worker import repair

The resumed controller could import the bundled checker, but the checker
launches each construction/data stage in a sanitized child process with
`PYTHONNOUSERSITE=1`.  The editable checker installation was available only
through user site-packages, so those child stages returned `CHK003` and the
old journal configuration admitted them fail-open.  This affected the
pre-existing stale jobs only; it is not a Predictor fallback.

The bundled `nn-model-preflight-checker/src` is now exported in `PYTHONPATH`
before the gate imports the checker.  A direct isolated child import using the
actual A10 virtual-environment interpreter succeeds, and focused loader tests
pass.  The reproducible Scheduler launcher now explicitly sets
`preflight.fail_open_on_internal_error=false`, so any future checker
infrastructure error is rejected instead of appearing in Scheduler evidence.
The active stale A10 job was left running; the repair will be loaded only at
the next safe same-journal resume.

## CUDA-process admission consistency repair

The resumed runtime selected `cuda_process`, but its inherited raw-script
submission default still named `mps_process`.  This made the scheduler reject
otherwise compatible CUDA-process packing candidates for the wrong reason.
The PetFinder launcher now explicitly sets the raw-job backend allowlist to
`[cuda_process]`, matching the selected scheduler backend, while leaving
incremental admission and `parallel_job_cap=null` unchanged.  The launcher
regression test verifies that the argument reaches `run.py`.
