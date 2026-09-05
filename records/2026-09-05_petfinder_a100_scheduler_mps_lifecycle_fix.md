# PetFinder A100 Scheduler: MPS Lifecycle-Label Fix

## Scope

- Task: PetFinder Pawpularity regression.
- Agent: Codex GPT-5.6 Terra Medium.
- Hardware: the available A100 on `ABA`, physical GPU 1.
- Scheduler policy: branch-profile based; no fixed `parallel_job_cap`.
- Execution transport: CUDA Multi-Process Service (MPS).
- Valid-node rule: only non-buggy nodes with execution time of at least 30 seconds count.

## Observed defect

The scheduler selected `mps_process` for the first job and the subprocess was
verified on GPU 1 together with `nvidia-cuda-mps-server`.  However, the
`job_started` event reported `backend_name: exclusive`.

The cause was local to `localml_scheduler.execution.worker_entry`: it passed
the literal string `exclusive` to all start, completion, and failure lifecycle
events, instead of reading the scheduler-selected placement in job metadata.
This was an observability/accounting defect, not a failure to use MPS.

## Fix and verification

`_runtime_backend_name(job)` now resolves `placement_backend` (falling back to
`effective_backend`, then `exclusive`) and validates it against the canonical
runtime backend taxonomy.  The worker uses that result for start, completion,
and failure events.

Regression coverage:

- MPS-selected jobs emit `mps_process`.
- Jobs with no selected packed backend retain `exclusive`.
- Targeted suite: 18 passed, 4 subtests passed.

The corrected worker entrypoint was synchronized to `/home/yufan/MLEvolve` on
`ABA` while the currently running first job remained untouched.  Thus later
workers in this run will carry the correct MPS label; the first job retains its
already-written legacy start event but has independent placement and GPU-process
evidence.

## Runtime-estimate correction

The runtime estimator also incorrectly restricted compatible branch-profile
reuse to `experiment_mode == hardware_aware`.  `origin` is a search/prompt mode;
when its Scheduler is explicitly enabled, it must still consume measured
branch-profile timing.  The estimator now excludes only the true `baseline`
mode.  A regression test proves that a same-branch, same-model-family `origin`
candidate receives the completed-profile estimate, while a baseline candidate
does not.  The affected scheduler tests pass (14 tests plus 4 subtests), and
the corrected estimator was synchronized to `ABA` for future worker/controller
starts.

## Live-epoch profiling correction

The generated PetFinder scripts print standard progress lines such as
`Epoch 1/30 ... valid_rmse=...`; they do not emit the optional structured
`MLEVOLVE_EPOCH_METRIC` marker. The runner previously ignored those lines, so
it could not write a runtime estimate or heartbeat until the job exited. It now
parses standard `Epoch completed/total` output, extracts a validation metric
when present, persists total epochs and observed seconds per epoch, and writes
the same live runtime profile/heartbeat used by structured markers.

The new regression test proves that this profile exists before a
standard-output script exits. Both structured-marker and standard-output
live-profile tests pass. The runner fix was synchronized to `ABA` for later
workers.

The first standard parser covered `Epoch current/total` output. A subsequent
PetFinder candidate used `Epoch current: ...` instead. The parser now accepts
both forms, and the runner extracts a literal `epochs`, `max_epochs`, or
`num_epochs` value from the generated source before launch when the job payload
lacks an epoch budget. This supplies the total required for a useful live ETA.
The associated live-profile regression tests pass.

The running checkout was later verified as
`/data1/downeyflyfan/MLEvolve_terra_a100` through the worker's `PYTHONPATH`.
The three modified scheduler files were synchronized to that checkout as well;
an earlier copy to `/home/yufan/MLEvolve` was not the code imported by this
experiment. Three mistakenly placed temporary files in the package root were
removed immediately. No active worker was terminated or restarted.

## Run state at the time of this record

Run root:
`/data1/downeyflyfan/MLEvolve_terra_a100/runs/petfinder_terra_a100_scheduler_mpsuuid_20260905T111228Z`

The initial job was running under the MPS daemon.  Additional agent-generated
candidates were in the scheduler's ready queue and intentionally awaited the
first runtime profile before incremental parallel admission.  No performance
conclusion is made here; the final comparison must use only completed valid
nodes and the same valid-node count from the original MLEvolve baseline.

## MPS bootstrap admission correction

The observed queue was more conservative than intended.  An unknown-runtime
MPS anchor had no allocation configuration, so it launched with the MPS
default of 100% active-thread percentage. NVIDIA MPS fixes that percentage at
CUDA-context creation. The incremental planner therefore correctly refused to
admit a second client because there was no immutable allocation configuration
to preserve, even while physical A100 memory was largely free.

The next controller launch now records a two-client MPS reservation on the
unknown anchor using the configured primary/secondary MPS percentages. The
anchor receives slot zero at launch; a later scheduler-approved incremental
candidate receives slot one. This does **not** impose a maximum parallel-job
count: admission still requires branch-profile/preflight evidence, live memory
headroom, and the time objective. It merely avoids making the first MPS client
incapable of safe incremental colocation.

Regression coverage verifies the planner records the reservation, the MPS
backend applies the reserved percentage to a one-job anchor, and dispatch
persists the anchor slot before worker creation. Focused scheduler/backend
suite: 110 passed, 4 subtests passed.

## Submission-time epoch extraction correction

Several generated PetFinder scripts keep their epoch budget in a literal
configuration dictionary, for example `{"epochs": 60}`, and train with
`config["epochs"]`. The submission-time introspector previously recognized
only assignment syntax such as `EPOCHS = 60`. It consequently submitted some
jobs without `max_epochs` or `planned_epochs`, preventing time-aware scoring
before the job first ran. The introspector now recognizes literal
`epochs`/`max_epochs`/`num_epochs` dictionary entries, so future scheduler
submissions receive their epoch budget before admission. Regression and
affected scheduler/executor suites: 120 passed.

## Runtime epoch-budget persistence correction

The live runner had a second, independent version of the same defect. When it
inferred epochs from a script at worker start, it saved only
`metadata.planned_epochs`. The time objective intentionally reads the
canonical `job.max_epochs` / `job.config.max_epochs` fields, so an active job
still reported an undefined remaining duration and incremental placement
returned no plan. The runner now persists all three fields atomically before
training begins. The live scheduler run's active job was safely backfilled
from its already-recorded 40-epoch script budget; only the `RUNNING` database
record changed, and no process was restarted. The `READY` queue was likewise
backfilled only where its immutable source contained a literal epoch budget.
The live profile now yields a concrete remaining duration (31 epochs at the
observed epoch time) instead of `None`.

The runner regression confirms that a source-inferred budget reaches both
canonical `max_epochs` locations before the script completes. Targeted runner,
scheduler, and executor suites: 97 passed.

## Static batch-configuration extraction correction

Generated scripts may set a known physical batch size as
`int(config["batch_size"])`. Although the dictionary value is static, the
source analyzer previously treated the `int(...)` wrapper as unresolved and
disabled batch probing. It now resolves a one-argument literal `int(...)`
expression recursively. This preserves safety: arbitrary calls are still not
evaluated. The associated batch-contract test and the existing runner,
scheduler, and executor tests pass (120 total). This update is synchronized to
`ABA` for subsequent submissions; it does not alter a submitted script or an
active worker.

## Canonical epoch reconciliation

The first epoch-budget persistence repair still skipped a job when
`metadata.planned_epochs` was already present. That is exactly how a newly
generated candidate reached the worker: it reported `planned_epochs: 50` but
left `max_epochs` empty, so the time objective could not compute remaining
epochs. The runner now reconciles, in order, canonical fields, submission
metadata, and source inference into all canonical fields before process launch.
The active 50-epoch job was safely reconciled in the scheduler database while
it continued training. No worker or controller restart occurred. Targeted
runner, scheduler, and executor suites: 97 passed.

## Rejected candidate evidence

One generated `quality_aware_multiscale_cnn_256` candidate failed before its
first epoch (14.9 seconds), so it is excluded by the >=30-second valid-node
rule. The failure is in generated model code, not scheduler placement:
`F.conv2d` received a 3-D Sobel buffer for a 4-D image tensor, producing the
documented stride/dimension runtime error. The scheduler recorded the exact
trace, marked the candidate failed, and continued with the next queued
candidate; no active process was stopped. This rejected candidate is not
included in any timing or scheduler-performance aggregate.

## Baseline context for later ETA and comparison

The completed original PetFinder baseline contains 21 valid nodes under the
same validity rule.  Its per-node execution-time distribution was: median
423.1 seconds, mean 671.7 seconds, and maximum 2701.0 seconds.  These values
are descriptive only; they are not a scheduler speedup claim.

## Deferred-generation branch-profile correction

The Scheduler execution loop previously interpreted an empty immediately
selectable search tree as terminal even when deferred Scheduler jobs were still
running. It then enlarged the root draft budget to meet `agent.steps` and
created more unrelated root architectures. Branch-profile estimates correctly
do not transfer across those architectures, so this erased the evidence needed
for incremental profile-based admission and produced an artificially serial
ready queue.

`_ensure_scheduler_generation_capacity` now receives the number of in-flight
Scheduler candidates. If no tree node is currently selectable but a candidate
is still running, it waits instead of making a new root draft. Once the result
is returned, normal same-branch improvement becomes selectable and can reuse
its profile. This does not set a parallel-job cap or relax the memory gate.
The lifecycle regression suite passed 8 tests. The already-running ABA
experiment is deliberately not restarted; the change applies to subsequent
Scheduler execution loops.

## Typed dataclass epoch-budget correction

An active PetFinder candidate expressed its training budget as a typed dataclass
field, `epochs: int = 45`. The extractor accepted plain assignments and
dictionary literals but not typed assignments; this left the canonical epoch
fields empty and incorrectly retained a transient completed-epoch value in
`planned_epochs`. The pattern now accepts a static annotation before the
integer assignment. The live job was verified to be running and was
transactionally backfilled to `max_epochs = config.max_epochs =
metadata.planned_epochs = 45`, with no worker restart. The combined relevant
test suite passed 117 tests.

## Live epoch update preserves canonical budget

The running worker keeps an in-memory job object. A safe external repair can
therefore update the SQLite canonical budget while that stale object still
contains an older provisional value. The per-epoch heartbeat previously used
the stale object first and overwrote `metadata.planned_epochs` with the current
completed epoch. It now reads the persisted job before calculating the total;
the canonical `max_epochs` and `config.max_epochs` therefore remain authoritative
through live updates. The runner's source inference now also recognizes typed
dataclass epoch fields. The updated runner and introspection source were synced
to ABA for future workers. Relevant tests passed: 119.

## Live runtime recalibration correction

The epoch-one prediction was kept unchanged for the remainder of a job. On a
PetFinder candidate whose later epochs were much slower than its first epoch,
this reported a 69-second full-run estimate after the process had already run
for minutes. Live epoch handling now recalibrates `seconds_per_epoch`, total
runtime, remaining time, confidence, and profile observation count at every
epoch marker. The current worker remains untouched; its profile was safely
recalibrated from 9 observed epochs and 611 seconds of worker elapsed time to
67.89 seconds/epoch, 3055 seconds total, and 2444 seconds remaining. No
comparison claim is made from this intermediate estimate. The relevant test
suite passed 120 tests.

## Batch-contract parsing correction

The active PetFinder candidate declares `physical_batch_size: int = 32` as an
entrypoint default and passes additional loader options with `**loader_kwargs`.
The static batch-contract analyser incorrectly classified this form as
unsupported because it did not use a function parameter's literal default when
there was no explicit caller argument. Consequently it recorded no batch-size
knob, disabled safe batch profiling for that candidate family, and prevented
later branch-profile reuse from receiving the correct default batch size.

The analyser now uses a statically resolvable default for the enclosing
function parameter only when no caller supplies that parameter. A regression
test covers a training `DataLoader` with `**loader_kwargs`; the full relevant
suite passed 121 tests. The exact active script now resolves to physical batch
size 32 with high confidence. The updated analyser was synchronized to ABA.
Already submitted jobs are preserved unchanged, so no active node was restarted
or re-queued; the correction applies to subsequently generated nodes.

## Safe queued-job batch backfill

The 11 ready jobs had been submitted before the batch-contract correction and
therefore retained the scheduler fallback of batch size 1. That fallback is
normally harmless for unsupported scripts because the runner cannot rewrite
their batch argument. It would, however, incorrectly override a script once
the corrected analyser proves its writable batch parameter.

Each ready script was re-analysed without execution. Exactly one script proved
a high-confidence `physical_batch_size: int = 32` default. Its existing READY
job record was transactionally backfilled to use that parameter and value 32;
the associated scheduler event records the source and reason. The remaining
ten unsupported scripts were intentionally left unchanged. No job was
re-queued, no active worker was interrupted, and no fixed concurrency cap was
introduced.

## Cold-start throughput diagnosis (live snapshot)

The requested primary validity rule is exactly `exec_time >= 30 seconds`; it
does not add a success-status filter. At the matched 19-node checkpoint,
Scheduler accumulated 16,327.3 seconds (mean 859.3 seconds/node), versus
12,032.2 seconds (mean 633.3 seconds/node) for the first 19 qualifying
baseline nodes: a 1.357 aggregate-time ratio. At the matched 20-node
checkpoint, Scheduler accumulated 17,515.9 seconds (875.8 seconds/node),
versus 15,551.5 seconds (777.6 seconds/node) for baseline: a 1.126 ratio.
These are descriptive, matched-count snapshots, not a speedup claim or final
experiment conclusion.

For transparency, the baseline contains four qualifying nodes that later
reported buggy status. A secondary successful-node-only statistic excludes
them: it is 1.225 at 19 nodes and 1.282 at 20 nodes. This secondary filter is
not used as the primary node-count rule because it was not part of the stated
30-second criterion. Earlier intermediate calculations that silently mixed the
two definitions are superseded by these explicitly labelled values.

The live scheduler state explains the missing throughput. One job is running
with a concrete epoch-derived estimate, but all ten queued jobs have distinct
model-family signatures and no compatible runtime or VRAM estimate. The
runtime database has zero pair profiles. A read-only reconstruction of the
placement decision against that exact state returned no incremental dispatch
plan (`PLAN None`). Consequently, the scheduler correctly does not create an
unprofiled colocated pair merely because VRAM is free. The absence of a fixed
parallel-job cap is verified: `parallel_job_cap` remains null.

The active 40-epoch capsule-regressor job is making progress (15 completed
epochs at this snapshot, with a recalibrated estimate of about 2,055 seconds
total). It remains running; no controller, worker, or queued job was stopped
or restarted. A fair final comparison will use equal valid-node counts.

The comparison also has an important search-state limitation. The first 19
valid Scheduler nodes are all `draft` nodes, whereas the first 19 baseline
nodes contain 9 `improve`, 4 `debug`, and 2 `evolution` nodes in addition to
4 drafts. The Scheduler run therefore received a materially different mixture
of agent-generated workloads after its old deferred-generation loop expanded
unrelated roots. Its higher observed runtime is real for this run, but it
cannot isolate scheduler transport overhead from the changed search-tree
mixture. The lifecycle fix above prevents that root expansion in a subsequent
run; the active run is intentionally preserved rather than restarted.

The currently running long candidate independently specifies a 40-epoch,
256-resolution capsule regressor with physical batch size 32, effective batch
size 128, and two persistent data-loader workers per loader. It has used about
1.1 GiB of the experiment A100 while its data/CPU pipeline is active. These
are agent-generated workload choices, not scheduler batch overrides or a
parallel-job cap. They further show why its duration cannot be attributed to
MPS transport alone.

## Equal-node visualization safeguard

The comparison plotter previously used `target_nodes` only for its axis label;
it still plotted every completed baseline node. That could visually compare an
unfinished Scheduler run against a longer baseline, contradicting the
equal-node requirement. `load_run` now orders budget-counted nodes and retains
only the first `target_nodes` for every run. A new regression test constructs a
three-node journal with a target of two and proves that only the first two
nodes reach the chart. The full plotter test module passes (3 tests).

The plotter also now supports the documented direct-script invocation rather
than requiring a package-module launch: it inserts the repository root before
importing `engine.node_accounting`. A subprocess regression test failed on the
former import error and then passed with the fix. The plotter suite now passes
4 tests.

`records/2026-09-05_petfinder_a100_equal_20node_interim.png` is the verified
20-node-versus-20-node interim artifact. It contains the required Gantt charts
above the metric-versus-node plots. The chart reports a 1.90-hour baseline span
and 4.90-hour Scheduler span at the same count; because the search-tree
mixtures differ, this is an observed run result, not causal proof that MPS
transport alone produced the difference.

The refreshed `records/2026-09-05_petfinder_a100_equal_21node_interim.png`
uses 21 qualifying nodes on each side. Under the primary `>=30s` rule,
baseline total execution time is 15,893.0 seconds (median 388.1 seconds) and
Scheduler total execution time is 17,813.8 seconds (median 628.1 seconds),
or a 1.121 aggregate-time ratio. The visible run spans are 1.90 and 4.98
hours respectively. This remains an interim observation because Scheduler is
still producing nodes and the search mixtures remain different.

## Argparse epoch-budget correction

A subsequent active candidate declared its training budget only through
`parser.add_argument("--epochs", type=int, default=40)`. Both submission-time
and worker-time epoch extractors missed this form, leaving canonical
`max_epochs` unset and causing a zero remaining-runtime estimate after the
first observed epochs. The introspector now recognizes literal argparse
defaults for `--epochs`, `--num-epochs`, and `--max-epochs`; the runner reuses
that parser as its source fallback. Regression tests cover both boundaries and
the relevant introspection, runner, and scheduler suite passes 115 tests.

The active job's source was read without execution, the literal default was
verified as 40, and only its persisted scheduler record was backfilled to
`max_epochs = config.max_epochs = metadata.planned_epochs = 40`. Subsequent
heartbeats retained the values and report a nonzero remaining time. No worker,
controller, or queued job was restarted.

## Run provenance limitation

The completed baseline and current Scheduler journals preserve their resolved
configuration but not a Git commit for the mutable runtime source directory.
They both use `codex_cli` with `gpt-5.6-terra` and origin-mode task semantics,
but baseline used `initial_drafts: 0` while Scheduler used
`initial_drafts: 1`. The source directory was also patched between the two
runs. Therefore their current traces demonstrate observed behavior but are not
a provenance-pinned causal Scheduler-only A/B comparison. The full-baseline
versus-current-Scheduler visualization is stored as
`records/2026-09-05_petfinder_a100_original_full_vs_scheduler_current.png` and
is explicitly not an equal-node timing claim.

## Guarded batch-size introspection correction

The next active Scheduler candidate used a configuration-derived loader batch:
`physical_batch_size = max(1, int(cfg["physical_batch_size"]))`. The
introspector could resolve the dictionary lookup and `int(...)`, but not the
outer `max(...)`, and therefore marked the script as batch-probe unsupported.
The job retained physical batch size 16 on an 80 GiB A100 while using only
about 1.5 GiB, so this was a FLOP-saturation gap rather than a fixed
parallel-job limit. The current active job was not stopped or modified.

`_resolve_static_int` now resolves constant-argument `min(...)` and `max(...)`.
A red test reproducing this guarded configuration expression failed before the
patch; after the patch, the complete relevant suite passed 116 tests. The
verified source was synchronized to the exact runtime
`engine/script_introspection.py` location for future submissions. Because the
controller has already imported its executor, the fix cannot retroactively
change the active job; it applies to a later controller or a newly started
experiment without requiring a fixed job cap.

## Counting definitions

Two quantities are intentionally independent. A **buggy node** is any node
whose completed run has `is_buggy = true`, regardless of how long it ran. An
**effective budget-counted node** is any non-root node with execution time at
least 30 seconds, regardless of whether its run was buggy. Thus a short failed
attempt remains a buggy node but does not consume an experiment node; a long
failed attempt is both buggy and budget-counted. All later reports and plots
must show these counts separately.
