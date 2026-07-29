# MLEvolve Time-Aware Packing Scheduler

## Goal Mode implementation plan for `hardware-awared`

### 1. Mission

Redesign the single-GPU scheduler as an online, time-aware packing algorithm.
Given a release-time trace

\[
\mathcal{J}=\{(J_1,a_1),(J_2,a_2),\ldots,(J_N,a_N)\},
\]

the scheduler must select:

1. which ready jobs run concurrently;
2. one of five batch-size choices for each newly dispatched job;
3. which concurrent backend to use;
4. when to drain active work for a high-priority exclusive branch probe; and
5. when a training job should stop because validation accuracy has stopped
   improving.

The primary system goal is to shorten the wall-clock time required to finish
the trace. The secondary goal is to reduce per-job turnaround time without
allowing an old job to wait forever.

Do not use GPU utilization, throughput, or slowdown as a placement input,
objective term, or rejection rule. Assume GPU utilization is always saturated.
GPU memory, runtime, backend availability, compatibility, release time,
priority, and early-stopping progress are the relevant control signals.

Work from the current `hardware-awared` branch. Preserve unrelated behavior and
existing user changes. Do not push, merge, or open a PR unless separately
requested.

---

## 2. Findings from the current branch

Before implementation, verify these observations against the current checkout:

- `scheduler/objective.py` currently rewards filling the safe VRAM budget.
  This is not aligned with the requested completion-time objective.
- `scheduler/candidate_generator.py` can generate power-of-two ranges, but it
  does not guarantee exactly the five requested choices.
- `scheduler/policies.py` already applies priority aging to queue ordering, but
  the group objective can still repeatedly choose groups that omit the oldest
  job. Queue aging alone therefore does not give a starvation guarantee.
- `scheduler/compatibility.py` rejects jobs based on predicted SM utilization.
  Remove this dependency from the new placement path.
- `scheduler/service.py::_enforce_packed_safety` reacts to the latest memory
  sample and pauses a packed job at the hard threshold. The new 90% rule is
  different: use average device memory to close admission to additional jobs;
  do not pause a job merely because one sample or the rolling average reaches
  90%.
- `parallel_auto_pack` can dispatch additional concurrent groups while work is
  active. The new live-memory admission gate and exclusive-probe drain state
  must cover this path.
- cached `CombinationProfile.objective_score` values use the old VRAM-fill
  meaning. Never compare these scores with the new time-based score. Version
  the objective and re-score cached candidates from raw measurements.
- pair slowdown prediction is intentionally deferred. Retain valid same-batch
  measurements as passive evidence, but never infer a value or consume it in
  placement.
- the resource estimator predicts average VRAM for a supplied batch size, but
  remaining runtime is not currently a batch-indexed estimate. Extend the
  estimator contract to return both memory and seconds per epoch for each
  candidate batch.
- `BatchResolution.apply(...)` persists a selected override into the job.
  Preserve an immutable originally requested batch size so the five-option
  search does not drift after a pause or redispatch.
- progress safe points already carry `epoch` and a metrics dictionary. This is
  the correct integration surface for the early-stopping watchdog.

---

## 3. Formal problem definition

### 3.1 Per-job inputs

For every job \(j\):

- \(a_j\): submission/release time;
- \(\pi_j\): base priority;
- \(E_j\): total configured epochs;
- \(e_j(t)\): completed epochs at decision time \(t\);
- \(b_j^0=2^{n_j}\): immutable originally requested batch size;
- \(K_j\): candidate batch-size set;
- \(\tau_{jk}\): predicted solo seconds per epoch at batch choice \(k\);
- \(m_{jk}\): predicted average VRAM at batch choice \(k\);
- \(p_{jk}(t)\): predicted remaining solo runtime.

Interpret the requested five choices as five powers of two around the original
exponent:

\[
K_j =
\left\{
2^{n_j-2},2^{n_j-1},2^{n_j},2^{n_j+1},2^{n_j+2}
\right\}.
\]

Clip values below 1, apply the job/configured maximum batch size, and deduplicate
after clipping. If the original batch is not a power of two, either reject the
job as invalid for this mode or use a documented policy such as
\(n_j=\operatorname{round}(\log_2 b_j^0)\). Prefer requiring an exact
power-of-two original batch because it avoids silently changing user intent.

For a not-yet-started job,

\[
p_{jk}(t) = E_j \tau_{jk}.
\]

For a running or resumed job,

\[
p_{jk}(t) =
\max(0,E_j-e_j(t))\tau_{jk}.
\]

Freeze the chosen batch after training starts unless the runner explicitly
declares that batch changes on resume are safe. Batch changes can otherwise
alter optimizer behavior and invalidate the original predictor estimate.

### 3.2 GPU and scheduler inputs

- \(V\): physical GPU VRAM;
- \(\rho_B\): configured predicted-memory budget fraction;
- \(B=\rho_B V\): predicted safe VRAM budget;
- \(P\): parallel job cap, where `null` means unlimited;
- \(A_b(t)\in\{0,1\}\): whether concurrent backend \(b\) is available;
- \(c_{ijb}\in\{0,1\}\): historical compatibility of jobs \(i,j\) on backend
  \(b\).

“Unlimited” must not trigger exhaustive enumeration. At a decision point, use
the effective bound

\[
P_{\mathrm{eff}} =
\min\left(
|Q_t|,
\left\lfloor
\frac{B-M_{\mathrm{active}}}
{\min_{j,k}m_{jk}}
\right\rfloor
\right),
\]

where \(Q_t\) is the candidate window. Apply an implementation safety cap only
to search complexity, not as a silent scheduling constraint.

### 3.3 Slowdown (deferred control input)

The current scheduler does not predict concurrent slowdown. For placement,
the completion offset and drain time are therefore

\[
d_{jg}=p_{jk_j},\qquad D_g=\max_{j\in S}d_{jg}.
\]

An observed packed run may record slowdown only when compared with an
exclusive profile at the same batch size. These measurements are passive
telemetry for future model design and must not affect compatibility,
feasibility, scoring, or cached decisions. Missing evidence remains missing;
the scheduler must not invent or persist a predicted value.

### 3.4 Hard feasibility constraints

A pack is feasible only when all conditions hold:

1. every member has arrived: \(a_j\le t\);
2. exactly one batch choice is selected for each member;
3. the parallel cap is respected:
   \[
   |S|+|A_t|\le P,
   \]
   unless \(P\) is unlimited;
4. predicted average memory is within budget:
   \[
   M_{\mathrm{active}}+\sum_{j\in S}m_{jk_j}\le B;
   \]
5. backend \(b\) is available and every member allows it;
6. every historical pair is compatible:
   \[
   c_{ijb}=1,\quad\forall i\ne j;
   \]
7. the live-memory admission gate is open;
8. there is no pending exclusive-probe drain reservation.

VRAM is a constraint, not a utilization reward.

### 3.5 Trace-level objective

For completion time \(C_j\), define:

\[
C_{\max}=\max_j C_j-\min_j a_j
\]

as trace makespan, and

\[
F_j=C_j-a_j
\]

as job flow/turnaround time.

Use both metrics:

\[
\min
\quad
\lambda_M
\frac{C_{\max}}{T_M}
+
\lambda_F
\frac{\sum_j w_jF_j}{T_F\sum_j w_j},
\quad
\lambda_M+\lambda_F=1.
\]

\(T_M\) and \(T_F\) are fixed normalization constants from a serial-FIFO
baseline, so the weights are dimensionless. Default to a makespan-oriented
policy such as \(\lambda_M=0.6\), \(\lambda_F=0.4\), but make both configurable.
Always report makespan and flow time separately; do not hide a regression in
one metric behind a combined score.

This offline objective is useful as an oracle for small test traces. The
production scheduler is online and cannot know future arrivals, so use the
rolling-horizon policy below.

### 3.6 Small-trace oracle

For tests and algorithm validation, enumerate feasible pack configurations
\(\mathcal{G}\). Each \(g\in\mathcal{G}\) contains a job subset, one batch
choice per job, a backend, member durations \(d_{jg}\), and drain time \(D_g\).

Use:

- binary \(x_g\): select pack \(g\);
- continuous \(S_g\): pack start time;
- binary \(y_{gh}\): precedence between selected packs;
- continuous \(C_j\): job completion time.

Required constraints:

\[
\sum_{g:j\in g}x_g=1,\quad\forall j
\]

\[
S_g\ge a_j-M(1-x_g),\quad\forall j\in g
\]

\[
C_j\ge S_g+d_{jg}-M(1-x_g),\quad\forall j\in g
\]

plus standard disjunctive non-overlap constraints for every selected pack pair
on the single GPU. Optimize the trace-level objective above. Use this only for
small fixtures because feasible-pack enumeration is exponential.

The general problem is an online, release-date, multiple-choice,
resource-constrained batch scheduling problem with incompatibility edges and
pack-dependent processing times. It is NP-hard. Do not attempt unrestricted
brute-force search in the production service.

---

## 4. Production algorithm

### 4.1 Decision events

Replan on:

- job arrival;
- job completion, failure, pause, or early stop;
- exclusive probe completion;
- a profile/predictor result becoming available;
- live-memory gate transition;
- configured periodic scheduler tick.

Do not preempt a running pack just because a higher-priority branch probe
arrives. Use drain-and-reserve behavior described below.

### 4.2 Prediction bundle

Add a single estimator API that returns all five choices in one call:

```python
@dataclass(frozen=True)
class BatchOptionEstimate:
    job_id: str
    batch_size: int
    avg_vram_mb: float
    seconds_per_epoch: float
    remaining_epochs: int
    remaining_runtime_seconds: float
    source: str                  # ml_predictor | branch_profile | probe
    confidence: float | None
    estimate_version: str
```

Requirements:

- reuse model feature extraction across the five ML-predictor calls;
- query exact batch-specific branch observations first;
- allow documented interpolation only within observed bounds;
- tag every estimate with source and confidence;
- never fall back from a missing runtime estimate to “zero”;
- retain per-job fallback from ML predictor to branch profile;
- cache by job/model shape, hardware, backend, batch size, and predictor/profile
  version;
- invalidate the cache when the underlying model, hardware, backend, or
  predictor version changes.

Pareto-prune a batch choice \(k_2\) if another choice \(k_1\) has

\[
m_{jk_1}\le m_{jk_2}
\quad\text{and}\quad
p_{jk_1}\le p_{jk_2},
\]

with at least one strict inequality. Keep the unpruned five-option bundle in
logs for auditability.

### 4.3 Fair candidate window

At time \(t\), define waiting time

\[
W_j(t)=t-a_j.
\]

Continue to use priority aging:

\[
\pi^{\mathrm{eff}}_j(t)
=
\pi_j
+
\Delta_{\mathrm{age}}
\left\lfloor
\frac{W_j(t)}{A_{\mathrm{age}}}
\right\rfloor.
\]

Build the planning window as the union of:

- the top `priority_window_size` jobs by effective priority/FIFO;
- the oldest `oldest_window_size` jobs by submission time;
- every pending exclusive probe; and
- every job that has crossed `starvation_timeout_seconds`.

This prevents an old low-priority job from disappearing outside a fixed
top-eight window.

Define positive scheduling weights inside the window:

\[
w_j(t)=
1+
\eta_p
\left(
\pi^{\mathrm{eff}}_j(t)
-
\min_{u\in Q_t}\pi^{\mathrm{eff}}_u(t)
\right).
\]

Add a hard starvation rule:

- if any normal job has waited at least `starvation_timeout_seconds`, choose
  the oldest such job as the mandatory anchor;
- only consider new packs containing that anchor;
- if no compatible pack exists, dispatch the anchor exclusively when the GPU
  is available.

This is stronger than a soft age bonus and gives a practical no-starvation
guarantee for finite-duration jobs.

### 4.4 Rolling-horizon cost

For a feasible candidate \(g=(S,\mathbf{k},b)\), compute:

\[
L_F(g)=
\sum_{j\in S}w_jd_{jg}
+
D_g\sum_{u\in Q_t\setminus S}w_u.
\]

The first term predicts completion offsets of selected jobs. The second term
charges the pack for delaying every unselected job until the next drain
boundary.

Normalize flow cost against running the mandatory anchor, or the head of the
effective-priority queue, exclusively:

\[
\operatorname{Score}(g)=
\frac{L_F(g)}{\max(\epsilon,L_F(g_{\mathrm{exclusive}}))}
\]

Minimize this score. Throughput/makespan terms, aggregate-gain gates, and
slowdown multipliers are intentionally absent. A small deterministic tie-break
sequence is acceptable:

1. lower score;
2. older mandatory/member submission time;
3. more jobs completed by the candidate;
4. lower predicted memory;
5. stable job-ID ordering.

Do not add a VRAM-fill reward. If two packs finish work equally quickly, prefer
the lower-memory pack.

### 4.5 Candidate search

The naive search size is

\[
\sum_{r=1}^{P_{\mathrm{eff}}}
\binom{|Q_t|}{r}5^r,
\]

so it must be bounded.

Use a hybrid strategy:

1. generate and Pareto-prune five batch choices per job;
2. for small windows and \(P_{\mathrm{eff}}\le3\), use exact enumeration;
3. otherwise use deterministic beam search or branch-and-bound:
   - seed with every feasible mandatory-anchor option, or each head candidate
     when no anchor exists;
   - add one job/batch option at a time;
   - prune immediately on memory, cap, backend, compatibility, probe, or
     live-admission failure;
   - compute an optimistic lower bound on the rolling-horizon score;
   - retain the best `beam_width` states at each depth;
   - emit every partial state as a legal candidate, so the search can choose
     fewer than the maximum number of jobs;
4. include an exclusive singleton fallback;
5. make search deterministic for identical inputs.

For admission beside already active groups:

- treat active jobs and their batch sizes as fixed;
- include their predicted memory and interference in every expansion;
- count active jobs against the parallel cap;
- do not mutate the batch size of an active job;
- dispatch only the new members as a compatible concurrent group;
- re-evaluate after any member exits.

---

## 5. High-priority exclusive branch probing

Introduce a first-class scheduling class instead of inferring probe importance
from `batch_probe.enabled`, because ordinary jobs may have probing enabled.

Suggested values:

```python
class SchedulingClass(str, Enum):
    NORMAL = "normal"
    EXCLUSIVE_PROBE = "exclusive_probe"
```

An `EXCLUSIVE_PROBE` job must be handled as follows:

1. when no work is active, dispatch the highest-effective-priority probe
   exclusively;
2. when a pack is active, set `exclusive_drain_requested=True`;
3. allow the current pack/groups to finish naturally;
4. while draining, admit no additional normal or packed jobs;
5. do not pause or preempt current pack members;
6. when all active groups are empty, dispatch the reserved probe exclusively;
7. after the probe persists its five batch-option measurements, clear the
   reservation and replan.

Order multiple reserved probes by effective priority, then submission time.
Expose drain state and the reserved probe ID in events and reports.

If an ordinary job lacks reliable candidate estimates, the scheduler may
request a probe reservation, but it must not create duplicate probes for the
same job/model-shape-hardware key.

---

## 6. Live average-memory admission gate

Use device-level memory samples. Define a rolling time-window average:

\[
\bar{u}(t)=
\frac{1}{|\mathcal{S}_L(t)|}
\sum_{r\in\mathcal{S}_L(t)}
\frac{\mathrm{memoryUsed}_r}{\mathrm{memoryTotal}_r},
\]

where \(\mathcal{S}_L(t)\) contains samples from the last
`admission_average_window_seconds`.

State machine:

- close packed admission when \(\bar{u}(t)\ge0.90\);
- while closed, do not add new concurrent jobs or groups;
- continue all active jobs;
- reopen only when \(\bar{u}(t)\le\rho_{\mathrm{resume}}\), default 0.85, for
  one complete averaging window;
- use hysteresis to avoid repeated open/close flapping;
- ignore a single peak if the rolling average remains below 90%;
- emit one event on each state transition, not on every scheduler tick.

Replace the current “latest sample at 90% pauses a fallback job” behavior for
this threshold. A genuine CUDA OOM remains a worker failure and may still mark
the combination incompatible. If a separate emergency pause threshold is
retained, it must have a different name, a higher default, average-based
semantics, and explicit documentation; do not silently reuse the 90% admission
threshold.

Validate configuration. Recommend
`predicted_budget_fraction <= live_admission_stop_fraction` by default. If the
user intentionally sets a higher predicted budget, warn clearly that a newly
created pack may begin above the live-admission stop threshold.

---

## 7. Early-stopping watchdog

### 7.1 Rule

For validation accuracy \(z_j(e)\), mode `max`, improvement occurs when

\[
z_j(e)>z_j^{\mathrm{best}}+\delta,
\]

where \(\delta\) is `min_delta`.

Maintain:

\[
q_j(e)=
\begin{cases}
0,&\text{if improved}\\
q_j(e-1)+1,&\text{otherwise}.
\end{cases}
\]

Stop when

\[
e\ge E_{\min}
\quad\land\quad
q_j(e)\ge P_{\mathrm{patience}}.
\]

Configuration:

```yaml
early_stopping:
  enabled: true
  metric_name: "accuracy"
  mode: "max"                 # max for accuracy; min for loss
  patience_epochs: 5
  min_delta: 0.0
  min_epochs: 1
  save_best_checkpoint: true
  restore_best_checkpoint: false
  missing_metric_policy: "ignore"
```

Rules:

- evaluate only once for each new epoch;
- persist best metric, best epoch, bad-epoch count, and last evaluated epoch;
- preserve this state across pause/resume and scheduler restart;
- ignore missing or non-finite metrics by default and emit a warning;
- support `mode=min` without duplicating logic;
- save/tag the best checkpoint when configured;
- stop at the epoch safe point;
- mark the job `COMPLETED`, not `CANCELLED` or `FAILED`;
- store reason `early_stopped_no_improvement`;
- emit best metric, best epoch, stop epoch, patience, and epochs saved;
- return a normal successful result to MLEvolve so an early-stopped candidate is
  not classified as a buggy node.

Prefer synchronous evaluation in the worker’s epoch safe-point path. A
scheduler-only poller can observe the metric too late and waste an additional
epoch. Implement the watchdog as a pure, testable state machine called by
`TrainingControlHook.safe_point(...)`. Add an `EarlyStopRequested` control-flow
exception caught by `execution/worker_entry.py`, and route it to a successful
completion helper with early-stop metadata.

When an early-stopped member leaves a pack, replan. A replacement may be added
only if the exclusive-probe drain flag is clear, the live-memory gate is open,
and all normal feasibility checks pass.

---

## 8. Configuration changes

Refactor or add configuration with backward-compatible defaults:

```yaml
gpu_scheduler:
  parallel_job_cap: null              # null = unlimited
  priority_window_size: 8
  oldest_window_size: 4
  starvation_timeout_seconds: 1800
  beam_width: 64
  exact_search_max_jobs: 3

  memory:
    gpu_vram_gib: null                # null = auto-detect
    predicted_budget_fraction: 0.85
    live_admission_stop_fraction: 0.90
    live_admission_resume_fraction: 0.85
    admission_average_window_seconds: 10

  objective:
    priority_weight: 0.10
    objective_version: "time_v3_flow_only"

  thresholds:
    pack_reject_max_slowdown: 1.30  # reserved; ignored by live placement

  batch_options:
    exponent_offsets: [-2, -1, 0, 1, 2]
    require_power_of_two_original: true

  exclusive_probe:
    enabled: true
    drain_without_preemption: true

early_stopping:
  enabled: false
  metric_name: "accuracy"
  mode: "max"
  patience_epochs: 5
  min_delta: 0.0
  min_epochs: 1
  save_best_checkpoint: true
  restore_best_checkpoint: false
  missing_metric_policy: "ignore"
```

Migration rules:

- accept the legacy integer `max_packed_jobs_per_gpu`;
- map it to `parallel_job_cap` when the new field is absent;
- treat `null` as unlimited;
- retain the old absolute `safe_vram_budget_gib` only as a compatibility input;
- when both absolute and fractional budgets are supplied, reject ambiguous
  configuration or define and test one explicit precedence rule;
- do not reuse old cached objective scores;
- preserve old scheduler modes until callers migrate, but route the requested
  time-aware mode through one new, clearly named mode such as
  `parallel_time_aware`.

---

## 9. Code change map

Keep the pure algorithm separate from process supervision.

### Domain and configuration

- `localml_scheduler/config/models.py`
  - add objective, live-memory admission, search, exclusive-probe, and
    early-stopping settings;
  - validate fractions, weights, patience, exponent offsets, and hysteresis.
- `localml_scheduler/configs/scheduler.example.yaml`
  - document the new settings and remove SM-utilization assumptions from the
    new mode.
- `localml_scheduler/domain/jobs.py`
  - add scheduling class/exclusive-probe intent;
  - preserve immutable requested batch size separately from dispatch override.
- `localml_scheduler/domain/progress.py`
  - add typed early-stop state or a stable metadata DTO.

### Prediction and profiles

- `localml_scheduler/scheduler/resource_estimator.py`
  - return batch-indexed memory and epoch-time estimates;
  - expose all five options in one API;
  - retain source/confidence/fallback metadata.
- `localml_scheduler/profiling/runtime_probe.py` and batch-profile storage
  - key epoch-time observations by hardware, workload shape, backend, and batch.
- profile domain/storage
  - persist selected batch vector, estimator version, and objective version;
  - persist slowdown only when measured against an exclusive profile at the
    same batch size; retain it as passive evidence.

### Pure planning algorithm

- `localml_scheduler/scheduler/candidate_generator.py`
  - exact five-option generation;
  - Pareto pruning;
  - fair dual-window selection;
  - exact small search and deterministic beam search.
- `localml_scheduler/scheduler/compatibility.py`
  - remove SM-utilization rejection from the new mode;
  - retain backend, explicit incompatibility, and cooldown gates;
  - ignore stored slowdown evidence.
- `localml_scheduler/scheduler/runtime_guardrail.py`
  - retain only the existing solo-runtime skew safeguard used by auto-pack;
  - do not predict or aggregate concurrent slowdown.
- `localml_scheduler/scheduler/objective.py`
  - replace memory-fill score with rolling-horizon time score;
  - return a full score breakdown for debugging.
- `localml_scheduler/scheduler/planner_types.py`
  - include member runtime estimates, predicted drain time, batch choices,
    score components, mandatory anchor, estimate sources, and objective version.
- `localml_scheduler/scheduler/placement_planner.py`
  - coordinate fairness, feasibility, search, and exclusive fallback;
  - never compare old and new cached objective scores.

Consider extracting pure helpers into:

- `scheduler/time_objective.py`;
- `scheduler/pack_search.py`;
- `scheduler/admission.py`;
- `scheduler/early_stopping.py`.

Do this only if it makes tests and ownership clearer; avoid a cosmetic
large-scale refactor.

### Service and execution

- `localml_scheduler/scheduler/telemetry.py`
  - add rolling time-window average and admission hysteresis.
- `localml_scheduler/scheduler/service.py`
  - maintain live-admission and exclusive-drain state;
  - stop using the 90% latest sample to pause a job;
  - prevent new dispatch during drain or closed admission;
  - replan after completion and early stop;
  - log actual trace timing and objective components.
- `localml_scheduler/scheduler/supervisor.py`
  - expose active group/member state required by admission checks;
  - preserve no-preemption drain semantics.
- `localml_scheduler/execution/control.py`
  - call the early-stopping state machine at epoch safe points.
- `localml_scheduler/execution/worker_entry.py`
  - treat `EarlyStopRequested` as successful completion.

### Observability

Record:

- release, first-dispatch, completion, waiting, and flow times;
- immutable requested and selected batch size;
- five candidate estimates and sources;
- predicted and actual average VRAM;
- measured per-member slowdown as passive telemetry;
- objective breakdown and rejection reasons;
- mandatory-anchor decisions;
- admission gate transitions;
- probe reservation/drain transitions;
- early-stop state and saved epochs.

Never label a short stress-test job buggy merely because it entered training
but the external stress timeout ended before full completion. Distinguish
`training_started`, `externally_timed_out`, `early_stopped_successfully`,
`failed`, and `completed`.

---

## 10. Required implementation sequence

### Phase 0 — Baseline and invariants

1. Run the existing scheduler and stress tests.
2. Capture current test results and benchmark metrics.
3. Add a feature flag/new scheduler mode so legacy modes stay usable.
4. Write hard-invariant tests before changing placement:
   - no job dispatched twice;
   - no candidate exceeds predicted memory budget;
   - backend and incompatibility rejections are honored;
   - active plus new jobs respect the cap.

### Phase 1 — Domain/config contract

1. Add validated settings and migration behavior.
2. Add immutable requested batch size and scheduling class.
3. Add versioned estimate and score DTOs.
4. Add serialization round-trip tests.

### Phase 2 — Five-option estimator

1. Generate exactly the requested exponent offsets.
2. Query ML predictor or branch profile for memory and seconds per epoch.
3. Add batch-aware runtime profile lookup.
4. Pareto-prune dominated options only after all estimates are logged.
5. Preserve safe fallback behavior for unsupported ML models.

### Phase 3 — Pure algorithm

1. Implement feasibility checks.
2. Implement fair candidate window and mandatory starvation anchor.
3. Implement exact search for small cases.
4. Implement deterministic beam search for larger/unlimited cases.
5. Implement rolling-horizon time score and stable tie-breaking.
6. Compare small random cases with the offline oracle.

### Phase 4 — Planner/service integration

1. Add `parallel_time_aware`.
2. Integrate active occupancy without changing active batch sizes.
3. Remove SM utilization from the new path.
4. Add structured rejection explanations.
5. Version cached combination objectives.

### Phase 5 — Memory gate and exclusive probes

1. Implement rolling average and hysteresis.
2. Replace 90% pause behavior with admission closure.
3. Add exclusive-probe reservation and drain state.
4. Verify no normal job is admitted between reservation and probe completion.

### Phase 6 — Early stopping

1. Implement the pure patience state machine.
2. Integrate at epoch safe points.
3. persist state across resume/restart.
4. mark early stop as successful completion.
5. log best metric and saved work.

### Phase 7 — Trace replay and evaluation

1. Build a deterministic trace simulator that accepts:
   - release times;
   - five per-job memory/runtime choices;
   - compatibility matrix;
   - realized-slowdown matrix for outcome simulation only;
   - backend availability changes;
   - live-memory samples;
   - validation-metric sequences.
2. Compare:
   - serial FIFO;
   - current fill-based packing;
   - new time-aware packing;
   - small-trace oracle.
3. Report:
   - trace makespan;
   - total, mean, median, and p95 flow time;
   - maximum waiting time and starvation count;
   - jobs/hour;
   - realized slowdown;
   - predicted versus actual average VRAM;
   - early-stopped epochs and wall time saved.

---

## 11. Test matrix

### Unit tests

- exact batch set for \(2^n\), lower-bound clipping, cap clipping, and dedup;
- immutable original batch after dispatch, pause, and resume;
- batch-specific runtime and memory estimate selection;
- dominated-option pruning;
- missing predictor result falls back per job to branch profile;
- combined memory equal to budget is accepted; above budget is rejected;
- unavailable backend, incompatible pair, and cooldown are rejected while
  stored slowdown values have no effect;
- SM utilization has no effect in the new mode;
- cap values 1, 2, 3, and unlimited;
- score chooses lower completion time instead of higher VRAM fill;
- old job becomes mandatory at starvation timeout;
- mandatory job falls back to exclusive when no compatible pack exists;
- deterministic beam-search output;
- stale VRAM-, slowdown-, or throughput-adjusted objective caches cannot
  influence `time_v3_flow_only`;
- one memory peak does not close admission;
- sustained average at 90% closes admission;
- admission reopens only below the resume threshold for the required window;
- closed admission never pauses active work;
- exclusive probe drains without preemption and runs next;
- no normal dispatch slips into a drain window;
- accuracy improvement resets patience;
- `min_delta`, `min_epochs`, `mode=min`, missing metric, and NaN behavior;
- early-stop state survives restart;
- early-stopped job is `COMPLETED`, not `CANCELLED` or `FAILED`.

### Property/invariant tests

For randomized traces:

- every dispatched member is released and runnable;
- each job has one selected batch;
- no job appears in two active groups unless explicitly supported;
- predicted memory never exceeds budget at admission;
- active plus new count never exceeds a finite cap;
- incompatible pairs never coexist;
- stored slowdown values never change an otherwise identical plan;
- urgent anchor is selected at the next legal drain/idle decision;
- identical inputs produce identical plans.

### Integration tests

- arrivals while a pack is running;
- pack member early-stops, then a new job is admitted;
- 90% rolling memory closes admission while active jobs finish normally;
- high-priority probe arrives during a pack and runs exclusively after drain;
- predictor failure for one of five options does not crash other jobs;
- service restart while admission is closed or a probe is reserved;
- real runner reports validation accuracy and early-stops successfully;
- stress-test timeout is classified separately from a model/training failure.

### Performance tests

Use seeded synthetic traces where the oracle shows a packing benefit. The new
policy must:

- obey every hard constraint in every run;
- show lower trace makespan and weighted flow time than serial FIFO on the
  designed packable fixtures;
- avoid the known case where the old VRAM-fill score chooses a fuller but
  slower pack;
- have bounded planning latency under the maximum configured queue window;
- produce a real A10 benchmark report with repeated runs and variance, without
  turning noisy performance percentages into brittle CI assertions.

---

## 12. Definition of done

The work is complete only when:

1. `parallel_time_aware` chooses jobs and batch sizes using predicted completion
   time, not GPU utilization or VRAM-fill reward.
2. Exactly five exponent-offset batch proposals are requested per eligible
   newly dispatched job before clipping/deduplication.
3. Predictor and branch-profile adapters return batch-specific average VRAM and
   seconds per epoch with source metadata.
4. All hard memory, cap, backend, and compatibility constraints are enforced;
   stored slowdown and throughput evidence cannot affect placement.
5. An old job cannot remain outside the candidate set forever and receives a
   mandatory-anchor fallback.
6. A high-priority exclusive probe drains current packs without preempting them
   and runs before any new normal pack.
7. sustained 90% average memory closes only new admission; a transient peak
   does not pause active work.
8. the early-stopping watchdog stops on configured accuracy patience and
   reports successful completion.
9. old VRAM-based cached objective values cannot influence the new score.
10. unit, invariant, integration, and existing regression tests pass.
11. trace replay reports both makespan and flow-time metrics against serial,
    old packing, and the small-trace oracle.
12. documentation explains the mathematical objective, configuration, fallback
    behavior, estimate sources, and limitations.

---

## 13. Required final report from the coding agent

Return:

1. a concise architecture summary;
2. the exact mathematical score implemented;
3. files changed and why;
4. configuration/migration notes;
5. test commands and results;
6. trace benchmark table for all compared policies;
7. remaining risks, especially runtime/memory prediction error, batch-size
   effects on model quality, and the deferred concurrent-slowdown model;
8. any behavior intentionally deferred.

Do not claim success from GPU utilization. Demonstrate success with trace
makespan, job flow time, waiting/starvation, hard-constraint compliance, and
correct early-stop outcomes.
