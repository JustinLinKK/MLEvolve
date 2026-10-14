# MLEvolve Non-ML Backend-Aware Colocation Trial Ranking Plan

> Historical design document. Its CUDA-stream and hybrid action domains were
> retired by [ADR 0001](adr/0001-retire-cuda-stream-backend.md). Current
> production ranking is restricted to the single configured `cuda_process` or
> `mps_process` mode; the sections below are retained only as design history.

## 1. Purpose

Implement a deterministic, source-informed colocation trial-ranking layer for the MLEvolve single-GPU scheduler.

The scheduler is hosted in one fixed backend mode, such as:

- `cuda_process`
- `mps_process`
- `cuda_stream`
- `mps_stream`

Every submitted training job is assumed to have already been adapted and optimized for the selected mode. The scheduler must not rewrite the training program or choose a different backend for individual jobs.

The new layer answers one question:

> Among the currently feasible jobs or job groups, which unknown colocation should be trialed first so that the scheduler is more likely to discover a low-slowdown, makespan-improving placement with fewer trials?

The primary system objective remains minimizing the completion span of the supplied single-GPU submission-time trace:

\[
C_{\max}=\max_i C_i-\min_i r_i
\]

where \(r_i\) is job \(i\)'s submission time and \(C_i\) is its completion time.

## 2. Hard constraints

The implementation must respect all of the following:

1. Run on one physical GPU.
2. The backend mode is fixed for the scheduler process.
3. The existing job predictor continues to provide only:
   - exclusive one-epoch training time;
   - exclusive VRAM requirement;
   - the values above for each supported batch-size candidate.
4. Do not add a neural network, learned reranker, regression model, classifier, embedding model, or training dataset.
5. Do not require new offline GPU profiling.
6. Reuse the existing live colocation trial as the final source of truth.
7. Static source analysis may order trials but must never be treated as proof that a placement is safe or beneficial.
8. Accuracy constraints and the existing five batch-size candidates remain unchanged.
9. Trial, setup, restart, and instrumentation time must be charged to the makespan objective.
10. The feature must be guarded by configuration and must preserve current behavior when disabled.

## 3. Non-goals

Do not implement the following as part of this work:

- A replacement for PerfSeer.
- A predictor for exact colocated epoch time.
- A predictor for actual SM utilization or memory bandwidth.
- Dynamic selection between MPS, streams, and process backends.
- Cross-GPU placement, migration, or distributed training.
- Kernel interception or an Orion-like operator scheduler.
- Automatic source-code optimization.
- Continuous tuning of heuristic weights.
- An exhaustive search over all MPS percentages or stream offsets.
- Skipping a required live trial solely because two source graphs appear similar.

## 4. Design overview

Add four components around the current placement planner:

1. **Static job analyzer**
   - Converts submitted source and known training configuration into a normalized, deterministic job fingerprint.
2. **Backend rule engine**
   - Converts one job or a candidate group into backend-specific risk components and reason codes.
3. **Trial-priority planner**
   - Filters infeasible or non-amortizable candidates, applies Pareto ranking, and orders the remaining trials.
4. **Exact result cache**
   - Reuses measured good/bad results for the same GPU, backend, job signatures, batches, and backend configuration.

The normal execution path becomes:

```text
ready jobs
  -> memory and compatibility filters
  -> exact profile lookup
  -> deterministic source-derived risk calculation
  -> Pareto trial ordering
  -> short live colocation trial
  -> measured piecewise-drain/makespan evaluation
  -> accept, reject, or try the next candidate
  -> persist exact result
```

## 5. Repository reconnaissance

Before changing code, the coding agent must inspect the newest `hardware-awared` branch and record the current commit hash.

At minimum, inspect:

- placement planner and candidate-window construction;
- time/makespan objective and piecewise-drain implementation;
- resource estimator and batch-candidate generation;
- colocation profile schema and storage;
- live trial lifecycle and abort/rollback path;
- backend registry and each backend launcher;
- source/model signature construction;
- configuration loading and validation;
- scheduler decision logging and existing tests.

Previously relevant paths included:

- `localml_scheduler/scheduler/placement_planner.py`
- `localml_scheduler/scheduler/time_objective.py`
- `localml_scheduler/scheduler/resource_estimator.py`
- `localml_scheduler/execution/backends.py`
- `localml_scheduler/execution/backend_registry.py`

Treat these as navigation hints. Verify current locations and interfaces before modifying anything.

Produce a short internal implementation map before coding:

| Concern | Current module/class/function | Planned integration point |
| --- | --- | --- |
| Ready-job ordering | Verify in repository | Replace only behind feature flag |
| Pair/group generation | Verify in repository | Add joint candidate enumeration |
| Batch enumeration | Verify in repository | Reuse unchanged |
| Memory admission | Verify in repository | Add conservative mode overhead |
| Trial launch | Verify in repository | Reuse unchanged |
| Trial evaluation | Verify in repository | Reuse measured piecewise objective |
| Profile persistence | Verify in repository | Extend exact cache identity if needed |

## 6. Proposed module boundaries

Use names consistent with the repository, but keep the following responsibilities separate.

```text
localml_scheduler/
  scheduler/
    source_fingerprint.py
    trial_candidate.py
    trial_priority.py
    pareto.py
    compatibility/
      base.py
      cuda_process.py
      mps_process.py
      cuda_stream.py
      mps_stream.py
```

Do not put source parsing, backend rules, and placement-state mutation into one large planner function.

### 6.1 Static analyzer interface

```python
class StaticJobAnalyzer(Protocol):
    def analyze(self, job: JobSpec, batch_size: int) -> StaticJobFingerprint:
        ...
```

Requirements:

- deterministic output;
- no GPU execution;
- no model training;
- cache by source hash, graph/config hash, and batch size;
- tolerate unsupported operations;
- return uncertainty indicators instead of failing the job;
- never execute untrusted arbitrary source merely to extract an AST.

### 6.2 Backend compatibility interface

```python
class CompatibilityPolicy(Protocol):
    backend_name: str

    def evaluate(
        self,
        active_group: Sequence[StaticJobFingerprint],
        candidate: StaticJobFingerprint,
        backend_config: BackendTrialConfig,
        hardware: HardwareSpec,
    ) -> CompatibilityAssessment:
        ...
```

The result should contain:

- hard rejection reasons;
- named continuous risk components;
- categorical reason codes;
- analysis confidence;
- suggested finite backend configurations;
- no scalar learned score.

### 6.3 Trial ranking interface

```python
class TrialPriorityPlanner(Protocol):
    def rank(
        self,
        scheduler_state: SchedulerState,
        candidates: Sequence[TrialCandidate],
        backend_name: str,
    ) -> Sequence[RankedTrialCandidate]:
        ...
```

## 7. Static job fingerprint schema

Create a versioned immutable schema. A suggested shape is:

```python
@dataclass(frozen=True)
class StaticJobFingerprint:
    schema_version: int
    source_hash: str
    graph_hash: str
    batch_size: int
    dtype: str

    steps_per_epoch: int | None
    predicted_epoch_seconds: float
    predicted_vram_bytes: int

    forward_flops: float | None
    training_step_flops: float | None
    estimated_bytes_per_step: float | None
    parameter_bytes: int | None
    activation_bytes: int | None
    gradient_bytes: int | None
    optimizer_bytes: int | None

    operator_count: int
    operator_histogram: Mapping[str, int]
    operator_flop_histogram: Mapping[str, float]
    largest_op_fraction: float | None
    small_op_fraction: float | None
    tensor_core_eligible_fraction: float | None
    reduction_fraction: float | None
    irregular_memory_fraction: float | None

    explicit_sync_count: int
    blocking_transfer_count: int
    async_transfer_count: int
    dataloader_worker_count: int | None
    cpu_augmentation_flag: bool | None
    checkpoint_frequency: float | None
    evaluation_frequency: float | None

    forward_phase: PhaseFingerprint | None
    backward_phase: PhaseFingerprint | None
    optimizer_phase: PhaseFingerprint | None

    unknown_operator_fraction: float
    dynamic_control_flow: bool
    analysis_warnings: tuple[str, ...]
```

### 7.1 Required source/configuration inputs

The analyzer should use information already available to the submitted job or scheduler:

- normalized model graph or model source;
- input shapes;
- batch size;
- datatype and AMP settings;
- optimizer type;
- gradient accumulation;
- activation checkpointing;
- dataset size or steps per epoch when known;
- training loop source;
- DataLoader configuration;
- evaluation/checkpoint cadence;
- use of fused, compiled, Flash Attention, or custom operators when detectable.

If only the model class is available and shapes are unknown, mark the analysis as low confidence. Do not invent dimensions.

### 7.2 Graph extraction precedence

Use this order:

1. Reuse an existing normalized graph/IR already generated by MLEvolve or PerfSeer.
2. Use an existing safe CPU/meta-tensor shape-propagation path if the repository already has one.
3. Use AST inspection for training-loop and DataLoader features.
4. Fall back to conservative unknown-operation markers.

Do not introduce a new dependency on a rarely maintained graph-shape package.

### 7.3 Operator normalization

Normalize equivalent framework calls into stable categories such as:

- GEMM/linear;
- convolution;
- attention;
- normalization;
- activation/elementwise;
- reduction/softmax;
- embedding/gather/scatter;
- pooling;
- data movement/layout conversion;
- recurrent;
- optimizer update;
- custom/unknown.

Include shapes and datatype in the graph hash. A ResNet with batch 8 and the same ResNet with batch 128 must not receive the same execution signature.

## 8. Deterministic analytical proxies

### 8.1 Step time

When steps per epoch are known:

\[
\tau_i = \frac{T^{epoch}_i}{N^{steps}_i}
\]

When they are unknown, keep `step_seconds=None`. Do not estimate it from a hidden constant.

### 8.2 Compute-pressure proxy

For known training-step FLOPs and step time:

\[
c_i = \frac{F_i}{P_{dtype}\tau_i}
\]

where \(P_{dtype}\) is the fixed GPU's documented peak throughput for the relevant datatype path.

### 8.3 Memory-traffic proxy

For known estimated bytes per step:

\[
m_i = \frac{D_i}{B_{GPU}\tau_i}
\]

where \(B_{GPU}\) is documented peak memory bandwidth.

These values are analytical lower-bound ratios, not claimed hardware utilization. Preserve values above one as an indication that the static estimate or assumptions are imperfect; do not silently clip before logging.

### 8.4 Operation granularity

Calculate at least:

\[
L_i=\frac{\max_o F_{i,o}}{\sum_o F_{i,o}}
\]

and a configurable small-operation fraction based on operation FLOPs or estimated lower-bound duration.

### 8.5 Resource class

Assign only an explanatory label:

- `compute_leaning`
- `memory_leaning`
- `balanced`
- `unknown`

Do not use the label as a hard placement decision. The continuous risk components remain authoritative.

### 8.6 Hardware specification source

Reuse the current hardware knowledge/configuration database. Add missing values only through versioned configuration:

- GPU model and architecture;
- VRAM capacity;
- memory bandwidth;
- FP32/FP16/BF16/Tensor throughput as applicable;
- backend/context memory allowance;
- supported MPS allocation templates.

No network lookup should occur in the scheduler hot path.

## 9. Candidate representation

```python
@dataclass(frozen=True)
class TrialCandidate:
    active_job_ids: tuple[str, ...]
    newcomer_job_ids: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    backend_name: str
    backend_config: BackendTrialConfig

    predicted_vram_bytes: int
    vram_headroom_bytes: int
    optimistic_makespan_gain_seconds: float
    estimated_trial_cost_seconds: float

    compatibility: CompatibilityAssessment | None = None
    exact_profile_status: str = "unknown"
```

Candidate identity must include:

- exact GPU model/architecture;
- backend name;
- backend configuration;
- job graph/source signatures;
- batch sizes;
- datatype;
- relevant framework/runtime compatibility versions already used by the profile store.

Do not include epoch count in the interference signature. Remaining epochs belong to the scheduling state, not the per-epoch compatibility identity.

## 10. Hard filters

Apply hard filters before heuristic ranking.

### 10.1 Backend compatibility

Reuse the existing backend eligibility contract. Since jobs are assumed pre-optimized for the hosted mode, incompatibility should indicate malformed metadata or a violated runtime contract, not a preference.

### 10.2 Conservative VRAM admission

\[
\sum_{i\in G} \widehat V_i
+\delta_{mode}(|G|)
+V_{safety}
\le V_{GPU}
\]

Use peak/reserved prediction if available. If the current predictor's VRAM target is not peak/reserved, apply an explicit configured guardband and document it.

The mode overhead must distinguish at least:

- ordinary process contexts;
- MPS clients;
- shared stream host;
- MPS plus stream topology.

### 10.3 Accuracy and batch constraints

Reuse all existing checks that determine whether a batch candidate preserves the job's accuracy contract.

### 10.4 Trial amortization

Use the existing trace/makespan simulator with an optimistic no-slowdown assumption to compute:

\[
\Delta C^{ideal}_{max}(a)
=C_{max}(baseline)-C_{max}(a,s_i=1)
\]

Reject an unknown trial when:

\[
\Delta C^{ideal}_{max}(a)
<\gamma T_{trial}(a)
\]

Start with configurable `gamma = 3.0`. This is a policy safety margin, not a learned parameter.

### 10.5 Known bad profile

Skip an exact configuration previously marked:

- OOM;
- backend failure;
- measured net loss;
- slowdown above the configured hard limit.

Allow expiry/invalidation when software, hardware, source, batch, datatype, or backend configuration changes.

## 11. Pareto ordering instead of a learned or weighted score

Do not combine risks using learned or manually tuned scalar weights.

For each backend, construct a named risk vector:

\[
R(a)=[r_1(a),r_2(a),...,r_k(a)]
\]

Candidate \(a\) dominates \(b\) when every risk component of \(a\) is no worse and at least one is better.

Compute deterministic Pareto fronts:

- Front 0: not dominated by another candidate.
- Front 1: not dominated after removing Front 0.
- Continue until all candidates are assigned.

Sort candidates using the tuple:

```text
(
  exact_profile_class,
  pareto_front,
  -optimistic_makespan_gain,
  analysis_uncertainty,
  -vram_headroom,
  starvation_or_priority_key,
  stable_candidate_id
)
```

Suggested exact profile classes:

1. measured-good and reusable;
2. unknown and eligible for trial;
3. measured-bad, which should normally be filtered rather than sorted.

The stable identifier is required so repeated runs make the same decision.

## 12. MPS-process compatibility policy

For candidate group \(G\), calculate:

### 12.1 Compute excess

\[
r_{compute}=\max(0,\sum_{i\in G}c_i-1)
\]

### 12.2 Memory-bandwidth excess

\[
r_{bandwidth}=\max(0,\sum_{i\in G}m_i-1)
\]

### 12.3 Same-resource conflict

For two jobs:

\[
r_{same}=c_ic_j+m_im_j
\]

For larger groups, use the sum of pairwise products.

### 12.4 Large-operation conflict

For two jobs:

\[
r_{blocking}=L_iL_j
\]

### 12.5 Cache/irregular-memory proxy

Use the combined irregular-memory and activation-working-set fractions as a separate risk component. Do not fold it into bandwidth risk.

### 12.6 Analysis uncertainty

Include:

- unknown operator fraction;
- dynamic-control-flow flag;
- missing shape or steps-per-epoch information;
- custom CUDA/fused operation flag when internal behavior is unknown.

### 12.7 Finite MPS configuration templates

Initially generate only:

- 50/50;
- 60/40;
- 40/60.

If an active process's MPS percentage cannot be changed safely after CUDA context creation, do not pretend it can be rebalanced. Either:

- launch the full group atomically with shares set before initialization; or
- use an existing supported checkpoint/restart boundary and charge its cost.

Do not implement a continuous MPS-share optimizer in this task.

### 12.8 MPS reason codes

Emit codes such as:

- `MPS_COMPUTE_EXCESS`
- `MPS_BANDWIDTH_EXCESS`
- `MPS_COMPUTE_MEMORY_COMPLEMENT`
- `MPS_BOTH_LARGE_OP_DOMINATED`
- `MPS_BOTH_UNDERFILLED_PROXY`
- `MPS_IRREGULAR_MEMORY_CONFLICT`
- `MPS_ALLOCATION_50_50`
- `MPS_ALLOCATION_60_40`
- `MPS_ANALYSIS_LOW_CONFIDENCE`

## 13. CUDA-stream compatibility policy

Assume the submitted job already satisfies the stream execution contract. The rule engine estimates degree of overlap rather than basic validity.

Risk components should include:

1. `sync_conflict`
   - explicit device-wide synchronization;
   - blocking host transfers;
   - forced default-stream dependencies when detectable.
2. `compute_excess`
   - same calculation as above.
3. `bandwidth_excess`
   - same calculation as above.
4. `large_operation_conflict`
   - pair/group large-operation product.
5. `phase_conflict`
   - deterministic comparison of forward/backward/optimizer resource labels.
6. `allocation_uncertainty`
   - unknown custom/fused operations and dynamic control flow.

### 13.1 Phase conflict

Represent each known phase using a short categorical sequence such as:

```text
[compute, compute, reduction, memory, optimizer_memory]
```

For each supported start offset, count positions where both jobs are predicted to stress the same resource. Normalize by compared positions.

Use only a finite offset menu, initially:

```text
0
0.25 * estimated_step_time
0.50 * estimated_step_time
```

If step time is unavailable, test offset zero first and do not manufacture an offset duration.

### 13.2 Stream reason codes

- `STREAM_LOW_SYNC_DENSITY`
- `STREAM_GLOBAL_SYNC_CONFLICT`
- `STREAM_ASYNC_COPY_COMPLEMENT`
- `STREAM_PHASE_CONFLICT`
- `STREAM_LARGE_OP_CONFLICT`
- `STREAM_OFFSET_ZERO`
- `STREAM_OFFSET_QUARTER_STEP`
- `STREAM_OFFSET_HALF_STEP`
- `STREAM_ANALYSIS_LOW_CONFIDENCE`

## 14. Ordinary CUDA-process compatibility policy

This mode should prioritize opportunities to hide host/input gaps and avoid expensive overlapping pressure patterns.

Risk components should include:

1. `continuous_gpu_conflict`
   - product of the jobs' compute/memory lower-bound pressure maxima.
2. `large_operation_conflict`
   - long/large operation proxies from graph shape.
3. `host_gap_alignment`
   - risk that both jobs perform CPU preprocessing, evaluation, or checkpointing at similar cadence.
4. `context_memory_pressure`
   - predicted VRAM plus per-process context allowance.
5. `synchronization_pressure`
   - explicit blocking operations and transfers.
6. `analysis_uncertainty`.

Positive reason codes can include:

- `PROCESS_CPU_GPU_COMPLEMENT`
- `PROCESS_DATALOADER_GAP_OPPORTUNITY`
- `PROCESS_CHECKPOINT_PHASE_OPPORTUNITY`
- `PROCESS_SHORT_BURST_PAIR`

Negative reason codes can include:

- `PROCESS_BOTH_CONTINUOUS_GPU`
- `PROCESS_LONG_KERNEL_CONFLICT`
- `PROCESS_CONTEXT_MEMORY_RISK`
- `PROCESS_HOST_PHASE_ALIGNMENT`

Do not use MPS-specific resource-allocation assumptions in this policy.

## 15. MPS-plus-stream compatibility policy

Treat this as a distinct hosted mode.

Do not add MPS and stream risk components into one weighted scalar.

Use hierarchical evaluation:

1. Evaluate inter-client MPS risk.
2. Reject candidates failing MPS hard gates.
3. Evaluate intra-client stream topology and synchronization risk.
4. Compute separate MPS and stream Pareto fronts.
5. Sort with:

```text
(
  mps_pareto_front,
  stream_pareto_front,
  -optimistic_makespan_gain,
  uncertainty,
  -vram_headroom,
  stable_candidate_id
)
```

The configuration identity must describe:

- number of MPS clients;
- job-to-client mapping;
- MPS percentage per client;
- streams per client;
- stream priority template;
- start-offset template.

Begin with one job per MPS client unless the current structured runner already safely supports several jobs inside one client process.

## 16. Source confidence and graceful degradation

Assign deterministic confidence classes:

- `HIGH`: shapes, datatype, graph, steps, optimizer, and training-loop features are known.
- `MEDIUM`: graph and shapes are known, but training-loop or phase features are incomplete.
- `LOW`: dynamic graph, many unknown/custom operations, missing shapes, or missing step count.

Confidence affects ordering only. It must not be interpreted as probability.

When confidence is low:

1. retain hard memory and amortization filters;
2. place the candidate behind comparable higher-confidence candidates;
3. keep the live trial requirement;
4. fall back to current time/priority ordering as the final tie-breaker;
5. log exactly which information was unavailable.

Never reject all candidates merely because source analysis is incomplete.

## 17. Exact measured-profile cache

The measured profile cache is lookup/memoization, not machine learning.

### 17.1 Key

Include:

- GPU architecture and exact GPU model;
- driver/CUDA/framework compatibility version already used by the repository;
- backend name;
- backend configuration;
- graph/source hashes;
- input shapes;
- datatype;
- batch sizes;
- group membership/topology.

Exclude:

- total epochs;
- submission time;
- current remaining epochs.

### 17.2 Value

Store:

- measured solo epoch/step time used as baseline;
- measured packed epoch/step time for every member;
- slowdown vector;
- measured packed peak memory when available;
- trial duration;
- setup/restart cost;
- measured piecewise drain gain;
- acceptance decision;
- failure/OOM classification;
- sample count and timestamp;
- configuration/schema version.

### 17.3 Reuse rules

- Exact known-good match: reuse if current predictor/runtime values are within the repository's existing skew tolerance.
- Exact known-bad match: skip until invalidated.
- Same graph but different shapes/batches: use only as a deterministic ordering hint; still require a trial.
- Any backend or topology change: treat as unknown.

Do not interpolate slowdowns between profiles in this task.

## 18. Trial-selection algorithm

Implement the following behavior.

```python
def choose_next_action(state, ready_jobs, fixed_backend):
    candidates = enumerate_candidates(state, ready_jobs, fixed_backend)

    feasible = []
    for candidate in candidates:
        if not passes_backend_contract(candidate):
            continue
        if not passes_accuracy_and_batch_constraints(candidate):
            continue
        if not passes_conservative_memory_gate(candidate):
            continue

        exact = exact_profile_cache.lookup(candidate.identity)
        if exact.is_bad:
            continue
        if exact.is_good:
            candidate.exact_profile_status = "good"
            feasible.append(candidate)
            continue

        candidate.optimistic_makespan_gain_seconds = simulate_ideal_gain(
            state, candidate
        )
        candidate.estimated_trial_cost_seconds = estimate_trial_cost(candidate)

        if candidate.optimistic_makespan_gain_seconds < (
            config.amortization_factor
            * candidate.estimated_trial_cost_seconds
        ):
            continue

        candidate.compatibility = policy_for(fixed_backend).evaluate(candidate)
        feasible.append(candidate)

    if not feasible:
        return current_safe_fallback(state, ready_jobs)

    ranked = deterministic_pareto_rank(feasible, fixed_backend)
    best = ranked[0]

    if best.exact_profile_status == "good":
        return launch_without_retrial(best)

    return launch_short_trial(best)
```

## 19. GPU-empty behavior

This is a required scheduler change.

When at least two jobs are ready and the GPU is empty, do not irrevocably select an SRT anchor before considering pair compatibility.

Instead:

1. Build a bounded ready-job window using current priority, age, starvation, and critical-path rules.
2. Enumerate feasible pairs jointly across that window.
3. Enumerate valid batch combinations.
4. Enumerate the fixed backend's finite configuration templates.
5. Apply exact-cache lookup and deterministic trial ranking.
6. Compare the best pair/trial action with the best exclusive action through the existing makespan objective.
7. Launch the pair atomically when required by the backend.
8. Use SRT only as a late tie-breaker or safe fallback.

Bound combinatorial cost with configuration, for example:

- ready window: 8-12 jobs;
- group size: 2 initially;
- five batch candidates per job, after current pruning;
- three MPS share templates;
- three stream offset templates.

Prune memory-infeasible combinations before source-risk calculation.

## 20. Active-group behavior

When one or more jobs are already active:

1. Preserve all backend membership/reconfiguration constraints.
2. Enumerate each eligible newcomer and batch choice.
3. Treat the active group as an aggregate fingerprint:
   - sum compute and bandwidth proxies;
   - sum memory predictions and mode overhead;
   - retain maximum large-operation fraction;
   - sum pairwise same-resource conflicts;
   - aggregate uncertainty conservatively using maximum or union.
4. Rank newcomer trials using the fixed backend policy.
5. Do not alter an active MPS client's immutable allocation unless the supported restart path is selected and charged.

Initially keep `max_group_size = 2` unless the current scheduler already has reliable trials for larger groups. Enable groups of three or more only after pair behavior is verified.

## 21. Measured trial evaluation

Do not change the principle that measured runtime overrides static ranking.

For each trial, measure:

- actual packed step or epoch time for every member;
- trial wall time;
- launch/setup/restart time;
- failure/OOM;
- peak/reserved memory if currently available;
- runtime skew from predicted/previous solo time.

Calculate:

\[
s_i=\frac{t^{packed}_i}{t^{solo}_i}
\]

Use the existing piecewise-drain logic: when the first colocated job completes, remaining work should return to the appropriate solo or smaller-group rate rather than retaining the full-group slowdown.

Accept only when measured projected net gain clears the configured margin:

\[
T_{packed,piecewise}
+T_{trial}
+T_{setup/restart}
<\frac{T_{sequential}}{1+\epsilon}
\]

Reuse the repository's current `min_gain` semantics if equivalent. Avoid introducing two conflicting gain thresholds.

## 22. Early abort and failure safety

At the earliest reliable step/epoch boundary, abort or reject the trial when:

- measured slowdown makes net gain impossible even under optimistic remaining execution;
- memory approaches the configured emergency limit;
- one job stops making progress;
- backend/runtime error occurs;
- measured solo/runtime skew invalidates the comparison;
- the remaining job durations are too short to amortize continuing the trial.

The trial manager must restore the scheduler to a known state through the current rollback/restart mechanism. Charge all lost work to the decision record.

## 23. Configuration

Add one versioned configuration block. Adapt names to the repository's configuration style.

```yaml
source_trial_ranking:
  enabled: false
  schema_version: 1
  policy: pareto

  ready_window_size: 10
  max_group_size: 2
  amortization_factor: 3.0
  require_live_trial_for_unknown: true

  source_analysis:
    prefer_existing_graph_ir: true
    allow_cpu_meta_shape_propagation: true
    execute_arbitrary_source: false
    cache_enabled: true
    max_unknown_operator_fraction_for_high_confidence: 0.05
    max_unknown_operator_fraction_for_medium_confidence: 0.25

  memory:
    safety_fraction: 0.10
    use_peak_or_reserved_prediction: true
    mode_overhead_bytes:
      cuda_process: 0        # replace with reviewed conservative configuration
      mps_process: 0         # replace with reviewed conservative configuration
      cuda_stream: 0         # replace with reviewed conservative configuration
      mps_stream: 0          # replace with reviewed conservative configuration

  mps_process:
    allocation_templates:
      - [50, 50]
      - [60, 40]
      - [40, 60]
    require_atomic_group_launch: true

  cuda_stream:
    offset_templates_in_steps:
      - 0.0
      - 0.25
      - 0.50

  cache:
    reuse_exact_good: true
    skip_exact_bad: true
    allow_interpolation: false
```

Do not ship zero mode-overhead defaults as production-safe values. The coding agent must locate existing measured/configured overhead handling or choose explicitly conservative documented values before enabling the feature.

## 24. Logging and observability

Every scheduling boundary should optionally emit a compact candidate table containing:

- stable candidate ID;
- active and newcomer job IDs;
- batches;
- fixed backend and configuration;
- predicted memory and headroom;
- optimistic makespan gain;
- estimated trial cost;
- exact profile status;
- Pareto front;
- named risk components;
- confidence class;
- reason codes;
- final rank;
- selected/rejected reason.

Example:

```json
{
  "candidate_id": "pair:job7:job9:mps:60-40",
  "backend": "mps_process",
  "pareto_front": 0,
  "optimistic_makespan_gain_s": 812.4,
  "vram_headroom_bytes": 4294967296,
  "risks": {
    "compute_excess": 0.0,
    "bandwidth_excess": 0.08,
    "same_resource_conflict": 0.12,
    "large_operation_conflict": 0.03,
    "analysis_uncertainty": 0.0
  },
  "reason_codes": [
    "MPS_COMPUTE_MEMORY_COMPLEMENT",
    "MPS_ALLOCATION_60_40"
  ],
  "selected": true
}
```

Do not log submitted source contents. Log hashes and derived numerical features only.

Add summary counters:

- candidate combinations generated;
- rejected by memory;
- rejected by amortization;
- exact good/bad cache hits;
- trials started;
- first-trial successes;
- trials aborted;
- trials producing net gain/loss;
- trial overhead seconds;
- makespan saved versus the scheduler's baseline estimate.

## 25. Unit tests

### 25.1 Source analyzer

Test:

- equivalent source formatting produces the same normalized hash;
- different shapes/batches/dtypes produce different execution signatures;
- operator normalization;
- FLOP/byte formulas for representative linear, convolution, attention, normalization, reduction, embedding, and optimizer operations;
- dynamic control flow produces warnings rather than crashes;
- unknown/custom operations increase uncertainty;
- explicit synchronization and blocking transfers are detected;
- DataLoader and checkpoint/evaluation features are extracted when present;
- no GPU is initialized during analysis.

### 25.2 Analytical proxies

Test:

- dimensional correctness;
- no division by zero;
- missing steps/time yields `None` plus uncertainty;
- values above one are preserved and logged;
- hardware dtype selection is correct;
- unsupported dtype falls back conservatively.

### 25.3 Pareto ranking

Test:

- dominance is correct;
- identical candidates use stable tie-breaking;
- ranking is deterministic across runs;
- optimistic gain is descending within the same front;
- low-confidence candidates are ordered behind otherwise equivalent high-confidence candidates;
- no hidden randomness exists.

### 25.4 Backend policies

Create hand-constructed fingerprints and verify:

- MPS compute-heavy plus memory-heavy ranks ahead of two compute-heavy jobs when other factors match;
- two underfilled small jobs remain eligible even if structurally similar;
- two large-operation compute-heavy jobs receive conflict reasons;
- stream global synchronization is penalized;
- async-copy/compute complement receives a positive stream reason;
- process CPU/GPU complement ranks ahead of two continuous-GPU jobs;
- MPS-stream applies MPS and stream fronts hierarchically.

### 25.5 Cache

Test:

- exact identity hits;
- batch, dtype, backend, topology, GPU, or source changes invalidate the hit;
- epochs do not affect cache identity;
- known-bad configurations are skipped;
- no interpolation occurs.

## 26. Integration tests

Build a fake backend and deterministic job runner so tests do not require a GPU.

Cover:

1. Feature flag disabled preserves current ordering and decisions.
2. GPU-empty planner selects a jointly evaluated pair instead of committing to an anchor prematurely.
3. Active-group planner ranks newcomers by backend policy.
4. Memory-infeasible candidate never reaches the backend.
5. Non-amortizable candidate never starts a trial.
6. Exact good profile bypasses retrial.
7. Exact bad profile is skipped.
8. First trial fails; scheduler tries the next ranked candidate.
9. Trial overhead changes acceptance from positive gross gain to negative net gain.
10. Runtime skew invalidates unsafe profile reuse.
11. MPS atomic launch receives the selected percentage before CUDA initialization.
12. Stream offset is applied only when step time exists.
13. Unknown source features fall back safely.
14. Starvation/priority constraints remain enforced.

## 27. Trace-replay evaluation

No ML dataset is needed. Use scheduler experiments and trace replay.

Compare:

1. Current SRT/newcomer trial order.
2. Random memory-feasible order with a fixed seed.
3. Time-and-VRAM-only ordering.
4. Non-ML backend-aware Pareto ordering.
5. A small exhaustive-pair oracle for evaluation only.

The oracle is not used in production and does not train anything. It provides a reference for whether good candidates appeared early.

### 27.1 Primary metrics

- Trace makespan, including trial/setup/restart overhead.
- Makespan regret relative to the exhaustive oracle.
- Fraction of scheduling events where the first unknown trial is beneficial.
- Mean and median number of trials required to find a beneficial pair.
- Time wasted in rejected/aborted trials.

### 27.2 Secondary metrics

- Top-1/top-3/top-5 recall of an oracle-near candidate.
- Mean reciprocal rank of the best or within-5%-of-best candidate.
- OOM/failure rate.
- Exact-cache hit rate.
- Mean job completion time as a secondary objective.
- Starvation incidents.
- Planner decision latency.

### 27.3 Required ablations

- remove graph operation features;
- remove compute/memory proxy;
- remove phase/synchronization features;
- remove exact profile reuse;
- replace Pareto fronts with current SRT order;
- disable joint pair selection at GPU-empty state;
- evaluate each backend separately.

This identifies whether complexity is producing real trial-order improvements.

## 28. Acceptance criteria

### Correctness gates

- No new ML model, training code, weights, or labeled-dataset pipeline is added.
- Feature disabled means no scheduler behavior change.
- Analysis does not initialize CUDA.
- Candidate ranking is deterministic.
- Memory-infeasible candidates never launch.
- Unknown candidates still require measured trials.
- Measured trial results override static ranking.
- Trial/setup/restart overhead is included in acceptance.
- Existing starvation and accuracy safeguards remain active.
- Exact cache identity excludes epochs and includes backend configuration.

### Performance targets

Treat these as evaluation targets, not reasons to weaken correctness:

- lower median trials-to-beneficial-placement than current SRT ordering;
- higher first-trial beneficial rate;
- lower trace makespan after charging all trial overhead;
- no systematic makespan regression on traces dominated by saturated jobs;
- planner overhead small relative to the shortest admitted trial;
- source analysis cached so repeated batch/configuration evaluation does not repeatedly parse source.

If the non-ML policy fails to improve trial order, retain exact cache reuse and hard amortization filtering, then simplify or disable source-risk ordering rather than adding an ML model.

## 29. Rollout plan

### Phase 0: Baseline and guardrail

- Record current branch commit.
- Run current scheduler unit/integration tests.
- Capture representative trace-replay outputs.
- Add feature flag defaulting to disabled.

### Phase 1: Schemas and analyzer

- Add versioned fingerprint and assessment schemas.
- Reuse existing graph representation.
- Implement deterministic feature extraction and caching.
- Add analyzer tests.

### Phase 2: Hardware proxies and common filters

- Connect hardware specification database.
- Calculate compute, bandwidth, granularity, memory, and uncertainty proxies.
- Implement conservative VRAM and amortization gates.
- Add tests.

### Phase 3: Exact cache improvements

- Review existing profile identity.
- Remove epoch count from interference identity if currently present.
- Add backend configuration/topology identity.
- Add exact good/bad reuse and invalidation tests.

### Phase 4: MPS-process policy

- Implement MPS risk vector and reason codes.
- Implement finite share templates.
- Ensure atomic launch or explicitly charged restart semantics.
- Add fake-runner integration tests.

### Phase 5: Pareto trial ordering

- Implement deterministic fronts and stable tie-breaking.
- Integrate with active-group newcomer ordering.
- Keep existing ordering as fallback.

### Phase 6: Joint GPU-empty pair selection

- Enumerate bounded pair/batch/configuration actions.
- Compare pair trial action with exclusive action using the existing time objective.
- Avoid premature SRT anchor commitment.

### Phase 7: Process and stream policies

- Add ordinary process risk vector.
- Add stream synchronization/phase risk vector.
- Add finite stream-offset templates.
- Test each backend independently.

### Phase 8: MPS-stream hierarchy

- Confirm current runtime topology actually supports this mode.
- Implement hierarchical MPS then stream evaluation.
- Do not silently expose the mode if the runtime contract is incomplete.

### Phase 9: Observability and trace replay

- Add candidate explanations and counters.
- Run baseline comparisons and ablations.
- Review every regression using logged reason codes.

### Phase 10: Controlled enablement

- Enable only for one backend first, preferably the backend with the most reliable live-trial lifecycle.
- Keep pair size at two.
- Expand to additional modes only after independent evaluation.
- Expand to groups above two only after pair behavior is stable.

## 30. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Source graph differs from executed kernels | Use analysis only for ordering; require measured trial |
| Compiler/fusion/custom kernels hide behavior | Increase uncertainty and defer behind comparable known candidates |
| Static byte estimates double-count/cache incorrectly | Keep compute, bandwidth, cache, and uncertainty as separate risks |
| Roofline ratios are not actual utilization | Label as proxies; never use as proof or an acceptance condition |
| Heuristic thresholds become hidden tuning | Prefer Pareto fronts and explicit policy margins |
| Candidate explosion | Bounded ready window, memory-first pruning, group size two, finite backend templates |
| MPS shares cannot be changed after launch | Atomic group launch or charged restart only |
| Stream source appears safe but runtime synchronizes | Live trial and early abort |
| Profile reuse becomes stale | Versioned exact keys and runtime-skew validation |
| Trials consume more time than they save | Optimistic amortization filter plus net-gain acceptance |
| Source analysis fails on dynamic programs | Low-confidence fallback; never block all scheduling |
| Rules hurt saturated-job traces | Compare against exclusive action and keep feature flag/fallback |

## 31. Deliverables

The coding agent should finish with:

1. Deterministic static job analyzer.
2. Versioned static fingerprint schema.
3. Common hard filters and amortization gate.
4. Backend-specific compatibility policies.
5. Deterministic Pareto-front implementation.
6. Joint GPU-empty pair selection.
7. Active-group newcomer ranking.
8. Exact measured good/bad profile reuse.
9. Candidate decision explanations and metrics.
10. Unit and fake-backend integration tests.
11. Trace-replay comparison report.
12. Configuration and operator documentation.
13. A final list of unsupported source patterns and backend limitations.

## 32. Required final coding-agent report

The agent's final report must state:

- branch and commit implemented;
- files changed;
- architecture and interfaces added;
- exact source features extracted;
- exact hard filters and risk components used;
- backend modes completed versus deferred;
- test commands and results;
- trace-replay results against each baseline;
- first-trial success and trials-to-benefit results;
- makespan including trial overhead;
- known limitations;
- whether the feature remains disabled by default;
- confirmation that no new ML model or dataset pipeline was introduced.

## 33. Research basis

The implementation is motivated by the following findings, but deliberately avoids their learned predictors:

- Horus shows that FLOPs, activations, batch size, shapes, and normalized computation-graph features contain useful scheduling information: <https://eprints.whiterose.ac.uk/id/eprint/173971/7/Horus_Interference-Aware_and_Prediction-Based_Scheduling_in_Deep_Learning_Systems.pdf>
- KACE shows that good MPS pairings depend on resource and kernel behavior and that coarse single metrics are insufficient: <https://bhan.im/static/publications/preprint/KACE_preprint.pdf>
- Orion shows that DNN iterations alternate between operators with different compute and memory requirements, motivating phase and granularity rules: <https://fotstrt.github.io/files/2024-orion.pdf>
- The EuroMLSys single-GPU training study shows benefits for underfilled and resource-diverse workloads and diminishing returns for saturated jobs: <https://itu-dasyalab.github.io/RAD/publication/papers/collocation_analysisi_euromlsys2024.pdf>
- NVIDIA documents MPS provisioning and the fact that allocation configuration affects client execution: <https://docs.nvidia.com/deploy/mps/when-to-use-mps.html>
- NVIDIA documents that stream overlap depends on non-default streams, synchronization, and available resources: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

The production design uses these physical insights only as deterministic trial-ordering rules. Actual colocation timing remains the decisive measurement.
