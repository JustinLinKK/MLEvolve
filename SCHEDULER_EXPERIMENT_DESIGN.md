# GPU-Normalized Experiment Design for the MLEvolve Scheduler

## 1. Purpose

This document defines a reproducible experiment for comparing the current
`parallel_time_aware` scheduler with:

1. true FIFO serial execution;
2. fixed FIFO two-process execution (`MP2`);
3. the same time-aware scheduler capped at two concurrent jobs; and
4. the production time-aware scheduler with adaptive concurrency.

The design is normalized by each job's fraction of GPU memory and compute
demand so that the same scenario definitions can be applied to GPUs with
different VRAM capacities and compute performance. An NVIDIA GeForce RTX 5090
with 32,607 MiB of detected VRAM is used as the worked example.

The intended headline claim is:

> On a fixed GPU and held-out workload distribution, the time-aware scheduler
> reduces mean job flow time relative to FIFO serial and fixed MP2, while
> remaining non-inferior in makespan and preserving or improving memory safety.

An additional adaptive-packing claim may be made only when the uncapped
scheduler safely achieves concurrency greater than two:

> Under workloads containing memory- and compute-light jobs, adaptive
> concurrency improves makespan and jobs/hour over a fixed two-process policy
> without increasing OOM or failure rates.

One GPU cannot establish a universal all-GPU result. The normalized design can
be repeated on any GPU; a cross-GPU claim requires results from more than one
GPU architecture and VRAM tier.

## 2. Motivation from Existing Results

Existing workload generators suppress the scheduler's main decision signals:

- `scheduler_benchmark_test/gen_trace_W3.py` assigns every job the same
  100-second duration estimate.
- `scheduler_benchmark_test/stress_bench/experiments/build_experiment2.py`
  adjusts batches per epoch so jobs reach one common target duration, then
  balances those jobs into equal-load groups.
- The RTX 5090 eight-job experiment intentionally normalized solo estimates to
  approximately 29--31 minutes.

These are useful throughput and regression tests, but they are weak tests of
shortest-remaining-time ordering. When all jobs have similar duration and MP2
keeps the GPU saturated, the scheduler mostly contributes profiling, trial,
reservation, and safety overhead.

The latest measured RTX 5090 experiment is consistent with this diagnosis: the
scheduler improved weighted flow time by 3.47%, but makespan was 1.36% worse.
That is a plausible SRT tradeoff, not evidence that SRT should be expected to
reduce makespan in every workload.

## 3. Research Questions

The experiment answers five separate questions.

### RQ1: Queue ordering

Does predicted shortest-remaining-time ordering reduce mean and total flow time
when a queue contains jobs of different lengths?

### RQ2: Fixed-cap placement

With both systems capped at two jobs, does time-aware admission avoid harmful
or infeasible pairings better than unconditional MP2?

### RQ3: Adaptive concurrency

When more than two jobs can safely and profitably share the GPU, does adaptive
concurrency improve makespan and throughput relative to fixed MP2?

### RQ4: Safety

Does predicted and live memory admission reduce OOMs, failed jobs, and
over-budget placements under mixed-VRAM workloads?

### RQ5: Prediction robustness

How much performance is lost between an oracle with exact timing information,
a warm scheduler with frozen measured profiles, and a cold scheduler that must
perform live trials?

These questions must be reported separately. A single aggregate number cannot
explain which scheduler mechanism caused an improvement or regression.

## 4. GPU-Normalized Resource Model

### 4.1 Hardware calibration

For GPU `g`, record:

- `M_phys(g)`: detected physical VRAM in MiB;
- `f_budget(g)`: configured scheduler prediction-budget fraction;
- `M_budget(g) = M_phys(g) * f_budget(g)`;
- `P_emp(g, precision)`: empirically measured sustained compute ceiling for
  FP32, TF32, FP16, and BF16 where supported;
- `B_emp(g)`: empirically measured sustained device-memory bandwidth;
- driver, CUDA, PyTorch, GPU clocks, power limit, persistence mode, and device
  temperature at run start.

Use empirical sustained ceilings rather than advertised peak specifications.
Run each calibration at least five times and report the median and coefficient
of variation.

### 4.2 Per-job resource vector

Profile each job alone and represent it as:

```text
R_j = (V_phys_j, V_budget_j, C_j, B_j, H_j, S_j)
```

where:

```text
V_phys_j   = 100 * peak_device_memory_delta_j / M_phys(g)
V_budget_j = 100 * predicted_admission_memory_j / M_budget(g)
C_j        = 100 * achieved_compute_throughput_j / P_emp(g, precision_j)
B_j        = 100 * achieved_device_bandwidth_j / B_emp(g)
H_j        = host/input stall percentage during steady-state epochs
S_j        = measured solo remaining runtime
```

Record all of the following memory values:

- peak device memory above the idle baseline;
- mean active device memory above the idle baseline;
- process peak allocated memory;
- process peak reserved memory; and
- the scheduler's predicted admission contribution.

`V_phys_j` supports cross-GPU reporting. `V_budget_j` explains actual scheduler
admission decisions.

### 4.3 Compute-demand percentage

`nvidia-smi utilization.gpu` is a busy-time indicator, not a percentage of
available SM capacity. It must not be used alone as `C_j`.

Prefer a profiler metric such as steady-state SM throughput as a percentage of
peak sustained throughput. Sample representative steady-state training steps,
excluding model construction, CUDA initialization, first-epoch compilation,
validation, and checkpoint I/O. If an instruction-level profiler is too
intrusive, report a resource vector instead of inventing one scalar:

```text
(SM throughput %, tensor throughput %, DRAM bandwidth %, host stall %)
```

Compute and bandwidth percentages are descriptive, not additive admission
rules. Two jobs at 40% solo compute demand do not necessarily consume 80% when
colocated. Pair and group timing must still be measured.

### 4.4 Resource-demand classes

Use these GPU-independent bins.

| Class | VRAM as % of physical GPU | Compute demand as % of empirical ceiling |
|---|---:|---:|
| 1: very low | 5--15% | 0--20% |
| 2: low | 15--30% | 20--40% |
| 3: medium | 30--45% | 40--60% |
| 4: high | 45--60% | 60--80% |
| 5: very high | 60--90% | 80--100% |

A job receives separate VRAM and compute classes. For example, `V2-C5` means
low VRAM but very high compute demand. Jobs below 5% VRAM may be retained as a
micro-job category, but they should not replace the main matrix.

Also record the bandwidth class using the same 20-percentage-point divisions.
Bandwidth distinguishes compute-saturated jobs from memory-bandwidth-saturated
jobs that have similar `nvidia-smi` busy percentages.

## 5. RTX 5090 Worked Example

The measured example device has:

```text
M_phys  = 32,607 MiB
f_budget = 0.95
M_budget = 30,976.65 MiB
```

### 5.1 RTX 5090 VRAM classes

| VRAM class | Physical-VRAM range | Approximate RTX 5090 range |
|---|---:|---:|
| V1 | 5--15% | 1,630--4,891 MiB |
| V2 | 15--30% | 4,891--9,782 MiB |
| V3 | 30--45% | 9,782--14,673 MiB |
| V4 | 45--60% | 14,673--19,564 MiB |
| V5 | 60--90% | 19,564--29,346 MiB |

Admission uses the 30,976.65 MiB scheduler budget, not the physical maximum.
For example, a job predicted at 8,000 MiB consumes:

```text
V_phys   = 24.53%
V_budget = 25.83%
```

Two and three such jobs are predicted to consume 51.65% and 77.48% of the
scheduler budget. Four consume 103.30% and must be rejected before dispatch.

### 5.2 Representative RTX 5090 scenarios

| Scenario | Per-job target | Expected discriminator |
|---|---|---|
| Light pack | 5--7 GiB, C1--C2 | Adaptive scheduler may admit 3--4 jobs while MP2 stays at 2 |
| Compute saturation | 4--8 GiB, C5 | Memory permits packing, but timing gain may reject harmful concurrency |
| Memory boundary | 14--16 GiB, any compute class | Some pairs fit and others exceed the 95% budget |
| Near-exclusive | 20--28 GiB, V5 | Scheduler should run one job and prevent MP2 OOM risk |
| Asymmetric | one 16--19 GiB job plus one 3--6 GiB job | Mixed pair may fit even though two large jobs do not |
| Complementary | compute-heavy/low-bandwidth plus bandwidth-heavy/lower-compute | Empirical pair gain determines whether resource complementarity helps |

Compute classes for the RTX 5090 must be assigned from measured sustained
ceilings on that device. They must not be inferred from model names.

## 6. Experimental Policies

Use the following policies in every applicable workload.

| ID | Policy | Queue order | Maximum concurrency | Backend |
|---|---|---|---:|---|
| P0 | True FIFO serial | release time, then queue sequence | 1 | exclusive |
| P1 | Fixed FIFO MP2 | release time, then queue sequence | 2 | CUDA processes |
| P2 | Time-aware cap 1 | current time-aware ordering | 1 | exclusive |
| P3 | Time-aware cap 2 | current ordering and admission | 2 | CUDA processes |
| P4 | Time-aware adaptive | current ordering and admission | production cap or unlimited | CUDA processes |
| P5 | Oracle simulator | exact solo and group timings | same as tested policy | simulated only |

P2 versus P0 isolates queue ordering. P3 versus P1 isolates time-aware
admission at equal concurrency. P4 versus P1 measures the complete benefit of
adaptive concurrency. P5 establishes whether the workload contains enough
theoretical opportunity to justify a real-GPU run.

Do not label the production time-aware scheduler with cap 1 as FIFO. It still
selects jobs by estimated remaining runtime.

## 7. Workload Suites

### 7.1 Suite A: SRT mechanism qualification

This is a positive-control experiment for queue ordering, not the final
generality claim.

Use 12 jobs with equal priority and fixed requested batch sizes:

| Duration class | Count | RTX 5090 target duration |
|---|---:|---:|
| Short | 6 | approximately 2 minutes |
| Medium | 4 | approximately 6 minutes |
| Long | 2 | approximately 12 minutes |

Construct duration classes by changing epoch count or dataset length while
keeping model, input, precision, and batch fixed where possible. This makes
remaining epochs the causal source of runtime variation.

Arrival pattern:

- release six jobs at `t=0`;
- release six jobs at `t=90 seconds`;
- randomize order within each burst;
- use at least ten held-out order seeds.

Run P0, P1, P2, and P3. The primary comparison is P2 versus P0 for mean flow
time. P3 versus P1 tests whether the ordering benefit survives two-process
execution.

### 7.2 Suite B: VRAM-compute matrix

The full characterization surface is a 5-by-5 matrix of VRAM and compute
classes. A minimum representative design uses nine cells:

```text
V1-C1  V1-C3  V1-C5
V3-C1  V3-C3  V3-C5
V5-C1  V5-C3  V5-C5
```

Use at least two distinct job identities per populated cell. Reprofile after
changing batch size, precision, input shape, or model width; those changes can
move the job into a different compute or bandwidth class.

For each populated cell, measure:

1. solo performance;
2. same-cell pairs;
3. neighboring-cell pairs;
4. asymmetric low/high-VRAM pairs; and
5. complementary compute/bandwidth pairs.

The full pair matrix is unnecessary if resources are limited. The following
pair archetypes are required:

| Pair archetype | Required examples | Purpose |
|---|---:|---|
| Low VRAM + low compute | 3 | Tests concurrency above two |
| Low VRAM + high compute | 3 | Tests rejection of compute contention |
| Medium VRAM + medium compute | 3 | Tests ordinary two-job packing |
| High VRAM + any compute | 3 | Tests admission boundaries |
| Large + small asymmetric | 3 | Tests nonuniform packing |
| Compute + bandwidth complementary | 3 | Tests multidimensional complementarity |

### 7.3 Suite C: Representative mixed workload

After Suites A and B qualify the mechanisms, build a 24-job held-out workload
with this composition:

| Resource category | Share |
|---|---:|
| V1--V2, C1--C3 | 25% |
| V1--V2, C4--C5 | 20% |
| V3, mixed compute | 25% |
| V4, mixed compute | 20% |
| V5, mixed compute | 10% |

Use the following duration distribution independently of resource category:

- 50% short;
- 33% medium; and
- 17% long.

Do not assign duration based on whether a resource category is favorable to the
scheduler. Cross resource and duration classes using seeded randomization.

Use three arrival regimes:

1. **Batch:** all jobs available at `t=0`;
2. **Bursty:** bursts of eight jobs with persistent backlog; and
3. **Poisson:** light, saturated, and overloaded offered-load levels.

Define serial offered load as:

```text
rho_serial = arrival_rate * mean_measured_solo_runtime
```

Use approximate targets of `rho_serial = 0.6`, `0.9`, and `1.2`. Report the
definition explicitly because effective GPU capacity under packing is
policy-dependent.

### 7.4 Suite D: homogeneous negative control

Use jobs with runtime ratio below 1.2, one VRAM class, and one compute class.
The scheduler should not report a large SRT advantage here. This measures
scheduler overhead and guards against conclusions driven by instrumentation or
cache imbalance.

### 7.5 Suite E: cold-start versus warm-profile operation

Run Suite C in two profile states:

- **Warm:** solo and group profiles are frozen from independent calibration;
- **Cold:** no colocation profiles are present and live trials are required.

Report calibration time, rejected-trial work, inconclusive trials, and profile
reuse. Do not average warm and cold results into one number.

## 8. Pair and Group Performance Model

For every measured pair or group, calculate:

```text
slowdown_j = colocated_runtime_j / matched_solo_runtime_j
```

and the scheduler-compatible piecewise drain gain:

```text
gain = sequential_piecewise_drain / packed_piecewise_drain
```

Interpretation:

- `gain > 1.10`: clearly beneficial placement;
- `0.95 <= gain <= 1.10`: neutral or within likely system noise;
- `gain < 0.95`: harmful placement;
- OOM or failed launch: infeasible placement.

These thresholds qualify the workload; the production scheduler continues to
use its configured `colocation.min_gain`.

Repeat solo timings at least five times and pair timings at least three times.
Require solo runtime coefficient of variation below 10% and pair runtime
coefficient of variation below 15% for a stable-profile experiment. Unstable
cells remain valid robustness tests but must not be used as trusted profile
evidence.

## 9. Workload Qualification Gates

Before an expensive repeated real-GPU comparison, a workload must satisfy:

| Gate | Requirement |
|---|---|
| Runtime spread | `p90 / p10 >= 4` for SRT suites |
| Runtime variability | coefficient of variation `>= 0.6` for SRT suites |
| Queue choice | at least three runnable candidates at 50% of dispatch boundaries |
| FIFO disagreement | FIFO chooses a non-shortest candidate at least 30% of boundaries |
| Prediction ranking | Spearman rank correlation with measured solo runtime `>= 0.8` |
| Solo stability | repeated solo runtime CV `<= 10%` |
| Pair diversity | at least 20% clearly beneficial pairs |
| Adverse diversity | at least 20% harmful or infeasible pairs |
| Oracle opportunity | at least 10--15% predicted mean-flow advantage over the relevant baseline |

If the oracle cannot beat MP2 using exact measured timings, the real scheduler
will not beat MP2 after adding dispatch and live-trial overhead. The workload
should then be retained as a negative result or the performance claim should be
changed; it should not be repeatedly reordered until the scheduler wins.

Qualification data may select resource strata, but final workload identities,
duration/resource crossings, and arrival orders must use held-out seeds.

## 10. Fairness and Causal Controls

For the scheduling-only comparison:

- use the same job code, dataset subset, seed, epochs, precision, and requested
  batch size under all policies;
- freeze each job to one batch size selected before policy execution;
- use the same CUDA-process mechanism for P1, P3, and P4;
- use the same release schedule and queue sequence;
- disable early stopping;
- exclude calibration from execution makespan but report it separately;
- prewarm every policy equally or run every policy cold;
- use a fresh scheduler/runtime directory for every run;
- give each job equal priority; and
- count failures, OOMs, and timeouts rather than dropping those jobs.

Run a secondary end-to-end experiment that allows scheduler-selected batch
sizes and early stopping. Label it as a system comparison, not a pure scheduling
comparison.

## 11. Experimental Procedure

### Phase 0: Environment validation

1. Reserve the GPU exclusively.
2. Record hardware and software versions.
3. Confirm no unrelated process holds GPU memory.
4. Fix power and clock policy where operationally permitted.
5. Define an acceptable run-start temperature band.

### Phase 1: Hardware calibration

1. Measure empirical compute ceilings by supported precision.
2. Measure empirical device-memory bandwidth.
3. Measure idle memory, power, and utilization baselines.
4. Repeat until calibration CV is acceptable.

### Phase 2: Solo job profiling

1. Run every job alone five times.
2. Exclude warmup and initialization from steady-state demand metrics.
3. Record memory, compute, bandwidth, runtime, and host-stall values.
4. Assign normalized resource classes.

### Phase 3: Pair/group profiling

1. Measure the required pair archetypes.
2. Calculate slowdown and piecewise drain gain.
3. Split observations into scheduler calibration and held-out evaluation sets.
4. Freeze the warm-profile database.

### Phase 4: Oracle screening

1. Replay each proposed trace using exact held-out solo and group timings.
2. Compare FIFO serial, FIFO MP2, time-aware cap 2, and adaptive placement.
3. Apply the workload qualification gates.

### Phase 5: Paired real-GPU execution

For each workload seed, run every policy as a paired block. Randomize or use a
balanced Latin-square order so that no policy is consistently first or last.
Wait for the GPU to return to the run-start temperature and idle-utilization
band between runs.

Use at least ten paired blocks per headline scenario. Conduct a pilot to
estimate variance, perform a power analysis, and freeze the final sample size
before examining headline results.

## 12. Metrics

### 12.1 Primary metric

```text
flow_time_j = finish_time_j - release_time_j
mean_flow_time = mean(flow_time_j)
```

Mean flow time matches the current shortest-remaining-time ordering objective.

### 12.2 Secondary performance metrics

- total flow time;
- median and p95 flow time;
- normalized turnaround: `flow_time_j / matched_solo_runtime_j`;
- queue wait, median queue wait, and p95 queue wait;
- makespan;
- jobs per hour;
- time at concurrency 0, 1, 2, and 3+;
- refill delay;
- per-job colocated slowdown;
- piecewise drain gain;
- scheduling-decision latency; and
- live-trial and calibration overhead.

### 12.3 Safety and resource metrics

- successful completion rate;
- OOM, launch failure, timeout, and cancellation count;
- predicted-budget violations;
- peak and p95 device memory;
- average and p95 compute demand;
- average and p95 device-memory bandwidth demand;
- energy per completed job; and
- jobs completed per kWh.

### 12.4 Prediction metrics

- solo runtime MAPE and rank correlation;
- VRAM prediction error;
- colocation-gain prediction error;
- accepted, rejected, and inconclusive trial count; and
- false-admit and false-reject rates.

## 13. Statistical Analysis

The independent observation is one complete trace-policy run, not an individual
job. Jobs within one trace share GPU state and are statistically correlated.

For every paired seed and baseline, calculate:

```text
metric_ratio = scheduler_metric / baseline_metric
relative_improvement = 1 - metric_ratio
```

Report:

- all raw paired runs;
- arithmetic mean for time totals;
- median and geometric-mean ratios;
- paired bootstrap 95% confidence intervals;
- a paired permutation test or Wilcoxon signed-rank test; and
- Holm correction across multiple headline scenarios.

Never encode a failed or timed-out run as zero seconds. Report failure rates and
use the timeout ceiling in a clearly labeled worst-case sensitivity analysis.

## 14. Pre-Registered Acceptance Criteria

### 14.1 Flow-time claim

Against both true FIFO serial and fixed MP2:

- mean flow time improves by at least 10%; and
- the upper bound of the paired 95% confidence interval for the scheduler to
  baseline ratio is below `1.0`.

### 14.2 Makespan claim

- scheduler makespan is better than FIFO serial; and
- scheduler cap-2 makespan is non-inferior to MP2 with a 5% margin, meaning the
  upper confidence bound of the ratio is below `1.05`.

For the adaptive-concurrency suite, claim makespan superiority over MP2 only if
the upper confidence bound is below `1.0` and concurrency above two is observed.

### 14.3 Safety claim

- no increase in failed or incomplete jobs;
- zero scheduler-caused predicted-budget violations;
- no OOM in a placement admitted as safe; and
- p95 wait and starvation remain within predeclared service limits.

### 14.4 Generalization claim

Report every populated VRAM-compute cell. A cross-scenario claim requires:

- improvement in the geometric mean across held-out cells;
- no unexplained severe regression in any safety-critical cell; and
- the acceptance criteria to hold on at least two GPU tiers before using
  language such as "across GPUs."

Recommended hardware tiers are:

1. a constrained-memory GPU;
2. a high-end consumer GPU such as the RTX 5090; and
3. a datacenter GPU with a different compute-to-bandwidth ratio.

## 15. Required Reports and Artifacts

Each experiment release must contain:

- immutable trace files and workload manifests;
- hardware and software manifests;
- solo and pair calibration observations;
- frozen scheduler settings;
- raw per-job release, start, and finish timestamps;
- raw GPU telemetry;
- scheduler decisions and objective breakdowns;
- failures and stderr logs;
- per-policy Gantt charts;
- VRAM-class by compute-class heat maps;
- flow-time versus makespan Pareto plots;
- paired-run scatter plots and confidence intervals; and
- a machine-readable JSON summary plus a Markdown report.

The main report must show per-scenario results before any aggregate. Aggregate
averages can otherwise hide that a policy helps low-VRAM jobs but harms
compute-saturated or near-exclusive jobs.

## 16. Interpretation Rules

1. A flow-time improvement with unchanged makespan supports the SRT claim.
2. A makespan improvement accompanied by concurrency above two supports the
   adaptive-packing claim, not necessarily the ordering claim.
3. Lower OOM/failure rates with similar performance support a safety claim.
4. A warm-profile win and cold-profile loss must be reported as a break-even or
   amortization result.
5. If homogeneous jobs show a large reported SRT gain, inspect cache, run order,
   instrumentation, or worker inequivalence.
6. If pair timings are nonstationary, do not attribute gains to precise
   colocation predictions.
7. If the oracle does not win, do not expect workload reordering to make the
   production scheduler win honestly.
8. If makespan superiority is mandatory but only flow time improves, change the
   scheduler objective or narrow the claim; do not redesign the workload solely
   around a favorable ordering.

## 17. Recommended First Implementation

The first practical experiment should use the RTX 5090 and proceed in this
order:

1. create the 12-job Suite A trace with a 2/6/12-minute duration distribution;
2. run P0, P1, P2, and P3 using fixed batches and ten arrival-order seeds;
3. profile a candidate pool into the nine representative VRAM-compute cells;
4. measure the required pair archetypes and perform oracle screening;
5. build one 24-job held-out mixed trace from qualified strata;
6. run P1, P3, and P4 in warm-profile mode;
7. repeat P4 cold to measure trial overhead; and
8. reproduce the normalized design on at least one materially different GPU.

This sequence first verifies that SRT ordering works, then tests admission at an
equal cap, then tests the complete adaptive system. It avoids spending long GPU
runs on workloads that contain no measurable scheduling opportunity.
