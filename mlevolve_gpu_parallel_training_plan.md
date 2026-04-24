# Parallel GPU Training Plan for an MLEvolve-Style Scheduler on a Local Linux Workstation

## Scope

This document lays out a practical implementation plan for adding **parallel GPU training / co-running** to your existing scheduler on a **single Linux workstation** first, then scaling the same design to larger NVIDIA platforms later.

**Target local validation box**

- CPU host: AMD Ryzen 9 9950X3D
- System RAM: 96 GB
- GPU: RTX 5090 (32 GB GDDR7)
- OS: Linux
- Framework: PyTorch
- Existing components already present:
  - RAM preload for baseline models
  - simple single-GPU queueing scheduler
  - MLEvolve-style agent loop that generates multiple candidate jobs

**Primary goal**

Maximize **effective GPU efficiency** by improving both:

1. **GPU memory utilization** — avoid leaving large VRAM slack when two compatible jobs can fit safely.
2. **GPU compute / SM utilization** — avoid running only one under-saturating training job when a second compatible job can overlap usefully.

**Non-goal for v1**

Do **not** attempt to solve full cluster scheduling, multi-node distributed training, or arbitrary many-way packing. The first production target should be:

- **pairwise packing only**
- **one packed pair at a time on one physical GPU**
- **fallback to exclusive execution whenever risk rises**

That is the highest-value, lowest-risk path for a workstation deployment.

---

## 1. What MLEvolve Already Gives You

The current MLEvolve codebase already has a useful execution skeleton that you should preserve rather than replace wholesale.

### 1.1 Current execution behavior

From the public repo:

- `run.py` creates a `ThreadPoolExecutor` sized from `interpreter.max_parallel_run`, generates initial drafts, then pipelines parallel execution tasks through the executor.
- `config/config.yaml` exposes `agent.search.parallel_search_num: 3` and `num_gpus: 1`.
- `engine/executor.py` already launches **multiple Python subprocesses**, pins CPU affinity per slot, and isolates some file outputs such as submission/model filenames.
- `run_single_task.sh` sets a single `CUDA_VISIBLE_DEVICES=$MEMORY_INDEX` and launches `python run.py`.

### 1.2 Why that matters

This means the **lowest-friction integration point** is **not** in the agent planner itself. The best integration point is around the **execution backend**:

- keep the agent search loop
- keep subprocess-based execution
- replace “launch every ready subprocess immediately” with:
  - **profile**
  - **admit**
  - **pack or isolate**
  - **monitor**
  - **fallback**

### 1.3 Key implication

You do **not** need to redesign MLEvolve’s graph search first.

You should introduce a **GPU-aware execution layer** underneath the existing `Interpreter` / subprocess model.

---

## 2. Core Recommendation

## Recommendation in one sentence

Build **pairwise empirical GPU packing** on top of the existing subprocess execution model, using **process-per-job + NVIDIA MPS** as the first backend, and use a **small family-specific compatibility matrix** rather than a heavy general predictor.

### Why this is the right first version

Because your queue is not an arbitrary cluster-scale workload. It is a small, repeated set of:

- **3–4 model families**
- each with **hyperparameter variants**
- repeatedly produced by agents in an MLEvolve-like loop

That is exactly the setting where a **lightweight empirical packer** beats a complicated general scheduler.

You do **not** need Horus-style full heterogeneity prediction on day 1.
You need:

1. **solo profiles** per family / batch / precision regime
2. **pairwise slowdown measurements**
3. a small **compatibility table**
4. **MPS-based execution**
5. **hard safety fallbacks**

---

## 3. Design Principles

### 3.1 Pairwise first, not 3-job packing

On a 32 GB RTX 5090, three-way packing is possible only for very small jobs, but the interference risk climbs quickly.
Your initial implementation should support:

- **exclusive mode**: 1 job
- **packed mode**: 2 jobs
- no 3-job mode in the first stable release

Treat 3-job packing as an offline experiment later.

### 3.2 Structured GPU jobs beat arbitrary Python blobs

The biggest engineering decision is this:

> Do not try to make arbitrary generated Python scripts the primary unit of packed execution.

Instead, introduce a **canonical training runner** for GPU-heavy jobs.

#### Best pattern

Let agents produce:

- model family
- dataset/task id
- hyperparameters
- optional small patch / training recipe diff

Then dispatch through a shared runner like:

```bash
python trainer_entry.py --job-spec /path/job.json
```

This gives you:

- consistent checkpointing
- consistent telemetry hooks
- reproducible model preload
- consistent profiling
- safe pause/resume later
- easier packing decisions

#### If you cannot refactor that much immediately

Then support two paths:

- **structured_runner** for GPU-heavy fine-tuning / training
- **legacy_script_runner** for arbitrary code

Only the structured runner should be eligible for aggressive packing in v1.
Legacy scripts can remain mostly exclusive unless they pass profiling gates.

---

## 4. Proposed Architecture

```text
Agents / search tree
    ↓
Job normalization layer
    ↓
Scheduler core
    ├─ solo profiler
    ├─ compatibility database
    ├─ pairwise packing policy
    ├─ safety / fallback controller
    ↓
Execution backend manager
    ├─ ExclusiveBackend
    ├─ MPSBackend   ← default packed backend
    └─ StreamBackend (experimental, later)
    ↓
Per-job training subprocesses
    ↓
Telemetry + profile store + event log
```

### Main new components

#### A. `TrainJobSpec`
Normalized representation of one trainable workload.

Fields:

- `job_id`
- `agent_id`
- `workflow_id`
- `model_family`
- `baseline_model_id`
- `dataset_id`
- `task_type`
- `precision_mode` (`fp32`, `fp16`, `bf16`, `amp`)
- `batch_size`
- `seq_len` or `image_resolution`
- `optimizer_family`
- `lora_or_full_ft`
- `num_workers`
- `estimated_steps`
- `priority`
- `runner_type` (`structured_runner`, `legacy_script_runner`)
- `entrypoint`
- `env`
- `checkpoint_policy`
- `metadata`

#### B. `SoloProfile`
Telemetry from isolated probing.

Fields:

- `peak_reserved_gib`
- `peak_allocated_gib`
- `steady_sm_active`
- `steady_sm_occupancy`
- `steady_dram_active`
- `tensor_active`
- `pcie_rx_mb_s`
- `pcie_tx_mb_s`
- `host_wait_ratio`
- `kernel_wait_ratio`
- `throughput_solo`
- `step_time_mean_ms`
- `step_time_p95_ms`
- `power_mean_w`
- `oom_seen`
- `profile_confidence`

#### C. `PairProfile`
Measured behavior for two jobs co-running.

Fields:

- `job_sig_a`
- `job_sig_b`
- `peak_reserved_pair_gib`
- `throughput_a_packed`
- `throughput_b_packed`
- `slowdown_a`
- `slowdown_b`
- `aggregate_gain`
- `oom_seen`
- `fallback_triggered`
- `notes`

#### D. `GpuPackingPolicy`
Determines:

- exclusive vs packed
- which two jobs to pair
- initial MPS resource knobs
- when to stop packing and fall back

#### E. `GpuExecutionBackend`
Launches actual subprocesses.

Backends:

- `ExclusiveBackend`
- `MPSBackend`
- later `StreamBackend`

---

## 5. The Fastest Viable Implementation Path

## Phase 0 — Freeze the execution contract

Before packing, standardize the training job interface.

### Deliverables

1. `trainer_entry.py`
2. `job_spec.json` schema
3. `profile_store.sqlite`
4. `event_log.sqlite`

### Requirement

Every pack-eligible job must be launchable through:

```bash
python trainer_entry.py --job-spec job.json
```

Even if the agent still generates code, wrap the code in a standard runner envelope so the scheduler can:

- set env vars
- attach telemetry
- identify model family
- record step metrics
- optionally checkpoint
- classify failures

---

## Phase 1 — Add solo profiling

The scheduler should not pack any job until it has seen a solo profile.

### 5.1 Probe strategy

For each new job signature:

1. **Warmup**
   - 20–50 steps or enough to stabilize CUDA context, kernels, allocator
2. **Solo probe**
   - 50–200 steps or 60–120 seconds
3. Persist results in profile DB

### 5.2 Metrics to collect

#### From PyTorch / job process
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`
- `torch.cuda.memory_snapshot()` or `_dump_snapshot()` during canary runs
- moving average step time
- moving average throughput
- optional trainer-level batch latency histogram

#### From device telemetry
Use **DCGM** if available, else fallback to `nvidia-smi`.

Collect:
- `PROF_SM_ACTIVE`
- `PROF_SM_OCCUPANCY`
- `PROF_DRAM_ACTIVE`
- tensor activity if exposed
- PCIe RX/TX
- device memory used
- utilization.gpu
- utilization.memory

#### From PyTorch profiler / HTA on selected canaries
Not necessarily every job.
Use on a few representative runs per family to estimate:
- host wait
- kernel wait
- queue length
- H2D bandwidth / memcpy patterns

### 5.3 Signature design

Because you only have a few repeated model families, bucket aggressively.

Recommended job signature key:

```text
(
  model_family,
  precision_mode,
  optimizer_family,
  batch_size_bin,
  seq_or_res_bin,
  finetune_mode   # lora/full/adapters
)
```

Example:
```text
("llama3-8b-lora", "bf16", "adamw", "bs16", "seq4k", "lora")
```

This keeps the profile table small and reusable.

---

## Phase 2 — Build the pairwise eligibility model

This is the heart of the scheduler.

## 6. Pairwise packing policy for the RTX 5090

The RTX 5090 has **32 GB GDDR7**.
For the first stable version, do **not** admit pairs against the full 32 GB.

### 6.1 Memory rule

Use a conservative **safe packing budget** first:

```text
safe_vram_budget_gib = 28.0
```

That is a starting engineering heuristic, not a hardware law.
It gives you room for:

- allocator fragmentation
- CUDA context overhead
- cuDNN / fused-kernel workspace fluctuations
- metadata / temporary tensors
- minor job-to-job drift

Then use:

```text
pack_memory_ok(i, j) :=
    peak_reserved_gib(i) + peak_reserved_gib(j) <= safe_vram_budget_gib
```

#### Later refinement

If you build iteration-aware runner hooks, you can split memory into:

- persistent
- ephemeral

and move toward a Salus-like admission rule later.
But for v1 on a workstation, **peak reserved memory** is the simplest robust gate.

### 6.2 Compute / bandwidth rule

Use solo profiles to classify jobs.

#### Good packing candidates
- `sm_active < 0.5`
- and not strongly DRAM bound
- or significant `host_wait_ratio`
- or short bursty kernels / intermittent GPU usage

#### Likely bad packing candidates
- `sm_active >= 0.8`
- plus high DRAM activity
- little host wait
- already near stable high throughput alone

### 6.3 CPU / input pipeline rule

On your 9950X3D workstation, GPU packing can fail because of CPU or loader contention even when VRAM fits.

Reject pairing if:

- total `num_workers` across the pair is too high
- total CPU thread caps exceed the host budget
- one or both jobs are data-loader bottlenecked and both need high host bandwidth simultaneously

### 6.4 Initial acceptance rule

For throughput-oriented background jobs:

```text
accept pair if:
  memory gate passes
  and no hard reject signal
  and pair_probe shows:
      max(slowdown_a, slowdown_b) <= 1.30
  and aggregate_gain > 1.10
```

For latency-sensitive / priority jobs:

```text
accept pair if:
  max(slowdown_a, slowdown_b) <= 1.15
```

### 6.5 Aggregate gain metric

Use:

```text
aggregate_gain = throughput_a_packed / throughput_a_solo
               + throughput_b_packed / throughput_b_solo
```

Interpretation:

- `aggregate_gain > 1.0` means the packed pair beats serialized execution in normalized throughput terms
- require `> 1.10` initially to justify risk

### 6.6 Pairwise compatibility matrix

Because you only have 3–4 families, maintain a small compatibility table.

Example conceptual table:

| A \ B | Family-1 small | Family-1 medium | Family-2 small | Family-3 small |
|---|---:|---:|---:|---:|
| Family-1 small | allow | maybe | allow | allow |
| Family-1 medium | maybe | no | maybe | maybe |
| Family-2 small | allow | maybe | maybe | allow |
| Family-3 small | allow | maybe | allow | maybe |

Store this empirically, not manually.

---

## 7. Execution Backend: Start with MPS

## 7.1 Why MPS should be your first packed backend

For your Linux workstation, **process-per-job + NVIDIA MPS** is the best first implementation because:

- MLEvolve already runs jobs as subprocesses
- MPS is designed for overlapping kernels and memcopies from different processes
- it reduces avoidable context switching / context storage overhead
- it requires far less refactoring than a one-process multi-stream design

### Use MPS as the default packed backend

**Exclusive mode** remains the fallback and default for unprofiled or risky jobs.

## 7.2 MPS operating model

### Backend modes

- `exclusive`: existing subprocess launch, one job only
- `mps_pair`: two subprocesses under a single MPS server

### Startup sequence

1. Set GPU compute mode:
```bash
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
```

2. Start MPS daemon:
```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

3. Scheduler launches packed jobs as ordinary subprocesses with:
- `CUDA_VISIBLE_DEVICES=0`
- optional `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
- thread caps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.)

### 7.3 First recommended resource split

Do **not** over-optimize MPS partitioning on day 1.

Use:
- primary job: 60%
- secondary job: 40%

only if:
- the primary job is higher priority
- both jobs are known compatible
- the driver + card behave well under MPS

Otherwise use symmetric:
- 50 / 50

### 7.4 Important MPS caveats

- Linux only
- telemetry attribution in `nvidia-smi` is to the MPS server, not cleanly per client
- `/dev/shm` matters because page-locked host memory limits can bite
- client early termination can destabilize the MPS server if not handled carefully

So the scheduler must treat MPS as a **managed runtime**, not a fire-and-forget toggle.

---

## 8. Backend API You Should Implement

```python
class GpuExecutionBackend(Protocol):
    def prepare(self, device_id: int) -> None: ...
    def launch(self, job: TrainJobSpec, lease: "GpuLease") -> "RunningJob": ...
    def stop(self, running_job: "RunningJob", reason: str) -> None: ...
    def cleanup(self) -> None: ...
```

### Implement these concrete backends

#### 8.1 `ExclusiveBackend`
- wraps current `subprocess.Popen` path
- exactly one running GPU job
- baseline and fallback backend

#### 8.2 `MPSBackend`
- manages MPS daemon lifecycle
- manages process env vars
- manages active thread percentages
- manages pair launch / pair teardown
- knows how to fall back when one client is killed or demoted

#### 8.3 `StreamBackend` (later)
Only for controlled structured jobs where both trainers live in one process.
Not a v1 priority.

---

## 9. The Most Important Refactor in MLEvolve

## Refactor `Interpreter` into a GPU-aware launcher

Current MLEvolve `engine/executor.py` already does:

- slot assignment
- CPU pinning
- subprocess launch
- output/model filename isolation
- timeout handling

That is exactly where GPU packing should hook in.

### Recommended refactor

Split current logic into:

- `SlotManager` — CPU slots, process ids, file isolation
- `GpuLauncher` — exclusive vs packed backend
- `GpuSchedulerClient` — asks central scheduler for a lease before launch

### New flow

```text
run.py
  -> ThreadPoolExecutor submits candidate execution task
  -> Interpreter asks GpuScheduler for launch lease
  -> GpuScheduler returns:
       - exclusive lease
       - or packed lease with partner + MPS config
  -> Interpreter launches subprocess with backend-specific env
```

This keeps the search loop mostly untouched.

---

## 10. Scheduling Logic

## 10.1 Do not pack “every pair that fits”

Decision order should be:

1. job priority
2. solo profile existence
3. memory gate
4. hard reject conditions
5. expected gain
6. pair probe
7. online monitor
8. fallback if breached

## 10.2 Recommended runtime policy

### If GPU idle
- choose highest-priority pending job
- if no solo profile: run solo probe / exclusive
- else:
  - scan top-K pending jobs for best partner
  - if partner score passes threshold, launch packed pair
  - else launch exclusive

### If one job already running exclusively
- optionally admit one secondary job only if:
  - running job is packable
  - incoming job is packable
  - pair score high enough
  - priority rules allow it

### If pair running
- monitor both
- if memory or slowdown trigger fires:
  - demote / kill / pause secondary
  - continue primary exclusively

---

## 11. Recommended Job Classification for Your Use Case

Because your workloads are repeated hyperparameter variants of a few model families, create three classes:

### 11.1 `solo_only`
Examples:
- full fine-tunes
- high batch size
- high sequence length / resolution
- jobs already near high SM and DRAM use

### 11.2 `pairable`
Examples:
- LoRA / adapter fine-tunes
- small / medium batch sweeps
- jobs with measurable host wait or intermittent GPU activity
- smaller family variants

### 11.3 `probe_required`
Examples:
- unclassified new family
- large config jump
- changed precision / optimizer
- uncertain memory footprint

This lets your scheduler behave conservatively without being static forever.

---

## 12. Concrete Local Defaults for the 9950X3D + 96 GB + RTX 5090

These are **recommended starting values**, not final truths.

### 12.1 Scheduler defaults

```yaml
gpu_scheduler:
  enabled: true
  backend_priority: ["mps", "exclusive"]
  max_packed_jobs_per_gpu: 2
  allow_three_way_packing: false

  profiling:
    warmup_steps: 30
    solo_probe_steps: 80
    pair_probe_steps: 60
    reuse_profile_if_confidence_ge: 0.8

  memory:
    safe_vram_budget_gib: 28.0
    hard_stop_memory_fraction: 0.90

  thresholds:
    pack_prefer_sm_active_lt: 0.50
    pack_reject_sm_active_ge: 0.80
    pack_reject_max_slowdown: 1.30
    latency_sensitive_max_slowdown: 1.15
    min_aggregate_gain: 1.10

  telemetry:
    device_poll_ms: 500
    pair_recheck_every_steps: 20

  mps:
    enabled: true
    compute_mode: EXCLUSIVE_PROCESS
    default_primary_active_thread_pct: 60
    default_secondary_active_thread_pct: 40
```

### 12.2 CPU defaults

For **two packed training jobs** on the 9950X3D host:

- `OMP_NUM_THREADS=6`
- `MKL_NUM_THREADS=6`
- `torch.set_num_threads(6)`
- dataloader workers per job: start with `2` or `4`, not `8+`
- reserve host capacity for:
  - scheduler
  - logging
  - data staging
  - OS noise

### 12.3 Precision defaults

For packable jobs:
- prefer `bf16` or `fp16` / AMP if numerically acceptable
- store precision mode in the profile key
- do not assume fp32 and bf16 pack the same

---

## 13. Instrumentation You Should Add Immediately

## 13.1 Minimal device telemetry loop

Sample every 500 ms or 1 s:

- `memory.used`
- `memory.total`
- `utilization.gpu`
- `utilization.memory`

Prefer DCGM when available for:

- `sm_active`
- `sm_occupancy`
- `dram_active`
- `pcie_tx_bytes`
- `pcie_rx_bytes`

## 13.2 Minimal per-job counters

Each training process should periodically emit:

- step count
- moving average throughput
- latest checkpoint path
- current phase (`load`, `train`, `eval`, `save`)
- last successful progress timestamp

### Why this matters under MPS

Because system tools won’t give you clean per-client attribution once jobs run under the MPS server.

---

## 14. Safety and Fallback Rules

You need these from day 1.

## 14.1 Hard stop triggers

Stop the secondary packed job immediately if any of these occurs:

- CUDA OOM
- repeated allocator growth toward the hard cap
- device memory > 90% for sustained interval
- pair slowdown threshold breached for multiple windows
- no progress heartbeat from one client
- MPS server fault / unhealthy client state

## 14.2 Degradation policy

Order of actions:

1. stop secondary
2. keep primary alive
3. mark pair as temporarily incompatible
4. cool down pair for N minutes / jobs
5. retry only after fresh solo profile or config change

## 14.3 Failure bookkeeping

Every failed packing attempt should persist:
- pair signature
- backend
- active thread split
- memory at failure
- exception type
- whether fallback succeeded

This will save you enormous debugging time.

---

## 15. Evaluation Plan for the Workstation

Your local box is enough for a strong first validation.

## 15.1 Benchmark set

Construct a matrix from your 3–4 job families.

For each family, define 2–3 bins:
- small
- medium
- large

based on:
- batch size
- sequence length or resolution
- precision
- LoRA/full

That gives maybe 8–12 representative job signatures.

## 15.2 Measure

### Solo
- throughput
- step time
- max reserved memory
- sm_active
- dram_active
- host wait on canaries

### Pairwise
Measure every pair:
- packed throughput per job
- slowdown per job
- aggregate gain
- OOM / failure count

Build:
- pairwise slowdown heatmap
- pairwise gain heatmap
- allowed / maybe / reject matrix

## 15.3 Compare policies

Compare at least:

1. **serialized**
2. **naive fit-by-memory only**
3. **your empirical packer**
4. **your empirical packer + MPS thread split**
5. **fallback enabled vs disabled**

## 15.4 Report

Metrics:
- makespan
- average JCT
- tail JCT
- aggregate throughput
- GPU active time
- effective VRAM utilization
- OOM rate
- fallback rate

---

## 16. Scale Path to Blackwell Data Center GPUs (Assuming You Mean B200)

I assume by “BE200” you mean **Blackwell B200** or a similar Blackwell-class data center target.

### Keep these abstractions portable

Do **not** hardcode:
- `32 GB`
- `1 GPU`
- GeForce-only assumptions
- consumer-only telemetry

### Instead query at runtime:
- total device memory
- compute capability
- MPS availability
- number of GPUs
- NVLink / topology if multi-GPU later
- MIG availability if future backend uses it

### What should stay the same
- `TrainJobSpec`
- `SoloProfile`
- `PairProfile`
- compatibility DB
- pack / reject / fallback logic
- execution backend interface

### What changes on B200-class systems
- much larger device memory budget
- different optimal batch-size bins
- possibly more aggressive packing
- multi-GPU placement policy later
- optional future datacenter-only backends

The point is: **make capacity and backend pluggable, not the scheduling logic.**

---

## 17. Recommended Implementation Order

### Milestone 1 — “Safe visibility”
- normalized job schema
- profile store
- telemetry loop
- solo profiling
- no packed execution yet

### Milestone 2 — “Controlled pair packing”
- pairwise matrix
- exclusive backend + MPS backend
- packed pair launch
- hard fallback
- local workstation experiments

### Milestone 3 — “Policy quality”
- automatic partner selection
- priority-aware packing
- family/bin generalization
- better slowdown prediction

### Milestone 4 — “Future systems work”
- checkpoint-aware demotion
- iteration-aware interleaving
- stream backend for structured runner
- datacenter backend variants

---

## 18. Recommended File / Module Layout

```text
scheduler/
  job_spec.py
  profile_store.py
  telemetry.py
  packing_policy.py
  compatibility.py
  gpu_scheduler.py
  fallbacks.py

backends/
  base.py
  exclusive_backend.py
  mps_backend.py
  stream_backend.py   # later

profiling/
  solo_probe.py
  pair_probe.py
  trace_tools.py

runners/
  trainer_entry.py
  runner_hooks.py
  checkpointing.py

mlevolve_integration/
  gpu_aware_interpreter.py
  executor_patch.py
```

### Existing repo files to modify first

- `engine/executor.py`
- `run.py`
- `config/config.yaml`
- `run_single_task.sh` (only lightly, if needed)

---

## 19. Pseudocode: Packing Decision

```python
def choose_launch_mode(pending_jobs, running_jobs, profiles, compatibility_db, device_caps):
    # single GPU v1:
    # allow either 1 exclusive job or 1 packed pair

    top = pick_highest_priority(pending_jobs)

    if top.signature not in profiles:
        return Exclusive(top, reason="needs solo profile")

    candidates = []
    for other in pending_jobs:
        if other.job_id == top.job_id:
            continue
        if other.signature not in profiles:
            continue

        if not memory_gate(top, other, profiles, device_caps):
            continue
        if hard_reject(top, other, profiles):
            continue

        score = compatibility_score(top, other, profiles, compatibility_db)
        candidates.append((score, other))

    if not candidates:
        return Exclusive(top, reason="no compatible partner")

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, partner = candidates[0]

    if best_score < PACK_SCORE_THRESHOLD:
        return Exclusive(top, reason="partner score too low")

    return PackedPair(top, partner, backend="mps")
```

---

## 20. Pseudocode: MPS Pair Launch

```python
def launch_packed_pair(job_a, job_b):
    ensure_mps_server_running(device_id=0)
    set_gpu_compute_mode(device_id=0, mode="EXCLUSIVE_PROCESS")

    env_a = base_env()
    env_b = base_env()

    env_a["CUDA_VISIBLE_DEVICES"] = "0"
    env_b["CUDA_VISIBLE_DEVICES"] = "0"

    env_a["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "60"
    env_b["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "40"

    env_a["OMP_NUM_THREADS"] = "6"
    env_b["OMP_NUM_THREADS"] = "6"
    env_a["MKL_NUM_THREADS"] = "6"
    env_b["MKL_NUM_THREADS"] = "6"

    proc_a = launch_subprocess(job_a.entrypoint, env=env_a)
    proc_b = launch_subprocess(job_b.entrypoint, env=env_b)

    return proc_a, proc_b
```

---

## 21. What I Would Build First If I Were Implementing This

If you want the shortest path to a working local prototype:

### Week 1 target
1. add `TrainJobSpec`
2. add solo profiler
3. add SQLite profile DB
4. refactor `engine/executor.py` into a pluggable backend launcher
5. keep everything exclusive

### Week 2 target
1. implement MPS daemon manager
2. allow 2-job pair launch
3. add pairwise probe mode
4. add kill-switch fallback
5. run 8–12 pair experiments locally

### Week 3 target
1. materialize compatibility matrix
2. auto-select best partner
3. compare against serialized baseline
4. tune safe VRAM budget and slowdown thresholds

That is the fastest route to something real and publishable.

---

## 22. Bottom-Line Recommendation

### Build this first

**Structured runner + solo profiles + pairwise empirical compatibility + MPS backend + fallback controller**

### Do not build this first

- 3-job packing
- fully arbitrary script packing
- stream-based co-training of unrelated code blobs
- cluster-wide scheduling
- fancy learned predictors
- Salus-style iteration scheduler

### Why

Because the workstation prototype you want is fundamentally a **systems engineering problem with repeated workload families**, not a generic cluster scheduling problem.

The best v1 is the one that:
- works reliably on your RTX 5090
- integrates cleanly with MLEvolve’s current subprocess model
- produces pairwise compatibility data
- is easy to port to B200 / larger NVIDIA systems later

---

## Sources / Reading

### MLEvolve code
- GitHub repo: https://github.com/InternScience/MLEvolve
- `run.py`
- `engine/executor.py`
- `config/config.yaml`
- `run_single_task.sh`

### NVIDIA docs
- MPS overview: https://docs.nvidia.com/deploy/mps/index.html
- When to use MPS: https://docs.nvidia.com/deploy/mps/when-to-use-mps.html
- MPS tools and interface reference: https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html
- nvidia-smi docs: https://docs.nvidia.com/deploy/nvidia-smi/
- DCGM feature overview: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html
- RTX 5090 official specs: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- DGX B200 overview: https://www.nvidia.com/en-us/data-center/dgx-b200/

### PyTorch docs
- CUDA memory usage: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
- `torch.cuda.memory_snapshot`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_snapshot.html
- multiprocessing best practices: https://docs.pytorch.org/docs/2.8/notes/multiprocessing.html
- Holistic Trace Analysis tutorial: https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html
- `torch.cuda.Stream`: https://docs.pytorch.org/docs/main/generated/torch.cuda.Stream_class.html

### Papers
- Salus (MLSys 2020): https://proceedings.mlsys.org/paper_files/paper/2020/file/d9cd83bc91b8c36a0c7c0fcca59228f2-Paper.pdf
- Gandiva (OSDI 2018): https://www.usenix.org/system/files/osdi18-xiao.pdf
- Horus (TPDS 2022): https://eprints.whiterose.ac.uk/173971/
- Muri (SIGCOMM 2022): https://xinjin.github.io/files/SIGCOMM22_Muri.pdf
