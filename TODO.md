# Novelty / Positioning

> Schedule the heterogeneous, LLM-generated training jobs that the search tree produces, where job utility = expected search value per GPU-hour (Value-of-Information), and the scheduler co-designs bidirectionally with the MCGS search controller. This combination is uncovered by all three neighboring fields: HPO/NAS (candidates are a fixed, homogeneous config space), classic GPU schedulers (jobs are externally-arriving black boxes), and agentic MLE systems (no scheduler at all).

# Agent System Plugin
## Findings from Hardware-Awareness Compare Run

- [ ] Disable actual scheduler batch probing in the MLEvolve Agent integration path for now. In the dogs-vs-cats compare run, every generated node had a different model/script structure, so probe cache hits were effectively unavailable.
- [ ] Replace probe-reuse assumptions with a predictive estimator for training time and memory. Inputs should include model/backbone family, framework, dataset size, image resolution, batch size, epochs/folds, AMP/dtype, dataloader settings, and feature-extraction vs fine-tuning style.
- [x] Keep exact probe cache only for identical or near-identical script signatures. Do not expect cross-node reuse when MLEvolve is free to change architecture.
- [ ] Fix scheduler-mode accounting so `initial_drafts=3` does not become 4 root drafts when the scheduler fills its first batch.
- [ ] Parallelize or pipeline node generation, not only node execution. Current same-branch generation is serial and scheduler mode spends substantial wall time waiting for a batch to finish before generating more runnable nodes.
- [ ] Treat batch-probe non-memory failures separately from real memory infeasibility. Script errors and warnings should not be recorded as "no feasible batch size" memory conclusions.
- [x] Verify scheduler actually packs jobs before claiming hardware-aware speedup. The baseline run used exclusive placement with `packed_dispatch_count=0`, so the scheduler added overhead without parallel packing benefit.
- [ ] Make runtime and memory predictors inspect the prepared dataset on disk. Do not rely only on task text, because prepared train/test sizes can differ from the public competition description.

## Agent System Plugin

- Feedback to Agent through MCP

- Try all kinds of agent systems: **MLEvolve**, 

## Hardware Optimization Feedback Loop

- [x] Add a feedback loop from the hardware optimization stage back into the model design stage.
- [ ] Let hardware optimization ask targeted feasibility questions before code is finalized, for example: "We have a faster training option using FP8. Can this model use FP8 without losing accuracy enough to matter?"
- [ ] Ask whether lower-precision options are safe for the proposed model: FP8, BF16, FP16, TF32, AMP, quantized optimizer states, or mixed precision only for selected layers.
- [ ] Ask whether the current model can keep the same architecture while changing execution/data choices: dtype, batch size, gradient accumulation, dataloader workers, pinned memory, channels-last layout, activation checkpointing, feature caching, or preprocessing cache.
- [x] Make the hardware-aware prompt prioritize minimizing training time while maintaining comparable accuracy.
- [ ] Prevent hardware-aware prompting from casually changing the proposed model structure. It should first optimize execution, dtype, data movement, and training parameters around the selected model.
- [ ] Record the answer to each feedback-loop question in node metadata so the scheduler can learn which optimizations were considered, applied, rejected, and why.

## Graph DB and Data accumulation on Cluster

- Test on different hardwares

- [ ] Make graph/result databases optional inputs to a predictor, not the only source of hardware estimates.
- [ ] Add fallback prediction when Neo4j/Postgres are unavailable, using local SQLite plus static model/script features.

## Pack Size Limit Setting

- [ ] Add a conservative pack-size limit when model structure is unknown or prediction confidence is low.
- [ ] Only increase pack size after the predictor or a selective probe indicates enough memory and runtime headroom.

## MLEAgent Optimization

- [ ] Use graph database feedback to estimate whether the job list generated from the current tree frontier can fully utilize available GPUs.
- [ ] If predicted GPU utilization is too low, allow MLEAgent to increase the number of tree nodes explored concurrently so the scheduler has enough runnable jobs to maximize GPU utility.
- [ ] Add safeguards for the expanded exploration count, including max concurrent node limits, per-GPU memory constraints, and rollback when probing shows the expanded job set is inefficient.
- [ ] Generate nodes ahead of execution so the scheduler always has a ready queue, instead of blocking generation on scheduler batch completion.
- [ ] Compare fair node budgets across origin, baseline, and hardware-aware modes: same total node count, same initial draft count, and same branch/debug opportunity.

## Log Database

- combination(tasks) -> Preheat time -> Stay time in GPU -> When out

- tasks: probe method, others ...

# Others

- Agent Generation Time also has to be considered in scheduler

- Parallel Parameter Tune

# Schema pruning: RTX 5000 Ada (objective = reduce training time, keep accuracy)

> File: `schema/hardware_feature_records/nvidia.ada_lovelace.rtx_5000_ada_generation.spec.yaml`

> Entries below never drove an accuracy-safe speedup decision in the simulated flow. Deletion is objective-scoped (training speed + maintained accuracy on Ada cc 8.9). Re-evaluate if objective changes to VRAM reduction or inference deployment.

- `fp8`, `fp8_e4m3`, `fp8_e5m2` — fp8 training degrades accuracy for general models; never selected under the maintain-accuracy constraint

- `int8` — quantization hurts training accuracy; not used

- `fp16` — bf16 strictly dominates on cc>=8.0 (no loss scaling, more stable); fp16 always replaced by bf16

- `tensor_cores` (generic) — subsumed by `tensor_cores_4gen`

- `sm_89` — duplicates `compute_capabilities: 8.9`

- `vram_type: gddr6` — bare label, no decision reads it (cannot derive bandwidth)

- `vram_clock_mhz: 2250` — bare clock, unusable without bus width

- `sm_count: 100` — occupancy auto-managed by CUDA, does not enter training code

- `toolkits: [cuda]` — constant across all NVIDIA records, zero discriminative signal

- `frameworks: [pytorch]` — constant across all records, zero signal

- `transformer_inference`, `vision_inference` (workload_types) — objective is training, inference workloads irrelevant

# Schema additions

## Optimizer-level (accuracy-neutral, PyTorch official)

- `fused optimizer (AdamW fused=True)` — fuses the whole optim step into one CUDA kernel, numerically equivalent, fewer launches. Requires CUDA params, same dtype; can pair with capturable=True for CUDA graphs

- `foreach multi-tensor optimizer (foreach=True)` — packs same-type params into one multi-tensor apply

- `capturable optimizer + CUDA graph` — optim step captured into an existing CUDA graph

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
```

## Data pipeline (accuracy-neutral)

- `non_blocking=True H2D + compute overlap` — overlaps host-to-GPU transfer with compute, beyond plain pin_memory

- `sequence/sample packing (no-pad batching)` — concatenate short sequences into one fixed-length sequence with block-diagonal / varlen attention mask so they do not attend across boundaries; removes pad-token waste, same accuracy, fewer FLOPs

- `length-sorted bucketing` — group similar-length samples per batch to cut padding

## Blackwell-specific precision (sm_120 only; schema currently marks fp8 inference-only)

- `NVFP4 training via TE NVFP4BlockScaling` — NVIDIA blog claims NVFP4 training accuracy near FP16 (block size 16 + e4m3 scale). MUST verify against Transformer Engine docs

- `MXFP8 training via TE MXFP8BlockScaling` — TE README claims mxfp8 training loss curves match bf16. MUST verify

- `2nd-gen Transformer Engine (2x fp8 throughput on 5gen TC)` — verify against Blackwell whitepaper

## Compile / graph refinements (schema only has max-autotune)

- `torch.compile(mode='reduce-overhead')` — automatic CUDA-graph integration, cuts launch overhead

- `dynamic=False / recompile guard` — fix static shape to avoid recompile storms. MLEvolve note below

- `torch.set_float32_matmul_precision('high')` — modern TF32 toggle (newer API than allow_tf32)

- `regional compile (per transformer block)` — compile only the repeated block to cut compile time

## F. Hardware numerics (gap; needs official-doc verification)

- `memory_bandwidth_GBps` — RTX 5090 GDDR7 approx 1790 GB/s (512-bit @ 28 Gbps). Enables memory-bound vs compute-bound detection (whether bf16 helps or the run is dataloader-bound)

- `l2_cache_MB` — Blackwell large L2, informs tiling/fusion benefit

- `bf16_tflops` / `fp8_tflops` (tensor) — peak throughput for roofline / compute-bound headroom

## Excluded (NOT accuracy-safe speedups)

- `gradient checkpointing / activation recompute` — compute-for-memory tradeoff, slower, not a speedup

- `int8 QAT in training` — degrades training accuracy

- `FSDP / DDP / NVLink` — RTX 5090 is PCIe only (no NVLink), single-GPU focus
- Scheduler observability detail
  - [x] Enable Postgres scheduler log DB in config and document `LOCALML_SCHEDULER_LOG_DSN`.
  - [x] Add a planner decision trace event/table before dispatch, including candidate packed groups, objective scores, rejection reasons, selected plan, expected runtime, VRAM budget, and active GPU occupancy.
  - [x] Expand batch probe selected/result logging with full profile metadata: search method, stop reason, failure batch size, average step time, warning message, and profile key.
  - [x] Persist runtime probe summaries with resolved batch size, backend strategy, confidence, and estimated total runtime.
  - [x] Link packed-job dispatch logs to run group members, probe profiles, batch overrides, and final placement reason.
  - [x] Record worker execution details for packed jobs: process command, stdout/stderr log paths, artifact paths, start/end timestamps, exit status, traceback, and runner result.
  - [x] Add read/query examples for reconstructing one scheduler run from startup -> probing -> final packed decision -> packed job execution.

## Model Feature for each GPUs

- [ ] Define model-structure features for prediction: backbone name, parameter count if available, framework, training mode, feature extractor vs end-to-end fine-tune, input shape, batch size, optimizer, epochs, folds, augmentation cost, and precision mode.
- [ ] Build a lightweight predictor from accumulated scheduler runs and batch-size observations.
- [ ] Track predictor confidence and trigger selective actual probing only when confidence is low or the estimated memory margin is unsafe.
- [ ] Design predictor-based scheduler mode. Intentionally undecided: do not add predictor planner behavior, APIs, schema, or config until the mode contract is specified.
