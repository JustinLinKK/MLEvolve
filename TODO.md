# Novelty / Positioning

> Schedule the heterogeneous, LLM-generated training jobs that the search tree produces, where job utility = expected search value per GPU-hour (Value-of-Information), and the scheduler co-designs bidirectionally with the MCGS search controller. This combination is uncovered by all three neighboring fields: HPO/NAS (candidates are a fixed, homogeneous config space), classic GPU schedulers (jobs are externally-arriving black boxes), and agentic MLE systems (no scheduler at all).

# Agent System Plugin

- Feedback to Agent through MCP

- Try all kinds of agent systems: **MLEvolve**, 

# Graph DB and Data accumulation on Cluster

- Test on different hardwares

# Log Database

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
