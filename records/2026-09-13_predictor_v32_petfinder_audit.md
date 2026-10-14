# Predictor v3.2 PetFinder audit

## Scope

This audit separates three different operations that were previously easy to
conflate:

1. **Candidate source capture**: construct the candidate model and capture its
   graph.  The PetFinder launcher requests this on the A10 GPU.
2. **Predictor student inference**: predict `train_epoch_ms` and
   `train_peak_vram_mib` from the captured graph.  The v3.2 worker deliberately
   calls the student with `device="cpu"`.
3. **Online runtime probing**: after a real job begins, update the Scheduler's
   total-runtime estimate from measured epoch progress.  This is not a
   Predictor-only estimate.

The evidence below is from the completed 2026-09-12 ten-budget-node,
one-A10 PetFinder run.  It is not a claim about the current rerun.

## Predictor execution placement and latency

| Quantity | Measured / configured value | Interpretation |
| --- | ---: | --- |
| Source-capture device | A10 CUDA | The candidate model and representative inputs are materialized on GPU. |
| Student-inference device | CPU | Explicit in `v32_source_worker.py`: `predict(..., device="cpu")`. |
| Student-only inference benchmark | 50 runs: mean 152.893 ms; median 146.264 ms; p95 185.729 ms | Loaded v3.2 student only; excludes process startup, source loading, and graph capture. |
| Predictor-covered actual epoch time | 5 completed jobs: weighted mean 35.73 s; unweighted mean 36.01 s; median 33.95 s; range 29.22--42.13 s | The measured candidate-training epoch duration matched to Predictor coverage. |
| Student-only cost as a fraction of mean training epoch | 0.42% | `0.152893 / 36.01`; CPU student inference is not the cause of multi-minute or multi-hour gaps. |

The end-to-end per-candidate prediction path was not timestamped separately in
the archived trace.  Therefore no honest mean can be reported for
"spawn + source import + CUDA capture + CPU student inference".  The next
rerun must persist those four components separately.

## Accuracy and coverage

| Statistic | Value | Valid interpretation |
| --- | ---: | --- |
| Submitted jobs with Predictor-model coverage | 10 / 16 (62.5%) | Static Predictor coverage was incomplete; unsupported or unextractable candidate graphs used a safe fallback. |
| Completed jobs with Predictor-model coverage | 5 / 8 (62.5%) | This is the population that can be paired with observed training epochs. |
| Online total-runtime mean absolute percentage error (MAPE), first probe | 81.69% over five completions | Calibration of the first observed-epoch extrapolation, not an accuracy result for the static Predictor alone. |
| Online total-runtime MAPE, latest pre-finish probe | 51.38% | The live estimate improved, but remains unsuitable for a strong accuracy claim. |
| One illustrative completed job | predicted 483.84 s; observed 493.46 s | 2.0% error for this one job only; it must not be generalized to the run. |

There is no persisted table pairing every static `train_epoch_ms` prediction
with its same-candidate actual epoch duration.  Consequently, a global static
epoch-time MAPE cannot be calculated from this trace.  Reporting the online
MAPE as if it were static Predictor accuracy would be incorrect.

## Meaning of `train_epoch_ms`

`train_epoch_ms` is the Predictor's target: predicted wall-clock milliseconds
for **one training epoch of the candidate task**.  It is neither the predictor
model's own training time nor its inference latency.  The Scheduler combines
this estimate with planned epochs and then replaces or refines it using live
epoch measurements once a job is running.

## Seed 42

`agent.seed=42` is the MLEvolve experiment/task seed.  It is applied before
the agent search begins to Python's `random` module and NumPy.  It does not
seed the public PetFinder data itself: that data is fixed.  Candidate scripts
may include their own `SEED = 42` for the train/validation split, model
initialization, and dataloader randomness.

At the time of the archived 2026-09-12 trace, `agent.seed` was **not** sent to
the local Qwen/vLLM code agent.  The current source now forwards that same
non-negative seed as the OpenAI-compatible vLLM request `seed` field for both
streaming and non-streaming local-Qwen calls.  An already-running controller
keeps its imported client module, so that change applies only to a fresh
controller.  PyTorch CPU/CUDA and dataloader seeding remain the responsibility
of the generated candidate program and must be recorded per node.

## Current local code-agent service check

On 2026-09-14 UTC, the idle A100 service was restarted with the existing
`bootstrap_qwen38_a100.sh` launcher before the rerun.  The service is
text-only Qwen3.8-27B-INT8-W8A16-MTP with the following verified runtime
configuration: one A100 80 GB PCIe, vLLM 0.27.1, Marlin W8A16 kernels,
FlashAttention 2, MTP with one speculative token, prefix caching, 262144-token
context, and vLLM compilation with full/piecewise CUDA graphs.  A warmup was
discarded before the measurement.

| Streaming measurement | Value |
| --- | ---: |
| Time to first token | 0.2642 s |
| Completion tokens | 125 |
| Generation interval after first token | 1.7683 s |
| Generation throughput | 70.69 tokens/s |
| End-to-end request time | 2.0325 s |

This is a fixed single-request service measurement, not a claim about
multi-request aggregate throughput or MLEvolve's end-to-end node latency.
