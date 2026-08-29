# Qwen3.8-27B INT8 on two A10 GPUs: serving qualification

## Scope

- Checkpoint: `Qwen3.8-27B-INT8-W8A16-MTP` (unaltered).
- Hardware: two NVIDIA A10 GPUs only (`CUDA_VISIBLE_DEVICES=0,1`); GPUs 2 and
  3 stayed at `0 MiB` throughout all runs.
- Workload: text only; no image request was submitted.
- Server: SGLang 0.5.18, tensor parallelism 2, `max_running_requests=1`.

## Evidence

SGLang loaded the compressed-tensors checkpoint successfully at 14.69 GB per
GPU.  The checkpoint contains 24.327 GB packed INT8 tensors and 7.289 GB BF16
tensors, hence its 31.64 GB on-disk size rather than 27 GB.

The standard hybrid Gated Delta Network (GDN) Triton path on A10 (SM86) then
failed in three independently observed ways:

1. Radix-cache state tracking produced `CUDA error: an illegal memory access`.
2. Disabling the Radix cache allowed a 64-token prefill, but the fused packed
   GDN decode produced no output while both GPUs stayed at 100% utilization.
3. Disabling packed decode and skipping the vision tower with
   `--language-model-only` still produced no first streamed token; the two GPUs
   stayed at 100% utilization for more than two minutes.

The SGLang source documents the available replacement GDN implementations as
FlashInfer for SM90+ and CuTe DSL prefill for SM100+.  Therefore no stable GDN
decode backend is available for this exact checkpoint on A10 SM86.

## Result

No TTFT (time to first token) or tokens-per-second value is recorded: every
real stream produced zero bytes, so a numeric value would be fabricated.  The
only measured successful stage was a 64-token prefill at 2.94 input tokens/s;
it is not generation throughput.

The final failed server process was terminated by exact process identifier;
after teardown, GPUs 0--3 each reported `0 MiB` usage.  No chart is emitted
because this is a failed serving-compatibility qualification, not an inference
experiment with valid runtime metrics.
