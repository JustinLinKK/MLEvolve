# Qwen3.8-27B INT8 on Two A10 GPUs: Serving Qualification

## Scope

- Checkpoint: `/root/downeyflyfan/qwen38-v100-int8/models/Qwen3.8-27B-INT8-W8A16-MTP`
- GPUs: NVIDIA A10 `0,1` only; GPUs `2,3` were kept unused.
- Requested metrics: time to first token (TTFT) and generation tokens per second (TPS).

## Result

No TTFT/TPS measurement is valid for this hardware constraint. The model did
not reach a successful generated token on exactly two A10 GPUs.

| Backend | Two-GPU layout | Observed failure |
| --- | --- | --- |
| vLLM 0.27.1 | tensor parallelism 2 | Qwen Gated Delta Network (GDN) kernel: CUDA illegal memory access on rank 1 |
| vLLM 0.27.1 | pipeline parallelism 2 | the MTP draft model does not support pipeline parallelism |
| vLLM 0.28.0 | pipeline parallelism 2, eager, local compile caches | same GDN kernel illegal memory access on rank 1 |
| Transformers 5.16.1 | 20/44 and 21/43 manual layer split | compressed-tensors runtime decompression OOM; closest case lacked 340 MiB on GPU 1 |

## Reproducible evidence

All vLLM compilation caches were redirected from Ceph to `/dev/shm`; this
eliminated the initial Ceph metadata stalls but not the GDN kernel fault.
The latest Transformers attempt used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
For the 21/43 split, a single one-token request failed at GPU 1 with a
340 MiB allocation while only 309 MiB was free. Moving another layer to GPU 0
would overflow GPU 0, which was already within about 0.25 GiB of capacity.

## Conclusion

Exactly two 23-GB A10 GPUs cannot serve this checkpoint through the available
backends. A valid local benchmark requires either a third A10 for the
compressed-tensors decompression peak, or a GPU architecture on which vLLM's
GDN kernel is supported. No metric is reported because the service never
produced a token.

## Persistent workload

The independent L40S random BF16 matrix multiplication workload remains an
intentional infinite loop and was not stopped or modified.
