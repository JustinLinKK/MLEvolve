# Nautilus A10 Inference Benchmark Plan

## Goal

Reserve four NVIDIA A10 GPUs as `Nautilus-A10`, then compare the exact
`Qwen3.8-27B-INT8-W8A16-MTP` checkpoint with vLLM and TensorRT-LLM under the
same tensor-parallel and request settings. Retain only the measured faster
backend for local agent serving.

## Fixed test contract

- Deployment: `gpu-dev-a10` in namespace `ecepxie`; four NVIDIA A10 GPUs.
- Model: `/root/qwen38-v100-int8/models/Qwen3.8-27B-INT8-W8A16-MTP`.
- Topology: tensor parallelism 4; no concurrent experiment workload during
  inference-engine comparison.
- Metrics: successful model load, first-token latency, generated tokens per
  second, peak GPU memory, and three post-warmup sequential requests.
- Selection: fastest successful backend at the same 128-output-token test.

## Execution order

1. Submit the four-A10 deployment and wait for a Running pod.
2. Verify GPU model/count and that the exact checkpoint is present on the PVC.
3. Run vLLM, collect JSON, logs, and the required Gantt/metric image.
4. Stop vLLM, build or load TensorRT-LLM for the same checkpoint, and run the
   identical request protocol.
5. Compare results, keep the faster successful server running, and record the
   measured settings and artifacts in `records/`.
