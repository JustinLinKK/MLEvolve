# V100 Local-Inference Engine Design

## Goal

Replace single-request Transformers `device_map="balanced"` inference with an
engine that performs real tensor parallelism on the V100 NVLink fabric and
measures Time to First Token and decode tokens per second.

## Evidence

- Current Qwen3.8-27B 8-bit integer quantized Transformers service measures
  1.744 decode tokens per second on three V100 GPUs.
- GPU 0 and GPU 1 share one NVLink bundle; GPU 2 has one NVLink to GPU 0 and
  two NVLinks to GPU 1; GPU 3 crosses the system interconnect to GPU 1/2.
- Generic `device_map="balanced"` assigns contiguous model portions and is not
  tensor parallelism.
- V100 is compute capability 7.0. 8-bit W8A8 acceleration paths requiring
  compute capability 7.5 or higher cannot be used.

## Chosen sequence

1. Probe current vLLM source installation against Python 3.12, PyTorch 2.6.0
   CUDA 11.8, and V100 compute capability 7.0. Do not replace the live service.
2. If it starts, benchmark an FP16 vLLM service with tensor parallel size 2 on
   GPU 0 and GPU 1, `max-model-len=4096`, 30 GiB maximum per V100, CUDA Graphs,
   continuous batching, and automatic prefix caching. GPU 2 and GPU 3 remain
   free for jobs.
3. If vLLM cannot run on V100, probe a pinned llama.cpp CUDA build and a Qwen
   3.8 GGUF conversion. Reject its current tensor-parallel mode if it hits the
   known Qwen3.8 Volta lockup or CPU sampling regression.
4. Add speculative decoding only after a tensor-parallel baseline succeeds;
   it consumes a free GPU and is a separate measurement.

## Non-goals

- Do not use W8A16/Marlin, FlashAttention-2, FP8, TensorRT-LLM, or
  bitsandbytes acceleration paths requiring compute capability above 7.0.
- Do not configure a scheduler Max Parallel Jobs cap.
- Do not stop the current three-V100 service until a replacement passes health
  and measurement checks.

## Success criteria

- Engine health endpoint reports live model and configured tensor parallelism.
- Each selected V100 stays below 31 GiB.
- GPU 3 remains free during the two-GPU tensor-parallel run.
- Persist raw results and one image with Gantt above and metric-node graph
  below in `records`.
