# PetFinder: A100 Agent and Single-A10 Scheduler Comparison

## Status

Running. The A100 agent deployment and serving benchmark are complete. The
fresh same-agent comparison is in the original-MLEvolve baseline phase; the
profile-based scheduler plus Hardware Knowledge Database phase will start only
after the baseline retains exactly 50 nodes.

## A100 agent deployment

- GPU: one NVIDIA A100-SXM4-80GB.
- Model: text-only `Qwen3.8-27B-INT8-W8A16-MTP`.
- Runtime: vLLM 0.27.1, compressed-tensors INT8 weights, Marlin linear kernel,
  FlashAttention 2, 32,768-token maximum context.
- Served name: `qwen3.8-27b-int8-a100`.
- Kubernetes service: `mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000`.
- Health evidence: HTTP 200 from both the A100 Pod and the A10 experiment Pod.
- Resident GPU memory after initialization: approximately 73,473 MiB.
- The owned `a100_goal_filler` process was terminated before model startup;
  the A100 Pod and persistent storage were not deleted or recreated.

## Serving benchmark

The benchmark used one warm-up request followed by three sequential streaming
requests. Each measured response contained 39 completion tokens.

| Metric | Median | Individual measurements |
| --- | ---: | --- |
| Time to First Token | 0.19697 s | 0.19807, 0.19697, 0.19665 s |
| Generation throughput | 13.2088 tokens/s | 13.2214, 13.1844, 13.2088 tokens/s |
| End-to-end request time | 3.14923 s | 3.14784, 3.15501, 3.14923 s |

Raw benchmark JSON is retained in
`.cache/qwen38_a100_benchmark_20260831/result.json`. The required combined Gantt
and metric-node image is
`records/2026-08-31_qwen38_a100_vllm_benchmark.png`.

## Controlled PetFinder comparison

- Execution GPU: exactly one NVIDIA A10, exposed as `CUDA_VISIBLE_DEVICES=0`
  inside `gpu-dev-a10-experiment-746c959459-2xwdd`.
- Task: PetFinder Pawpularity Score.
- Both phases use the same A100 Agent endpoint, model, seed 42, and 50 retained
  nodes.
- Baseline phase disables scheduler, preflight, stage review, hardware context,
  pipeline decisions, and Hardware Knowledge Database.
- Modified phase enables hardware-aware mode, profile-based runtime prediction,
  preflight, scheduler, and Hardware Knowledge Database.
- `agent.search.parallel_search_num=null` in both phases. The modified phase
  also sets `parallel_job_cap=null`; live admission owns concurrency.
- Stage-review or preflight rejections that avoid GPU execution are discarded
  and do not count toward the 50-node target.
- The sequence controller validates the retained journal count and refuses to
  advance after an early exit with fewer than 50 nodes.

Active comparison root:

`/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_comparison_20260831_190026`

The superseded L40S diagnostic run was stopped without deleting its journal or
artifacts. It is excluded from the final same-agent comparison.

