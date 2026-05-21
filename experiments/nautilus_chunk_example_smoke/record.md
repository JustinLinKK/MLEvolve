# Nautilus chunk-example smoke run -- 2026-05-21

## Goal

- Run every `example_code` in `schema/code_doc_chunks/*.yaml` on a real GPU

- Prove the snippets actually execute, or that their capability-gates trigger the right SKIP path

## Setup

- Cluster: Nautilus, namespace `ecepxie`

- Requested GPU class: Ada or Ampere workstation

- No Blackwell `sm_120` publicly schedulable on Nautilus

- The only two Blackwell nodes `bw6600-01,bw6600-02.ts.fresnostate.edu` carry the `nautilus.io/reservation` taint with `0` allocatable GPU

- Landed on: `NVIDIA RTX 5000 Ada Generation`, compute capability 8.9

- Compute capability 8.9 = Ada Lovelace, sm_89, 4th-generation Tensor Cores, fp8 native, no fp4 native

- Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

- Image contents: torch 2.5.1 + CUDA 12.4 + cuDNN 9, no `transformer_engine` installed

- Resources: 1 GPU, 4 cpu, 16 GiB RAM, 10 GiB ephemeral

- Wall time on cluster: 31 s container runtime + ~50 s image pull

- Files in this directory:

> `build_runner.py` -- embeds 14 chunk example_codes into one script

> `run_chunk_examples.py` -- auto-generated runner (subprocess per chunk, 60 s timeout, JSON output to stdout)

> `pod.yaml` -- Pod spec (nodeAffinity Ada/Ampere, ConfigMap-mounted)

> `pod.log` -- raw kubectl logs

> `results.json` -- parsed JSON block from the runner

## Per-chunk result

| chunk_id | exit | duration | verdict | notes |
|---|---|---|---|---|
| `cuda.graphs.training_loop.001`              | 0 | 2.08s | PASS | warmup + CUDAGraph capture + replay |
| `flash_attn.v3.usage.001`                    | 0 | 1.83s | PASS | bf16 SDPA path |
| `nvidia.ada_lovelace.architecture.001`       | 0 | 1.68s | PASS | asserted cc==(8,9) -- exact arch match |
| `nvidia.fp4.blackwell.001`                   | 1 | 1.58s | EXPECTED_SKIP | capability gate `fp4 unsupported on CC 8.9 (requires Blackwell sm_100 or sm_120)` -- correct behaviour |
| `nvidia.fp8.e4m3_vs_e5m2.001`                | 1 | 0.01s | KNOWN_LIMITATION | `ModuleNotFoundError: transformer_engine`; snippet imports te unconditionally |
| `nvidia.int8.quantization.training.001`      | 0 | 1.81s | PASS | `quantize_dynamic` on Linear runs on CPU |
| `nvidia.tensor_cores.generations.001`        | 0 | 1.58s | PASS | reports `4gen (Ada)` for cc=8.9; archs from torch wheel: sm_50..sm_90 |
| `nvidia.tf32.matmul.policy.001`              | 0 | 1.93s | PASS | tf32 flags set + 256x256 matmul |
| `pytorch.amp.autocast.bf16.training.001`     | 0 | 3.06s | PASS | bf16 autocast forward + backward + SGD step |
| `pytorch.compile.max_autotune.001`           | 0 | 5.58s | PASS | inductor compile path warned but compiled+ran |
| `pytorch.cudnn.benchmark.policy.001`         | 0 | 2.15s | PASS | conv with benchmark=True |
| `pytorch.dataloader.pin_memory_workers.001`  | 0 | 1.84s | PASS | pin_memory=True (cuda available) |
| `pytorch.vram_probe.budget.001`              | 0 | 2.03s | PASS | `reset_peak_memory_stats` + matmul + `max_memory_allocated` |
| `transformer_engine.fp8.inference.001`       | 0 | 1.95s | PASS | uses native torch fp8 dtypes (`torch.float8_e4m3fn`, `torch.float8_e5m2`); no te needed in torch 2.5.1 |

- Score: 12 PASS, 1 EXPECTED_SKIP (fp4 by design), 1 KNOWN_LIMITATION (fp8.e4m3_vs_e5m2 te import)

## Conclusion

- All snippets that target features available on this GPU (compute capability 8.9) ran green

- 9 features pass: cuda_graphs, torch.compile, AMP bf16, cuDNN benchmark, DataLoader, tf32, int8 quant, fp8 native dtypes, vram_probe

- fp4 capability gate works as designed: snippet exits with the right error message when `cc < (10, 0)`

- No further code runs after the gate (no spurious te import); this is the expected SKIP path

- `nvidia.fp8.e4m3_vs_e5m2.001` snippet has a packaging bug: imports `transformer_engine` and assumes `te_model`, `x` exist in scope

- That snippet is runnable only inside the user's own training script

- Fix options: (a) make it self-contained, building a tiny te model when te is installed, else skip; (b) note in the chunk `text` that the snippet is a fragment, not directly runnable

- The fp8 dtype check that actually exercises hardware lives in `transformer_engine.fp8.inference.001` (passed using native torch fp8 dtypes, no te required)

- No Blackwell (sm_120) GPU was reachable; fp4 was therefore only verified by capability-gate, not by running a real MXFP4 GEMM

- Action item: re-run on Blackwell when a 5090, RTX PRO 6000, or B200 becomes schedulable for namespace `ecepxie`

## Reproduce

```bash
# regenerate runner from yamls
.venv/bin/python experiments/nautilus_chunk_example_smoke/build_runner.py

# create ConfigMap + Pod
kubectl create configmap chunk-example-runner \
    --from-file=run_chunk_examples.py \
    -n ecepxie \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f experiments/nautilus_chunk_example_smoke/pod.yaml

# wait + collect
kubectl wait --for=condition=Ready=False pod/chunk-example-smoke \
    -n ecepxie --timeout=600s
kubectl logs chunk-example-smoke -n ecepxie > pod.log
awk '/RESULT_JSON_BEGIN/,/RESULT_JSON_END/' pod.log \
    | sed '1d;$d' > results.json

# cleanup
kubectl delete pod chunk-example-smoke -n ecepxie
kubectl delete configmap chunk-example-runner -n ecepxie
```
