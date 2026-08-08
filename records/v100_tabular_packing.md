# V100 Tabular Packing Sweep

## Goal

- Find MLEBench-Lite tasks whose agent solutions let 4-5 concurrent jobs fill V100 compute while staying under the scheduler's 1.15x slowdown gate

## Motivation

- The CNN sweep in [v100_packing_results.md](v100_packing_results.md) covered only image tasks, where every model reached 100% SM solo and packing failed

- MLEBench-Lite also contains tabular and text tasks whose agent solutions are MLPs, not CNNs; those were untested

## Setup

- GPU: Tesla V100-SXM2-32GB (Volta, 80 SMs), Nautilus, **no MPS**

- Model: $x(b, f) \rightarrow [\textbf{Linear} \rightarrow \textbf{BatchNorm} \rightarrow \textbf{ReLU}] \times \textbf{depth} \rightarrow \textbf{Linear} \rightarrow \textbf{logits}(b, c)$

- Loss: CrossEntropyLoss; Optimizer: SGD, lr = 1e-3

- Measurement: 50 warmup + 300 timed steps per worker, multiprocessing spawn

- SM%: `nvidia-smi dmon -s u`, averaged over 8 s while all workers sustain training

## Results

| Config | f | w | depth | b | VRAM MiB | SM@1 | SM@4 | SM@5 | sd@2 | sd@3 | sd@4 | sd@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tps_w128_d1_b512 | 54 | 128 | 1 | 512 | 1.7 | 11.0 | 99.1 | 99.0 | 0.94 | 0.95 | 1.00 | 1.01 |
| tps_w256_d2_b1024 | 54 | 256 | 2 | 1024 | 6.8 | 23.0 | 99.0 | 99.0 | 0.99 | 0.98 | 1.00 | 1.07 |
| tps_w512_d3_b2048 | 54 | 512 | 3 | 2048 | 35.6 | 49.6 | 100.0 | 100.0 | 1.08 | 1.66 | 2.26 | 1.94 |
| tps_w1024_d4_b4096 | 54 | 1024 | 4 | 4096 | 177.3 | 99.9 | 100.0 | 100.0 | 2.38 | 3.58 | 4.68 | 6.04 |
| tps_w2048_d4_b4096 | 54 | 2048 | 4 | 4096 | 385.6 | 100.0 | 100.0 | 100.0 | 2.32 | 3.45 | 4.65 | 5.84 |
| tps_w2048_d6_b8192 | 54 | 2048 | 6 | 8192 | 978.6 | 100.0 | 100.0 | 100.0 | 2.32 | 3.51 | 4.68 | 5.85 |
| text_w512_d3_b1024 | 5000 | 512 | 3 | 1024 | 49.8 | 66.8 | 100.0 | 100.0 | 1.14 | 2.45 | 2.59 | 4.05 |
| text_w1024_d3_b2048 | 5000 | 1024 | 3 | 2048 | 136.1 | 100.0 | 100.0 | 100.0 | 2.33 | 3.48 | 4.69 | 5.62 |

- `sd@N` is per-job slowdown at N concurrent jobs relative to the solo rate; the gate is 1.15x

## Configurations Meeting Both Targets

> Fill compute at N=4-5 and stay under the gate

| Config | Solo SM | SM at N=4 | Slowdown at N=4 | Slowdown at N=5 |
|---|---|---|---|---|
| tps_w128_d1_b512 | 11.0% | 99.1% | 1.00x | 1.01x |
| tps_w256_d2_b1024 | 23.0% | 99.0% | 1.00x | 1.07x |

- `tps_w256_d2_b1024` is the better of the two: it reaches the same 99% device utilization with fewer co-runners of headroom to spare, and 5 jobs still land at 1.07x

- Both configurations correspond to the **tabular-playground-series** MLEBench-Lite tasks (dec-2021: 54 features / 7 classes; may-2022: 31 features / 2 classes)

## Threshold

- Solo SM utilization at or below roughly 25% packs 4-5 jobs at about 1.0x and drives the device to 99%

- Solo SM utilization at or above roughly 50% fails the gate by N=3

- `tps_w512_d3_b2048` at 49.6% solo SM is the crossover: it passes at N=2 (1.08x) and fails from N=3 (1.66x)

- No MPS was used; on V100 these models are small enough that concurrent contexts do not contend

## Correction to the Earlier Conclusion

- [v100_packing_results.md](v100_packing_results.md) concluded that V100 cannot pack MLEBench-Lite jobs; that holds for the image tasks it measured but not for the benchmark as a whole

- The earlier phase-1 tension ("models that pack do not fill compute") does not hold on V100: `tps_w256_d2_b1024` packs at 1.00x and fills the device to 99% at the same time

- The distinguishing factor is solo SM utilization, not architecture family

## Reproduction

- Sweep script: `scheduler_benchmark_test/v100_tabular_sweep.py`

```bash
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 -u v100_tabular_sweep.py
```
