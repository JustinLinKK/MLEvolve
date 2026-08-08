# V100 CNN Packing Concurrency Test

> Scope: this record covers **image** MLEBench-Lite tasks only. The tabular and text tasks were measured separately in [v100_tabular_packing.md](v100_tabular_packing.md) and **do** pack 4-5 jobs on V100. Do not read the conclusion below as applying to MLEBench-Lite as a whole.

## Goal

- Determine if 4-5 concurrent CNN training jobs can pack on V100-SXM2-32GB within scheduler's 1.15x slowdown gate

## Setup

- GPU: Tesla V100-SXM2-32GB (Volta, 80 SMs, compute capability 7.0)

- Location: Nautilus cluster, 4x V100 available

- Models: timm pretrained=False, SGD lr=1e-3, CrossEntropyLoss

- Measurement: 20 warmup + 100 timed steps per worker, multiprocessing spawn

## Solo Profiling

| Model | BS | step_ms | VRAM_MiB | Params_M | SM% (dmon) |
|-------|-----|---------|----------|----------|------------|
| efficientnet_b0 | 32 | 52.8 | 2779 | 4.01 | 100 |
| efficientnet_b0 | 64 | 98.3 | 5518 | 4.01 | 100 |
| resnet50 | 32 | 95.0 | 2811 | 23.52 | 100 |
| resnet50 | 64 | 180.1 | 5441 | 23.52 | 100 |
| convnext_small | 32 | 284.8 | 5589 | 49.46 | 100 |
| mobilenetv3_large_100 | 32 | 32.5 | 1425 | 4.21 | 100 |
| mobilenetv3_large_100 | 64 | 60.4 | 2805 | 4.21 | 100 |
| resnet18 | 32 | 29.7 | 825 | 11.18 | 100 |
| resnet18 | 64 | 55.7 | 1524 | 11.18 | 100 |
| resnet18 | 128 | 105.7 | 2953 | 11.18 | 100 |

- SM% measured via `nvidia-smi dmon -s u -d 1 -i 0` during continuous training

- All models show 100% SM utilization at N=1

- pynvml `nvmlDeviceGetUtilizationRates` reports 2-3% (incorrect on Volta — use dmon instead)

## Concurrent Packing WITHOUT MPS

| Model | BS | N=2 | N=3 | N=4 | N=5 |
|-------|-----|-----|-----|-----|-----|
| efficientnet_b0 | 32 | 2.39x FAIL | 3.59x FAIL | 4.79x FAIL | 5.98x FAIL |
| resnet50 | 32 | 2.41x FAIL | 3.62x FAIL | 4.84x FAIL | 6.07x FAIL |
| mobilenetv3_large_100 | 64 | 2.41x FAIL | 3.60x FAIL | 4.82x FAIL | 6.71x FAIL |
| resnet18 | 64 | 2.41x FAIL | 3.63x FAIL | 3.81x FAIL | 4.96x FAIL |

- Near-perfect linear degradation: N jobs → ~Nx slowdown

- GPU context switching dominates (same as Ampere without MPS)

## Concurrent Packing WITH MPS

| Model | BS | N=1 base | N=2 | N=3 | N=4 | N=5 |
|-------|-----|----------|-----|-----|-----|-----|
| efficientnet_b0 | 32 | 5.1* | 1.29x FAIL | 0.68x* PASS | 0.93x* PASS | 1.13x* PASS |
| resnet50 | 32 | 10.7 | 1.87x FAIL | 2.81x FAIL | 3.74x FAIL | 4.65x FAIL |
| mobilenetv3_large_100 | 64 | 16.6 | 1.72x FAIL | 2.47x FAIL | 3.31x FAIL | 4.09x FAIL |
| resnet18 | 64 | 18.4 | 1.83x FAIL | 2.68x FAIL | 3.57x FAIL | 4.44x FAIL |

> *efficientnet_b0 N=1 baseline drops from 18.9 to 5.1 with MPS — 3.7x overhead from MPS init. N=3 being faster than N=1 is physically impossible; baseline is contaminated. These results are unreliable.

- MPS reduces slowdown slightly vs no-MPS (e.g., resnet50 N=2: 1.87x vs 2.41x)

- Still far from 1.15x gate for all realistic models

## Conclusion

- **V100 cannot pack CNN training jobs within 1.15x gate**

- Every CNN model saturates V100 at 100% SM utilization solo

- Packing 4-5 CNN jobs → 3-6x slowdown regardless of MPS

- Volta MPS provides marginal improvement (unlike Ampere where MPS solved the problem for small models)

- Contrast with A10 (Ampere): w128_d1_b512 tabular MLP packed 4 jobs with MPS at 1.05-1.14x

## Implication for Scheduler

- On V100 with CNN workloads, colocation gain < 1 for all packing attempts

- Scheduler correctly rejects packing → runs jobs serially

- Scheduler value on V100: job ordering, priority, early stopping, queue management

- Packing requires either Ampere+ GPU or models that don't saturate SMs (tabular MLPs)
