# V100 Compute Saturation Search

## Goal

- Find MLEBench-Lite configs where 4-5 concurrent jobs fill V100 **compute**, not VRAM

## Why the Earlier Tabular Result Was Wrong

- [v100_tabular_packing.md](v100_tabular_packing.md) reported `tps_w128_d1_b512` at solo SM 11%, N=4 SM 99.1%, slowdown 1.00x, and called that "fills compute at N=4"

- Both numbers were misread. `nvidia-smi dmon -s u` reports the fraction of **time** at least one kernel was resident, not the fraction of SM capacity used. Four tiny kernels launching constantly read 99% while occupying a handful of the 80 SMs

- A slowdown of 1.00x at N=4 is itself proof the device was **not** full. A saturated device cannot absorb a fourth job for free

- Direct measurement settles it. Throughput is independent of model width:

| Config | steps/s solo |
|---|---:|
| tps_w256_d2_b1024 | 471.5 |
| tps_w320_d2_b1024 | 453.8 |
| tps_w384_d2_b1024 | 514.1 |
| tps_w448_d2_b1024 | 492.9 |

- `w448` has roughly 3x the FLOPs of `w256` and runs no slower, so compute is not the bottleneck. At ~2 ms/step these MLPs are CUDA-launch bound. At MLEBench-Lite's 54 and 31 input features, no tabular MLP can saturate a V100

## Corrected Criterion

- A plateau test alone is not enough. A config whose aggregate throughput ceilings at 2 has plateaued while packing nothing

- Define aggregate throughput against the solo rate:

$$
\begin{equation}
\begin{aligned}
\textbf{aggregate}(N) =
\frac{N}{\textbf{slowdown}(N)}
\end{aligned}
\end{equation}
$$

- Two conditions must hold together:

- real work, solo step time at or above 5 ms, so the measurement reflects GPU compute rather than Python and launch overhead

- near-linear scaling, per-job slowdown within the 1.15 gate at N=4-5, so aggregate really approaches N

- Saturation is confirmed when aggregate peaks at N and falls at N+1

## MPS Is Required

- Without MPS the V100 time-slices whole process contexts, so aggregate ceilings regardless of job size

- Same config, same GPU model, 3-repeat medians:

| N | no-MPS slowdown | no-MPS aggregate | MPS slowdown | MPS aggregate |
|---|---:|---:|---:|---:|
| 2 | 1.06 | 1.88 | 0.92 | 2.16 |
| 3 | 1.12 | 2.67 | 0.99 | 3.04 |
| 4 | 1.48 | 2.70 | 1.21 | 3.29 |
| 5 | 1.99 | 2.51 | 1.08 | 4.63 |
| 6 | - | - | 1.69 | 3.55 |

- Without MPS aggregate peaks at 2.70 and falls, so the device never delivers more than about 2.7 jobs' worth of work no matter how many are added

- With MPS aggregate reaches 4.63 at N=5, then falls at N=6. That is genuine saturation at N=5

## Candidate Rejected on Re-measurement

- The exploratory sweeps suggested `resnet18` at resolution 64, batch 32 under MPS peaked at an aggregate of 4.63x at N=5. It did not survive a rigorous re-run

- Exploratory runs used a 100-step window, about 1.5 s per sample, giving a solo coefficient of variation of 13%. A 1.15 gate cannot be resolved at that precision. Re-measured at 600 steps per sample, 7 repeats per level, every CV fell to between 1.6% and 6.6%

| N | slowdown | aggregate | CV |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 5.6% |
| 2 | 0.973 | 2.056 | 6.6% |
| 3 | 1.087 | 2.759 | 1.6% |
| 4 | 1.381 | 2.896 | 3.2% |
| 5 | 1.676 | 2.983 | 3.1% |
| 6 | 1.962 | 3.058 | 2.4% |

- The clean curve is monotone and ceilings near 3.0. Slowdown passes the gate only through N=3. The earlier 4.63x at N=5 was measurement noise

- So this config saturates at about N=3, not 4-5, and does not meet the requirement

- MPS still matters, but less than it first appeared: the aggregate ceiling moves from about 2.70 without MPS to about 3.06 with it

## Step Time Is Not a Valid Test of Compute Boundedness

- Every wrong answer in this search came from using a time-based proxy for "is the GPU actually working". Three configs, all at about 1.9 ms per step, span the entire range of device demand:

| Config | GFLOP/step | achieved | demand |
|---|---:|---:|---:|
| text f=5000, w=256, b=512 | 4.14 | 2.20 TFLOP/s | 14.0% |
| text f=5000, w=384, b=512 | 6.35 | 3.23 TFLOP/s | 20.6% |
| text f=20000, w=512, b=512 | 31.5 | 16.1 TFLOP/s | ~100% |

- The first is overhead-bound with the GPU mostly idle. The last is compute-bound at fp32 peak and fills the device with a single job, confirmed by an aggregate of only 1.09 at N=2. Identical step times, opposite situations

- The correct test is achieved FLOP rate against the device peak:

$$
\begin{equation}
\begin{aligned}
\textbf{demand} =
\frac{\textbf{GFLOP per step} \times \textbf{steps per second}}
{\textbf{peak TFLOP per second}}
\end{aligned}
\end{equation}
$$

- A Linear of shape (in, out) at batch $b$ costs $2 b \cdot in \cdot out$ forward, and backward adds input-gradient and weight-gradient GEMMs of the same size, so a training step costs $6 b \cdot in \cdot out$. Predicted demand matched measurement to within 1 point on both text configs

- Peak used is 15.7 TFLOP/s, the V100-SXM2 fp32 figure. Volta has no TF32 path

## Text MLPs Do Pack, CNNs Do Not

- The determinant is kernel structure, not task family and not VRAM. A shallow MLP over a TF-IDF vector issues a few large GEMMs, which MPS co-schedules efficiently. resnet18 issues about twenty small convolution and norm kernels per step, and those ceiling near an aggregate of 3

- `text_f5000_w256_d2_b512`, measured at 5 repeats per level, solo demand 14.0%:

| N | slowdown | aggregate |
|---:|---:|---:|
| 2 | 1.00 | 2.00 |
| 3 | 1.06 | 2.84 |
| 4 | 1.01 | 3.97 |
| 5 | 1.05 | 4.77 |
| 6 | 1.09 | 5.51 |

- Near-linear through N=6 at 1.09x, driving the device to 77% of fp32 peak. This config packs well but is too light to be full at 4-5, needing about seven jobs

- `text_f5000_w384_d2_b512`, solo demand 20.6%, is the configuration that meets the requirement:

| N | slowdown | aggregate | fraction of fp32 peak |
|---:|---:|---:|---:|
| 2 | 0.97 | 2.06 | 42.4% |
| 3 | 1.03 | 2.91 | 59.9% |
| 4 | 1.04 | 3.86 | 79.5% |
| 5 | 1.11 | 4.49 | 92.5% |
| 6 | 1.21 | 4.96 | 102.0% |

- Five concurrent jobs drive the device to 92.5% of fp32 peak while each job still runs at 90% of solo speed, inside the 1.15 gate. Per-job VRAM is 26.3 MiB, so the fill is entirely compute

- Both shapes are ordinary TF-IDF solutions for the MLEBench-Lite text tasks `jigsaw-toxic-comment-classification-challenge` and `spooky-author-identification`

## Score by Largest Pack Within the Gate, Not by Aggregate Peak

- Aggregate throughput can keep creeping upward past the gate by making every job slower, which is not a usable placement. For `w=384` it rose from 4.49 at N=5 to 4.96 at N=6 only by pushing per-job slowdown from 1.11 to 1.21

- The operative quantity is the largest N whose slowdown stays within the gate, together with the device fraction that pack drives:

| Config | solo demand | largest N in gate | slowdown there | fraction of peak |
|---|---:|---:|---:|---:|
| text f=5000, w=256, b=512 | 14.0% | 6 | 1.09 | 77.3% |
| text f=5000, w=384, b=512 | 20.6% | 5 | 1.11 | 92.5% |
| text f=5000, w=512, b=512 | 28.3% | 4 | 1.13 | 100.5% |
| text f=8000, w=384, b=512 | 32.5% | 4 | 1.12 | 116% |

- Two configurations satisfy the requirement, one at each end of it:

- 4 jobs fill the device with `f=5000, w=512, b=512`, reaching 100.5% of fp32 peak at 1.13x per-job slowdown

- 5 jobs fill the device with `f=5000, w=384, b=512`, reaching 92.5% of peak at 1.11x

- `w=512` was briefly dismissed as too heavy on the strength of its aggregate peak at N=6, which is exactly the scoring error this section corrects. Judged by the largest pack inside the gate it is the better of the two

- `f=8000, w=384` is past the useful point: 4 jobs demand 116% of the device, so the pack is oversubscribed even though the measured slowdown still passes

## Blocker for Trace Recording

- Neither text dataset is present on Nautilus. `/root/downeyflyfan/mle-bench-data/` holds only `aerial-cactus-identification` and `leaf-classification`, and there are no Kaggle credentials on that host

- Both available datasets are image tasks, which the measurements above show cannot pack. Recording a trace that exercises the packing scheduler needs the text task data first

## Configs Rejected

| Config | Reason |
|---|---|
| tps_w128 to w2048, all depths | launch-bound, ~2 ms/step, throughput independent of width |
| may_w384, may_w512 | same, 31-feature tabular |
| resnet18 r64 b64 | slowdown 2.20x at N=3 |
| resnet18 r48 b64 | slowdown 1.24x at N=3, 1.44x at N=4 |
| image models at r=224 | 100% SM solo, 3.8-4.8x slowdown at N=4, see [v100_packing_results.md](v100_packing_results.md) |

## Reproduction

- Sweeps: `scheduler_benchmark_test/v100_saturation_sweep.py`, `v100_resolution_sweep.py`, `v100_low_res_sweep.py`

- Confirmation: `scheduler_benchmark_test/v100_confirm_winner.py`

- MPS daemon, bound to one GPU:

```bash
export CUDA_VISIBLE_DEVICES=3
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_g3
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_g3
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
```

- MPS clients address the daemon's single visible device as index 0, so they run with `CUDA_VISIBLE_DEVICES=0` plus the same pipe directory
