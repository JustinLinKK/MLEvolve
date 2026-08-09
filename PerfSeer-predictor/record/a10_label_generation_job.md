# A10 Label Generation Job

## Goal

- Queue a Kubernetes Job that waits for A10 capacity and, as soon as it schedules, generates PerfSeer labels

- Label mapping: model file in, avg/peak VRAM and training time per epoch out

## Cluster

- Nautilus2 is the `gpu-dev2` deployment in namespace `ecepxie` on the same cluster reached by `kubectl` (port-forward helper `~/.local/bin/nautilus-pf2`, ssh on localhost:2223)

- There is no `Nautilus2` entry in `~/.ssh/config`, so `ssh Nautilus2` does not resolve; the Job was submitted with `kubectl` against the same namespace instead

- 35 nodes advertise `nvidia.com/gpu.product=NVIDIA-A10`, of which 17 are schedulable for this namespace

## Pending Behaviour

- The Job sets a `requiredDuringSchedulingIgnoredDuringExecution` node affinity on `NVIDIA-A10`

- With no A10 free the pod sits in `Pending`, and the scheduler admits it the moment capacity appears; no polling or manual retry is involved

- `completions: 16`, `parallelism: 1`, one GPU per pod, so a single free A10 is enough to start

- `--resume` plus `--num-shards 16 --shard-index $JOB_COMPLETION_INDEX` makes the run restartable and evenly sharded

- Requests were reduced from cpu 4 / memory 24Gi to cpu 2 / memory 12Gi; at the larger size 160 nodes rejected the pod on CPU alone

## Inputs

- All inputs are reused from the prior A10 calibration run on `perfseer-real-dataset-pvc` (300Gi, RWX, `rook-cephfs-central`), which still holds 12.8G under `real-a10-0629060430`

| Path | Contents |
|---|---|
| `real-a10-0629060430/repo` | PerfSeer repo checkout used by the prior run |
| `real-a10-0629060430/pack_ready.tar.gz` | 10006 model files (the label inputs) |
| `real-a10-0629060430/workload_specs_real_ready.tar.gz` | 10005 workload specs |

- The pod copies `repo` to `/tmp/repo` before running, so the shared copy on the PVC is never modified

- Workload specs carry absolute dataset paths under `/mnt/output/real-a10-0629060430/repo/datasets/prepared/...`, so the PVC mount is required; the inputs cannot be profiled off-cluster

## Outputs

- Written to `/mnt/output/perfseer-a10-vram-time/results/a10`, separate from the prior run's 257 shard files

- The scheduler-facing label is `label_v3_shard<N>.jsonl`, built by `nrp_calibration_pack/workload.py::scheduler_label`

| Requested label | Field emitted |
|---|---|
| training time per epoch | `train_epoch_ms` |
| avg VRAM | `train_avg_vram_mib`, `infer_avg_vram_mib` |
| peak VRAM | `train_peak_vram_mib`, `infer_peak_vram_mib` |

- `--label-time-mode measured_epochs --time-label-warmup-epochs 1 --time-label-measured-epochs 2` makes the time label a measured epoch; verified labels report `epoch_time_source: measured_epochs`

- VRAM is sampled by `NvmlSampler` at `--sample-interval 0.01` with `--min-phase-seconds 20` and `--min-sampler-samples 100`, so avg and peak come from a sustained window rather than a single reading

## Changes Required

> Label schema lacked avg VRAM

- `SCHEDULER_TARGET_NAMES` in `nrp_calibration_pack/workload.py` carried `train_peak_vram_mib` and `infer_peak_vram_mib` but no average, so the requested avg VRAM was not part of the scheduler label

- Added `train_avg_vram_mib` and `infer_avg_vram_mib`, both sourced from the sampler's existing `avg_mem_usage` series, and bumped `SCHEDULER_LABEL_VERSION` from 3 to 4

- No code outside `workload.py` referenced `SCHEDULER_TARGET_NAMES` or these field names, so nothing downstream broke

> Job deleted its own working directory

- The container sets `workingDir: /tmp/repo`, and the script began with `rm -rf /tmp/repo`, leaving the shell on a deleted inode

- `pip` then failed with `FileNotFoundError: [Errno 2] No such file or directory` from `os.getcwd()`, before any profiling started

- Fixed by `cd /` before the delete and `cd /tmp/repo` after the copy

> torch_geometric pin

- The unpinned install resolved to a build calling `torch.serialization.add_safe_globals`, absent in this torch; pinned to `torch_geometric==2.5.3`

## A10 Result

- The pod stayed `Pending` for roughly 22 h, then the scheduler admitted it as soon as A10 capacity freed, which is the behavior the required-affinity design was chosen for

- It landed on an `NVIDIA A10` with 23028 MiB and began profiling after the dependency install

> First label row, read from `results/a10/label_v3_shard0.jsonl`

| Field | Value |
|---|---|
| `scheduler_label_version` | 4 |
| `hardware_id` | a10 |
| `train_epoch_ms` | 253.81 |
| `train_avg_vram_mib` | 778.56 |
| `train_peak_vram_mib` | 778.56 |
| `infer_avg_vram_mib` | 778.56 |
| `infer_peak_vram_mib` | 778.56 |
| `epoch_time_source` | measured_epochs |

- All four requested outputs are populated on real A10 hardware, and the epoch time is measured rather than extrapolated

## Validation

- The A10 job could not be validated directly, since no A10 was free; an identical Job with the node affinity removed was run instead and landed on a GTX 1080

- That run exercised the same script path, the same PVC inputs, and the same `workload.py` overlay

- Result: profile points returned `ok`, and the emitted rows carried `scheduler_label_version: 4` with every requested field populated

| Field | Sample value |
|---|---|
| `train_epoch_ms` | 168.65 |
| `train_avg_vram_mib` | 666.81 |
| `train_peak_vram_mib` | 666.81 |
| `infer_avg_vram_mib` | 666.81 |
| `infer_peak_vram_mib` | 666.81 |

- Average and peak VRAM are equal in this sample because `NvmlSampler` reads whole-device memory, which is dominated by the roughly 640 MiB CUDA context and stays flat for these `tiny` batch-size-8 workloads; the two diverge on larger workloads

## Note on Power

- An earlier revision of this task asked for GPU power. `run_profile.py` has no power instrumentation, so power fields were added to `NvmlSampler` and `phase_label_v2` and unit-tested (avg 120.0 W, peak 140.0 W, p95 138.0 W from a synthetic three-sample series)

- The requirement was then restated as VRAM and training time per epoch, so that change was reverted; the profiler in use is unmodified apart from the `workload.py` label-schema change above

## Commands

```bash
kubectl apply -f record/a10-label-generation-job.yaml
```

```bash
kubectl get pods -n ecepxie -l app=perfseer-a10-label-gen
```

```bash
kubectl logs -n ecepxie -l app=perfseer-a10-label-gen --tail=50
```
