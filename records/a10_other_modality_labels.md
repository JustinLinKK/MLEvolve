# A10 other-modality label generation

## Scope

- `other` here means every modality that is neither CV nor NLP

- CV is finished and NLP belongs to someone else, so both are filtered out

- Counts come from `workloads.jsonl` in `workload_specs_real_ready.tar.gz`, 10005 specs total

| modality | specs | dataset | owner |
|---|---|---|---|
| image | 4002 | cassava, pothole, taco | done |
| text | 1334 | jigsaw, spooky | someone else |
| audio | 1334 | animal_audio_classification | this run |
| time_series | 1334 | store_sales_time_series | this run |
| graph | 1334 | ogbn_products | this run |
| tabular | 667 | credit_card_default | this run |

- This run covers 4669 specs, and the precision sweep is `auto`, which yields 4 precisions, so 18676 labels

## Why regenerate at all

- `/mnt/output/real-a10-0629060430/results` already holds 40020 rows on a10, all `ok`, covering all 10005 specs at 4 precisions

- That run is `scheduler_label_version: 3` with 10 targets

- It has no `train_avg_vram_mib` and no `infer_avg_vram_mib`

- Average VRAM is a required predictor label, so the v3 set cannot be used as is

- The CV runs already moved to v4 by overlaying `patch/workload.py`, and this run uses the same overlay, so all modalities end up on one schema

| run | dir | schema | targets | rows |
|---|---|---|---|---|
| 0629 full-modality | `real-a10-0629060430/results` | 3 | 10 | 40020 |
| CV bs8 | `perfseer-a10-vram-time/results/a10` | 4 | 12 | 20859 |
| CV batch sweep | `perfseer-a10-vram-time/results/a10_bs` | 4 | 12 | 4800 |
| other | `perfseer-a10-vram-time/results/a10_other` | 4 | 12 | 18676 target |

## Cluster capacity, measured

- A10 capacity was probed with throwaway busybox Deployments, because `kubectl` in this namespace cannot list pods cluster-wide and node `allocatable` reports total capacity rather than idle capacity

- 35 A10 nodes exist, 26 untainted, 34 of them with 8 GPUs and 1 with 7

- No A10 entry in the namespace `ResourceQuota`, so quota is not the limit

| request | result |
|---|---|
| 1 GPU | binds on gpu-14 |
| 2 GPU | no capacity |
| 3 GPU | no capacity |
| 4 GPU | no capacity |

- The pool is fragmented down to at most one free GPU per node

- Memory was probed separately at 1 GPU

| request | result |
|---|---|
| cpu 2, memory 12Gi | binds |
| cpu 3, memory 12Gi | binds |
| cpu 2, memory 12Gi, ephemeral-storage 16Gi | binds |
| cpu 2, memory 16Gi | no capacity |
| cpu 4, memory 24Gi | no capacity |

- Memory is the constraint, not CPU and not ephemeral storage

- The job therefore requests cpu 2, memory 12Gi, ephemeral-storage 16Gi, 1 GPU, which is also the footprint the CV runs held for 28h without trouble

## Shape of the run

- Four Jobs, one GPU each, over disjoint shard indices, rather than one Job with 4 GPUs

| job | shards |
|---|---|
| `perfseer-a10-other-s0` | 0-3 |
| `perfseer-a10-other-s1` | 4-7 |
| `perfseer-a10-other-s2` | 8-11 |
| `perfseer-a10-other-s3` | 12-15 |

- Shard files are per index, so the four writers never touch the same file

- Every job rebuilds the filtered spec file with the same filter and the same input order, because sharding is by position in that file

- Profiling builds tensors from `input_specs` rather than loading real datasets, so no dataset staging is needed and the graph and audio workloads do not need more resident memory than the image ones did

## Reproduction

```bash
kubectl apply -f scheduler_benchmark_test/gpu_dev_a10_deployment.yaml
```

```bash
./scheduler_benchmark_test/apply_a10_other_labels.sh
```

## Status at 2026-08-22

- Launched 2026-08-22

- `perfseer-a10-other-s1` running on `gpu-14.nrp.mghpcc.org`, shard 4 of 16, resume checkpoint empty

- Spec filter emitted audio 1334, graph 1334, tabular 667, time_series 1334, total 4669

- First point `calib_4003::animal_audio_classification::tiny::bs8::adam::fp32_ieee::a10` returned `ok`

- `perfseer-a10-other-s0`, `s2`, `s3` Pending, waiting for A10 capacity

- `gpu-dev-a10` Pending, waiting for 4 free A10 on one node

## Nautilus-A10 access

- New deployment `gpu-dev-a10`, 4x A10, reachable as `ssh Nautilus-A10` on localhost:2224

- `gpu-dev` stays pinned to `Tesla-V100-SXM2-32GB` so `ssh Nautilus` keeps reaching V100

| host | deployment | port | GPU |
|---|---|---|---|
| `Nautilus` | `gpu-dev` | 2222 | 4x V100 |
| `Nautilus2` | `gpu-dev2` | 2223 | 4x A100 |
| `Nautilus-A10` | `gpu-dev-a10` | 2224 | 4x A10 |

- `/root` is the same `yuw-home` PVC on all three, which is `ReadWriteMany` on rook-cephfs, so dotfiles, keys and conda envs are shared rather than copied

- The init script is the same `nautilus-init` ConfigMap, so sshd comes up identically

- The image differs: `gpu-dev` is stuck on `pytorch:24.05-py3` because newer NGC builds dropped Volta, and A10 is Ampere, so `gpu-dev-a10` runs `pytorch:26.04-py3`

- Helper scripts are `~/.local/bin/nautilus-connect-a10` and `~/.local/bin/nautilus-pf-a10`, cloned from the `Nautilus` pair

- `ssh Nautilus-A10` blocks rather than failing while the pod is Pending, because the ProxyCommand waits for a Running pod

- The pod sat Pending for 3h23m before 4 free A10 appeared on one node, then bound to `gpu-02.nrp.mghpcc.org`

- Verified end to end on 2026-08-22

```
$ ssh Nautilus-A10 'hostname; whoami; nvidia-smi --query-gpu=name --format=csv,noheader'
gpu-dev-a10-84bd5b95d4-pmkh5
downeyflyfan
NVIDIA A10
NVIDIA A10
NVIDIA A10
NVIDIA A10
```

- Home resolves to `/root/downeyflyfan` with the same contents as on `Nautilus`, confirming the shared PVC

- `ssh Nautilus` still reaches 4x `Tesla V100-SXM2-32GB` on `gpu-dev`, unchanged

- All three tunnels coexist on 2222, 2223 and 2224
