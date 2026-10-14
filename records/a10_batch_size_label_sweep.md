# A10 batch-size label sweep

## Why

- Shipped workload spec file pins `training.batch_size = 8` on all 4002 computer-vision specs

- Audit of produced labels confirms it

```
batch_size in CV labels: Counter({8: 16910})
input shapes: Counter({(8, 3, 224, 224): 10905, (8, 3, 512, 512): 6005})
```

- Every non-model axis is constant: `subset_id = tiny`, `optimizer = adam`, `num_workers = 4`, three datasets, one batch size

- Only varying axes are `model_id` (4002 unique) and `precision` (4, swept at runtime)

- Predictor trained on this has no gradient signal on batch size, yet `train_epoch_ms` and `*_vram_mib` both scale roughly linearly with it

## Existing bs8 run, job `perfseer-a10-cv-labels`

| | |
|---|---|
| Started | 2026-08-12 05:20 UTC |
| Specs | 4002 CV, filter `dataset.modality == "image"` |
| Precision | `--precision-sweep auto`, yields 4 |
| Output | `/mnt/output/perfseer-a10-vram-time/results/a10` |
| Unique CV profile points at 2026-08-17 | 16008 of 16008, 100% |
| Status | all `ok`, 0 err |
| Coverage by precision, unique | fp32_ieee 4002, tf32 4002, bf16_amp 4002, fp16_amp 4002 |

- Job finished; reaped from the queue by `ttlSecondsAfterFinished: 86400`

- 17508 CV rows against 16008 unique points, so 1500 rows are duplicates re-emitted by restarts under append mode

- v3 label files also hold 3351 non-CV rows from an earlier full-modality run, same files, append mode

- 20859 rows in the directory in total

## Batch-size run, job `perfseer-a10-bs-labels`

- Launched 2026-08-14 17:53 UTC on `gpu-13.nrp.mghpcc.org`, a different node from the bs8 job, so the two do not contend for one GPU

| | |
|---|---|
| Batch sizes | 16, 32, 64, 128 |
| Models | 600 of 4002 |
| Sampling | even stride of 100 through each of the 6 `(dataset, architecture_family)` pairs |
| Families | resnet_cnn, efficientnet_cnn, vgg_cnn, vit_encoder, unet_encoder_decoder, yolo_detector |
| Datasets | cassava_leaf_disease, pothole_image_segmentation, taco_yolo_object_detection |
| Precision | pinned `fp32_ieee,fp16_amp` |
| Spec lines | 2400 |
| Expected labels | 4800 |
| Shards | 8 |
| Output | `/mnt/output/perfseer-a10-vram-time/results/a10_bs` |

- Budget spent on models rather than precision, 600 x 4 x 2 = 4800, against the alternative 300 x 4 x 4

- Precision already covered 4-wide across all 4002 models at bs8; architecture coverage is not recoverable from the bs8 set at a new batch size

- `profile_point_id` and `workload_hash` are dropped from each rewritten spec so `normalize_workload_spec` regenerates them

- A stale id would carry the `bs8` tag and collide with existing rows under `--resume`

- Verified: 2400 specs normalize to 2400 unique ids, 600 per batch size, tagged e.g. `calib_0000::cassava_leaf_disease::tiny::bs128::adam::fp32_ieee::a10`

- Spec rewrite also restates `input_shape`, `dataset_input_shape`, `input_specs`, `dataset_input_specs`, `input_dim0`, `input_bytes_per_batch` at the new batch size

- Profiler derives the real tensor from `training.batch_size` alone: `run_profile.py:1835` reads it and `with_batch_size` overwrites dim0 of the input specs

- Those fields are therefore metadata, but they are copied verbatim into the emitted label

## Results

- Finished 2026-08-15 22:28 UTC, elapsed 28 h 25 min

- 4800 rows, 4800 unique `profile_point_id`, `status = ok` on all, `scheduler_label_version = 4`

- 1200 per batch size, 2400 per precision, 3200 cassava / 800 pothole / 800 taco

- Medians per (batch size, precision)

| bs | precision | n | train_epoch_ms | train_peak_vram_mib | train_avg_vram_mib | infer_peak_vram_mib |
|---|---|---|---|---|---|---|
| 16 | fp16_amp | 600 | 152.8 | 788.6 | 788.6 | 788.6 |
| 16 | fp32_ieee | 600 | 108.3 | 780.6 | 780.6 | 780.6 |
| 32 | fp16_amp | 600 | 79.8 | 788.6 | 788.6 | 788.6 |
| 32 | fp32_ieee | 600 | 60.5 | 782.6 | 782.6 | 782.6 |
| 64 | fp16_amp | 600 | 47.3 | 790.6 | 790.6 | 790.6 |
| 64 | fp32_ieee | 600 | 31.9 | 800.6 | 800.6 | 800.6 |
| 128 | fp16_amp | 600 | 25.5 | 898.6 | 898.6 | 896.6 |
| 128 | fp32_ieee | 600 | 19.6 | 802.6 | 802.6 | 798.6 |

- Median per-model ratio against the same model at bs8, fp32_ieee, 600 matched pairs at each batch size

| bs | train_epoch_ms | train_peak_vram_mib |
|---|---|---|
| 16 | 0.458 | 1.000 |
| 32 | 0.244 | 1.003 |
| 64 | 0.131 | 1.026 |
| 128 | 0.079 | 1.046 |

- Per-step medians, fp32_ieee

| bs | train_step_gpu_ms | train_step_wall_ms | train_avg_sm_util_percent | train_peak_torch_allocated_mib |
|---|---|---|---|---|
| 16 | 1.722 | 1.723 | 10.4 | 19.3 |
| 32 | 1.918 | 1.920 | 9.0 | 21.4 |
| 64 | 1.991 | 1.995 | 10.0 | 25.5 |
| 128 | 2.457 | 2.466 | 10.2 | 33.7 |

## Reading of the results

- `train_epoch_ms` ratios are 0.458, 0.244, 0.131, 0.079 against 0.5, 0.25, 0.125, 0.0625 for exact halving

- Epoch time is `steps_per_epoch x step_time` and `steps_per_epoch` halves with each batch doubling, so the epoch axis moves almost entirely through the step count

- `train_step_gpu_ms` moves 1.722 to 2.457 for an 8x batch increase, and `train_avg_sm_util_percent` sits near 10 percent at every batch size, so these workloads are launch-bound at 16x16

- `train_peak_vram_mib` moves 780.6 to 802.6 over the same 8x, a 2.8 percent change, because the CUDA context floor of roughly 780 MiB dominates

- `train_peak_torch_allocated_mib` moves 19.3 to 33.7, so the tensor-level allocation does scale, but device-level peak VRAM does not resolve it

- Same failure mode already recorded for V100 in `records/v100_compute_saturation.md`: at small tensors, step time is set by kernel launch overhead and is not a function of the work per step

## Caveat on resolution

- `model.input_shape` is `[8, 3, 16, 16]`, not the `[8, 3, 224, 224]` in `dataset_input_shape`

- Calibration models profile at 16x16, so the batch axis is measured on small tensors

## Reproduction

- Job manifest builds its spec file inline from the shipped tarball, so it needs no pre-staged file

```bash
kubectl apply -f a10_bs_job.yaml
```

## Labels emitted, schema v4

- `train_step_wall_ms`, `train_step_gpu_ms`, `train_epoch_ms`, `train_avg_sm_util_percent`, `train_avg_vram_mib`, `train_peak_vram_mib`, `train_peak_torch_allocated_mib`

- `infer_step_wall_ms`, `infer_step_gpu_ms`, `infer_avg_sm_util_percent`, `infer_avg_vram_mib`, `infer_peak_vram_mib`

## V100

- Stopped, 0 `run_profile` processes, all 4 GPUs at 0% and 0 MiB

- Partial results left in place at `/root/downeyflyfan/perfseer_label_validate/results/v100/`
