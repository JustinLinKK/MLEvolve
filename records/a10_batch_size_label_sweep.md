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
| Unique CV profile points at 2026-08-14 17:50 UTC | 15419 of 16008, 96.3% |
| Status | all `ok`, 0 err |
| Coverage by precision | fp32_ieee 4002, tf32 4002, bf16_amp 4002, fp16_amp 3398 |

- v3 label files also hold 3351 non-CV rows from an earlier full-modality run, same files, append mode

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
