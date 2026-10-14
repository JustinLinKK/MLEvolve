# A10 NLP Label Generation

- Scope: text modality only.

- Hardware: exactly two NVIDIA A10 GPUs.

- Admission: one Job requesting two GPUs.

- Queue condition: NVIDIA-A10 node affinity.

- Workload source: `workloads.jsonl` from the established A10 pack.

- Expected source specifications: 1,334 text workloads.

- Precision sweep: automatic supported precision selection.

- Expected labels: 5,336 profile points.

- Sharding: 16 total shards.

- GPU 0 runs shards 0 through 7.

- GPU 1 runs shards 8 through 15.

- Resume: completed profile points are skipped.

- Output: `/mnt/output/perfseer-a10-vram-time/results/a10_nlp`.

- Manifest: `deployments/perfseer_a10_nlp_2gpu_labels.yaml`.

## Result

- Job completed after 16 hours on two NVIDIA A10 GPUs.

- Generated 5,336 of 5,336 labels with status `ok`.

- Generated 5,336 unique profile point identifiers.

- All labels use scheduler label schema version 4.

- All labels report A10 hardware identity.

- All labels include training epoch time and average/peak VRAM.

## Local Archive

- Archive: `.cache/a10_cv_other_nlp_labels_models.zip`.

- Labels: 16,008 CV, 18,676 other, 5,336 NLP.

- Sources: 10,005 `calib_*.py` model files.

- SHA256: `50313ce9f9b2cfe74e4452dd9c1522fcabba3186448b77ab8b596a357072bfe1`.
