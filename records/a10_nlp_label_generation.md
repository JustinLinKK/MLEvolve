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
