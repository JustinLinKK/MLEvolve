# ABA Predictor A100 Training

- Date: 2026-09-05.
- Host: `ssh ABA`, two NVIDIA A100 80GB GPUs.
- Training source: `/home/yufan/Predictor`.
- Device: two A100 ranks on GPU 0 and GPU 1.
- Dataset: bundled v3.2 prepared label corpus.
- Split: train, validation, and held-out test.
- Model: v3.2 teacher, twelve-output SeerNet.
- Labels: twelve measured training and inference resource labels.
- Metric: validation per-label relative-error accuracy within five percent.

- Gate: every label must reach 95 percent validation accuracy.
- Zero ground truth uses denominator `1e-6`.
- Metric is logged after every epoch.
- Active output: `/data1/yufan/perfseer_v32_a100_training_20260905_mb16`.
- Execution: host virtual environment with CUDA PyTorch.
- vLLM: project process stopped before training.

- Raw target coverage: 40,020 prepared samples, 12 labels each.

- Distributed backend: NCCL Socket transport.

- NCCL direct peer transport is disabled for this host.

- Training launch: two A100 ranks, 126 global steps per epoch.

- Epoch logs store per-label five-percent accuracy and gate status.

- Active microbatch: 16 per rank; global effective batch remains 256.
- Feature cache: reused from the prior output; prior checkpoints preserved.

## Epoch 1

- Gate result: failed; threshold remains 95 percent for every label.
- Training GPU wall: both A100 GPUs at approximately 100 percent.
- `train_step_wall_ms`: 28.22 percent.
- `train_step_gpu_ms`: 30.51 percent.
- `train_epoch_ms`: 30.27 percent.
- `train_avg_sm_util_percent`: 24.10 percent.
- `train_avg_vram_mib`: 98.73 percent.
- `train_peak_vram_mib`: 97.15 percent.
- `train_peak_torch_allocated_mib`: 90.49 percent.
- `infer_step_wall_ms`: 46.93 percent.
- `infer_step_gpu_ms`: 44.10 percent.
- `infer_avg_sm_util_percent`: 37.71 percent.
- `infer_avg_vram_mib`: 98.37 percent.
- `infer_peak_vram_mib`: 99.20 percent.

## Continuation

- Epoch 30 ended because the six-epoch early-stop limit fired.
- The gate had not passed; therefore training was resumed.
- Continuation disables early stopping and retains the 95-percent gate.
- Resume begins at epoch 31 and ends only on gate pass or epoch 600.
- GPU 0 runs the continuation; GPU 1 belongs to another user.
- Checkpoint random state is trimmed to visible GPUs during single-GPU resume.

## Epoch 32

- Gate result: failed; training continued to epoch 33.
- Time-label accuracy range: 40.71 to 68.73 percent.
- SM-utilization accuracy range: 37.17 to 58.61 percent.
- VRAM-label accuracy range: 98.93 to 99.76 percent.

## Loss Correction

- Seven validation training-SM targets are exactly zero.
- The original relative loss assigned these labels disproportionate weight.
- SM-loss denominators now have a one-percent minimum.
- The strict within-five-percent validation gate remains unchanged.
- Training resumed safely from epoch 34, step 97.

## Epoch 34

- Gate result: failed; training continued to epoch 35.
- Mean validation relative error: 4.99 percent.
- Maximum validation relative error: 11.97 percent.
- Strict per-label five-percent accuracy remains below 95 percent.

## Epoch 35

- Gate result: failed; training continued to epoch 36.
- Mean twelve-label within-five-percent accuracy: 70.33 percent.
- Lowest label accuracy: 36.41 percent for training SM utilization.

## Epoch 36

- Added per-label within-ten-percent accuracy reporting.
- Gate result: failed; training continued to epoch 37.
- The gate remains based solely on within-five-percent accuracy.

## Epoch 37

- Gate result: failed; training continued to epoch 38.
- Mean twelve-label within-five-percent accuracy: 70.78 percent.
- Lowest label accuracy: 37.85 percent.

## Epoch 38

- Gate result: failed; training continued to epoch 39.
- Mean twelve-label within-five-percent accuracy: 70.33 percent.
- Lowest label accuracy: 35.73 percent.

## Epoch 39

- Gate result: failed; training continued to epoch 40.
- Mean twelve-label within-five-percent accuracy: 70.41 percent.
- Lowest label accuracy: 37.54 percent.

## Epoch 40

- Gate result: failed; next epoch starts automatically.
- Mean twelve-label within-five-percent accuracy: 70.27 percent.
- Mean twelve-label within-ten-percent accuracy: 84.80 percent.

## Epochs 41 and 42

- Both gates failed; training advanced to epoch 43.
- Epoch 41: mean five-percent accuracy 68.86 percent.
- Epoch 42: mean five-percent accuracy 69.39 percent.

## Capacity Check

- Evaluated the epoch-45 checkpoint on train and validation splits.
- Both splits had similar low time and SM accuracies.
- The model is underfitting rather than overfitting the held-out split.
- Temporary evaluator was removed; training resumed at epoch 45, step 80.

## Two-GPU Continuation

- GPU 1 became free during epoch 45.
- Training resumed from step 87 with two A100 ranks.
- Both GPUs are active; accuracy gate and per-epoch checks are unchanged.

## Epoch 45

- Gate result: failed; training continued to epoch 46.
- Mean five-percent accuracy: 70.11 percent.
- Mean ten-percent accuracy: 85.15 percent.
