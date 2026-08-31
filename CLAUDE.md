# Rules

- `ssh Nautilus` to connect to Nautilus, where you have access to 4 `V100` GPUs

- Claude Code is available on Nautilus

- Every trace is saved in this repo

- Only Branch-Profile based method is adopted in this experiment.

- Memory upperbound is 31GB for this scheduler

- Use 3 V100 with vllm to run `Qwen3.8-27B-int8` model (text only) as agent and the other one to run experiments.

- Never configure a fixed `Max Parallel Jobs` / `parallel_job_cap` for baseline or scheduler experiments. Incremental admission determines safe concurrency from branch profiles and live VRAM telemetry.

- Draw a image with Gantt Chart above and Metric-Node graphs below to show the schedule and performance of jobs after **every experiment**. Put them in 1 png image

- If multiple comparison experiments are conducted, put all results in 1 image

# Scheduler improvement

- Labels for Predictor Training: Training Time per epoch, Avg/Peak VRAM, Avg/Peak Power(Optional)

- Scheduler Target: Saturate Flops not VRAM

- Backend: MPS, Cuda Process

## Persistent execution preferences

- Do not pause or stop an active task when the user interrupts with a correction. Treat it as a scope update and resume immediately unless the user explicitly cancels or replaces the task.
