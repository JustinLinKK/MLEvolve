# Rules

- Every trace must be saved in this repo

- Never configure a fixed `Max Parallel Jobs` / `parallel_job_cap` for baseline or scheduler experiments. Incremental admission determines safe concurrency from branch profiles and live VRAM telemetry.

- Draw a image with Gantt Chart above and Metric-Node graphs below to show the schedule and performance of jobs after **every experiment**. Put them in 1 png image

- In Gantt chart, blue lines represent successful jobs, red lines represent buggy nodes.

- If multiple comparison experiments are conducted, put all results in 1 image

- Set seed for petfinder task (for both local LLM and code, including torch/numpy) as `42`

# Server

- Use Nautilus A100 to run Qwen-3.8-27B-int8 (vllm) agent; Use A10 to run MLEvolve

- Claude Code is available on Nautilus

# Scheduler improvement

- Important Labels for Predictor Training: Training Time per epoch, Avg/Peak VRAM, Avg/Peak Power(Optional)

- Scheduler Target: Saturate Flops not VRAM

- Backend: MPS, Cuda Process

## Persistent execution preferences

- Do not pause or stop an active task when the user interrupts with a correction. Treat it as a scope update and resume immediately unless the user explicitly cancels or replaces the task.
