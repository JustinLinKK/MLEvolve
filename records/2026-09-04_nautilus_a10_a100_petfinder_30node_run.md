# Nautilus A100-Agent / A10-Scheduler PetFinder (30 nodes)

## Active configuration

- Agent GPU: one NVIDIA A100 80 GB on Nautilus, serving `qwen3.8-27b-int8-a100` through vLLM at the in-cluster `mlevolve-qwen-a100` service. The server reports `max_model_len=262144`; no agent-context cap was introduced.
- Execution GPU: one NVIDIA A10 (23,028 MiB physical memory) in `gpu-dev-a10-experiment`.
- Task: PetFinder Pawpularity Score; lower root mean squared error (RMSE) is better.
- Scheduler: hardware-aware mode, Branch-Profile prediction, CUDA process backend, `parallel_job_cap=null`, and live A10 telemetry rather than a fixed 31 GiB budget that would exceed the A10's physical memory.
- Run target: 30 generated nodes. Global memory is disabled and retrieval embeddings run on Central Processing Unit (CPU).

## Repaired launch failures

1. The A10 Secure Shell (SSH) helper still targeted the retired `gpu-dev-a10` Deployment. It now targets the live `gpu-dev-a10-experiment` Deployment and the port-forward helper uses the same target.
2. A clean A10 pod had no private vLLM cache salt, so configuration validation aborted before model invocation. A per-run, mode-600 random salt is supplied through the environment and is not recorded here.
3. Starting `run.py` outside its source directory made cold-start guidance resolve `engine/coldstart/competition_tag_classified.json` relative to the wrong directory. The controller is now launched from the source root.
4. Scheduler SQLite write-ahead-log files on the Ceph persistent volume caused the controller to enter `ceph_mdsc_wait_request` before candidate generation. Scheduler and hardware-knowledge runtime roots now use `/dev/shm`; controller logs and final artifacts remain under the persistent run root and will be copied back after completion.
5. The SentenceTransformer/Hugging Face cache is placed under `/dev/shm` to avoid the same Ceph metadata path. The BAAI `bge-base-en-v1.5` retrieval model loaded successfully from that cache.

## Live evidence

At 2026-09-04 17:44 UTC, the repaired controller had initialized the Scheduler and started Step 1/3 (`model_design`) of the first candidate. The A100 vLLM metric had one active request and a monotonically increasing generation-token count. No Scheduler job exists until generation and review complete; the A10 has therefore remained reserved and idle rather than running an unreviewed candidate.

At 17:48 UTC the first candidate entered `training_evaluation`.  It was still
streaming after seven minutes, with no returned candidate or Scheduler job.
The request had the configured 16,384-token completion budget.  This is a
generation-latency defect, not a context-limit failure: the vLLM server still
reports a 262,144-token context window and the request continued to make
forward progress.  The unfinished controller was terminated before it
generated a node and restarted at 17:52 UTC with
`vllm_client.default_completion_tokens=8192`.  That constrains only each
completion; it does not constrain the Agent context.  A first restart exposed
the already-known relative-coldstart-path failure because it invoked `run.py`
by absolute path outside the source directory; it exited before selecting a
node.  The active retry is launched from the source root and reached Step 1/3
at 17:53 UTC.

## Results

Pending: retain this section until 30 terminal nodes are present, then compare an equal number of valid scored nodes with the recorded PetFinder baseline, generate the combined Gantt/metric graphic, and push the final artifacts.
