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
6. The first all-`/dev/shm` retry still stalled directly after FAISS initialization. The remaining path was SentenceTransformer's separate default cache under `$HOME` on Ceph. Setting `SENTENCE_TRANSFORMERS_HOME`, `HUGGINGFACE_HUB_CACHE`, and `HF_DATASETS_CACHE` under the existing `/dev/shm` cache completed the isolated BGE load and allowed the active controller to pass retrieval and begin A100 generation.
7. The first generated candidate reached the Scheduler but failed before training because its validation split compared string `Id` values with positional values from `np.arange`, producing an empty validation partition. This was not a Scheduler or GPU failure. A deterministic pre-submission training-contract check now rejects that exact identifier/position-domain mismatch and requires an `iloc`-based split plus nonempty-partition checks. The failed candidate is excluded from all reported comparisons.
8. The experiment budget now counts a failed runtime attempt only when it lasted at least 30 seconds. CPU preflight/container rejections returned in under 30 seconds are excluded, so the final target is 30 meaningful Scheduler-execution nodes rather than 30 generated snippets. A running node is investigated only after two hours without completion.
9. The CPU preflight import-safety rule incorrectly treated the pure configuration expression `GRAD_ACCUM_STEPS = max(1, ...)` as an import-time side effect. That caused a safe candidate to be rejected despite passing construction, data-contract, abstract-forward, CPU-training, validation, and memory checks. `max` and `min` are now explicitly recognized as pure top-level configuration calls; the exact candidate is accepted by the repaired static inspection. The controller that had imported the old rule was stopped before it submitted any A10 job, and a clean run using the corrected rule was started.

## Live evidence

At 2026-09-04 17:44 UTC, the repaired controller had initialized the Scheduler and started Step 1/3 (`model_design`) of the first candidate. The A100 vLLM metric had one active request and a monotonically increasing generation-token count. No Scheduler job exists until generation and review complete; the A10 has therefore remained reserved and idle rather than running an unreviewed candidate.

At 17:48 UTC the first candidate entered `training_evaluation` and reached
the merge stage at 17:51 UTC.  It was stopped before merge completed, so it
did not prove a generation hang.  A diagnostic restart with an 8,192-token
completion budget was therefore incorrect: after reaching
`training_evaluation` it remained active longer than the original and never
entered merge.  The code-extraction retry path explains that behavior: a
truncated completion can be regenerated up to three times.  The default
16,384-token completion budget is restored; it is independent of the
262,144-token vLLM context window and does not impose an Agent context cap.
A first restart also exposed the already-known relative-coldstart-path failure
because it invoked `run.py` by absolute path outside the source directory; it
exited before selecting a node.  The next retry is launched from the source
root.

## Results

Pending: retain this section until 30 terminal nodes are present, then compare an equal number of valid scored nodes with the recorded PetFinder baseline, generate the combined Gantt/metric graphic, and push the final artifacts.
