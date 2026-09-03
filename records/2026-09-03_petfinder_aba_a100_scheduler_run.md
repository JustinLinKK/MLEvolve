# PetFinder Scheduler Run on ABA A100

## Configuration

- Host: `ssh ABA` (`ecepxiegpu1.ucsd.edu`)
- Agent GPU: A100 80GB PCIe GPU 0, local vLLM serving `qwen3.8-27b-int8-a100`
- Experiment GPU: A100 80GB PCIe GPU 1
- Task: `petfinder-pawpularity-score`
- Budget: 50 terminal, budget-counted nodes
- Scheduler: branch-profile prediction, `parallel_job_cap=null`, 31 GiB admission budget
- Agent context: checkpoint-native 262,144 tokens; no artificial context-length limit

## Observed event

At 2026-09-03 16:45 UTC, the first generated candidate was rejected by CPU model preflight before GPU execution. It attempted to fetch torchvision EfficientNet-B0 ImageNet weights while preflight deliberately disables network access. The candidate also contained import-time directory creation. The stage repair attempted one targeted fix, but its integration patch was syntactically invalid; the final admission report therefore remained `FAIL` with `CON001` and `MLE_IMPORT001`.

This candidate was excluded from the budgeted node count, as intended. Per experiment instruction, no generation constraint or repair policy was changed after this single rejection. The run continued to the next candidate, and the event remains available in `pipeline.sqlite3`, the preflight report, and `scheduler.log` under the remote run root.

At 2026-09-03 16:53 UTC, the second candidate was also rejected before execution. Its isolated construction attempted to load a Hugging Face checkpoint absent from the local cache, and static inspection found import-time side effects. The candidate generation itself completed and produced this next node within the normal observed interval, so no generation-stall repair was triggered.

At 2026-09-03 17:07 UTC, the third candidate was rejected before execution. Its preflight construction check found a candidate-local error: `NameError: name 'DEVICE' is not defined` at module import. This is distinct from the first two candidate defects. Generation completed, review completed, and the next candidate began at 17:08 UTC; GPU 0 remained busy serving the agent and GPU 1 correctly remained idle because no candidate had been admitted.

At 2026-09-03 17:17 UTC, the first admitted candidate was launched but failed at `model.to(cuda)` with `torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable`. This was reproducible with an independent one-tensor CUDA allocation under `CUDA_VISIBLE_DEVICES=1`. Root cause: the search process constructed SentenceTransformer without forwarding its configured CPU device; SentenceTransformer therefore created a CUDA context on GPU 1 before being moved to CPU. GPU 1 was in `Exclusive_Process` compute mode, so the scheduler worker could not create its own CUDA context. The targeted fix passes `device=self.device` to SentenceTransformer at construction, preventing the transient CUDA context. The local regression test and related configuration tests passed (20 tests).

## Watch condition

Treat an extended period with no newly generated candidate/node as a generation-stall bug. A single preflight rejection is recorded but is not itself a reason to restart the experiment.
