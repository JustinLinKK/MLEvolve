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

## Restart after worker-GPU repair

The repaired run began at 2026-09-03 17:21 UTC under the same task, model, branch-profile scheduler, 31 GiB budget, no fixed parallel-job cap, and uncapped native model context. After retrieval-model initialization, GPU 1 had only 4 MiB used and no process, proving it had not retained the prior search-process CUDA context. The first admitted worker then ran on GPU 1 and produced the first valid terminal node at 17:35 UTC: node `18810afc794540b8b9b4515e4af8771d`, validation RMSE `19.5385`. Its job completed successfully and the result was added to the journal and top-1 submission.

The optional lesson-profile endpoint at `127.0.0.1:5005` was unavailable after this terminal node and failed open; the primary node parser validated the submission and metric successfully, so the run continued. This warning is recorded for follow-up but did not invalidate the terminal node.

At 2026-09-03 17:48 UTC, a second valid terminal node was recorded: `0fc98d77ed5d457ab0d5404fc4bc3537`, RMSE `19.412435`, improving the current best. The third candidate, `ba15d31ccf374d0f80e6a59c071fcaba`, was rejected by static code review before execution at 17:53 UTC and correctly excluded from the budget. Its retained artifact is the draft prompt only; no preflight report directory was created. The scheduler immediately continued the search, so this is recorded as an exclusion event rather than a generation stall.

The fourth candidate, `b6b55caf826141328a9152cd4f714879`, was rejected by CPU preflight at 18:04 UTC (`CHK001`, `GPU003`, `MLE_IMPORT001`). Its one permitted integration-stage repair attempt produced an invalid-syntax patch, so preflight remained `FAIL` and no GPU execution was attempted. This is a candidate-local repair failure; the active run continued to the next candidate without a generation stall.

This exposed a repair-engine defect: retry logic retried malformed patch envelopes, but an otherwise well-formed patch that failed only after application and Python syntax validation was not retried. A regression test now supplies an invalid-syntax patch followed by a valid patch and verifies that the second one is applied. The repair engine validates a candidate patch against the current source before accepting it; application/syntax failures consume a configured repair retry instead of causing immediate candidate exclusion. Targeted stage-repair and preflight integration tests passed before the next restart.

## Active repair-retry run

At 2026-09-03 18:15 UTC, the experiment was restarted from an empty search tree after committing the repair retry fix (`941bd034`). The vLLM server on physical A100 GPU 0 was preserved; the scheduler process has exclusive access to physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`. The command retains `agent.steps=50`, `parallel_job_cap=null`, branch-profile prediction, the 31 GiB scheduler budget, and no agent-context limit. A persistent five-minute monitor records budget-counted nodes and both GPU utilization; before the first journal entry it reports zero nodes, which is the expected state during initial agent generation.

The first draft of this run, `821612d19d1b42cb9399d079c8761bf1`, completed stepwise generation at 18:23 UTC after approximately seven minutes. Code review completed two repair rounds without unresolved critical issues. CPU preflight then rejected the candidate before GPU execution: the initial report contained `CHK001`, `CON001`, `GPU003`, and `MLE_IMPORT001`; the one applied preflight repair left `AUT002`, `GPU003`, and `MLE_IMPORT001`. The remaining `AUT002` is a confirmed candidate failure rather than a checker false positive: the model trains BatchNorm on the actual last batch of size one and raises `ValueError: Expected more than 1 value per channel`. A second repair output was rejected by syntax validation and did not modify the candidate. The candidate was correctly excluded from the budget at 18:28 UTC; GPU 1 remained unused, and the scheduler immediately began the next candidate. This is a candidate-quality rejection, not a generation stall.

The second draft, `1279c4d6252b46f0a8b12f6bd476b7da`, was admitted after one preflight repair; only the non-blocking static `GPU003` warning remained. Scheduler job `d628b2f4-133a-4f24-9c69-e51cd51f2694` started on GPU 1 at 18:41 UTC. At 18:42 UTC it wrote a `runtime_probe_profiled` event with branch-profile estimate `estimated_total_runtime_seconds=1229.3785`, confidence `0.75`, and resolved batch size `32`. This is direct live evidence that scheduler estimation is populated rather than `None`. The initial placement is exclusive only because this is the first observed runtime signature; no fixed parallel-job cap is configured.

The job completed successfully at 18:44 UTC with return code zero, validation RMSE `18.8938`, and actual execution time `207.1114` seconds. Early stopping ended training after seven epochs (best epoch two); this is normal candidate behavior, not a crash. The scheduler updated the same profile with the observed runtime and confidence `0.95`. The result was parsed as valid, added to the journal, and saved as the current top-1 submission. The optional lesson-profile service at port 5005 was unavailable but failed open after three retries; primary parsing and scheduler accounting were unaffected.

### Equal-node baseline checkpoint

At one valid scored node, the historical A10 baseline journal's first valid node (`3ca4b1d03bd042f381b8f5941032ecee`) has RMSE `18.1561`; the current A100 scheduler run's first valid node has RMSE `18.8938`. Thus the scheduler is worse by `0.7377` RMSE at this one-node checkpoint. This is intentionally reported without an elapsed-time or throughput claim: the baseline was executed on A10 while the current scheduler job ran on A100, so wall-clock comparison would not be a controlled hardware comparison. The checkpoint is provisional and will be recomputed at every common count after more scheduler nodes complete.

## Shared-host resource observation

At 2026-09-03 18:52 UTC, physical GPU 1 (A100 80GB PCIe, `Exclusive_Process`) was occupied by an unrelated active process owned under `/data1/kylehu/...` using 414 MiB. The process is not part of this experiment and was left untouched. This does not affect vLLM on GPU 0, but it can defer a later scheduler worker launch until the external process exits. It is recorded as shared-host contention, not an MLEvolve failure.

## Watch condition

Treat an extended period with no newly generated candidate/node as a generation-stall bug. A single preflight rejection is recorded but is not itself a reason to restart the experiment.

## Removal of client-side generation caps

At 2026-09-03 19:29 UTC, the sixth candidate's `training_evaluation` step took 8 minutes 40 seconds. It remained an active vLLM request throughout, then completed at 19:28 UTC; therefore it was not a deadlock. Its completion nevertheless exposed a configuration defect: the local Qwen/vLLM client injected `max_tokens=8192` whenever the caller did not supply a limit, and the MetaAgent merge injected `max_tokens=16384`. These are client-side generation caps, contrary to this experiment's uncapped-agent requirement. The 8,192-token step ended without an extractable code block, which is consistent with truncation.

The fix removes both implicit caps for local Qwen requests through the vLLM provider. Explicit caller limits and the legacy non-local defaults are unchanged. A regression test verifies that MetaAgent merge passes no `max_tokens`; a second verifies recognition of the uncapped local-vLLM Qwen route. Both targeted tests passed. The active scheduler will be restarted from a fresh search tree after the amended client code is synchronized, while preserving the vLLM service and all prior run artifacts for audit.

The uncapped restart began at 2026-09-03 19:33 UTC in remote run root `/dev/shm/mlevolve_a100_agent/runs/petfinder_scheduler_aba_a100_uncapped_20260903T193333Z` (scheduler PID `249201`). The prior scheduler PID `164271` was verified by its full command line, then terminated cleanly; its run root and journal remain intact. The vLLM process was not restarted. The new command retains `agent.steps=50`, native 262,144-token model context, `parallel_job_cap=null`, branch-profile prediction, and a 31 GiB admission budget.

## Dataset-grounded review repair

During the uncapped run, the reviewer rejected candidate `ce1ecda1095a4e789da1063ec31eec8f` on the assertion that test JPEGs were absent and test rows reused train image IDs. Direct inspection of the experiment data disproved that assertion: both `prepared/public/train/` and `prepared/public/test/` contain JPEG files. This was a review false positive caused by grounding the reviewer in task text but not the live data preview.

The review prompt now includes the same observed data manifest used by generation and explicitly prohibits findings that contradict it without concrete code evidence. This preserves valid path/inference checks while preventing ungrounded layout claims. The focused stage-review, hardware-context, and Qwen/vLLM tests passed (`93 passed`). The scheduler will be restarted with this correction; the vLLM server and prior run artifacts remain preserved.

The data-grounded restart began at 2026-09-03 20:12 UTC in `/dev/shm/mlevolve_a100_agent/runs/petfinder_scheduler_aba_a100_data_grounded_20260903T201244Z` (scheduler PID `283432`). The immediately prior scheduler was verified and stopped cleanly; vLLM remained live. The new run preserves the 50-node target, native context, branch-profile scheduler, `parallel_job_cap=null`, and 31 GiB admission budget.

The old monitor deliberately exited when its watched scheduler PID was replaced during that restart. At 20:23 UTC, a new read-only five-minute monitor was attached to the current scheduler PID and run-root pointer (monitor PID `291953`). It records budget-counted nodes and both GPU states without creating, modifying, or terminating experiment work.
