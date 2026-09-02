# Petfinder scheduler zero-node diagnosis and long-context recovery

## Scope

- Experiment: Petfinder scheduler + Hardware Knowledge Database (HWKD), using
  branch-profile runtime prediction and no fixed `parallel_job_cap`.
- Agent: Qwen3.8-27B INT8 served by vLLM on one NVIDIA A100-SXM4-80GB.
- Training: one NVIDIA A10.
- Failed run PID: `270338`.
- Failed run root:
  `/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_current_allextras_20260901_183411`.

## Observed failure

After 4 hours 21 minutes, the run had zero budget-counted nodes. Five drafts
had been generated and reviewed, but four were rejected for unresolved critical
review issues. The remaining draft entered model pre-flight and was rejected.
Consequently no candidate reached scheduler execution and the A10 remained
idle.

The failed draft prompts were 257,898--273,987 characters. The actual merge
prompt extracted from candidate `3558a597ca484c7cad685456756dc79c` was 107,090
characters and exactly 26,438 tokens under the deployed Qwen tokenizer. The
vLLM service requested an 8,192-token output while its launch configuration
artificially limited the total context to 32,768 tokens. The required 34,630
tokens exceeded that limit, causing repeated context-rejection retries and
truncated merged scripts.

The model configuration advertises a native context window of 262,144 tokens.
The running vLLM instance reported a Key--Value cache capacity of 641,524
tokens, so the 32,768-token limit was not imposed by available A100 memory.

The one candidate that reached pre-flight initially failed because its adapter
tried to retrieve an unavailable pretrained weight while network access was
disabled. Its repair passed construction, data contract, abstract forward,
CPU training, validation, and memory stages, but introduced executable
import-time code outside the main guard (`MLE_IMPORT001`). Pre-flight therefore
correctly refused that repaired candidate; warnings `GPU003` and `MEM001` were
not the final critical rejection reason.

## Fix

- Changed the persistent A100 vLLM bootstrap default from `32768` to `131072`
  tokens.
- Added a behavioral bootstrap test that launches the script with controlled
  fake runtime and health endpoints and verifies that its default context can
  accommodate the observed long merge request.
- Preserved the failed run directory and all prompt, review, pre-flight, and
  hardware-monitor evidence.
- Stopped only the verified zero-progress scheduler PID and its verified Qwen
  vLLM PID. The first graceful vLLM termination remained blocked by an orphaned
  request after the API endpoint had closed, so that exact PID was force-killed
  and the existing monitor restarted it.

## Verification

- Local tests: `6 passed` across the A100 bootstrap, monitor, and container
  entrypoint tests.
- New vLLM PID: `11497`.
- Exposed model context: `131072` tokens.
- Model: `qwen3.8-27b-int8-a100`.
- A100 GPU after load: 73,471 MiB / 81,920 MiB.
- Exact previously rejected merge prompt probe: HTTP 200, 26,448 server-counted
  prompt tokens, 16 generated tokens, 12.02 seconds.

## Relaunched run

- PID: `314952`.
- Run root:
  `/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_longctx_20260901_231234`.
- MLEvolve run:
  `20260901_231242_petfinder_scheduler_profile_hkwd_a100_agent_a10_longctx`.
- Prediction mode: `branch_profile`.
- Fixed parallel cap: `null`.
- Pre-flight: enabled, target `nvidia/a10_24gb`.
- HWKD and hardware-aware pipeline: enabled.
- CUDA documentation mode: `debug_cached`.
- Code review: enabled.

At the time this record was written, initialization, metric-direction detection,
scheduler startup, hardware context, and A100 API calls had succeeded. The run
was alive and preparing its first long-context candidate; no training node was
yet counted, so completion was not claimed.

## Persistent zero-progress watchdog

A persistent watchdog was installed inside the A10 experiment pod:

- Script: `deployments/monitor_petfinder_scheduler_progress.sh`.
- PID: `318030`.
- Poll interval: 120 seconds.
- Stall threshold: 3,600 seconds without an increase in the canonical
  `Scheduler-controlled progress: N/T budget-counted nodes` value.
- State: `/root/downeyflyfan/.cache/mlevolve_scheduler_watchdog_a10_v1`.

Pre-flight/review rejections are not counted as effective nodes. When the
threshold is reached, the watchdog verifies that the recorded PID is a matching
Petfinder scheduler process, captures the runner log, process state, GPU state,
pipeline database, journal, and node records under the run's
`watchdog_stalls/` directory, writes `STALL_DETECTED.json`, and stops only that
verified scheduler. It leaves the A100 vLLM service and all experiment evidence
untouched.

The watchdog tests cover both the positive stop path and refusal to stop an
unrelated process. The combined A100/bootstrap/watchdog suite passed `8` tests.
A second heartbeat was observed at `2026-09-01T23:30:52Z`, 121 seconds after
the initial tick, confirming that the deployed monitor is persistent.

### Watchdog timing correction

The first deployed version initialized `last_progress_epoch` when the watchdog
was installed, not when the monitored run started. This incorrectly delayed the
zero-progress deadline for an already-running experiment. A regression test was
added using a literal runner-log start timestamp, and the watchdog now parses
the canonical `Starting run` timestamp whenever it begins monitoring a new run.

The corrected watchdog was redeployed as PID `319165`. For the active run it
resolved the start time as `2026-09-01 23:12:43 UTC`; at its first corrected
heartbeat (`23:34:17 UTC`) it reported 1,294 seconds without progress, instead
of resetting the counter to zero. The corresponding 60-minute deadline is
`2026-09-02 00:12:43 UTC` if no effective node completes before then.

## 2026-09-02 A100 vLLM decode acceleration and clean restart

The stalled scheduler was stopped after its first candidate was rejected by
preflight with `Dataset` undefined and import-time execution. The A100 model
was verified against the still-running L40S service: both used the same
`Qwen3.8-27B-INT8-W8A16-MTP` directory and compressed-tensors Marlin W8A16
runtime. A controlled, single-request, 1,024-output-token comparison measured
19.105 and 19.318 tokens/s on L40S, versus 11.218 and 12.290 tokens/s on the
old A100 configuration. The previously observed L40S 50+ tokens/s figure was
aggregate concurrent throughput rather than single-request decode throughput.

The A100 root cause was its vLLM launch configuration: `--enforce-eager`
disabled both `torch.compile` and CUDA Graphs, prefix caching was disabled, and
the checkpoint's Multi-Token Prediction (MTP) head was not configured
(`speculative_config=None`). The persistent bootstrap now removes eager mode,
enables prefix caching, and enables one-token MTP speculative decoding. It also
places future vLLM compile caches under `/tmp/vllm-cache-a100`, outside a home
directory. Startup logs confirmed `enforce_eager=False`,
`CUDAGraphMode.FULL_AND_PIECEWISE`, and `SpeculativeConfig(method='mtp')`.

With the same 1,024-token request, optimized A100 measurements were 47.272,
77.921, and 77.978 tokens/s; the first sample included first-use graph overhead
and the two warm samples were stable near 78 tokens/s. MTP counters reported
3,040 accepted draft tokens from 3,311 drafts (91.8% acceptance). A 56,544
prompt-token probe plus 16 output tokens returned HTTP 200 in 28.195 seconds,
so the optimization retained the required 131,072-token context.

A fresh scheduler + Hardware Knowledge Database + review + preflight run was
started at `2026-09-02T00:18:09Z` on the A10 with scheduler PID `325754` and
watchdog PID `325756`. The run root is
`/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_vllmopt_20260902_001809`.
It keeps `parallel_job_cap=null`, uses branch-profile admission, and retains a
one-hour no-effective-node stop-and-diagnostics watchdog. The A100 server was
already processing two requests with no waiters after launch.
