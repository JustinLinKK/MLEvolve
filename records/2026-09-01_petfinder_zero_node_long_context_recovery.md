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

## 2026-09-02 merge truncation diagnosis and native-context restart

The optimized run produced two reviewed drafts but still reached zero
budget-counted nodes. The second draft entered the stepwise merge at 00:58:53
UTC. All three merge attempts returned about 32 KiB and exhausted the default
8,192 output-token allowance with an opening Python fence but no closing fence.
Code extraction therefore could not succeed, and retrying the same bounded
request consumed about 14 minutes without changing the failure condition. The
one-hour watchdog preserved diagnostics and stopped only verified scheduler
PID `325754` at 01:18 UTC.

`MetaAgent.merge` now supplies Qwen models with a 16,384-token merge output
budget, while non-Qwen providers retain their existing default. A regression
test simulates the exact unclosed-fence failure below 16,384 tokens and proves
that the increased budget yields extractable code. The A100 bootstrap's
remaining 131,072-token artificial context cap was also removed: its default
is now the model's advertised native 262,144-token window. The combined
merge/context/bootstrap/watchdog regression set passed 13 tests in 13.59
seconds.

The same exact Qwen3.8-27B INT8 checkpoint was restarted on the existing
`NVIDIA A100-SXM4-80GB` Pod. The live process arguments and `/v1/models`
endpoint both report `max_model_len=262144`; logs confirm text-only mode,
Marlin W8A16 kernels, one-token Multi-Token Prediction, prefix caching,
`enforce_eager=False`, and full/piecewise CUDA Graphs. No Pod, model, or
experiment evidence was deleted.

A new 50-node scheduler run started at `2026-09-02T01:32:41Z`:

- scheduler PID: `338074`;
- watchdog PID: `338076`;
- run root:
  `/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_native262k_merge16k_20260902_013233`;
- branch-profile prediction, `parallel_job_cap=null`, Hardware Knowledge
  Database, pre-flight, pipeline decisions, code review, and cached CUDA
  documentation are enabled;
- the A100 service processed initialization and candidate-generation calls
  successfully with no waiting requests or API errors; two concurrent calls
  delivered about 54--57 aggregate generated tokens/s during the first draft.

At the last checkpoint the scheduler was alive and generating its first
candidate. The canonical completed-node count was still 0/50, so completion is
not claimed.

### Live merge proof and obsolete-controller retirement

The first candidate exercised the repaired path in the real run. Its three
specialist stages completed, merge began at 01:41:33 UTC, and the single merge
request completed at 01:45:09 UTC without an extraction retry. Draft node
`7d4fe5c7018243e884dcf80e28abb2cc` was created, reviewed, and automatically
modified. The old 8,192-token behavior could not reach this point for the same
class of response.

An older comparison monitor was then found repeatedly restarting the
superseded sequence-v5 scheduler against the same A10. The duplicate controller
and child were identified by their exact script, old comparison root, and
`resume_journal` command line before termination. The monitor itself was also
identified by exact command line and stopped; sequence-v5 state is now marked
`superseded`. The monitor now treats both `complete` and `superseded` as
terminal states. A test first reproduced the unwanted restart, then the full
sequence-recovery suite passed 4/4 and shell syntax validation passed. Only the
new native-context scheduler PID `338074` remains.

## 2026-09-02 configured preflight repair rounds

The native-context run remained at 0/50 after two candidates. The second
candidate exposed a concrete admission failure: its first targeted repair
moved device resolution into a function but left `DEVICE` referenced by an
import-time default argument, so the recheck still raised `NameError`. Although
`preflight.max_repair_rounds` is configurable, `_run_node_preflight` used a
single `if` and therefore stopped after one repair for every value above one.

A regression test reproduced the defect with a candidate that requires two
repairs. It failed before the implementation change and now proves the gate is
run at attempts 0, 1, and 2 and admits the candidate after the second repair.
The implementation now loops only while the gate is failing and stops at the
configured bound. The preflight and review workflow suites passed 78 tests in
39.30 seconds.

The previous run was terminated only after verifying scheduler PID `338074`
and watchdog PID `338076`; its complete artifacts remain under its original
root, marked with the replacement reason. The patched file was synchronized
to the A10 experiment container with matching SHA-256 checksums. Two immediate
restart attempts failed before experiment initialization because the unified
configuration path was not explicit; both failure logs were retained and no
node was submitted. Setting `MLEVOLVE_CONFIG` to the repository's actual
`config/config.yaml` fixed startup. A fresh 50-node run started at
`2026-09-02T02:14:42Z` with `preflight.max_repair_rounds=2`:

- scheduler PID: `345390`;
- watchdog PID: `345456`;
- run root:
  `/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_native262k_merge16k_preflight2_20260902_021436`;
- scheduler prediction remains branch-profile based and
  `parallel_job_cap=null`;
- the one-hour no-effective-node watchdog remains active.

At restart the canonical completed-node count was 0/50. Completion is not
claimed.

## 2026-09-02 preflight lightweight-configuration false positive

The two-round run exercised the repaired loop exactly as intended, but its
first candidate still failed after both rounds. The admission report identified
top-level lines containing `os.environ.get(...)` and `torch.device(...)` as
unsafe execution. Those are lightweight configuration assignments explicitly
allowed by the candidate contract; the checker had classified every function
call inside an assignment as unsafe, including pure environment reads and
device-value construction. This false positive caused the repair model to
rewrite unrelated adapter and directory code instead of admitting the module.

Regression tests now prove that `os.environ.get`, `os.getenv`, `os.path.join`,
`pathlib.Path`/`Path`, `torch.device`, and `torch.cuda.is_available` are allowed
only as top-level configuration expressions, while a side-effecting assignment
such as `RESULT = train_model()` remains rejected. The full preflight and
review workflow suite passed 80 tests in 39.60 seconds. Running the corrected
inspection against the real rejected candidate produced a complete adapter,
an existing main guard, and zero unsafe top-level lines.

The 0/50 predecessor was stopped only after exact scheduler/watchdog identity
verification and remains preserved. The corrected checker was synchronized to
the A10 container with matching SHA-256 checksums. The replacement run started
at `2026-09-02T02:32:53Z`:

- scheduler PID: `347703`;
- watchdog PID: `347706`;
- run root:
  `/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_native262k_merge16k_importsafe_20260902_023245`;
- model context remains the native 262,144 tokens;
- scheduler prediction remains branch-profile based with
  `parallel_job_cap=null`;
- preflight permits two bounded repair rounds and the one-hour effective-node
  watchdog remains active.

At replacement startup the canonical completed-node count was 0/50.
Completion is not claimed.

### Targeted adapter-context repair guidance

The first import-safe candidate passed the corrected import contract but still
needed two model-preflight repairs. The first added a same-family
`pretrained=False` fallback for offline isolated construction. The recheck then
reproduced `KeyError: 'precision'`: the checker supplies a valid partial context,
while the adapter indexed it as if every private default key were present. The
second repair changed the default context but did not merge defaults into the
caller-supplied mapping, so the same failure remained and the candidate was
rejected.

Preflight diagnostics now add exact repair guidance for these two recurring
cases: preserve the real model family while making isolated construction
network-free, and merge a checker-supplied partial context over adapter defaults
before reading optional keys. The generation contract now states the same
partial-context rule so new candidates should implement it initially. Focused
tests first failed on both missing instructions; the complete prompt,
preflight, and review set then passed 85 tests in 39.60 seconds. Both patched
files were synchronized to the A10 container with matching SHA-256 checksums.

The active process had already imported the preceding code, so these prompt and
diagnostic improvements apply after its next restart. It remains alive while a
second candidate is evaluated; the canonical count is still 0/50, and the
one-hour watchdog remains authoritative for a no-progress stop.

The second candidate was rejected by code review after its configured two
repair rounds with two critical issues still unresolved; it never reached
preflight or scheduler submission. Since both evaluated candidates had consumed
their bounded repair budgets and no effective node existed, the old process was
stopped after exact scheduler/watchdog identity checks and its artifacts were
preserved. The replacement loads the new partial-context prompt and targeted
diagnostics, and allows up to four bounded code-review and preflight repair
rounds so sequential independent defects can be repaired rather than silently
discarding an otherwise viable candidate.

The replacement started at `2026-09-02T03:02:33Z`:

- scheduler PID: `351000`;
- watchdog PID: `351003`;
- run root:
  `/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/runs/a100_agent_a10_scheduler_native262k_guided_repair4_20260902_030225`;
- `agent.review.max_repair_rounds=4` and
  `preflight.max_repair_rounds=4`;
- native 262,144-token model context, branch-profile prediction, and
  `parallel_job_cap=null` remain unchanged.

At startup the canonical count remained 0/50; completion is not claimed.

## 2026-09-02 dormant-AMP introspection false positive

The first candidate of the four-round run generated a valid TF32/FP32 path:
`USE_AMP=False`, `USE_TF32=True`, with FP16 code present only as a dormant
fallback. The static introspector nevertheless searched the entire source for
`torch.float16`, reported `uses_amp=true` and `precision_mode=fp16`, and caused
the deterministic precision guard to override five successive reviewer
decisions that had explicitly approved the candidate. Four repair rounds then
oscillated between comments and the TF32 flag before rejecting the candidate;
no GPU job was submitted.

The introspector now resolves explicit module-level AMP/TF32 boolean flags
before scanning dormant helper and fallback code. `USE_AMP=False` therefore
reports no active AMP and resolves to TF32 when `USE_TF32=True`; an explicit
`USE_AMP=True` still resolves its declared AMP dtype. Regression tests cover
both paths. The script-introspection, precision-policy, stage-review, and model
preflight suites passed 124 tests in 39.13 seconds. Applying the corrected
introspector to the real rejected candidate changes its metadata from
`uses_amp=true, precision_mode=fp16` to
`uses_amp=false, precision_mode=tf32`, matching the actual executable path.

The active run loaded the old module before this repair and cannot be trusted
to admit later candidates consistently. It will be replaced after exact
process-identity verification; all artifacts remain preserved. Completion is
not claimed.
