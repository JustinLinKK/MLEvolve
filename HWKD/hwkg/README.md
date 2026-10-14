# Isolated HWKG treatment — native feedback and calibrated advice v8

Base: InternScience/MLEvolve `9c5c8a3b23f0361708b59a401452dddc00f97189`.
This checkout adds the frozen hardware graph, optional hardware prompt advice,
its delivery/audit hooks and tests. Search, scheduling, scoring, model backends,
budgets and code execution remain upstream. It does not implement an optimizer
or force generated code to adopt advice.

The current graph SHA256 is
`ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.
The historical `MLEvolve-hwkg-isolated` sibling and its recovered v2 overlay are
unchanged. This branch's default evidence policy is now v8. Explicit `legacy`,
`v2`, `v4`, `v5`, `v6` and `v7` retain their historical policies for reproducibility.
Completed experiment snapshots are preserved. Unknown policy names fail closed.

The local v7.3 revision rejects violated input contracts before performance
advice and only uses v2 runtime windows with uniform full-model trainability,
one observed forward/backward per update, and a passed observed input contract.
Head-only, mixed freeze/unfreeze, stale-gradient and legacy unverified windows
cannot drive measured sync/batch advice. This does not certify learning quality
or the complete task recipe. This revision only changes the HWKG component's
interpretation of optional evidence; it does not add runtime observers, enforce
input dimensions, repair IPC, change scoring, or alter candidate execution.
The proposed common-adapter repairs were withdrawn on the user's clarification.
Existing v1 receipts therefore cannot activate measured sync/batch cards;
normal host-specific capability/configuration advice remains available. Missing
measurement coverage is left unknown instead of changing the baseline to obtain it.
The completed October 12 r3 study remains frozen at v7.2. No new performance
claim is made from these local regression checks.

Future repairs are confined to HWKG and its own tests/documentation. The upstream
baseline and any common adapters affecting its behavior must not be modified.

The local v7.2 revision distinguishes source-attributed CUDA loss scalar reads
from finite/conditional checks; unclassified historical read totals do not drive
logging advice. Exact observed training dtype/layout override unused syntax, and
an applicable CNN layout trial takes precedence over a capacity reminder. These
changes retain one performance intervention and the exact host graph. The external
runtime adapter can optionally supply call-site purpose counters; without them
sync advice remains unknown. The local v7.1 revision suppresses hardware debug cards when the parent execution
and first numeric update succeeded: a failed output-quality check is not evidence
of a NaN/BF16 problem. The original debug agent handles task correctness. Draft
cards describe applicable host capabilities; measured advice still requires a
matched, eligible normal-training window. The archived 15+15 study used the earlier
v7 snapshot, not this post-study repair. This revision has targeted regression
checks; no additional performance run or gain is claimed.

## Current implementation: v8.19 across the hardware catalogue

The current implementation branch is `hwkd-impl` in `AYPatrL04/MLEvolve`.
The catalogue contains 75 profiles: 67 NVIDIA and 8 AMD. Initial context and
parent precision/optimizer trials now use the actual host target, installed
CUDA/ROCm backend, native dtype availability and fused-kernel registration.
No RTX5080 model-name gate remains in capability exploration. NVIDIA Volta/Turing
use FP16/GradScaler when eligible; native BF16 requires its graph edge and the
runtime check without emulation. AMD uses `gcnArchName`/ROCm identity and runtime
gates, never an NVIDIA `sm_*` or TF32 assumption. FP16 fused trials remain excluded.

The two new runtime-conditional graph features have edges for all 75 profiles.
Their verification refers to framework trial eligibility, not measured speed or
per-device kernel execution. Unsupported runtime choices are omitted.
[SUPPORTED_HARDWARE.md](SUPPORTED_HARDWARE.md) lists every profile.

Owned calibration files can be supplied with `MLEVOLVE_HWKG_CALIBRATION_PATH`
and `MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH`. Numerical projections still require
an exact GPU/architecture/backend/version/workload match. ROCm keys include HIP
version and `gfx` target. Existing RTX5080 records and their hashes are unchanged.
The standalone precision calibration runner can record other visible GPUs and
omits BF16 profiles when native BF16 is unavailable; it never runs during advice.

One GPU must be visible to this component. Multi-GPU platform profiles use the
member's per-GPU memory and target; this does not launch distributed training or
verify rack topology. Driver-identical memory variants use common features only;
variant bandwidth and numerical calibration are withheld.

The remainder of this document describes earlier policy revisions and their
historical studies. Current validation and scope are in
[RELEASE_V8_19.md](RELEASE_V8_19.md).

## Enable

Use the same external configuration as your baseline; add only HWKG settings:

```bash
export MLEVOLVE_HWKG_GRAPH_PATH="$PWD/hwkg/graph.json"
export MLEVOLVE_HWKG_GRAPH_SHA256=ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1
export MLEVOLVE_HWKG_HARDWARE_ID=nvidia.blackwell.geforce_rtx_5080.spec
export MLEVOLVE_HWKG_EVIDENCE_POLICY=v8
export MLEVOLVE_HWKG_AUDIT_DIR=/absolute/path/to/this-run/hwkg-audit
```

For an A10, use `nvidia.ampere.a10.spec`. The adapter verifies the actual visible
device against the pinned graph (name, compute capability, approximate total
memory, unique match). Exactly one visible CUDA GPU is required. Memory is total,
not available memory. Graph verification is not a performance measurement.
Unset `MLEVOLVE_HWKG_GRAPH_PATH` for the exact no-injection path: no GPU probe,
prompt modification or HWKG audit. An initial solution that makes no generation
call also needs no probe.

## Native feedback and calibrated advice v8

v8.1 strengthens the candidate budget guard after a real usability candidate
ran extra cloned-model FP32/BF16 training benchmarks. Hardware timing must reuse
existing permitted updates; no calibration/warmup/backward/BN-stat passes may be
added outside the task budget. Standalone component calibration remains separate.
This changes advice only, without rewriting generated code or modifying execution.

Only `hwkg/`, its tests and this documentation change. Baseline code, common
adapters, model backends, execution, scoring and generated candidates are not
patched. The frozen v7.3 native 15+15 study and its launcher remain v7.3; that
study is not evidence of v8 performance.

`THROUGHPUT_JSON` / `HWKG_RUNTIME_JSON` in successful parent output are now
consumed directly, without requiring a shared execution adapter. The component
checks counter arithmetic, window boundaries, optional identity fields, static
batch size and known dtype/layout values. Missing, ambiguous or inconsistent
windows remain unknown. Accepted records are **source-bound self-reports**, not
trusted runtime observers. Static full-model screening is conservative and is
not proof of execution. SAM, freezing/detach, custom heads, reported partial work
and skipped updates do not establish comparable full-model single-forward work.
These logs cannot establish a bottleneck, quality equivalence or measured gain.

Planner advice describes one applicable implementation and its cost. Coder
advice supplies short BF16 or model+input channels-last snippets, retaining the
existing transfer, backward and update order. Debug advice requires a hardware
failure signature; ordinary generated-code errors get no hardware tuning.
Merge gets no new intervention. A checked native report can contradict unused
source configuration, but any resulting suggestion remains a conditional trial.
Original response requirements stay last, and chat system/prefill fields remain
unchanged. At most 1100 characters are injected. One decision per candidate,
phase and returned-plan context prevents repeated injection into phase retries;
planner and coder contexts are refreshed separately. All original provider calls,
responses and internal retry behavior remain unchanged; there is no new API call.
Provider wall time is recorded, but marginal token/reasoning cost is unknown.

The bundled `calibrations/rtx5080-resnet50-b64.json` records this workstation's
bounded public-training calibration and is considered only for a visible RTX
5080 and must match its pinned SHA256. `MLEVOLVE_HWKG_CALIBRATION_PATH` can override it with an owned regular JSON
file from the standalone component calibration; an empty value disables profiles.
No match means no numerical speed estimate. Nothing benchmarks implicitly
inside guidance or adds auxiliary GPU work during a search. The matcher requires
the recorded GPU name/capability, Torch/CUDA and backend settings, direct model,
weights, classifier, optimizer/options, loss, batch, shape and forwards/update.
Candidate backend changes or incomplete syntax leave matching unknown. The
calibration uses public training minibatches with H2D and full-model updates,
rotates three profile orders, and checks all losses, initial gradients and final
parameters for finiteness. It does not measure augmentation/evaluation, task
quality, full candidate startup or LLM overhead. Source matching still cannot
certify semantic equivalence or reproduce every runtime decision.

```bash
python -m hwkg.calibration \
  --images /absolute/path/public/train_images.npy \
  --csv /absolute/path/public/train.csv \
  --weights /absolute/path/resnet50-11ad3fa6.pth \
  --output /absolute/path/new-calibration.json --batch 64
export MLEVOLVE_HWKG_CALIBRATION_PATH=/absolute/path/new-calibration.json
```

The shipped recipe is deliberately narrow: RTX 5080, ResNet50 V2, 100-way head,
128x128 public training images, AdamW LR 0.001 / weight decay 0.0001, one forward
per update. It reads no validation/test labels and never downloads a model.
The estimate allows one precision or layout change, requires at least a 5% step
improvement and positive projected saving after extra setup and any declared
overhead. Warmup/cold-process and marginal LLM costs remain unmeasured, so the
estimate is conditional; only equal-work full-candidate timing and acceptable
task quality can establish a benefit. GPU busy percentage is not useful throughput.
Candidate return audits record source adoption and recipe syntax drift without
rewriting or rejecting upstream candidates; runtime/semantic proof remains false.

## Measured parent feedback v7

v7 retains v6's exact GPU/graph/backend validation and adds selection from a
source- and GPU-matched parent execution receipt. It suppresses repeated BF16,
channels_last and generic capacity cards after configuration/observation; syntax
is explicitly distinguished from execution. It selects at most one performance
intervention using the actual host graph: measured scalar logging reads or
observed memory headroom. Read counts are not a measured stall duration; larger
physical batches must preserve the intended update/effective-batch schedule or
be skipped. No new applicable intervention means no injection tokens. Failed
parent correctness defers hardware tuning.

The optional shared-runtime-v1 receipt comes from an external experiment adapter
applied identically to baseline and treatment. That adapter guards degenerate
image grids and the first optimizer update and observes normal training; it is
not a HWKG-only correctness improvement. Without it, measurements remain unknown.
New paired comparisons must disclose this adapter and freeze its source along
with the task, provider, configs and both source snapshots. Existing completed
studies are unchanged. v6 remains available explicitly for historical behavior.

## Historical host specialization v6

v4/v5 confirmed GPU identity but replaced graph context with fixed generic cards,
discarding hardware patterns and most capability-specific choices. That failed to
test the intended hardware-specialization component. v6 was the corrected default;
explicit older-policy settings still select their historical behavior.

v6 queries the exact host node, its feature edges, recommended patterns and local
runtime. Each selected intervention records graph fields/patterns, feature IDs,
technical sources and why it applies. Rejected and budget-omitted entries are
audited. The 1600-character limit remains. Concrete CPU implementations receive
no appended CUDA advice; planning may still consider a host-fit neural approach.

For the actual RTX 5080, this includes its `sm_120` target, GDDR7/bandwidth and
actual total/free-memory snapshot, native BF16 when the runtime supports it, and
the verified TF32 alternative missing from v4/v5. Compatible transformer attention
can trial framework SDPA with mask/bias/dropout preservation and actual backend
eligibility checks. Custom-kernel candidates receive the graph target and relevant
architecture limits; ordinary PyTorch candidates do not receive kernel-writing
instructions. Memory/batching and CNN layout cards are selected from host patterns
for the corresponding workload. Model and kernel choices remain the agent's;
external scoring, data and training budgets are preserved.

Host specificity means decisions depend on the host's capabilities and limits.
Some GPUs legitimately share BF16/TF32 paths. A shared operation can be applicable
to both devices while memory limits, target and advanced recipes differ. The
counterfactual tests switch RTX 5080/A10 and remove graph edges/patterns or runtime
support; the resulting options must change, not just the displayed GPU name.

FP8/MXFP8/NVFP4 recipes require their host edge, an installed backend, a long LM
workload and `MLEVOLVE_HWKG_OPTIMIZATION_MODE=aggressive`, matching their pinned
graph scope. Package presence is only eligibility for a conditional trial, not
proof the backend supports this exact target/module/shape. Generic FP4 is not
NVFP4. On this workstation Transformer Engine/torchao were absent, so those entries
were audited as unavailable. A single-GPU run gets no FSDP all-gather advice.

Planner, coder, joint plan-and-code, debug and merge use the existing phase hooks
and refresh context on each dispatch. Parent source and returned plan are hints;
actual adoption and runtime measurements still need separate verification. The
new policy has local route tests and real-device context construction, not a new
performance result. The archived v4 pair remains a measurement of v4.

API references: [TF32-capable FP32 matmul](https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html),
[SDPA and its input limits](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html),
[CUDA targets and architecture limits](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

## What v4 fixes

- Candidate-scoped delivery covers draft, improve, evolution, debug, fusion and
  aggregation, including planners, diff coders/retries, full-code fallback,
  stepwise coders and merging. Context is appended at `llm.generate`, after
  intermediate prompt rebuilding. Feedback/evaluation queries are unchanged.
- Thread-local context is reset on return or failure; shared prompts are never
  mutated. Hooks alone remain in agent code; tests compare its AST to upstream.
- Hardware advice does not choose validation splits, thresholds, truncation,
  folds, ensemble sizes or training budgets. Stage/task requirements take
  precedence. CPU methods are not asked to become GPU methods.
- At most 1600 characters, prioritizing numerical safety. BF16 advice needs
  verified AMP/BF16 graph edges and a runtime capability check in generated code.
  Finite checks include post-update parameters; dynamic padding preserves
  truncation/order and aligns every stream. Debug gets safety, not unrelated
  optimization. AST clues are treated as inference, not execution evidence.
- Runtime advice specifies counts and elapsed time from the **same** warmed
  training window, with CUDA synchronization at both boundaries. GPU busy
  percentage is never converted into useful throughput.

## Evidence and limitations

`prompt-audit.jsonl` records context construction. `dispatch-audit.jsonl` records
provider dispatch/return/failure with a call ID, candidate trace ID, context and
prompt hashes; candidate-return events link the trace to actual node/code IDs.
Dispatch proves delivery to the backend, not API acceptance or adoption. The
unchanged backend may retry HTTP requests internally. Its request policy is not
an HWKG feature. The one-candidate usability harness separately prohibits retries.
`SearchNode.prompt_input` remains upstream's base prompt; use the dispatch audit
for the final injected context. It does not log credentials or full task data.

Advice is conditional; an LLM can ignore it or change the proposed model. It is
not a correctness gate. v6 refreshes applicability using the returned planner
hint, but unrecognized custom models can be missed. All performance claims require independent
execution evidence. Single visible GPU and conservative graph matching are
intentional limitations. Sparse advice does not search the graph for every
architecture-specific recipe.

Sources for conditional trials: [PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html),
[HF dynamic padding](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/data_collator),
[PyTorch tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
These establish API behavior, not a measured benefit for this task.

Run local regression checks from this checkout:

```bash
../MLEvolve/.venv/bin/python -m pytest tests/test_hwkg_injection.py tests/test_hwkg_v4.py tests/test_hwkg_v5.py tests/test_hwkg_v6.py -q
```

## Optional v5 stage advice

Set `MLEVOLVE_HWKG_EVIDENCE_POLICY=v5` to reproduce this intermediate policy;
the completed RTX 5080 pair used an independent frozen v4 snapshot. Its measured
results do not evaluate v5. This revision has local routing/regression validation,
not a new paid generation or performance experiment.

v5 uses existing call boundaries; it does not introduce planning calls or alter
the upstream search. Planner calls receive cost-aware change selection. Coding
calls receive implementation and numerical checks. Debug receives failure
diagnosis, and merge receives preservation of guards/collation/counters. A joint
plan-and-code call receives the combined implementation advice. Each dispatch
rebuilds advice within the same 1600-character budget, with numerical safety
prioritized; omitted cards are recorded.

The planner's returned text is retained as an unverified applicability hint. If
a CPU parent plans a transformer, coding advice is refreshed instead of retaining
CPU-only context. Parent failure output is bounded and used only as a hint. A
changed model still needs its own runtime checks; planner claims cannot establish
dtype, numerical stability, or a cause of failure.

The numerical card requests an explicit FP32 parameter cast and actual dtype
logging, finite loaded parameters/loss/gradients/first post-update parameters,
and immediate failure before inference. BF16 requires the verified graph edge
and a runtime capability check; it is not presented as a cure for NaNs. Throughput
requires actual completed counters and skipped-step counts. These remain prompts,
not an enforced execution guard: generated code can ignore them.

Audit records include phase, plan/failure hashes and conservative AST observations.
Comments and literal `dtype=fp32` messages do not prove adoption. Source observations
always say `runtime_verified=false`; a call named `isfinite` does not prove a
post-step guard. Technical source URLs are attached only to applicable cards;
planner/diagnostic/merge rules are identified separately as local policy.
# v8.2 layout reporting guard (local component revision)

The frozen local v8.1 three-plus-three fixture produced one generated-code failure
before training: `Parameter.memory_format` is not an attribute. The layout card
already recommends `is_contiguous(memory_format=torch.channels_last)`; v8.2 also
puts that warning into precision coder and combined plan/code advice. Planner
advice remains concise. This changes advisory text only: no generated-code rewrite,
baseline/shared adapter repair, added provider call, or claimed enforcement.
The frozen v8.1 comparison is preserved and does not evaluate v8.2. See the separate
component regression fixture for generation/runtime evidence; it is unpaired and
cannot establish improved baseline comparison scores.

# v8.3 stateless retry delivery and precision scope

The v8 decision cache now caches applicability without suppressing advisory delivery
on subsequent independent provider requests. Each retry receives the same cached
decision, while an already augmented prompt receives only one copy of the same
block. Calls/responses, retry policy, baseline adapters and generated code remain
unchanged. Cached no-advice decisions still pass through without injection.

Precision guidance targets existing training forwards/loss only, preserves parent
evaluation precision and requests FP32 metric passes for drafts. Recipe preservation
explicitly includes batch, RNG and sample order. These are advisory constraints,
not enforced runtime guarantees. Original output-format instructions remain intact.
The paired local component screen is separate from frozen v8.1/v8.2 records and
does not claim to run the full native MLEvolve search.

## v8.6 EMA/checkpoint confounding and native layout evidence

The v8.5 native H3-to-H4 change introduced ModelEMA and checkpoint selection as
well as channels_last. The previous selected-call recipe screen missed those
changes. v8.6 includes EMA/SWA helpers, constructors, directly bound updates,
averaging parameters and checkpoint load/copy/save syntax. Confounded observations
are recorded and cannot suppress a later layout/precision suggestion based only
on their wall time or metric difference. Candidate failures and precision safety
warnings remain run-local advice warnings. Nodes, rewards and search are unchanged.
Parent advice explicitly preserves EMA/checkpoint selection within the same
750-character budget. This is a request to the generator, not an enforcement gate.

The throughput parser now accepts the two exact descriptive NCHW labels observed
in native logs. It still checks counters, source/batch/GPU identity and layouts;
the result remains a source-bound self-report, never verified useful throughput.
Dynamic alias/control/data flow and unselected recipe syntax remain uncertified.
The original calibration bundle and its narrow workload matching are unchanged.

The current regression suite is `../MLEvolve/.venv/bin/python -m pytest
tests/test_hwkg*.py -q`. Historical frozen studies and reviewed archives are
preserved. These fixes alone do not establish improved generation or performance;
the autonomous campaign records each fresh test and its size decision separately.

## v8.7 precise calibration coverage and useful context

Observed native R50/AdamW parents used literal 128px resize/allocation and
CrossEntropyLoss(label_smoothing=.1), which the prior syntax screen left unknown.
v8.7 recognizes only consistent literal square RGB geometry and direct unweighted
mean CE with a finite literal smoothing value; ambiguous geometry, dynamic loss,
SAM and parameter groups still cannot borrow a direct AdamW record. Literal
legacy matmul.allow_tf32 assignment is matched to its observed precision effect
only in the tested Torch2.14.1+cu130 runtime. Mixed API setters remain unknown;
see [PyTorch CUDA precision controls](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
No baseline or generated candidate is repaired or rewritten.

The new bundle preserves all five v8.4 records and adds one separately measured
public-train R50/B256/smoothing.1/lr.0003/WD.05/benchmark+TF32 record: six profiles,
three repeats,25 normal updates plus5 warmup per trial. No agent/API study overlaps
this calibration. No augmentation or task-score comparison is part of it.
New repeat ranges gate projected net savings using the fastest observed baseline
and slowest target repeat. This is a spread guard, not a confidence interval;
older records without ranges remain explicitly median-only projections. The
declared5s reserve is not measured marginal LLM overhead.

The new NHWC precision comparison is borderline: median FP16 projects about5.19s
savings across936 updates, approximately0.19s after the reserve; the observed
repeat-spread projection is negative and therefore suppresses this suggestion.
This does not prove the host has no other useful optimization. Native tasks
already declaring this exact GPU/capability/CUDA build receive no duplicate
parentless host facts; other tasks can still receive the compact factual card.
Parented hardware trials remain conditional on applicability and expected cost.

## v8.8 actual provider prompt identity gate

The v8.7 native screen did not activate its duplicate-host gate: upstream's
clean_task_desc removed host details from agent.task_desc, while the native
data preview put the original description back into the delivered prompt.
v8.8 checks the actual pre-injection provider prompt across its original chat
roles. The archived native prompt's canonical hash matches the dispatch's base
hash; this is a regression fixture from the real call path. Prompt content is
not copied into the component audit. Parentless cache keys distinguish prompt
identity changes, and only previously delivered component blocks are removed
before checking reused prompts or withdrawing now-unneeded advice. Native task
text, roles/prefill, requests/kwargs, returned responses, parser and retries are
preserved. Tests use both the native archived prompt and changing retry contexts.
This is an injection-delivery fix, not a repair of 16K generation failures or
evidence of performance improvement. The v8.7 calibration bundle is unchanged.


## v8.9 recipe syntax coverage from actual native source

Native v8.8 source replay exposed three false negatives: parameter-group LR
constants, destructured BATCH settings, and a custom nn.Module loss formula.
The audit now follows declarative module assignments referenced by selected
model/optimizer/loss/scheduler calls, including scalar/tuple/list bindings and
alias chains; it includes custom Module definitions and their constructors.
Executable model/tensor factories are not followed as declarative settings,
so a layout conversion or DataLoader-worker-only change is not automatically
misclassified as a quality recipe change. Cyclic references terminate.

This only changes the component's syntax audit and resulting feedback memory.
It does not reject, repair, rewrite or rescore native candidates. Dynamic
helper/control/dataflow equivalence remains unknown; a false-negative repair
does not certify semantic preservation or measurable hardware performance.
The v8.8 root is partial STOP after GPU/NVML query timeout and lease watchdog.
Its numerical 85.90% candidate exceeds six gradient-training epochs by an
additional standalone20-step window and is excluded from protocol-valid gains.
No API/executor/parser/SDK/search/shared adapter was changed. All GPU work is
local; this revision adds no calibration trials. Real performance retesting
requires the current device/driver access failure to be recovered first.

## v8.10 native window counter safeguards

Actual v8.8 source uses tuple-assigned BATCH, which source_spec previously left
unknown despite recipe_changes recognizing it. Unambiguous module literals,
tuple/list destructuring, annotated settings and aliases now bind batch/shape/
epoch and direct optimizer/loss settings. Rebinding, shadowing, conditional or
executable assignments remain unknown; no code execution or dynamic evaluation.
Custom losses/models/groups retain their calibration mismatch safeguards.

Actual v8.9 FP16 source logs skipped_updates=0 without scale-delta accounting.
Native windows remain self-reports; potentially active GradScaler work now
requires a narrow linked get_scale comparison/counter syntax screen to support
a matched-work calibration projection. Literal-zero/unknown counters add a
reason and suppress that projection. Recognized syntax is still uncertified;
no window resets, successful kernels or runtime dataflow are proven. This does
not change upstream candidates, scoring, execution, parsing or API requests.
No new calibration or task-quality/performance gain is established by this fix.

## v8.11 scoped precision alias audit

Actual v8.8 native source uses AMP_DTYPE for both train and evaluation. Changing
its module assignment previously left both call ASTs unchanged and escaped the
precision audit. Unambiguous declarative dtype/enabled bindings now join each
reachable scope's signature. Torch dtype aliases resolve only under the narrow
source syntax screen; dynamic/rebound/shadowed values remain unknown. Common
bindings moved into a pure component helper, without changing integer semantics.
Evaluation precision drift excludes time/quality comparison eligibility and
marks feedback confounded for any hardware intervention. No native node/reward,
score/search/execute/API or generated source is changed. This is not a semantic
or runtime equivalence proof, and does not claim generation/performance gains.

## v8.12 normal-window source scope

Native counter arithmetic alone can accept a separate timing gradient loop
followed by epoch iterator reset. A narrow component AST screen now annotates
embedded epoch windows, standalone gradient windows before epoch iterator reset,
and unknown forms. It follows unique unshadowed direct backward helpers and
source boundary/counter fields; no candidate execution or complete dataflow proof.
The affirmative standalone pattern adds a reason blocking matched calibration
projection. Unknown forms stay unknown; embedded syntax never certifies useful
throughput, full dataset coverage, actual epoch count, or the task budget.
Native source/outputs/scoring/search/API/executor remain untouched. This only
protects component evidence; no generation, score or speedup gain is established.

## v8.13 feedback work-scope eligibility

A selected recipe signature can miss changes to a separate gradient timing
loop. Trial-memory now audits before/after window scopes, excluding affirmative
standalone-gradient/epoch-reset syntax from time/quality comparison. This scope
exclusion alone never blocks a hardware profile: extra or unmatched work cannot
be blamed on a dtype/layout intervention. Independent candidate failures and
precision warnings still apply. No native fields/rewards/scoring are changed.
Unknown/embedded scope remains syntax only, with no complete work or budget proof.

## v8.14 source-scoped fused AdamW trial

The RTX5080 graph contains a new measured optimizer implementation entry, bound
by checksum to R50V2/BF16/b64/128px/AdamW.0003wd.05/CE.1 and exact runtime flags.
A successful matched parent can receive one compact fused=True suggestion when
its conditional six-epoch cost passes the declared reserve/repeat-spread screen.
FP16 and unmatched/unknown workloads are excluded. Do not infer a generic fused
speedup or task-quality gain. Primary long-window signal is thin,earlier short
range conflicted,and LLM/full-pipeline costs are unknown; source-only syntax and
normal microtimings cannot certify actual candidate adoption or useful learning.
All original request/output/search/score/execution behavior stays unchanged.
# v8.15 source-scope repair

Known model forward/call mutation or execution-hook registration leaves ordinary
model calibration scope unknown. Exact native NCHW spelling is accepted only as
self-report; ambiguous layout remains unknown. See [release notes](RELEASE_V8_15.md).
No native component gain is established by the zero-advice v8.14 5+5 comparison.

## v8.16 capability-guided hypotheses

Initial RTX5080 drafts retain precision/optimizer options even when the task already
provides host identity. Successful parents may receive one unmeasured implementation
or TRAIN precision hypothesis without requiring an exact profitable calibration.
Measured projections remain strict; source/runtime/quality/benefit are separate.
See [release notes](RELEASE_V8_16.md). Baseline stays immutable; followups reuse the
prior baseline and run HWKD only. Native performance and one live GPU check pending.

## v8.17 initial context provenance

Initial missing-host identity is supplied by the component from the actual visible
device/runtime, with header and suggestion characters counted together. Shared
dataset goals need no hardware preset. See [release notes](RELEASE_V8_17.md).
The user's later request authorizes one fresh baseline/HWKD Cassava comparison
using shared public train/validation data; earlier CIFAR results remain historical.
Native v8.17 task performance is pending.
