# HWKD `hwkd-impl` integration design — 2026-10-14

Status: design for review; no product-code migration has been performed under this document.

## Objective and source of truth

Install the hardware knowledge graph (HWKG) from `AYPatrL04/MLEvolve`, branch
`hwkd-impl`, into Lightning-Agent's hardware-knowledge directory (`HWKD/`), then make it usable by
both local agents: MLEvolve and ML-Master. The verified source revision is
`a25c6413b5181cc9043cae1a150b16606137bc2f`. Its `hwkg/` tree has 38
files; its `hwkg/graph.json` SHA-256 (Secure Hash Algorithm 256-bit digest) is
`ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.
The graph contains 75 hardware profiles. Pin the commit and graph digest in
the implementation record; do not silently track the branch head later.

This is an opt-in, advisory prompt component, not a scheduler, performance
guarantee, code rewriter, hardware allocator, or substitute for the preflight
checker. Candidate execution, scoring, search budgets, and model backends
remain the agents' native behavior.

## Existing state and migration boundary

The working-tree `HWKD/` directory is empty, while its older, different
contents are staged as additions/deletions in Git. Historical material also
exists under `records/legacy_hwkd_2026-10-09/`. Neither the Git index nor the
historical record may be reset, deleted, or overwritten to make room. The new
source occupies a distinct `HWKD/hwkg/` path, with a minimal `HWKD/__init__.py`
only if package importability requires it. Before copying, record file hashes;
after copying, compare all 38 files byte-for-byte with the pinned Git tree.

The local MLEvolve agent starts from upstream base
`9c5c8a3b23f0361708b59a401452dddc00f97189`. Its relevant local files
matched that base when inspected. ML-Master has a local opt-in plugin that
currently imports `HWKD.injection`, which cannot resolve in the empty working
tree; it only enriches the task once at startup. This is insufficient for
candidate-specific guidance during improvement and debugging.

## Package and import contract

Keep the upstream `hwkg/` files unchanged beneath `HWKD/hwkg/`; use one
canonical runtime import path, `HWKD.hwkg.*`. Put the Lightning-Agent root on
Python's module search path at each agent's startup through a small explicit
bootstrap, not by copying the graph package into either agent. The existing
ML-Master plugin path helper already exposes this root; MLEvolve needs an
equivalent startup/import path. Avoid loading the same package both as `hwkg`
and `HWKD.hwkg` in one process, because that would split context-local routing
state. Adapt upstream tests' import path in the test harness if needed, without
editing the pinned package files.

Configuration follows the source package: `MLEVOLVE_HWKG_GRAPH_PATH` points
to the installed graph, `MLEVOLVE_HWKG_GRAPH_SHA256` pins its bytes,
`MLEVOLVE_HWKG_HARDWARE_ID` selects the expected graph node, and the optional
`MLEVOLVE_HWKG_AUDIT_DIR` captures dispatch evidence. The source's default
evidence policy is v8. Disabling the treatment means no graph path; a disabled
plugin must leave prompts and provider calls unchanged. An explicitly enabled
treatment with an invalid graph, digest, node, or visible device must surface
the verification error before generation, not silently produce a supposedly
treated baseline. No credentials or secrets belong in the audit directory.

## MLEvolve integration

Reproduce the pinned branch's small hook changes, adjusting only the import
prefix from `hwkg.routing` to `HWKD.hwkg.routing`:

- Candidate-scope hooks in draft, improve, debug, evolution, fusion, and
  aggregation agents carry task, parent code, outcome, and stage.
- Phase hooks in both planners and the base, stepwise, and diff coders mark
  planner/coder/model-call boundaries.
- The **12th affected file**, `llm/__init__.py`, wraps `generate` with
  `with_hardware_prompt`. Without this provider-dispatch hook, candidate and
  phase annotations alone do not deliver advice. Preserve the native prompt,
  provider request, retries, response parsing, and error behavior when advice
  is absent.

This is 12 local MLEvolve files total: 11 under `agents/` and one under `llm/`.
Do not port unrelated source-branch experiments or replace local agent changes
outside these hooks. The pinned routing code records *delivery* of advice,
not compliance by the model or measured speedup.

## ML-Master integration

Keep `LIGHTNING_PLUGINS=hwkd` as the opt-in switch. Replace the broken
`HWKD.injection` import with the shared `HWKD.hwkg` package. Route hardware
advice at the three candidate-generation stages in
`Agent/ML-Master/agent/mcts_agent.py`: `_draft`, `_improve`, and `_debug`.
Draft uses the task; improve and debug also pass the parent candidate code,
with debug receiving the failure output and parent execution status where
available. Use the source's stage names and a single stage-aware plugin
interface rather than duplicating policy logic in ML-Master.

Insert optional advice into the user-facing instructions before each backend's
prompt serialization, including its chat and steerable-reasoning forms. Keep
response-format rules, assistant prefill, native retries, and model calls
unchanged. Avoid double insertion: remove or bypass the current one-time
startup task enrichment when stage-aware delivery is active, and make retry
delivery idempotent. With the plugin disabled, the task and every generated
prompt should be byte-identical to their current form. Audit must distinguish
context constructed, prompt dispatched, and candidate outcome; none alone
proves hardware advice was followed.

## Hardware and evidence limits

The pinned source currently requires exactly **one graphics processing unit
(GPU) visible through PyTorch's `torch.cuda` interface** for guidance. This
interface is used on both Compute Unified Device Architecture (CUDA) and
Radeon Open Compute (ROCm) builds; it does not imply NVIDIA-only support.
It verifies the visible device against the chosen graph profile by name,
compute capability, and approximate total memory. Total memory is not free
memory. A multi-GPU platform profile describes one visible member and does
not verify topology or launch distributed training. Do not work around this
gate, claim support for a two-GPU run, or infer performance from utilization.
The two agents can still run normally with the optional component disabled.

The component's graph and conditional suggestions are source evidence only.
Calibrated numerical advice requires a matching owned calibration; none is
invented during integration. Actual benefit would require a separately
authorized matched-workload experiment with complete runtime and quality
measurements.

## Verification and acceptance gates

1. Confirm the source commit and all copied package file bytes, plus graph
   digest, against the pinned source; confirm the old staged Git state and
   historical records remain intact.
2. Run the source component's relevant unit/regression tests using the
   installed package; add narrow import-routing tests for the local path.
   Test disabled behavior, bad/missing graph settings, digest mismatch,
   mismatched hardware, unsupported prompt types, and duplicate/retry delivery.
   Use controlled fixtures or mocks where a real GPU is unnecessary.
3. Test MLEvolve draft/improve/debug and planner/coder dispatch paths with a
   stub provider: one advisory block when enabled, none when disabled; no
   extra provider calls, no changed retries, no altered parsed output.
4. Test ML-Master draft/improve/debug across supported prompt formats with
   representative parent code and failure output. Verify exactly one block,
   distinct stage context, unchanged response-format directives, and exact
   disabled-path equality.
5. On a compatible actual host, perform a bounded smoke check of graph/device
   verification and audit records. Do not launch a training job, allocate a
   different GPU type, or claim throughput improvement as part of this gate.
6. Record settings, tests, outcomes, and unresolved limitations in `records/`.
   Do not call the integration complete until both agents' enabled and disabled
   paths pass. Do not push without the user's separate approval.

## Rollback and out-of-scope

The feature is off when its opt-in configuration is absent. If enabled-path
verification fails, switch it off and retain native agent behavior while
repairing only the integration. Revert only the new `HWKD/hwkg/` snapshot and
the scoped hooks through a reviewed patch if full rollback is needed; leave
pre-existing staged HWKD content, unrelated agent edits, and checkpoints/data
untouched. No legacy HWKD rewrite, scheduler change, new GPU job, model-weight
download, or unrelated repository cleanup is part of this migration.
