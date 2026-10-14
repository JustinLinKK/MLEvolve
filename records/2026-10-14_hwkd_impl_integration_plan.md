# Pinned HWKD Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user authorized native execution without further approval prompts.

**Goal:** Install the pinned hardware knowledge graph (HWKG) and deliver opt-in, stage-aware hardware advice in MLEvolve and ML-Master without changing their disabled behavior.

**Architecture:** Preserve the 38-file source package at `HWKD/hwkg/`, imported canonically as `HWKD.hwkg`. Reproduce the upstream MLEvolve candidate, phase, and provider hooks; add a narrow ML-Master plugin interface at draft/improve/debug prompt construction. Keep existing staged HWKD content untouched.

**Tech Stack:** Python, PyTorch, Git, pytest/unittest; no new dependencies.

**Spec:** `records/2026-10-14_hwkd_impl_integration_design.md`

**Execution status (2026-10-14):** All four tasks were implemented and verified;
see `records/2026-10-14_hwkd_impl_integration_results.md` and the companion
ledger. The source-fidelity check exposed one device-detection defect on the
local NVIDIA RTX 5070 Ti: 37 of 38 files remain byte-identical, while
`HWKD/hwkg/device_support.py` carries a tested two-condition compatibility
fix. The 50 original source tests dependent on source-root or historical paths
are explicitly reported rather than counted as passing.

## Global Constraints

- Source commit: `a25c6413b5181cc9043cae1a150b16606137bc2f`; graph SHA-256: `ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.
- No unreviewed deletion/reset of existing staged HWKD files, historical records, data, or checkpoints; no GPU Job, performance claim, or push.
- Optional treatment is off without its graph path; invalid explicitly configured graph/device must error before model generation.
- Runtime must use a single `HWKD.hwkg` module identity; preserve native retries, output parsing, and provider call count.

## Review Focus

1. Enabled plugin but no graph path: ML-Master must keep the original task and prompts; test in Task 3.
2. Wrong graph digest or hardware: the enabled path must not silently call a provider as a treated run; test in Tasks 2 and 3.
3. Multiple prompt representations (plain text, chat, steerable reasoning): one advisory block, intact response format; test in Task 3.
4. MLEvolve planner/coder fallback and retries: no duplicate block or added provider call; test in Task 2.
5. Existing staged HWKD files and unrelated local agent edits: byte/state preservation after migration; test in Task 1 and final verification.

---

### Task 1: Install pinned package and prove provenance

**Files:** Create `HWKD/hwkg/**` (the 38 source files), optionally `HWKD/__init__.py`; test `tests/test_hwkd_import_path.py`.

**Interfaces:** Produces `HWKD.hwkg.injection.hardware_guidance(stage, task_description='', candidate_code='', **stage_context)` and routing decorators `hardware_context`, `hardware_phase`, `with_hardware_prompt`.

- [ ] **Step 1: Write a failing import test.** It imports `HWKD.hwkg.routing`, checks `hardware_guidance('draft') is None` without `MLEVOLVE_HWKG_GRAPH_PATH`, and asserts the graph's SHA-256 equals the pinned digest.
  ```python
  def test_pinned_package_import_and_graph(monkeypatch):
      monkeypatch.delenv("MLEVOLVE_HWKG_GRAPH_PATH", raising=False)
      from HWKD.hwkg.injection import hardware_guidance
      from pathlib import Path
      import hashlib
      assert hardware_guidance("draft") is None
      graph = Path("HWKD/hwkg/graph.json")
      assert hashlib.sha256(graph.read_bytes()).hexdigest() == "ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1"
  ```
- [ ] **Step 2: Run the test.** `python -m pytest tests/test_hwkd_import_path.py -q`; expect import failure before installation.
- [ ] **Step 3: Install exactly the 38 upstream `hwkg/` files into `HWKD/hwkg/`.** Preserve their bytes, use the pinned Git tree, and add only minimum local import bootstrap. Before and after installation record `git diff --cached --name-status -- HWKD` and compare every installed file with `git show a25c6413:hwkg/<relative-file>`.
  ```bash
  git archive a25c6413b5181cc9043cae1a150b16606137bc2f hwkg | tar -x -C HWKD
  ```
- [ ] **Step 4: Re-run import test and source tests.** `python -m pytest tests/test_hwkd_import_path.py -q`; run pinned source `tests/test_hwkg_*.py` in an import-compatible harness without changing the installed package.
- [ ] **Step 5: Verify migration boundary.** `git diff --cached --name-status -- HWKD` must still show the pre-existing staged set, and all 38 installed file hashes must equal the pinned source.

### Task 2: Connect MLEvolve's candidate and provider paths

**Files:** Modify `Agent/MLEvolve/run.py`, 11 upstream-matched `Agent/MLEvolve/agents/**` files, and `Agent/MLEvolve/llm/__init__.py`; test `tests/test_hwkd_mlevolve_hooks.py`.

**Interfaces:** The source routing decorators receive MLEvolve candidate stage/phase and wrap `llm.generate(prompt, cfg, ...)` without changing its return type or native retry arguments.

- [ ] **Step 1: Write failing hook tests.** Assert the six candidate stage functions carry the source decorators, planner/coder phases are wrapped, and `llm.generate` has the provider wrapper. Use a stub provider to assert one delivered advice block and one native call; repeat with no graph path and assert the exact original prompt.
  ```python
  def test_mlevolve_generation_receives_one_advisory_block():
      # Invoke the real MLEvolve llm.generate routing boundary with a stubbed
      # outbound provider; assert the captured user prompt has one block.
      assert captured_prompt.count("# Hardware knowledge (advisory)") == 1
      assert provider_call_count == 1
  ```
- [ ] **Step 2: Run tests.** `python -m pytest tests/test_hwkd_mlevolve_hooks.py -q`; expect missing hook assertions.
- [ ] **Step 3: Apply the pinned branch's 12 hook edits.** Change only `from hwkg.routing` to `from HWKD.hwkg.routing`; add a startup import bootstrap in `run.py` before MLEvolve agent imports. Do not port unrelated branch changes.
  ```python
  from pathlib import Path
  import sys
  lightning_root = Path(__file__).resolve().parents[2]
  if str(lightning_root) not in sys.path:
      sys.path.insert(0, str(lightning_root))
  # Pinned imports then use: from HWKD.hwkg.routing import hardware_context
  ```
- [ ] **Step 4: Test enabled and disabled behavior.** Re-run focused tests, including wrong digest/device and fallback/retry prompts. Verify model-call count, native return, and audit distinction between context creation and dispatch.
- [ ] **Step 5: Review diffs.** Compare the 12 hook changes against `git diff 9c5c8a3 FETCH_HEAD -- agents llm`, accounting only for the import prefix and bootstrap.

### Task 3: Make ML-Master stage-aware without double injection

**Files:** Modify `Agent/ML-Master/plugins/hwkd.py`, `Agent/ML-Master/plugins/manager.py`, `Agent/ML-Master/main_mcts.py`, and `Agent/ML-Master/agent/mcts_agent.py`; test `tests/test_hwkd_mlmaster_hooks.py`.

**Interfaces:** Add `PluginManager.hardware_guidance(stage: str, task_description: str, candidate_code: str = '', **context) -> str | None`. It calls the one shared HWKG function only when the plugin and graph are enabled. The agent's three generation methods consume this guidance once per candidate prompt.

- [ ] **Step 1: Write failing manager tests.** Disabled or missing-graph configuration returns `None`; enabled invalid digest/visible device raises before a stub model query; `draft`, `improve`, and `debug` forward distinct stage/parent context.
  ```python
  def test_disabled_manager_returns_no_guidance(monkeypatch):
      monkeypatch.delenv("MLEVOLVE_HWKG_GRAPH_PATH", raising=False)
      manager = PluginManager(cfg=None, enabled=set())
      assert manager.hardware_guidance("draft", "task") is None
  ```
- [ ] **Step 2: Run tests.** `python -m pytest tests/test_hwkd_mlmaster_hooks.py -q`; expect missing interface/stage assertions.
- [ ] **Step 3: Replace the broken import and startup enrichment.** Make `plugins/hwkd.py` call `HWKD.hwkg.injection.hardware_guidance`; manager exposes the interface; `main_mcts.py` passes the manager into `MCTSAgent` and no longer inserts one-time HWKD task text.
  ```python
  def hardware_guidance(self, stage: str, task_description: str,
                        candidate_code: str = "", **context) -> str | None:
      if "hwkd" not in self.enabled:
          return None
      return HardwareKnowledgePlugin().guidance(stage, task_description,
                                                candidate_code, **context)
  ```
- [ ] **Step 4: Insert once at each candidate stage.** In `_draft`, `_improve`, `_debug`, append the advisory block to user instructions before serialization for all supported prompt branches; leave response-format directives and assistant prefill in their original order. Add the parent code and debug failure text only for the corresponding stages.
  ```python
  # Before branch-specific prompt serialization:
  advice = self.plugins.hardware_guidance(stage, self.task_desc,
      candidate_code=parent_node.code if parent_node else "") if self.plugins else None
  if advice:
      prompt["Instructions"]["Hardware knowledge (advisory)"] = advice
  ```
- [ ] **Step 5: Run contract tests.** Check text/chat/steerable prompts, one block per attempt, idempotent retries, disabled-path byte equality, native call count/return, and configured-invalid fail-closed behavior.

### Task 4: Whole-path verification and record

**Files:** Create `records/2026-10-14_hwkd_impl_integration_results.md`; only revise scoped integration files if verification finds a defect.

**Interfaces:** Evidence of source identity, behavior, audit, and remaining limits for both agents.

- [ ] **Step 1: Run focused suites and static compilation.** `python -m pytest tests/test_hwkd_import_path.py tests/test_hwkd_mlevolve_hooks.py tests/test_hwkd_mlmaster_hooks.py -q`; compile changed Python files without generating tracked artifacts.
- [ ] **Step 2: Verify source and Git boundaries.** Recompute graph and 38 file hashes; compare staged HWKD name/status to the pre-migration snapshot; inspect `git diff --check` and changed paths.
- [ ] **Step 3: Perform a bounded live-device smoke test only if a compatible device is visible.** Verify name, capability, total memory, and graph match; otherwise record that this gate is unverified rather than simulating hardware as proof.
  ```bash
  .venv/bin/python -c 'import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_properties(0) if torch.cuda.device_count() == 1 else "not a one-device host")'
  ```
- [ ] **Step 4: Write concise settings/results/conclusions.** Record exact commands, pass/fail counts, disabled-path proof, source commit/digest, live-device coverage, and limitations in `records/2026-10-14_hwkd_impl_integration_results.md`.
- [ ] **Step 5: Finish without push.** Report only verified claims and the status of both agents; do not treat copied files alone as integration success.
