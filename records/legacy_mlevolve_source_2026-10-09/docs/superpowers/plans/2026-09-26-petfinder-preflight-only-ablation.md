# PetFinder Preflight-Only Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a complete 20-node PetFinder ablation on ABA that enables the preflight checker and no other treatment feature, with a verified prompt surface before launch.

**Architecture:** Add feature-aware prompt composition so scheduler-only contracts are absent when the scheduler is disabled. For this matched ablation, keep the generator request byte-compatible with the frozen baseline and run the preflight checker only after generation. Candidates that already define CandidateAdapter receive the full CPU checker; baseline-style candidates without it receive the static-source and hardware fallback, without adapter-only LLM repair. Add a dedicated ABA launcher that preserves the completed baseline, uses the existing Qwen service on GPU 1, executes candidates on GPU 0, and refuses to launch unless an offline prompt/configuration gate proves the treatment is preflight-only.

**Tech Stack:** Python, pytest, Hydra/OmegaConf configuration, Bash, SSH, vLLM, NVIDIA A100.

**Spec:** `records/2026-09-26_petfinder_preflight_only_ablation.md`

## Global Constraints

- Baseline is frozen and must not be rerun or changed.
- Treatment budget is exactly 20 budget-counted nodes.
- Scheduler, Predictor, HWKD, context cache, lesson profiles, review, CUDA-doc retrieval, and pipeline decision features are disabled.
- Preflight checker is enabled and is the only treatment feature.
- Seed is 42 for Qwen requests and Python/NumPy/PyTorch candidate execution.
- No fixed maximum-parallel-jobs setting is allowed.
- ABA GPU 1 remains the local Qwen3.8-27B INT8 vLLM service; ABA GPU 0 runs MLEvolve candidates.
- Work in the current checkout; do not create a worktree, commit, push, or disturb unrelated remote processes.

## Review Focus

- A disabled scheduler must not add backend/model-family scheduler contracts to prompts.
- A disabled preflight checker must not add CandidateAdapter contracts unless the required ML Predictor explicitly needs them.
- Preflight-only matched mode must omit CandidateAdapter and every scheduler/HWKD instruction from generation. It must use the full checker when an adapter is naturally present and the static/hardware fallback otherwise; adapter absence alone must not cause repair.
- The launcher must fail before generation if the prompt/configuration gate detects forbidden treatment features.
- Runtime traces, preflight reports, generated candidates, metrics, and the final combined Gantt/metric figure must persist locally under `records/`.

---

### Task 1: Feature-aware prompt composition

**Files:**
- Modify: `agents/draft_agent.py`
- Modify: `agents/prompts/impl_guideline.py`
- Modify: `localml_scheduler/root_tests/test_draft_agent_preflight_prompt.py`
- Modify: `localml_scheduler/root_tests/test_impl_guideline.py`

**Interfaces:**
- Consumes: `agent.cfg.scheduler.enabled`, `agent.cfg.preflight.enabled`, and Scheduler prediction mode.
- Produces: prompt sections containing only contracts required by enabled features.

- [x] Write tests proving baseline, preflight-only, and ML-Predictor prompt surfaces.
- [x] Run the focused tests and verify the new tests fail on unconditional prompt injection.
- [x] Implement the smallest feature-gating helpers and use them in both prompt builders.
- [x] Run the focused tests and existing prompt/preflight suites to green.

### Task 2: Dedicated ABA preflight-only launcher and prompt gate

**Files:**
- Create: `localml_scheduler/deployments/run_petfinder_aba_a100_preflight_only.sh`
- Create: `localml_scheduler/root_tests/test_petfinder_aba_a100_preflight_only_launcher.py`

**Interfaces:**
- Consumes: existing ABA data, Qwen endpoint, current repository, and feature-aware prompt builder.
- Produces: durable 20-node run root plus pre-launch configuration/prompt audit artifacts.

- [x] Write launcher-contract tests for GPU separation, seed 42, 20 nodes, preflight on, all other treatments off, and no parallel cap.
- [x] Run the launcher test and verify it fails because the launcher is absent.
- [x] Implement the launcher with GPU/process safety checks, durable runtime copying, and prompt/configuration gate.
- [x] Run launcher and focused prompt tests to green.

### Task 3: ABA deployment and verified launch

**Files:**
- Update: `records/2026-09-26_petfinder_preflight_only_ablation.md`

**Interfaces:**
- Consumes: Tasks 1-2 and idle ABA GPU 0 with healthy Qwen on GPU 1.
- Produces: a live, monitored experiment handle and prompt/configuration hashes.

- [x] Synchronize only changed source and launcher files to `/home/yufan/MLEvolve` on ABA.
- [x] Run focused tests and the prompt/configuration gate on ABA.
- [x] Verify no unrelated GPU process owns GPU 0 and launch with `nohup`.
- [x] Record the run root, PID, exact prompt audit, and initial progress.

### Task 4: Completion, persistence, and figure

**Files:**
- Update: `records/2026-09-26_petfinder_preflight_only_ablation.md`
- Create: `records/2026-09-26_petfinder_baseline_vs_preflight_only.png`
- Create: `records/2026-09-26_petfinder_preflight_only_raw/`

**Interfaces:**
- Consumes: completed 20-node ABA run and frozen baseline trace.
- Produces: locally durable traces, statistics, and required combined Gantt/metric-node PNG.

- [x] Monitor the live process; stop and diagnose if no new node is generated for an anomalous interval or the process fails.
- [x] Verify exactly 20 budget-counted nodes and clean process exit.
- [x] Synchronize complete durable traces without bulky duplicated working caches.
- [x] Render and visually inspect one comparison PNG with Gantt charts above and metric-node plots below.
- [x] Audit every objective requirement against saved configuration, logs, reports, and artifacts.
