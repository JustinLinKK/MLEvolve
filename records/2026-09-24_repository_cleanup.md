# Repository cleanup — 2026-09-24

## Contract

- Reference repository: `InternScience/MLEvolve`, commit `9c5c8a3b23f0361708b59a401452dddc00f97189`.
- Preserve the upstream MLEvolve top-level source layout.
- Preserve exactly four feature additions: `HWKD`, `localml_scheduler`,
  `nn-model-preflight-checker`, and `PerfSeer-predictor`.
- Preserve `.git`, `.venv`, `.cache`, `AGENTS.md`, `.gitmodules`, and
  `records` as repository metadata, runtime state, instructions, and evidence.
- Move additional project documents into `records/project_documents`.
- Preserve experiment evidence; do not treat `records` or model/data caches as
  disposable source folders.

## Consolidation map

- Hardware-knowledge packages, schema, graph tooling, lesson profiles, and
  prompt-context caching move under `HWKD/`.
- Scheduler benchmarks, deployment launchers, replay sources, project tests,
  and scheduler support scripts move under `localml_scheduler/`.
- Predictor label datasets move under `.cache/predictor_labels/`.
- Existing trace files move under `records/traces/`.
- The remaining text-only Kubernetes draft moves to
  `records/project_documents/k8s/ablation_study.yaml`; the now-empty extra
  top-level `k8s/` directory is removed.
- Rebuildable Python/test caches and abandoned download waiters are removed.
- The disposable local and ABA `.pytest_cache` directories are removed after
  verification.
- The canonical 8,920-train / 992-test PetFinder image dataset is moved from
  an old captured workspace into `.cache/petfinder_public`; `records` retains
  experiment evidence rather than another copy of the 1.0 GB input dataset.
- Before removing ABA's stale 1.7 GB `runs/` tree, 268 text artifacts
  (journals, logs, prompts, generated programs, tables, and configuration)
  were copied with their relative paths into
  `records/precleanup_runs_20260924`. Non-text duplicated images, checkpoints,
  and SQLite runtime files were removed with the stale run tree.

## Verification required

- Root directory allowlist check against the upstream tree and the four named
  feature directories.
- Predictor submodule URL and requested feature-branch check.
- Python import checks after path consolidation.
- Relevant Scheduler, Predictor, and pre-flight integration tests before the
  ABA experiment is launched.

## Verification result

- A fresh fetch of `InternScience/MLEvolve` main at
  `9c5c8a3b23f0361708b59a401452dddc00f97189` showed no missing upstream source
  directories.
- Excluding operational `.git`, `.cache`, `.venv`, and `records`, the only
  additional source directories are exactly `HWKD`, `localml_scheduler`,
  `nn-model-preflight-checker`, and `PerfSeer-predictor`.
- No upstream top-level file is missing. The only additional top-level regular
  file is `.gitmodules`, required for the two submodules.
- `CLAUDE.md` is archived under `records/project_documents`; the root
  `AGENTS.md` symlink now resolves to that archived document.
- A final root allowlist check passes: the only source additions beyond the
  upstream directories are exactly the four named feature directories, and
  no upstream source directory is missing.
