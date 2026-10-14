# Lightning Agent repository restructure — 2026-10-09

## Source and layout

- `Agent/MLEvolve/`: exact file snapshot of `InternScience/MLEvolve` at
  `9c5c8a3b23f0361708b59a401452dddc00f97189` (76 tracked files). A
  recursive comparison with the shallow upstream checkout showed no differences.
- `Agent/ML-Master/`: upstream `sjtu-sai-agents/ML-Master` `main` at
  `6a7bf550074d09162fa955adcc4ed81017c0e697` (123 imported files), plus
  local plugin hooks and adapters.
- Root `HWKD/`: isolated hardware knowledge graph code from
  `AYPatrL04/MLEvolve` branch `hwkg-isolated-vs-upstream-s42` at
  `a7d3ac6cc5588ab1aba091ad56dee683486787c6`. Graph SHA-256:
  `0d422cfe81e2531980204e71b49fa9e5ffed9c2606a9a5569e7a43ebfc0078b4`.
- Root `PerfSeer-predictor/`: Git submodule on `main` at
  `747ccdd936362133c4b91a70b282b11d8a298a2f`.
- Root `nn-model-preflight-checker/`: existing submodule at
  `57f199c5446ba5c36f509fd8a808e67848853919`.
- Root `localml_scheduler/`: existing scheduler, with the script introspector
  copied inside its adapter package to remove a dependency on customized
  MLEvolve source.
- Previous customized MLEvolve source, hardware knowledge database, and
  feature-branch Predictor checkout are preserved in `records/legacy_*_2026-10-09/`.
  Existing experiment records and model files remain in root `records/` and
  `.cache/`.

## Plugin behavior

`Agent/ML-Master/main_mcts.py` now calls `PluginManager` for task context and
candidate execution. Plugins are opt-in through `LIGHTNING_PLUGINS`:

- `preflight`: static source and target-hardware checks, with a required
  explicit `LIGHTNING_GPU_PROFILE`. It cannot perform a full representative
  CPU training check without a candidate adapter and task fixture.
- `scheduler`: submits generated scripts to the root scheduler and returns an
  ML-Master execution result. It uses no fixed parallel-job cap. Opaque scripts
  are not admitted to checkpoint-dependent co-location trials.
- `predictor`: the main-branch PerfSeer student model predicts source-model
  resource metrics when `LIGHTNING_PREDICTOR_ENTRY` and
  `LIGHTNING_PREDICTOR_INPUT_SHAPES` are set. It records predictions but does
  not yet feed them to scheduler admission for arbitrary generated scripts.
- `hwkd`: injects the isolated graph's verified guidance only when its pinned
  graph environment variables and one matching visible GPU are available.

## Local verification

- Syntax compilation passed for plugin files, ML-Master entry point, and HWKD.
- ML-Master entry point imported successfully in its local Python 3.14 virtual
  environment after installing only small missing import dependencies.
- Preflight returned `PASS` with no diagnostics for a simple legal script
  using the explicit A10 profile. This is an adapter smoke test, not a real
  model validation.
- A syntactically invalid script was rejected as `PreflightRejected`; the
  fallback execution callback was not invoked.
- Predictor main `student_A10.pt` returned six numeric predictions for a
  one-layer `Net` source; e.g. predicted `train_time=3.2727` in the model's
  A10 label units. This is a pipeline smoke test, not accuracy validation.
- Scheduler executed `print(123)` and the manager's combined preflight plus
  scheduler path executed `print(456)`, each returning `exc_type=None`.
- The scheduler's raw-script runner test file passed: 15 tests, 0 failures.
- The scheduler service stopped after each test; no worker was left running.
- No full ML-Master search or target-GPU experiment was run. A local Neo4j
  mirror was unavailable, so the scheduler used its SQLite primary store.

## Issues fixed during verification

1. The scheduler worker initially could not resolve the relocated MLEvolve
   `utils.seed` module. Its worker Python path now includes original MLEvolve.
2. Its raw-script runner also depended on a customized MLEvolve
   `engine.script_introspection`; the introspector now lives in the scheduler
   adapter package.
3. Cache warming tried to unpickle Python source as a checkpoint. Opaque
   script jobs are now excluded from cache warming unless given an explicit
   preload source.
4. The service initially had a startup-heartbeat race; the plugin waits for
   readiness before submitting jobs.

No Git push was made.
