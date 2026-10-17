# Lightning Agent

This repository keeps two machine-learning agents under `Agent/`:

The shared components live at the repository root: `HWKD/` (Hardware Knowledge
Database), `Preflight_checker/`, and `PerfSeer-predictor/` are Git submodules;
`localml_scheduler/` belongs to this repository. Initialize the dependencies
after cloning with `git submodule update --init --recursive`.

ML-Master plugins are opt-in through `LIGHTNING_PLUGINS`, e.g.
`LIGHTNING_PLUGINS=preflight,scheduler,predictor,hwkd`. The preflight adapter
performs static-source and hardware checks only; arbitrary generated scripts
do not provide the model/fixture contract needed for a full CPU check. The
preflight target must be set explicitly with `LIGHTNING_GPU_PROFILE`; guessing
a GPU would make hardware checks misleading. The

Predictor adapter needs an explicit model entry class and JSON input shapes
(`LIGHTNING_PREDICTOR_ENTRY` and `LIGHTNING_PREDICTOR_INPUT_SHAPES`) before it
can estimate a candidate. Its predictions are recorded, not yet used for
scheduler admission. Hardware guidance requires the isolated HWKD branch's
pinned graph environment variables. Scheduler execution uses an opaque script
runner and does not permit unsafe checkpoint-dependent co-location trials.

# Experiments

- Make sure you agree to consents on Kaggle before experiments
