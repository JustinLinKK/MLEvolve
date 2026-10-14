# PetFinder Predictor v3.2 Scheduler Integration Repair

## Scope

Target experiment: ten budget-counted PetFinder nodes on one NVIDIA A10 with
the MLEvolve scheduler, Predictor v3.2, and the pre-flight checker.  Hardware
Knowledge Database (HWKD) is disabled.  No fixed parallel-job limit is used.

## Defects repaired

1. Scheduler submissions omitted the static model contract needed by Predictor
   v3.2.  `engine/executor.py` now supplies a `build_model` entry point and
   PetFinder image/tabular input shapes when those facts are statically known.
2. Predictor v3.2 source capture restored only CPU random-number-generator
   state.  Dropout made strict CUDA graph replay fail.  The v3.2 capture source
   now restores both CPU and CUDA random-number-generator state.
3. Source capture had no configurable CUDA path.  Scheduler prediction now
   accepts `v32_capture_device=cuda`, moves the model and all nested inputs to
   CUDA, and returns both estimated epoch time and peak video-memory use.
4. The active legacy controller stores high-frequency SQLite write-ahead-log
   files on Ceph.  It showed `ceph_mdsc_wait_request` / page-write waits.  The
   next-run launcher uses `/dev/shm` for scheduler and pipeline SQLite state,
   then persists the final databases and trace to the durable run directory.

## Verification

| Check | Result |
| --- | --- |
| Local scheduler bridge, Predictor configuration, and lifecycle tests | `32 passed, 2 subtests passed` |
| A10 strict CUDA dropout replay | passed |
| A10 Predictor v3.2 worker with CUDA source capture | passed; returned positive epoch-time and peak-video-memory estimates |
| Legacy run decision metadata | invalid for Predictor evaluation: all submitted jobs used `branch_profile`, with no `perfseer_model` contract |

## Active legacy trace status

At 2026-09-11 05:32 UTC the legacy run had nine completed valid nodes and two
runtime-buggy nodes.  A twelfth candidate was rejected by pre-flight for a
confirmed matrix-dimension mismatch and was not counted.  This trace is kept
for debugging only; it must not be presented as a Predictor v3.2 performance
result because its decisions fell back to branch profiles.

## Reproducible next run

`deployments/run_petfinder_scheduler_v32_a10.sh` launches the valid experiment
after the active controller is terminal.  It specifies `ml_predictor`, the
Predictor v3.2 source and weights, CUDA capture, `hardware_knowledge.enabled=false`,
pre-flight checking, and scheduler-owned live admission without a job-count cap.
