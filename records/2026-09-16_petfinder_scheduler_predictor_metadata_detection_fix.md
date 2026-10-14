# PetFinder Predictor metadata detection repair

## Scope

Scheduler-only PetFinder run on one Nautilus A10, with the local Qwen-3.8-27B
INT8 agent served on Nautilus A100.  The launch uses seed `42`, disables the
Hardware Knowledge Database, leaves `parallel_job_cap=null`, and requires a
Predictor v3.2 estimate for admission.

## Observed failure

Candidate `8d2b9f800bcd4920b44baaf179665d52` used the real multimodal contract
`DualBranchModel(images, tabular)`, with
`transforms.Resize(256)` followed by `transforms.CenterCrop(224)`.  Its
execution never reached the A10 because the scheduler recorded
`job_rejected_ml_prediction` with `source=branch_profile` and a null error.

The immediately following candidates with tuple-style image transforms did
produce Predictor v3.2 estimates and were admitted.  In particular, the live
run dispatched `e9dbb418-1d46-486f-bb4c-6706dc077069` and then placed
`b5a22f1f-717f-4186-a9d5-3c8d68f73c98` concurrently using the CUDA-process
backend.

## Root cause

`detect_input_resolution` recognized `Resize((H, W))` but not scalar
`Resize(H)` or `CenterCrop(H)`.  Consequently the executor supplied no
`perfseer_model` metadata for this valid candidate; Predictor v3.2 could not
construct the model specification.  A second reporting defect built the model
specification outside the Predictor exception boundary, so the scheduler
recorded the rejection with a null error rather than the missing-input-contract
diagnostic.

## Repair and verification

1. Geometry detection now recognizes scalar `Resize` and `CenterCrop`, using
   the last geometry transform as the delivered square resolution.  The actual
   `Resize(256) -> CenterCrop(224)` pattern therefore yields `224`.
2. Predictor V3.2 now records model-specification failures in `last_errors`
   before required-ML admission rejects the candidate.
3. Regression coverage: `60 passed, 2 subtests passed` for
   `tests/test_script_introspection.py`, `tests/test_perfseer_model_metadata.py`,
   and `localml_scheduler/tests/test_ml_prediction.py`.
4. A separate, inactive Nautilus A10 staging checkout was verified against the
   saved real candidate.  It now produces the Predictor contract
   `[["$batch", 3, 224, 224], ["$batch", 12]]` without allocating the A10.

The already-running remote controller intentionally remains unchanged.  This
repair must be staged before the next controlled run; it is not claimed as a
retroactive fix for its in-flight candidates.
