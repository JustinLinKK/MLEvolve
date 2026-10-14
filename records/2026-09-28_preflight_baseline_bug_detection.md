# Preflight checker evaluation on all recorded baseline nodes

## Scope

- Checker: `nn-model-preflight-checker`, using its static-source fallback for
  generated candidates that do not implement `CandidateAdapter`.
- Baseline journals:
  - `.cache/petfinder_a100_a10_live/baseline_final_50_20260902/journal.json`
  - `.cache/petfinder_terra_a100_20260905/upstream_60_mpsuuid/journal.json`
- Evaluation unit: every unique non-root node with non-empty saved source code.
- Ground truth: a node is buggy only when execution recorded a non-empty
  `exc_type`. This follows the project definition that a buggy node must have
  produced a runtime error. A confirmed checker error counts as a rejection;
  warnings and inconclusive results do not.
- Target hardware profile: the A10 execution environment used by these
  baseline experiments.

## Source-code preservation

The two journals preserve generated source code for every evaluated node:

| Baseline journal | Nodes with saved code |
|---|---:|
| September 2 | 108 |
| September 5 | 52 |
| **Combined** | **160** |

The durable replay artifact is
`records/2026-09-28_preflight_all_baseline_nodes.json`, and the reproducible
evaluator is `nn-model-preflight-checker/scripts/evaluate_baseline_journal.py`.

## Checker coverage added

The checker now detects the concrete failure classes found in the recorded
baseline programs, including dependency and local-repository failures, invalid
TorchVision/timm/scikit-learn application programming interface usage, tensor
rank/dtype/indexing errors, test-target leakage, undefined or definition-order
errors, incompatible backbone dimensions, invalid model identifiers, unsafe
mixed-precision scaler usage, and A10-specific memory/runtime/numerical-risk
configurations observed in the baseline.

Safe counterpart cases are covered by regression tests so that these rules do
not reject the recorded non-error nodes.

## Results

| Corpus | Nodes | Runtime-error nodes | Errors caught | Recall | Non-error nodes | False rejects |
|---|---:|---:|---:|---:|---:|---:|
| September 2 | 108 | 75 | 75 | **100%** | 33 | **0** |
| September 5 | 52 | 25 | 25 | **100%** | 27 | **0** |
| **Combined** | **160** | **100** | **100** | **100%** | **60** | **0** |

Three September 2 entries have `is_buggy=true` but no `exc_type`; they completed
without an uncaught runtime error and are therefore counted as non-error nodes,
not bugs. This prevents inflating the ground truth with a legacy metadata flag.

## Verification

- Full checker test suite: `78 passed`.
- Full journal replay: `100/100` runtime-error nodes rejected.
- False rejections: `0/60` recorded non-error nodes.
- Patch whitespace validation: passed.

The result establishes complete detection for the two recorded baseline
corpora under their A10 profile; it is not a claim that static analysis can
detect every possible future Python or machine-learning failure.
