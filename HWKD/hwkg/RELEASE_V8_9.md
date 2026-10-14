# HWKD v8.9 snapshot

This branch contains the original InternScience/MLEvolve baseline at
`9c5c8a3b23f0361708b59a401452dddc00f97189`, plus the optional HWKD/HWKG
component, its own tests/documentation/calibration records, and twelve
imports/decorators whose removal yields the original AST. Native baseline
execution, scoring, search, parsing, SDK/API settings and generated candidates
are unchanged. No common behavior adapter is added.

The current policy selector is `MLEVOLVE_HWKG_EVIDENCE_POLICY=v8`; its
implementation revision is v8.9. See [component instructions](README.md) for
graph/host/calibration settings. Hardware advice is conditional, candidate
and stage scoped, and audited separately from source adoption, execution and
measured performance. The disabled component is a no-op.

v8.8 checks the actual provider prompt for repeated host facts, rather than
the intermediate task description produced by the original cleaner. v8.9
extends recipe syntax auditing to parameter-group LR constants, tuple batch
settings and custom Module loss definitions, while avoiding classification
of worker-only or ordinary model-layout changes as quality recipe changes.
Syntax auditing does not certify complete semantic/dataflow equivalence.

Validation on the recovered workstation:244 offline component tests plus4
subtests,12 native compatibility checks, and194 protected hashes passed. One
existing live-GPU calibration-identity check remains pending because the
local NVIDIA driver/device access failed. No new performance test or gain
is claimed for v8.9. The partial v8.8 run produced a numerical85.90% result
that exceeds its six-epoch training budget; it is excluded from valid gains.

Some regression tests optionally reference historical local source/prompt
fixtures and skip when those records are absent. Experiment roots, complete
archives, datasets, pretrained weights and credentials are intentionally kept
out of this branch. Bundled calibration JSON is component microcalibration,
not task-quality or whole-pipeline evidence. The autonomous test campaign is
user-paused.
