# HWKD v8.19 — catalogue-wide capability routing

The user requested promoting the RTX5080-specific implementation to all listed
GPUs and renaming the implementation branch to `hwkd-impl`.

## Resulting behavior

- All 75 hardware profiles share runtime-gated precision and optimizer trial
  routing: 67 NVIDIA/CUDA profiles and 8 AMD/ROCm profiles.
- Initial context uses the actual architecture, backend, installed framework and
  per-GPU memory. Parent advice retains stage routing, failure suppression,
  train-only fitting reminders, source/precision scope and negative trial memory.
- BF16 needs native runtime support; NVIDIA additionally requires its native
  graph edge. FP16 requires AMP and GradScaler. Fused AdamW requires a graph
  trial edge, the API parameter and a compiled backend kernel registration.
  These checks do not execute training or make extra provider calls.
- ROCm identity uses the `gfx` target and HIP runtime; CUDA targets remain `sm_*`.
  Driver-identical GDDR6/GDDR6X variants use their shared feature intersection and
  do not receive variant-specific bandwidth or numerical projections.
- Multi-GPU platform profiles refer to one visible GPU's target and memory;
  aggregate memory is never presented as one allocation. Execution remains native.
- The precision calibration reader and standalone runner accept other GPUs;
  owned optimizer records have a separate optional path. Records retain exact
  host/runtime/recipe matching; existing RTX5080 measurements are unchanged.

Graph SHA256: `ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.

## Validation scope

The new test matrix covers every catalogue row at policy and actual component
dispatch boundaries, parent trial selection, unsupported capability gates,
ROCm identity/version mismatches, driver-identical variants and owned calibration
matching. Synthetic calibration records in tests are fixtures, not measurements.
Full component regression, native hook equivalence and protected baseline checks
are recorded independently before publication. Cross-GPU physical performance
validation is pending. The completed 30+30 study remains v8.18.

Publication checks passed: 671 offline component tests plus 4 subtests, one local
GPU calibration-identity check, 26 native/public-task checks, 194 protected-file
hashes, 76 exact upstream files and 12 AST-equivalent hooks. The four existing
calibration files retain their original SHA256 identities.

All edits are limited to `hwkg/` and component-owned tests/documentation.
Baseline execution, scoring, search, SDK/API parameters, parser, generated
candidates and common adapters are unchanged. Prior frozen roots are unchanged.

## API sources

- [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.14/notes/hip.html)
- [PyTorch AMP](https://docs.pytorch.org/docs/2.14/amp.html)
- [Native BF16 capability API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.is_bf16_supported.html)
- [AdamW implementation options](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
