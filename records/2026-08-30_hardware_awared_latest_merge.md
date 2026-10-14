# Hardware-aware latest-branch integration

Date: 2026-08-30

## Inputs

- Target branch: `hardware-awared`
- Latest remote base: `origin/hardware-awared` at `dcb0611`
- Local work safety snapshot: `stash@{0}` (`codex-pre-latest-merge-2026-08-30`)

## Integration decisions

- Fast-forwarded the branch base to `origin/hardware-awared`, then replayed the local workspace changes.
- Kept the latest canonical scheduler/runtime surface and removed the conflicting legacy backend-mode migration test.
- Kept local preflight admission, lesson-profile context, physical CUDA-device mapping, and experiment assets.
- Kept `parallel_job_cap` unset. Incremental admission remains responsible for deciding safe concurrency.
- Restored the latest comparison script and canonical module locations required by the remote test contract.

## Verification

- `git diff --check`: passed.
- Changed Python files parsed successfully with the Python abstract syntax tree parser.
- Full suite on an isolated Python 3.14 / CUDA 13 verification environment:
  - `630 passed`
  - `4 skipped`
  - `8 subtests passed`
  - `0 failed`
  - elapsed time: `263.89 s`
- The temporary verification environment was deleted after the run.

## Environment diagnosis

The repository's older `.venv` contains PyTorch `2.6.0+cu118`, which cannot execute kernels for the local RTX 5070 Ti (`sm_120`). The system CUDA stack and the isolated verification environment both executed an actual CUDA tensor kernel successfully. The earlier worker failures were therefore environment failures, not scheduler regressions.

## Conclusion

The local workspace is integrated on top of the latest `hardware-awared` remote base, with the latest canonical scheduler behavior and the compatible local features preserved. The resulting source passes the complete test suite in a CUDA-compatible environment.
