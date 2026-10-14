# Hardware-Aware Latest-Branch Integration

## Source and integration method

- Local base: `ede4705`
- Remote target: `origin/hardware-awared` at `420e989`
- Isolated three-way integration commit: `a434981`
- Comparison clone: `../MLEvolve-upstream-420e989`

The remote update was first cloned into the comparison directory. The current
tracked worktree changes were reconstructed there as a temporary commit, then
merged with the remote target. Only the resolved delta from that temporary
commit to the merge commit was applied to this working tree. No local file was
overwritten and the pre-existing `PerfSeer-predictor` submodule state was not
altered.

## Resolved overlaps

- Preserved both the local lesson-profile prompt context and the incoming CUDA
  documentation context in draft, improve, debug, review, and stepwise agents.
- Adopted the incoming authoritative `packing_backend` mode and MPS/CUDA
  process backend migration, including CUDA Model Context Protocol documents,
  cache, router, and bridge code.
- Restored the incoming Nautilus Job manifests that were locally deleted while
  retaining the local RTX 5090 pressure benchmark that upstream deleted.
- Removed `parallel_job_cap` from the example scheduler configuration and from
  production placement decisions. Legacy deserialization remains supported but
  cannot cap incremental admission.
- Updated placement-replay tests to finalize a staged observation only after
  the simulated jobs complete, matching the scheduler lifecycle.

## Verification

Command:

```text
pytest -q localml_scheduler/tests/test_backend_mode_contract.py \
  localml_scheduler/tests/test_backend_mode_migration.py \
  localml_scheduler/tests/test_process_backends.py \
  localml_scheduler/tests/test_decision_replay.py \
  localml_scheduler/tests/test_time_aware_scheduler.py \
  localml_scheduler/tests/test_cuda_docs_schema_roundtrip.py \
  localml_scheduler/tests/test_cuda_mcp_backend_contract.py
```

Result: `122 passed`.

The broader hardware-context test remains environment-blocked because the
system Python lacks the existing project dependency `backoff`; no dependency
or CUDA installation was changed during this integration.
