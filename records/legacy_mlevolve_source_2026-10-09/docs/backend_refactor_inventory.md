# Backend refactor baseline and inventory

- Reviewed branch: `hardware-awared`
- Baseline commit: `8ca263a85347bb52176aa022b119114d9453aca2`
- Baseline targeted tests: 25 passed in 1.10 seconds
- Command: `pytest -q localml_scheduler/tests/test_persistent_stream_backend.py localml_scheduler/tests/test_backend_aware_trial_ranking.py localml_scheduler/tests/test_client_surface.py localml_scheduler/tests/test_feature_filter.py`

The pre-change inventory found retired runtime/config/planner behavior in
`config/models.py`, `execution/backends.py`, `execution/backend_registry.py`,
`execution/stream_host.py`, `scheduler/backend_aware_planner.py`,
`scheduler/backend_compatibility.py`, `scheduler/trial_candidate.py`,
`scheduler/trial_priority.py`, `scheduler/trace_simulator.py`, active examples,
and scheduler tests. Prompt/backend inference was in
`agents/hardware_context.py`; backend-soft evidence retrieval was in
`client.py` and `graph_knowledge.py`; vector list filtering was in
`code_knowledge/store.py`.

Repository benchmark outputs, replay captures, and trace archives also contain
historical identifiers. They are retained as historical evidence and are not
part of the active runtime configuration or planner action space.

## Implementation verification

- Removed active identifiers: `stream`, `cuda_stream`, `mps_stream`, and
  `stream_mps`; legacy `mps` is read only as a deprecated configuration alias
  or as provenance-gated migration input.
- Final maintained suites: `443 passed` from
  `pytest -q tests localml_scheduler/tests scheduler_benchmark_test`.
- Final prompt/schema/contract focus: `77 passed` from the hardware-context,
  prompt-preview, graph-validation, backend-contract, and guidance suites.
- `compileall` and `git diff --check` completed successfully.
- The final active-source scan found retired names only in rejection constants,
  actionable validation messages, migration/history support, and ADR source
  references. It found no active execution, planning, prompt, or knowledge rule.
- The controlled migration fixture reported one provable `mps` profile, one
  retired `stream` profile, one schema-v2 rekey, one `stream_mps` configuration
  reference, one `cuda_stream` cache entry, and one historical stream event.
  The first execute changed three rows and the second execute changed zero.
