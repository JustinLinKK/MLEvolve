# SDD ledger — plan: records/2026-10-14_hwkd_impl_integration_plan.md

Pre-flight: Task 1 produces `HWKD.hwkg` for Task 2 MLEvolve and Task 3 ML-Master; both use the same canonical package name. Task 2 and Task 3 produce independent opt-in prompt paths for Task 4 verification.

Ruling: Keep the ledger in `records/`, not an isolated workspace — the user's project instruction says to work in the current folder and not create a sandbox — cost if wrong: skill-specific scratch tooling cannot be used for this run.

Task 1: Ruling: 38-byte identity conflicts with a real NVIDIA device misclassified as ROCm by the pinned source — keep 37 exact files plus a two-condition compatibility fix in `device_support.py`, preserve the exact Git source commit and graph digest — cost if wrong: a future source update needs conflict review.

Task 1: complete (import/provenance tests green; source component tests 598 passed, 24 skipped, 50 source-root-dependent deselected; graph hash verified; old staged HWKD checksum unchanged).

Task 2: complete (12 hook files match the pinned source after import-prefix substitution and blank-line normalization; MLEvolve model-dispatch tests pass in its own Python 3.14 environment; entrypoint imports).

Task 3: Ruling: two agents' historical top-level package names collide inside one pytest process — remove cached agent-local imports when switching test subjects, matching separate production processes — cost if wrong: shared-process embedding needs a separate namespace design.

Task 3: complete (ML-Master draft/improve/debug tests pass for text and steerable prompt forms; disabled byte identity and bad digest before model call verified; successful parent measurements forwarded; per-native-attempt dispatch/return/failure and candidate-return audits verified; entrypoint imports).

Task 4: Ruling: the 50 original source tests require original source-root `agents/`, `llm/`, or historical experiment files not installed here — report unfiltered failures and run the 598 relocatable component cases without fabricating those paths — cost if wrong: any behavior uniquely covered by excluded tests remains unverified.

Task 4: complete (17/17 root integration tests pass in each agent environment; 67 Python files parsed before final review changes; real RTX 5070 Ti graph/device smoke and three MLEvolve audit events verified; temporary test/audit files removed; results recorded).

Final review: complete. Read-only reviewer identified the two Task 3 gaps above; both were fixed with failing-first regression tests and the full 17-test suite re-run in each agent environment.
