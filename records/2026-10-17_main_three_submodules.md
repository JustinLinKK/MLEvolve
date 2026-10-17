# Three shared submodules on `main`

## Settings

- Parent target: `JustinLinKK/MLEvolve`, branch `main`.
- `HWKD`: `AYPatrL04/MLEvolve`, branch `hwkd-only`, commit `62d3c797235cb29301c2868a4c6336e38754f2df`.
- `Preflight_checker`: `JustinLinKK/nn-model-preflight-checker`, branch `main`, commit `db61a408684265311d92aedba17f2fd6e4368b75`.
- `PerfSeer-predictor`: `JustinLinKK/PerfSeer-predictor`, branch `main`, existing commit `747ccdd936362133c4b91a70b282b11d8a298a2f`.
- The ordinary `HWKD` and `Preflight_checker` trees were replaced by Git links in preceding commits. Their old contents remain in parent Git history; this change does not delete remote experiment data or checkpoints.

## Verification and result

- Initialized all three submodules in a separate clean worktree and checked their fixed commit identities.
- Removed the stale `.gitmodules` entry for the nonexistent `nn-model-preflight-checker` path; updated the README, deployment paths, and path assertions to use the actual `Preflight_checker` submodule.
- Confirmed that `Preflight_checker.preflight.check_source` and `HWKD.hwkg.routing.hardware_phase` import from the initialized submodules.
- Preflight static-check tests: 4 passed. A10 resume-path tests: 3 passed. Modified shell scripts passed `bash -n`; parent diff passed `git diff --check`.
- The old model-preflight integration test could not collect because this checkout lacks its historical `agents.review_contracts` package. A direct Predictor import also requires an external `perfseer` dependency not installed in the test environment; its existing Git link was not changed. These are not claimed as passing integration tests.

## Rollback

The parent commit immediately before the migration and the original ordinary directory contents remain in Git history. Restoring that parent commit reverses the Git-link conversion; fresh clones of the new parent commit require `git submodule update --init --recursive`.
