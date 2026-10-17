# HWKD standalone submodule migration

## Requested source and scope

- Source: `git@github.com:AYPatrL04/MLEvolve.git`, branch `hwkd-only`, verified remote head `62d3c797235cb29301c2868a4c6336e38754f2df`.
- Parent path: `HWKD`. The parent Git index now has one `160000` gitlink at that exact commit, and `.gitmodules` pins the requested branch. No running remote experiment or baseline was changed.
- The previous ordinary tracked directory was moved intact to `.cache/HWKD_pre_submodule_20261017_0439` before changing the parent index. This is the rollback copy; preserve it until the migration is reviewed. The old directory held 39 tracked files and was 1.7 MiB. Its apparent local modifications included executable-bit changes on all 39 files and a 42-line graph delta; no material was silently discarded.

## Compatibility and verification

- Existing agent call sites import `HWKD.hwkg.*`, which still resolves through the new submodule's `hwkg` package. A fresh import of `hardware_guidance` and the routing decorators passed. With no graph configured, `hardware_guidance("draft")` returned `None` as expected.
- The new graph SHA-256 is `91e52fd2f7299b24c782f63f600ed4e2fa429a7159f9743d912b1dcf922efd3a`, matching the branch's pinned configuration. It has 115 nodes and 1,343 edges. Any future enabled run must use this new digest, not the old local graph digest `d5af9de663a304855624b4736d35ea28a6fc494a5d68cb24dab96de4e78014ba`.
- The branch deliberately omits historical release notes, workstation calibration JSON, and a task fixture that were in the prior directory. It also differs in `hwkg/device_support.py`: the prior local copy required an AMD architecture name to start with `gfx`, while the source branch treats any nonempty `gcnArchName` as ROCm. The old copy remains available for review and rollback. No claim of byte-identical behavior is made for those paths.
- Focused tests: `localml_scheduler/tests/test_mlevolve_hwkd_ablation_config.py` and `test_hardware_features.py` passed, **11/11**. `git diff --check --cached` passed. Two older `root_tests` could not be collected because their unrelated `agents` package is absent in this checkout, so no full-suite pass is claimed.
- The submodule checkout itself is clean on `hwkd-only`; no new Python package was installed and no GPU experiment was run. The parent change is staged but uncommitted and unpushed. Existing unrelated parent changes were left alone.

## Reversibility

The pre-migration tree is retained at `.cache/HWKD_pre_submodule_20261017_0439`, and the original tracked tree remains available in parent Git history. Do not delete the rollback copy or contract the older calibration/documentation assets without a separate decision. A clean checkout of the new parent state must initialize submodules before running the hardware-knowledge plugin.
