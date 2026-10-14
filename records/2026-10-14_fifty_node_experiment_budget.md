# Fifty-node full-experiment budget (2026-10-14)

## Rule

New full experiments use 50 generated nodes total, not 50 successful nodes. A generated node remains in the count when it is preflight-rejected, fails during execution, or receives a buggy result. The virtual root is not a generated node. A failure before a draft exists is not counted as a node.

## Settings changed

- ML-Master's default `agent.steps` and the MLEvolve scheduler example default are now 50 instead of 500.
- The A100 scheduler comparison and preflight-only launchers now default to 50, matching the A100 baseline; the preflight-only launcher also enforces that number in its existing guard.
- The A10 seeded baseline full-run launcher now defaults to 50.
- Historical ten-node gates, diagnostic smoke runs, and existing experiment records were not rewritten. Other launchers retain their explicit diagnostic override mechanisms; callers must not use those overrides for a full experiment.

Both normal runners advance `completed` from `len(journal) - 1`, so persisted buggy/preflight-rejected nodes consume the budget as they do successful nodes. The checker does not launch or train any node.

## Verification

- Ten focused Pytest checks passed, including the new 50-node configuration and journal-count assertions plus the Pyright integration checks.
- Ten existing launcher assertion functions passed when invoked directly. Their package-level Pytest collection was unavailable in the focused virtual environment because importing the entire scheduler package requires PyTorch; no GPU package was installed just to run these text-only assertions.
- Shell syntax checks passed for the three changed launchers.
- No experiment was launched and no historical results were changed.
