# Pyright source preflight (2026-10-14)

## Scope and settings

- Kept the original `check_source_syntax_only` behavior available in `Preflight_checker/preflight.py`; the committed parent revision is the complete previous version.
- `check_source` first parses syntax, then asks the Pyright language server to analyze an in-memory candidate using the caller's workspace and Python interpreter. It never executes the candidate.
- The server has an eight-second default deadline. Missing, failed, or timed-out analysis retains syntax-only admission but records `analysis_status` and a visible diagnostic; it is not represented as a Pyright pass.
- Reject only selected error-severity diagnostics: missing imports, undefined/unbound names, and selected call, argument, or attribute errors. Other diagnostics are retained as advice in the report.
- ML-Master and MLEvolve pass their actual working directories to the checker. Their existing rejected-candidate paths return the diagnostic text to the agent and persist the JSON report.
- Install the server in the repository cache with `npm install --prefix .cache/pyright --no-audit --no-fund pyright@1.1.414`, or set `PYRIGHT_LANGSERVER` to an installed `pyright-langserver` executable. The checker also searches `PATH`.

## Verification

- Local test environment: Python 3.12.13, Pyright 1.1.414, `pytest` 9.1.1.
- `Preflight_checker/test_source_pyright.py`: eight tests passed. These cover syntax-only compatibility, valid source, undefined names, missing imports, unavailable server, syntax failure, ML-Master report persistence, and MLEvolve's `PreflightRejected` feedback. The undefined-name test proves that a candidate file-creation statement was not executed.
- An additional call without `PYRIGHT_LANGSERVER` used the cached server and rejected `print(missing_name)` with `reportUndefinedVariable`.
- Compilation of the three changed Python modules and `git diff --check` passed.

## Limits

- This is static analysis, not a training correctness claim. It does not prove package runtime compatibility, object picklability, absence of process races/deadlocks, memory sufficiency, or valid loss trajectories.
- In the earlier five-node Google Quest run, the `transformers`/`kernels` runtime incompatibility and multiprocessing failures would not be reliably caught by Pyright alone. The generated source files from that run were not available here for a direct replay, so verification used controlled regression candidates rather than claiming those nodes now pass or fail.
- Missing-import conclusions assume the preflight interpreter and candidate worker environment match. A different remote environment requires matching configuration or treating that result as advisory.
