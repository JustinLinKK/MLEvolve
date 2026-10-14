# Single-file preflight class refactor — 2026-10-13

## Settings and scope

- Refactored `Preflight_checker/preflight.py` as one module organized around four workflow classes: `DatasetValidator`, `CompetitionValidator`, `NodeValidator`, and `PreflightApp`. Kept small stateless TensorFlow Record and syntax helpers outside the classes.
- Dataset formats and competition row rules are methods on their validators; the controlled model-node rules are methods on `NodeValidator`; command-line modes belong to `PreflightApp`. The number of top-level functions fell from 68 to 33.
- Preserved the public `check(path, cache)`, `check_source(source)`, `DatasetContract`, `dataset_issues`, checker registration, report shapes, finding codes, and command-line exit semantics. Did not run candidate training code.

## Verification

- Rebuilt the last experiment's synthetic corpus from its recorded generators, with the generator's TensorFlow Record checksum import redirected to the consolidated module. The rebuilt manifest contains 482 nodes: 22 clean and 460 controlled buggy nodes.
- Before and after the refactor, checking every node produced the same findings-map 256-bit Secure Hash Algorithm (SHA-256) digest: `76fddb5a2818bcace4efdb4a8a2520d20d23c4763e9642eab8ca83ba20f9eea4`. All 460 buggy nodes were flagged; all 22 clean nodes had no findings.
- Direct competition and dataset command-line checks returned clean or expected findings on generated fixtures. A deliberately expired competition deadline returned `audit_incomplete` and exit code 2.
- The ML-Master plugin admitted valid Python, rejected malformed Python, and wrote reports in a temporary workspace; MLEvolve's `check_source` import returned the same decisions. The temporary workspace was removed after the test.

## Conclusion and limits

The single file now has four stateful workflow classes, with the original public entry points and controlled-corpus behavior preserved. This is a static check against declared contracts, not a proof that arbitrary training code or real competition datasets will run correctly. The rebuilt corpus is temporary verification data, not retained project data.
