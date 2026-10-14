# Single-file preflight checker — 2026-10-13

## Settings and scope

- Consolidated the prior dataset, competition, modality, and TFRecord helpers
  into `Preflight_checker/preflight.py`; removed the four helper files. Renamed
  two overlapping helper names so the merged module preserves both dataset
  and competition validation paths.
- Preserved `check(path, cache)`, the command-line interface, and
  `check_source(source)`. The latter remains syntax-only for arbitrary agent
  code (`assurance: syntax_only`); it does not certify training or hardware.
- ML-Master and MLEvolve still use their existing optional
  `LIGHTNING_PLUGINS=preflight` execution hooks.

## Verification results

- Reconstructed the last experiment's `nodes/wrong_column/mlp.py` and its clean
  counterpart from the recorded final corpus generator and mutation definition.
  The buggy node's source 256-bit Secure Hash Algorithm (SHA-256) digest was
  `c2e6bf10cf2c40ac59dfade33de514896728bdfd04cf10481641a165079fc06d`.
- Before and after consolidation, checking the buggy node without its deleted
  dataset gave `['missing_file', 'wrong_column']`. With the original 32-row comma-separated values (CSV)
  fixture temporarily regenerated from that generator, the clean node gave
  `[]` and the buggy node gave exactly `['wrong_column']`.
- Focused parity checks matched the former helpers for declared dataset kinds,
  a competition missing-file case, an image-modality case, and a TFRecord
  checksum. Direct-file and package-style entry points worked. With the four
  helpers removed, both agents rejected invalid syntax and allowed valid code
  through their existing hooks.

## Conclusion and limit

The requested single-file layout and the sampled old-corpus behavior are
verified. The original 482-node corpus was not rerun because the saved node and
fixture directory was deleted. No generated node training was executed; these
checks do not prove arbitrary model-training correctness.
