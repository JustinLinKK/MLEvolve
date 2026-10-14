# Pinned HWKD integration: settings, results, conclusions — 2026-10-14

## Scope and source

- Source: `AYPatrL04/MLEvolve` `hwkd-impl` at
  `a25c6413b5181cc9043cae1a150b16606137bc2f`.
- Installed 38 `hwkg/` files under `HWKD/hwkg/`. The graph's SHA-256 is
  `ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.
- Of those 38 files, 37 match the source Git blobs byte-for-byte. The only
  local delta is `HWKD/hwkg/device_support.py` (local SHA-256
  `79e49dffbe650021b2552ad66583bf83cd98215eee0a423b43f479f017456eca`).
  It treats `gcnArchName` as a Radeon Open Compute (ROCm) architecture only
  when the value begins with `gfx`. On PyTorch 2.11, NVIDIA's device property
  unexpectedly holds the plain device name in that field; the pinned source
  otherwise reported an NVIDIA GeForce RTX 5070 Ti as `unknown, ROCmNone`.
  The local two-condition patch restores `sm_120, CUDA13.0`. It does not alter
  the graph, evidence policy, or calibration data. Regression tests cover
  both NVIDIA-name and AMD-`gfx` values.
- All 12 MLEvolve hook files match the pinned source after substituting the
  `HWKD.hwkg.routing` import path and ignoring blank-line-only differences.
  `Agent/MLEvolve/run.py` adds only a project-root import bootstrap.
- ML-Master's one-time task enrichment is no longer used at startup. Its
  plugin is passed to the agent and consulted at draft, improve, and debug
  prompt construction. It still requires `LIGHTNING_PLUGINS=hwkd`.

## Verification performed

| Check | Result |
| --- | --- |
| Root integration tests in MLEvolve's Python 3.14 virtual environment | 17 passed; 17 dependency deprecation warnings |
| Same tests in ML-Master's Python 3.14 virtual environment | 17 passed; 1 dependency deprecation warning |
| Relevant relocated source HWKG tests | 598 passed, 24 skipped, 4 subtests passed; 50 source-root-dependent cases deselected |
| Initial unfiltered source suite | 598 passed, 24 skipped, 50 failed; failures were missing original `agents/`/`llm/` paths or unavailable historical experiment files in this relocated checkout |
| Python syntax parsing | 67 package/agent/test files parsed |
| MLEvolve and ML-Master entry imports | Both imported in their own virtual environments |
| Real local device | One NVIDIA RTX 5070 Ti, compute target `sm_120`, PyTorch 2.11.0+cu130; one-element GPU add returned 2.0; graph/device check produced `CUDA13.0` guidance |
| Offline provider dispatch on that device | One provider call, one advisory block; audit recorded `context_built`, `provider_dispatch`, `provider_return` |
| Offline ML-Master draft on that device after review | Real graph/policy plus stubbed native model response produced a draft node; prompt audit recorded `context_built`, dispatch audit recorded `provider_dispatch`, `provider_return`, `candidate_return` |
| Existing staged HWKD state | Name/status checksum `38be633c965954c332f694d10beb2c15fb261d095f55c05f25753fd7d98f0b29` before and after |

The local suite specifically checks disabled/no-graph prompt identity for
ML-Master's text, Qwen-style and DeepSeek-style prompt branches; stage-specific
draft/improve/debug content; successful parent measurement forwarding; bad graph digest failure before a model call;
MLEvolve's one-block/one-call dispatch and disabled exact prompt; source
graph digest; and the device-property compatibility fix. Temporary source test
files and temporary dispatch audit files were removed after verification.

An independent read-only review found two missing ML-Master links. They were
repaired and covered by red/green regressions: improve now passes parent
execution output (including successful measurement windows) to the v8 policy,
and ML-Master emits `provider_dispatch`, `provider_return` or
`provider_failure` for each native query attempt, followed by a separate
`candidate_return` after plan/code parsing. The audit records hashes and
identifiers rather than full prompts or task data. A controlled two-attempt
query produced two dispatches, two returns, one candidate event and the same
trace identifier; a controlled provider exception produced dispatch plus
failure without a candidate event. These events prove local backend-boundary
delivery, not remote service acceptance or adoption. The source package's
prompt `context_built` record remains distinct.

MLEvolve's virtual environment was created with system site packages using
Python 3.14.4 from `/opt/miniconda3/bin/python`. Only the missing pinned
startup dependencies were installed there: `backoff`, `coolname`,
`dataclasses-json`, `funcy`, `genson`, and `shutup` (plus their resolver
dependencies). The old Lightning-Agent root `.venv` uses PyTorch 2.6.0+cu118
and explicitly warns that it does **not** support this host's `sm_120` target;
do not use that interpreter for this device. No model weights or data were
downloaded.

## Rulings and remaining limits

- Source-byte fidelity versus truthful device identity: keep the exact source
  commit and graph as provenance, but carry the smallest local
  `device_support.py` compatibility fix. Cost if wrong: future upstream
  updates could conflict with this one-file delta, so re-run both device
  regression tests and blob comparison before updating.
- The 50 repository-shape/history cases from the original source test suite
  are not relocated or fabricated merely to make a green count. Cost if wrong:
  a behavior covered only by one of those cases may remain unverified here;
  local integration and the remaining source component tests are the current
  evidence.
- No full candidate training, quality comparison, throughput measurement, or
  multi-GPU proof was performed. Source advice is conditional. The local GPU
  smoke proves dispatch and device identity, not a performance gain.
- Existing old staged HWKD files, unrelated working-tree changes, tokenized
  data, and checkpoints were not reset or deleted. No remote push or new GPU
  Job was performed.
