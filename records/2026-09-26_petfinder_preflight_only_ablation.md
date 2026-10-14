# PetFinder preflight-only ablation

## Objective

Keep the completed original-MLEvolve baseline frozen. Run one new 20-node
PetFinder experiment on ABA with the preflight checker as the only enabled
treatment feature.

## Required configuration

- Agent: local Qwen3.8-27B INT8 through vLLM on ABA physical GPU 1.
- Candidate execution: ABA physical GPU 0, NVIDIA A100 80GB PCIe.
- Seed: 42 for Qwen requests and candidate Python/NumPy/PyTorch execution.
- Node budget: 20 budget-counted nodes.
- Enabled: preflight checker only.
- Disabled: Scheduler, Predictor, Hardware Knowledge Database (HWKD), context
  cache, lesson profiles, review, CUDA documentation retrieval, pipeline
  decision, and hardware prompt context.
- Concurrency: incremental execution logic only; no fixed Max Parallel Jobs or
  `parallel_job_cap`.

## Prompt acceptance gate

The code-generation request must match the frozen baseline prompt rather than
containing CandidateAdapter/preflight text. A generated candidate that already
defines CandidateAdapter receives the full CPU checker; a baseline-style
candidate without it receives the static-source and hardware fallback. Missing
adapter instrumentation alone must not trigger an LLM repair. The saved
configuration and prompt audit must prove that Scheduler, Predictor, HWKD,
lesson-profile, review, CUDA-doc, and preflight contracts are all absent from
the generator prompt.

## Status

Live run started on ABA at 2026-09-26 18:00:00 UTC.

- Launcher PID: `3244583`
- MLEvolve PID at launch: `3244721`
- Durable run root:
  `/data1/yufan/MLEvolve/.cache/experiments/petfinder_aba_a100_preflight_only_seed42_20260926T180000Z`
- Volatile run root:
  `/dev/shm/petfinder_aba_a100_preflight_only_seed42_3244583/experiment`
- Prompt gate: accepted. The only feature section was
  `CPU preflight adapter contract`; both `Backend design contract` and
  `Scheduler Model Family Contract` were absent.
- Initial hardware check: GPU 0 idle for candidate execution; GPU 1 held the
  healthy Qwen vLLM service at about 73.4 GiB.
- Initial request: metric direction determination completed successfully;
  draft 1 generation started.

The first pre-launch attempt exited before any model generation because one
nested prompt file was synchronized to the wrong destination. The misplaced
file and empty attempt directory were removed, the remote hashes were matched
to the local source, and the prompt gate passed before this live run began.

The 18:00 UTC attempt was then stopped before any node received a score. Its
first three drafts proved the treatment-section gate, but a saved-prompt audit
showed that the launcher had disabled the task-description cleaner whereas the
frozen baseline used it. That attempt is setup-invalid and is retained only as
diagnostic evidence. The corrected launcher enables the same cleaner before
the replacement run.

At 18:08 UTC, ABA GPU 0 was occupied by an unrelated `kylehu` training
process. The launcher's safety check refused to overlap it. The original
waiter was paused while the frozen prompt-core compatibility gate was added.
After the remote audit passed, a replacement non-destructive waiter (PID
`3259771`) was started at 18:24 UTC. It will execute the corrected launcher
only after physical GPU 0 falls to at most 256 MiB. It does not signal or
otherwise modify the unrelated process.

## Frozen prompt-core compatibility

The frozen baseline journal preserves the exact first request: system 714
characters, user 18,620 characters, assistant prefix 2,845 characters, and
22,605 serialized characters. The corrected treatment now selects
`agent.prompt_compatibility_profile=original_mlevolve_20260924`. This keeps
the baseline task-cleaning suffix, task/metric-alignment text, implementation
guideline core, and assistant prefix while adding only the preflight adapter
contract during the superseded diagnostic design. The final matched design
removes that contract too: the generated request is byte-identical to the
baseline and the checker runs only after generation. The default/current
prompt remains unchanged for other experiments.

The compatibility change was developed test-first. Five new assertions first
failed for missing behavior, then the focused prompt, cleaner, launcher, and
preflight suites reached 99 passes. One unrelated pre-existing integration
test still fails because its synthetic `AgentSearch` omits `acfg`; that failure
predates and does not exercise prompt composition.

## Corrected live run

GPU 0 released at 18:31 UTC and waiter PID `3259771` atomically transitioned
into the corrected launcher. The live MLEvolve process is PID `3263562` and
the durable run root is:

`/data1/yufan/MLEvolve/.cache/experiments/petfinder_aba_a100_preflight_only_seed42_20260926T183126Z`

The actual first saved draft request was compared with node 1 of the frozen
baseline journal, not only with synthetic marker checks:

- system role: exact match, 714 characters, SHA-256 prefix
  `c2669b24c5156a00`;
- assistant role: exact match, 2,845 characters, SHA-256 prefix
  `c848a55951307133`;
- task-description content: exact match after ignoring four trailing blank
  lines in the treatment rendering;
- forbidden treatment markers absent: Backend design, Scheduler model-family,
  HWKD/lesson, Pipeline Decision, and CUDA documentation;
- intentional treatment delta present: CPU preflight generation contract and
  CPU model-preflight implementation contract.

That conclusion was too weak. The complete request was 26,080 serialized
characters rather than the frozen baseline's 22,605, so the run was stopped
before it could count as treatment evidence. The differences included two
preflight contracts, the displayed 20-versus-50 node budget, package wording,
four trailing task-description blank lines, and a removed multimodal
recommendation. The diagnostic run was persisted with exit code 2; Qwen vLLM
was left running.

The replacement design keeps generation byte-compatible with the frozen
baseline. This initial replacement still moved the adapter requirement to the
checker's post-generation repair path; the later diagnosis below shows why that
was also an invalid confound. Regression tests cover the prompt-treatment flag,
baseline budget surface, task boundary, seeded package shuffle, and multimodal
guidance.

## Byte-identical replacement run

The valid replacement run started at 2026-09-26 18:56:40 UTC:

- controller PID: `3279347`;
- MLEvolve PID: `3279480`;
- durable root:
  `/data1/yufan/MLEvolve/.cache/experiments/petfinder_aba_a100_preflight_only_exactprompt_seed42_20260926T185640Z`;
- actual first saved generation request: 22,808 UTF-8 bytes, SHA-256
  `dcd3ebc35a3fad7be385e2e9739955b05df41de551d5077d450014abf04438f8`;
- frozen baseline first request: the same byte count and SHA-256.

This is a byte-for-byte comparison of the live request artifact, not a marker
or normalized-text comparison. Preflight is enabled with
`require_adapter_for_generated=true`, while
`prompt_contract_enabled=false` keeps its adapter request in the
post-generation repair path.

## 2026-09-26 first-candidate divergence diagnosis

- The replacement treatment prompt was byte-identical to the frozen reconstructed
  baseline prompt: 22,808 UTF-8 bytes, SHA-256
  `dcd3ebc35a3fad7be385e2e9739955b05df41de551d5077d450014abf04438f8`.
- Nevertheless, the historical baseline's first logged generation and the treatment's
  first logged generation differed before preflight repair:
  - baseline response SHA-256: `6aa0be20286670539965544cd48041a9c1289c537fd90eda9d0f242be5e39f03`
  - treatment response SHA-256: `27148bcf462c48298a5b912d9084122406a50febd7101839c32355a408fb3732`
- The old and current request payload forms were replayed in isolation twice each.
  All four replays produced the same response SHA-256
  `e3fe6796e34ad68da448b13a367c548884857828fadc4dfb98a5af3040505a31`.
  Therefore the `enable_thinking` payload representation is not the cause.
- The baseline and treatment used the same continuously running vLLM process and
  unchanged weights. The remaining uncontrolled variable is the inference execution
  state. This vLLM process has batch-invariant execution disabled (the installed vLLM
  default), and it uses prefix caching plus Multi-Token Prediction speculative decoding.
  Seed 42 fixes sampling random numbers but does not make floating-point logits and
  scheduling/cache execution paths bitwise invariant. At temperature 0.7, a tiny logits
  difference can select a different token and then cause the entire generated program
  and first metric point to diverge.
- The invalid treatment controller and its preflight workers were stopped. The Qwen
  vLLM service was deliberately left running.

## Relaunch queue

At 2026-09-26 19:17 UTC, ABA physical GPU 0 was again occupied by an unrelated
`kylehu` candidate process (PID `3276597`, approximately 966 MiB VRAM). It was
not signalled. A single-lock, non-destructive waiter (PID `3291334`) now checks
the exact GPU UUID every 30 seconds and atomically launches the full 20-node
preflight-only experiment only after GPU 0 has no compute owner and at most
256 MiB in use. The resident Qwen vLLM process on GPU 1 remains untouched.

At 2026-09-26 19:34:34 UTC the unrelated GPU 0 process released the device and
the same waiter atomically became the corrected launcher. The authoritative
run root is
`/data1/yufan/MLEvolve/.cache/experiments/petfinder_aba_a100_preflight_only_seed42_20260926T193434Z`;
launcher PID `3291334` owns MLEvolve PID `3300421`. The saved prompt/configuration
audit is accepted and confirms 20 nodes, seed 42, preflight enabled,
`require_adapter_for_generated=false`, Scheduler/Predictor/HWKD/context cache/
review/CUDA-doc/pipeline decision disabled, and no fixed parallel cap. Initial
draft generation is live; the complete first-request hash will be checked as
soon as the first journal is persisted.

At 2026-09-26 19:46 UTC the journal was persisted. The earliest generated
draft by `created_time` is node `501a864c620d4dd089280e76f5eb0236` at
19:37:38 UTC; its saved `prompt_input` is 22,808 UTF-8 bytes with SHA-256
`dcd3ebc35a3fad7be385e2e9739955b05df41de551d5077d450014abf04438f8`,
an exact match to the frozen baseline request. Three budget steps had completed
at that checkpoint, all runtime-buggy and without a finite metric; three more
tasks were running concurrently. The admitted drafts used
`static_hardware_fallback`, confirming that absence of CandidateAdapter no
longer causes a checker-only rewrite.

At 2026-09-26 19:53 UTC the run reached 9/20 completed budget steps with three
more tasks active. One node produced a finite RMSE of 19.1063 and eight were
runtime-buggy. All nine had preflight reports (2 PASS, 7 INCONCLUSIVE), were
admitted under the balanced static/hardware fallback, and GPU 0 was actively
training at approximately 99% utilization. No configuration or prompt invariant
had failed, so the experiment was allowed to continue.

At 2026-09-26 20:16 UTC the run reached 14/20 completed steps with three more
tasks active. Two nodes were valid and twelve were runtime-buggy; the best RMSE
improved from 19.1063 to 18.719164241544952. Both GPUs remained actively used
(candidate training on GPU 0 and local-agent generation on GPU 1), with no
prompt/configuration invariant failure.

## Adapter-repair confound removed

The stopped byte-identical diagnostic produced no budget-counted nodes because
the first five baseline-style drafts did not define CandidateAdapter. With
`require_adapter_for_generated=true`, that instrumentation absence alone forced
each draft into an LLM repair round. Several repaired drafts added an adapter
but retained module-level dataset reads, so isolated import failed with
`FileNotFoundError`. This changed candidate code for a reason unrelated to the
checker finding a model bug and therefore was not a clean preflight-only
ablation.

The final launcher sets `require_adapter_for_generated=false`. The checker is
still enabled for every generated candidate: naturally instrumented candidates
run the full CPU checker, while candidates without an adapter run the existing
`static_hardware_fallback`. Under the balanced policy, PASS and INCONCLUSIVE
fallback reports are admitted; genuine syntax, dependency, precision, and
hardware failures remain rejectable. Focused regression evidence:

- launcher contract: 4 passed;
- generated-candidate adapter/fallback integration checks: 3 passed, including
  proof that a real syntax failure remains blocked with adapter enforcement off;
- combined launcher, prompt, seed, coder, and preflight regression suite:
  109 passed;
- the final generator prompt remains byte-identical to the frozen baseline.

## Final result

The corrected run finished cleanly at 2026-09-26 20:33:12 UTC with exit code
0 and exactly 20 budget-counted nodes. The complete 175-file, 5.4 MiB durable
artifact tree is saved under
`records/2026-09-26_petfinder_preflight_only_raw/`. It contains the journal,
filtered journal, configuration, hardware samples, pipeline database, every
prompt, all 20 preflight reports and attempt reports, generated workspaces,
submissions, and the best solution.

Matched first-20-completed-node results on the same A100 80GB PCIe hardware:

| Result | Frozen baseline | Preflight only |
|---|---:|---:|
| Wall-clock span | 2.5472 h | 0.9258 h |
| Peak execution concurrency | 3 | 3 |
| Successful/scored nodes | 10 | 5 |
| Runtime-buggy nodes | 10 | 15 |
| Median execution time | 256.04 s | 89.12 s |
| Best validation RMSE (lower is better) | 18.1502 | 18.6012928673 |

The treatment span is 2.751x shorter (63.65% less wall time), but this is not a
valid causal speedup claim: it produced half as many scored nodes and 50% more
runtime failures, whose shorter executions reduced the span. Its best RMSE is
0.45109 higher (2.49% worse). All 20 candidates used
`static_hardware_fallback`; preflight returned 3 PASS and 17 INCONCLUSIVE and
admitted all 20, so it filtered none of the 15 eventual runtime bugs. The
prompt-matched checker-only ablation therefore does **not** demonstrate a useful
improvement in its present fallback configuration.

The required combined figure is
`records/2026-09-26_petfinder_baseline_vs_preflight_only.png` (1860x1500,
SHA-256 `0af2a825af270255a00c14bcf3276eb2cfc5a86420d0c293bab7587038e7509d`).
It was rendered with Gantt charts above and metric-node graphs below, then
visually inspected. A plotting bug was found and fixed test-first: generic
`time limit is 2 hours` context had incorrectly turned runtime exceptions gray.
The corrected chart shows every actual execution exception in red and only
explicit SIGTERM-style external stops in gray; its focused suite passes 11/11.
