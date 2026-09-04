# PetFinder multimodal preflight fixture repair

## Trigger and root cause

The cached-vision Scheduler run generated five candidates over 57 minutes but
produced zero budget-counted nodes. The repeated deterministic failure was not
a Scheduler admission or Qwen generation limit: model preflight supplied only
`batch_size` to `CandidateAdapter.build_train_batch`. A PetFinder candidate
therefore constructed a vector batch (`[B, 1281]`) and passed it to an
EfficientNet convolution, which reproduced `RuntimeError: Expected 3D or 4D
input to conv2d`.

## Repair

- PetFinder manifests now declare image rank 3, float32 regression target, and
  the representative fixture: `image=[3,256,256]`, `tabular=[12]`, and scalar
  target.
- The bundled Model Preflight checker now preserves named fixture shapes from
  YAML through its normalized manifest and into every CPU scenario passed to
  the adapter.
- Generation and targeted repair prompts name the same PetFinder batch
  contract. This supplies data-interface information only; it does not limit
  generation length, model choice, search branching, execution time, or job
  concurrency.

## Verification

The regression first failed because the old manifest contained only the task
name. After the repair, focused tests passed:

```text
10 passed
```

The modified checker was synchronized to ABA and a remote Python assertion
verified that `ScenarioConfig.cpu_scenarios()` retains the image and tabular
fixture. Commits: parent `8472ca72`; checker `57f199c`.

## Live restart

Two fresh starts exposed stale ABA configuration paths under `/root`, first
for the task description and then for PetFinder data. Both exited before any
candidate was generated or GPU1 worker was created. The active replacement
uses the readable task description in `/dev/shm/mlevolve_a100_agent/` and the
existing readable PetFinder workspace under `/home/yufan/MLEvolve/runs/`.

Active run root:

`/dev/shm/mlevolve_a100_agent/runs/petfinder_scheduler_aba_a100_multimodalfixture_20260904T055109Z`

It retains 50 target nodes, branch-profile scheduling, `parallel_job_cap=null`,
a 31 GiB admission budget, CPU embeddings, and Qwen3.8-27B-INT8 vLLM on GPU0
with GPU1 reserved for the experiment. At 05:52 UTC its first candidate entered
stepwise model-design generation; GPU0 was 99% utilized and GPU1 used 4 MiB.

## Scalar-target schema correction

The first post-repair candidate completed stepwise generation and review but
preflight correctly surfaced a manifest integration error before it could
reach GPU1: JSON Schema required every `fixture` shape to have at least one
dimension, while the regression target was encoded as scalar shape `[]`.
The candidate was excluded and no scheduler worker was created. The fix omits
the target from the named input fixture; `task.target_dtype=float32` remains
the checker-owned target contract and generated adapters still construct the
real float32 batch target. The regression test first failed with the obsolete
`target: []` entry and then passed with the focused suite (`10 passed`).

The verified controller was restarted only after synchronizing parent commit
`1538fca7`. The active run root is now
`/dev/shm/mlevolve_a100_agent/runs/petfinder_scheduler_aba_a100_multimodalfixturefix_20260904T060630Z`.
vLLM on GPU0 was preserved; the new controller reached stepwise model design
at 06:07 UTC with GPU1 still idle.
