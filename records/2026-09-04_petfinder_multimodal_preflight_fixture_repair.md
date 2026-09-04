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

## Offline generation-contract contradiction

The first candidate from that restarted controller completed generation and
review, but its isolated construction selected a pretrained EfficientNet and
then a pretrained ResNet fallback. Both required a network download and were
correctly rejected as `CON001` because model preflight disables network access.
This was not a Scheduler rejection: the generated-code prompt had contradicted
the worker contract by stating that `torch.hub`, Hugging Face, and arbitrary
pretrained models were available during development, while the implementation
guideline made the same claim.

Both prompt surfaces now state that generated candidates run offline in both
preflight and experiment execution, must not download weights or data, and may
use only an explicitly supplied local checkpoint or verified cache. This
changes neither the model-search space nor Scheduler policy; it removes a
false environment guarantee. A regression test first failed against the old
prompt and then passed. Focused prompt, cold-start, and preflight tests passed:

```text
13 passed
```

The controller that had loaded the contradictory prompt was stopped only after
its verified command line was inspected; it had zero budget-counted nodes and
its artifacts remain in place. The replacement runs the synchronized source
from `/dev/shm/mlevolve_a100_agent/repo/run.py` (rather than ABA's stale
`/home/yufan/MLEvolve/run.py`), preserves vLLM on GPU0, and has run root
`/dev/shm/mlevolve_a100_agent/runs/petfinder_scheduler_aba_a100_offlinecontract_20260904T062522Z`.
An initial replacement invocation exposed the stale-source import mismatch
(`ModuleNotFoundError: agents.lesson_context`) before candidate generation;
the command was corrected and a direct import of `engine.agent_search` from
the synchronized source succeeded before the final launch.

### Liveness monitor

The historical watchdog was bound to a retired experiment name and path, so a
current-run-specific monitor was attached under
`/dev/shm/mlevolve_a100_agent/scheduler_watchdog`. Every five minutes it
records the budget-node count, active vLLM request count, and GPU0 utilization.
It captures diagnostics and terminates only the verified current Scheduler
controller after all three show no progress for one hour. It neither imposes a
generation limit nor touches vLLM or an unrelated GPU process. At attachment,
the controller had zero budget nodes, one active vLLM request, and 99% GPU0
utilization, so its no-progress timer was correctly reset.

## Offline-review consistency correction

The first candidate after the offline-contract restart was rejected by stage
review before preflight. Its initial review correctly flagged an uncached
`torchvision` EfficientNet download. A repair changed this to
`pretrained=False`, which is a legitimate offline, train-from-scratch path;
however, the next review incorrectly called the trained model an ``untrained
network'' solely because it had `weights=None`. This made the review contract
internally inconsistent: a remote pretrained weight was forbidden, while a
fresh model trained by the candidate was also forbidden.

The review environment facts now say that only an explicit local checkpoint or
verified cache may be treated as pretrained, and explicitly state that a fresh
trainable model is valid when the script trains it before validation and test
inference. The inference-integrity rubric now prohibits treating
`weights=None`/`pretrained=False` as dummy inference merely due to initialization.
The repair does not select or replace any model architecture. A regression test
failed against the old reviewer prompt; the prompt, review workflow, offline
contract, and relevant preflight tests then passed:

```text
52 passed
```
