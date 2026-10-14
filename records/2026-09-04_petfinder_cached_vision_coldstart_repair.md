# PetFinder A100 cached-vision cold-start repair

## Scope

- Agent device: A100 80 GiB GPU 0, local vLLM `qwen3.8-27b-int8-a100`.
- Experiment device: isolated A100 80 GiB GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
- Scheduler admission budget: 31 GiB; `parallel_job_cap=null`.
- Agent context remains unbounded. No generation-length cap was introduced.

## Observed failure

Before this repair, candidates generated from the bundled SigLIP2 template were rejected before GPU execution. Their model-preflight reports identified a concrete construction failure: `from_pretrained` requested a checkpoint that was unavailable to the network-disabled worker. The candidate `d7c97db177d5416ebb651bedf81fa415` retained that `CON001` failure across its two targeted repair rounds.

## Diagnosis

`engine/coldstart/models_guidance_classified.json` advertised a SigLIP2 `AutoModel.from_pretrained` template even though the local SigLIP2 cache contains only an incomplete snapshot. A direct ABA probe succeeded for `tf_efficientnet_b0.ns_jft_in1k`, but that probe alone was not sufficient evidence that the preflight worker would find every required artifact with network disabled.

```python
timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=True)
```

## Repair and verification

Replaced the general-image cold-start template with the verified cached timm EfficientNet-B0 backbone and an explicit offline warning. Added a regression test that requires the bundled template to use this exact cached backbone and not contain `AutoModel.from_pretrained`.

Verification:

```text
pytest tests/test_coldstart_guidance.py -q
4 passed
python -m json.tool engine/coldstart/models_guidance_classified.json
git diff --check
```

Commit: `090eb8ab fix: use cached timm backbone in vision coldstart`.

The existing controller was then terminated with `SIGTERM` only after the source was synchronized; vLLM was preserved. A clean replacement run began at `2026-09-04T04:42:30Z` under `petfinder_scheduler_aba_a100_cachedvision`. It retains the same task, hardware mapping, 31 GiB admission budget, unbounded Agent context, and no fixed parallel-job cap.

## Status

The replacement run is active. Its first generated candidate initially received the same offline `CON001`, but preflight repair round 1 changed it to a data-contract warning; the scheduler then ran it on GPU 1 and correctly rejected its generated `KeyError: 'Paws'`. Its second generated candidate was rejected before execution after all two repair rounds retained `CON001` for assigning a 12-element feature list to a 8,920-row DataFrame column. These are candidate-code failures, not a scheduler deadlock: neither invalid candidate is budget-counted and GPU 1 remains free for an admissible candidate.

This record establishes only the repaired offline-model admission path; it does not claim a valid PetFinder metric or Scheduler advantage yet.
