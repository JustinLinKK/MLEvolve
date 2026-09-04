# PetFinder A100 cached-vision cold-start repair

## Scope

- Agent device: A100 80 GiB GPU 0, local vLLM `qwen3.8-27b-int8-a100`.
- Experiment device: isolated A100 80 GiB GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
- Scheduler admission budget: 31 GiB; `parallel_job_cap=null`.
- Agent context remains unbounded. No generation-length cap was introduced.

## Observed failure

The first two generated PetFinder candidates were rejected before GPU execution. The model-preflight reports identified a concrete construction failure: generated code called a Hugging Face `from_pretrained` checkpoint that was not fully cached on ABA while outgoing network traffic was disabled. No rejected candidate was budget-counted and GPU 1 was not used.

The second candidate (`d7c97db177d5416ebb651bedf81fa415`) demonstrated that the configured two targeted preflight repair rounds alone did not remove the root cause: `report_attempt_0.json`, `report_attempt_1.json`, and `report_attempt_2.json` all retained a `CON001` construction failure for an unavailable pretrained checkpoint.

## Diagnosis

`engine/coldstart/models_guidance_classified.json` advertised a SigLIP2 `AutoModel.from_pretrained` template even though the local SigLIP2 cache contains only an incomplete snapshot. ABA does have complete local timm checkpoints, including `tf_efficientnet_b0.ns_jft_in1k`, and direct ABA verification succeeded with:

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

The replacement run is active. This record establishes only the repaired offline-model admission condition; it does not claim a valid PetFinder metric or Scheduler advantage yet.
