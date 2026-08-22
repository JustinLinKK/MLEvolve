# CPU Model-Preflight Admission

MLEvolve runs the pinned `nn-model-preflight-checker` after stage-aware review and before any
scheduler submission or GPU batch probe. The gate is enabled by default for
`experiment.mode: hardware_aware`.

## Checkout and installation

Clone recursively, or initialize submodules in an existing checkout:

```bash
git clone --recurse-submodules <MLEvolve repository URL>
git submodule update --init --recursive
./install_dependencies.sh
python -c "import jsonschema, model_preflight; print(model_preflight.__version__)"
```

The checker is installed editable from `./nn-model-preflight-checker`. Its gitlink is pinned;
project-specific orchestration and GPU profiles remain in MLEvolve.

## Admission flow

Every candidate is copied to `workspace/working/preflight/<node-id>/candidate/candidate.py` with
a generated manifest. Batch scenarios and precision are derived from the same script
introspection used by the scheduler. The schema-validated checker report is stored as
`report.json`; `admission_summary.json` records MLEvolve's compact decision, source hash,
diagnostic codes, repair count, and GPU-canary requirement.

Newly generated candidates must expose this no-argument class and remain safe to import:

```python
class CandidateAdapter:
    def build_model(self, context): ...
    def build_optimizer(self, model, context): ...
    def build_train_batch(self, scenario, device): ...
    def build_validation_batch(self, scenario, device): ...
    def training_step(self, model, batch, context): ...
    def validation_step(self, model, batch, context): ...

if __name__ == "__main__":
    main()
```

Adapter batch builders can locate run input through `MLEVOLVE_INPUT_DIR`, honor
`scenario["batch_size"]`, and must reuse the candidate's real transforms, collate path, model,
loss, and optimizer. `training_step` returns a differentiable scalar loss.

Adapter-enabled scripts receive all eight checker stages. Legacy, replay, and initial-solution
scripts without an adapter receive `static_source` and `hardware` only. Unknown auto-detected
GPUs skip hardware and memory with a warning; explicit `target_profile` values may be bundled
names or custom YAML paths.

`FAIL` receives one targeted stage-owned repair by default and is rechecked once; an unresolved
failure is journaled through the existing rejected-node path without a GPU job. `PASS` proceeds.
With the default balanced policy, `INCONCLUSIVE` proceeds with `gpu_check_required`. Checker
infrastructure errors fail open by default and are never labeled as candidate defects. The
source hash is checked again immediately before execution, so modified code cannot reuse a
stale report.

The scheduler's short batch probe remains the target-GPU canary: CPU preflight cannot prove
CUDA-kernel, distributed, stream/MPS, mixed-precision, or concurrent-runtime correctness.

## Profiles and configuration

`target_profile: auto` selects bundled V100-16/32, A10-24, and A100-40 profiles or MLEvolve's
`config/preflight_profiles/a100_80gb.yaml` and `rtx_5090_32gb.yaml`. Set a bundled name such as
`nvidia/a10_24gb`, or a YAML path, to override detection. All time, memory, process, output,
network, abstract-fallback, policy, repair, and fail-open settings are documented in
`config.example.yaml` under the top-level `preflight` key.

The project-owned values follow NVIDIA's [A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf),
[RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
and [Blackwell architecture guide](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

## Validation and archived replay

Run the checker baseline, MLEvolve integration tests, and archived-script measurement with:

```bash
pytest -q nn-model-preflight-checker/tests
pytest -q tests/test_model_preflight_integration.py
python scheduler_benchmark_test/model_preflight_replay.py \
  --output reports/model_preflight_archived_replay.json
```
