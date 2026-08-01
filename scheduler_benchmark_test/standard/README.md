# Stress Test Data v1.0

The active 100-model predictor stress-test dataset is:

`scheduler_benchmark_test/fixtures/stress_test_data_v1.0/`

It contains 20 model structures with five variants each. Every job carries a
complete `perfseer_model` source specification and is restricted to operation
identities represented by the deployed `53/3/40` student interface.

Regenerate the deterministic list and manifest:

```bash
python -m scheduler_benchmark_test.standard.stress_test_data
```

Check for fixture drift:

```bash
python -m scheduler_benchmark_test.standard.stress_test_data --check
```

Run all 100 source conversions and CPU TorchScript predictions:

```bash
python -m scheduler_benchmark_test.standard.stress_test_data \
  --check \
  --verify-predictions \
  --output-report /tmp/stress-test-data-v1.0-verification.json
```

Verify the RTX PRO 6000 Blackwell CPU artifact explicitly:

```bash
python -m scheduler_benchmark_test.standard.stress_test_data \
  --check \
  --verify-predictions \
  --artifact PerfSeer-predictor/models/nvidia_rtx_pro_6000_blackwell/student_rtx_pro_6000_blackwell_cpu.torchscript.pt
```

Acceptance requires:

- exactly 100 jobs;
- no operation outside the current student vocabulary;
- exact `53/3/40` input tensors;
- finite positive `train_mem` output for every job;
- CPU-only predictor tensors and no change in CUDA allocation.

This is a deployment compatibility fixture. It intentionally does not pretend
to cover operations that require the proposed next-generation student schema.
See
`PerfSeer-predictor/docs/student_operation_coverage_and_dataset_redesign.md`
for that redesign plan.
