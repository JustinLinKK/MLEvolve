# Legacy fixture location

The historical 100-model `standard_histopath_v1` fixture was introduced in
commit `9df6343` and later removed during the parent-branch merge. Its model
variants used operations outside the deployed student's `53/3/40` vocabulary,
including BatchNorm, GroupNorm, InstanceNorm, LeakyReLU, ELU, transpose,
multiply, reduce, and the `MultiHeadAttention`/`Attention` naming mismatch.

Do not use old generated files or remaining Python caches from this directory
as evidence of ML-predictor compatibility.

The active 100-model compatibility stress-test dataset is:

`scheduler_benchmark_test/fixtures/stress_test_data_v1.0/`

Verify all 100 models with:

```bash
python -m scheduler_benchmark_test.standard.stress_test_data \
  --check --verify-predictions
```
