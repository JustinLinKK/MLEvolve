# Backend mode v2 migration

Before:

```yaml
gpu_scheduler:
  backend_priority: [mps, exclusive]
```

After:

```yaml
gpu_scheduler:
  packing_backend: mps_process
  exclusive_fallback_enabled: true
  mps_unavailable_policy: exclusive
```

Use `packing_backend: cuda_process` for independent CUDA job processes without
MPS. The deprecated `mps` configuration alias is accepted for one migration
window and immediately normalized; new settings and profiles use
`mps_process`.

Preview persisted legacy data without writing:

```bash
python -m localml_scheduler.cli scheduler migrate-backend-modes \
  --settings config.yaml --dry-run
```

Use `--execute` only after reviewing the report. Execution is idempotent. It
normalizes only MPS rows with positive launch provenance and annotates retired
stream rows as non-selectable without deleting them. Ambiguous MPS rows remain
unchanged for operator review. The report gives exact per-identifier counts for
configuration files, cache entries, profile tables, events, and active
knowledge records. It also reports identity-schema rekeys and conflicts; these
rekeys add the canonical backend and subprocess runner contract to profile
namespaces so version-1 cache keys cannot be silently reused.
