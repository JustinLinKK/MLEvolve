# Scheduler Observability

The scheduler writes canonical state to SQLite and mirrors selected evidence to Neo4j. For append-only analytics, enable the Postgres log database:

```yaml
scheduler:
  settings:
    log_db:
      provider: postgres
      dsn_env: LOCALML_SCHEDULER_LOG_DSN
      schema: scheduler_logs
      enabled: true
```

Set `LOCALML_SCHEDULER_LOG_DSN` to a psycopg-compatible DSN before starting the scheduler:

```bash
export LOCALML_SCHEDULER_LOG_DSN='postgresql://scheduler:password@localhost:5432/scheduler'
python -m localml_scheduler.cli scheduler start --config config.yaml
```

The log store is best-effort. If the DSN or `psycopg` is unavailable, jobs continue and the scheduler logs a warning.

## Main Tables

- `scheduler_logs.scheduler_sessions`: one row per scheduler service lifecycle, including runtime root, host identity, and config snapshot.
- `scheduler_logs.planner_decision_log`: one row per planner decision before dispatch, including candidate groups, rejection reasons, objective scores, selected plan, expected runtime, VRAM budget, and active occupancy.
- `scheduler_logs.probe_activity_log`: batch probe and runtime probe events, including `batch_probe_selected`, `batch_probe_result`, and `runtime_probe_profiled`.
- `scheduler_logs.runtime_probe_summaries`: latest summary per runtime profile key, including resolved batch size, backend, strategy, confidence, and estimated total runtime.
- `scheduler_logs.run_groups` and `scheduler_logs.run_group_members`: packed dispatch groups and per-job membership with placement reason, batch overrides, probe profile, and runtime profile metadata.
- `scheduler_logs.worker_execution_log`: worker launch and finish events with process command, log paths, artifacts, timestamps, exit status, traceback, and runner result.
- `scheduler_logs.gpu_metric_samples` and `scheduler_logs.job_metric_samples`: sampled GPU occupancy and job progress/runtime estimates.

## Reconstruct One Run

Pick a scheduler session:

```sql
SELECT session_id, started_at, stopped_at, runtime_root, config_json->'gpu_scheduler' AS gpu_scheduler
FROM scheduler_logs.scheduler_sessions
ORDER BY started_at DESC
LIMIT 5;
```

Follow startup and job admission:

```sql
SELECT created_at, job_id, event_type, payload_json
FROM scheduler_logs.job_activity_log
WHERE session_id = :session_id
ORDER BY created_at, activity_id;
```

Inspect probing for one job:

```sql
SELECT created_at, event_type, payload_json
FROM scheduler_logs.probe_activity_log
WHERE session_id = :session_id
  AND job_id = :job_id
ORDER BY created_at, probe_id;
```

Find the final planner decision that selected the packed group:

```sql
SELECT created_at,
       selected_mode,
       selected_backend,
       selected_job_ids_json,
       selected_reason,
       expected_runtime_seconds,
       safe_vram_budget_mb,
       active_vram_mb,
       active_sm_utilization,
       payload_json->'candidates' AS candidates
FROM scheduler_logs.planner_decision_log
WHERE session_id = :session_id
  AND selected_job_ids_json ? :job_id
ORDER BY created_at DESC
LIMIT 1;
```

Read the packed run group and its members:

```sql
SELECT rg.group_id,
       rg.mode,
       rg.backend_name,
       rg.opened_at,
       rg.closed_at,
       rg.fallback_triggered,
       rg.fallback_reason,
       rg.metadata_json AS group_metadata,
       jsonb_agg(rgm.metadata_json ORDER BY rgm.role) AS member_metadata
FROM scheduler_logs.run_groups rg
JOIN scheduler_logs.run_group_members rgm USING (group_id)
WHERE rg.session_id = :session_id
  AND rgm.job_id = :job_id
GROUP BY rg.group_id;
```

Inspect worker execution for each packed member:

```sql
SELECT created_at,
       job_id,
       event_type,
       backend_name,
       placement_mode,
       pid,
       exit_status,
       stdout_path,
       stderr_path,
       payload_json->'process_command' AS command,
       payload_json->'artifact_paths' AS artifacts,
       payload_json->'runner_result' AS runner_result,
       payload_json->'traceback' AS traceback
FROM scheduler_logs.worker_execution_log
WHERE session_id = :session_id
  AND group_id = :group_id
ORDER BY created_at, execution_id;
```

Correlate GPU occupancy during the packed run:

```sql
SELECT created_at,
       memory_used_mb,
       memory_total_mb,
       gpu_utilization,
       memory_utilization,
       job_ids_json
FROM scheduler_logs.gpu_metric_samples
WHERE session_id = :session_id
  AND group_id = :group_id
ORDER BY created_at;
```
