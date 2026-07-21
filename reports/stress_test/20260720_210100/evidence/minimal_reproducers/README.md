# Minimal Reproducer Notes

Primary timeout replay target:
- Run one saved `workspace/runfile_*_<node_id>_*.py` under the same prepared Dogs vs Cats dataset with `exec.timeout=120`.
- Compare direct execution with scheduler exclusive execution.

Important paths:
- Primary full stress run: runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass
- Primary scheduler DB: runs/stress_workflow_fix20_pass/scheduler_runtime/db/scheduler.sqlite3
- Fresh KG-off direct/scheduler retry: reports/stress_test/20260720_210100/matrix/kg_off_exclusive_retry

Observed replay proxy:
- KG off, scheduler off: 2/2 fresh nodes hit execution timeout after nonzero training phase time.
- KG off, scheduler on, exclusive placement: 2/2 fresh nodes hit execution timeout after batch probes succeeded.
