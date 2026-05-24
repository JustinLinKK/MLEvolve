# TODO

## Agent System Plugin

- Feedback to Agent through MCP

- Try all kinds of agent systems: **MLEvolve**, 

## Graph DB and Data accumulation on Cluster

- Test on different hardwares

## Pack Size Limit Setting

## MLEAgent Optimization

- [ ] Use graph database feedback to estimate whether the job list generated from the current tree frontier can fully utilize available GPUs.
- [ ] If predicted GPU utilization is too low, allow MLEAgent to increase the number of tree nodes explored concurrently so the scheduler has enough runnable jobs to maximize GPU utility.
- [ ] Add safeguards for the expanded exploration count, including max concurrent node limits, per-GPU memory constraints, and rollback when probing shows the expanded job set is inefficient.

## Log Database

- combination(tasks) -> Preheat time -> Stay time in GPU -> When out

- tasks: probe method, others ...

- Scheduler observability detail
  - [ ] Enable Postgres scheduler log DB in config and document `LOCALML_SCHEDULER_LOG_DSN`.
  - [ ] Add a planner decision trace event/table before dispatch, including candidate packed groups, objective scores, rejection reasons, selected plan, expected runtime, VRAM budget, and active GPU occupancy.
  - [ ] Expand batch probe selected/result logging with full profile metadata: search method, stop reason, failure batch size, average step time, warning message, and profile key.
  - [ ] Persist runtime probe summaries with resolved batch size, backend strategy, confidence, and estimated total runtime.
  - [ ] Link packed-job dispatch logs to run group members, probe profiles, batch overrides, and final placement reason.
  - [ ] Record worker execution details for packed jobs: process command, stdout/stderr log paths, artifact paths, start/end timestamps, exit status, traceback, and runner result.
  - [ ] Add read/query examples for reconstructing one scheduler run from startup -> probing -> final packed decision -> packed job execution.

## Model Feature for each GPUs
