# Experiment Plan

## Trace Generation

- Recorded: real MLEvolve on the 4 Nautilus V100, `parallel_search_num=2`, claude CLI backend. `run_traced.py` stamps `ready_at`, `dispatch_at`, `exec_duration_s`, `peak_vram_mib`, `avg_sm_util_percent`, `concurrent_jobs`, `ran_solo`. Device-level NVML only; per-PID hidden by container PID namespace

- Config: budget 31000 MB, cap 5, trial 2 epochs, min gain 1.0

## Replayability

$$
\begin{equation}
\begin{aligned}
t^{\textbf{arrive}}_{n} =
t^{\textbf{exec-end}}_{\textbf{parent}(n)} + d^{\textbf{gen}}_{n}
\end{aligned}
\end{equation}
$$

- Re-record must stamp `gen_start_at`, `gen_end_at`, `parent_node_id`, `chain_id`. Implemented and verified in `scheduler_benchmark_test/run_traced.py`

- Generation timings are keyed by node id, not by thread. MLEvolve generates drafts in one phase and executes them in a later one, so a hook at the executing call site records a time after generation already finished; hooking `AgentSearch.step` that way measured 0.001 s. The working hook wraps the six generator modules (`draft_agent`, `improve_agent`, `debug_agent`, `evolution_agent`, `fusion_agent`, `aggregation_agent`), whose `run` performs the LLM work and returns the node

- `parent_node_id` comes from `node.parent`, so it is the real dependency edge rather than a guess at which node the branch ran last

- Verified on a 2-branch smoke run: `gen_duration_s = 330.5` against `exec_duration_s = 5.4`, with distinct `chain_id` per branch and no cross-branch contamination

- Generation dominates on light tasks, 330 s of LLM work against 5 s of training, so a replay that omits `gen_duration_s` would show an almost idle GPU

- Record solo training time. Colocated `exec_duration_s` would double-count slowdown

- Raise `parallel_search_num`. At 2 branches only 2 jobs ever pending, so `parallel_cap = 5` unreachable and packing untestable

## Strategies

- `serial` — priority-FIFO baseline

- `Our Scheduler`: Check codes to see our strategy

## Output

- `makespan`, `mean_flow`, `p95_flow`, `max_wait`, `average_slowdown`, `slowdown_rejections`, `starvation_count`; headline = makespan speedup vs serial

- Gantt via `run_trace_experiment.py <cnn|tabular>` to `records/scheduling_gantt_v100_<workload>.png`
