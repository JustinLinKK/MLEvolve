# Colocation Probe Cost

## Question

- Is the trial-epoch probe that decides whether a job may join the running set worth what it costs?

## Mechanism

- `simulate_recursive_time_aware` admits a candidate by running a trial before committing, in `localml_scheduler/scheduler/trace_simulator.py`

- Three properties of that trial drive its cost

> The trial waits for every active member, not just the candidate

- The loop condition is `any(measured_epochs[job] < colocation_trial_epochs for job in active)`, so the trial only ends once the slowest member has completed the required epochs

> Every member runs degraded for the whole trial

- All members execute at the packed rate during the trial, so the anchor pays the colocation penalty before the decision is made

- Trial duration is approximately

$$
\begin{equation}
\begin{aligned}
\textbf{trial\_seconds} &\approx
\textbf{colocation\_trial\_epochs} \times
\max_{j \in \textbf{active}} \textbf{packed\_epoch\_seconds}(j) \\
\textbf{packed\_epoch\_seconds}(j) &=
\textbf{solo\_epoch\_seconds}(j) \times \textbf{slowdown}(j)
\end{aligned}
\end{equation}
$$

- Because the packed epoch time contains the slowdown, the probe becomes more expensive exactly when the colocation being tested is worse; discovering a bad pack costs more than discovering a good one

> A minimum evidence window applies

- `trial_evidence_timeout_min_seconds` is 300, so the deadline never falls below 300 s even when a verdict could be reached sooner

## Measured Cost on a Packable Workload

- Trace: `traces/mlebench_tabular_v100_100jobs.jsonl`, pair slowdown 1.017, serial makespan 9094.4 s

| `colocation_trial_epochs` | Makespan (s) | vs serial | Trial epochs burned | Rejections |
|---|---|---|---|---|
| ~0 | 2090.4 | 4.350x | 0.3 | 0 |
| 0.25 | 2089.2 | 4.353x | 88.8 | 0 |
| 0.5 | 2098.2 | 4.334x | 187.4 | 0 |
| 1 | 2280.0 | 3.989x | 391.0 | 0 |
| 2 (current default) | 2748.3 | 3.309x | 636.2 | 0 |
| 3 | 3133.0 | 2.903x | 691.0 | 0 |
| 5 | 3652.8 | 2.490x | 808.8 | 0 |

- The default setting costs 657.9 s of makespan against a near-zero probe, a 31 % increase, and lowers the speedup from 4.350x to 3.309x

- Across every setting the probe produced zero rejections, so on this workload all 636.2 trial epochs bought no decision that changed an outcome

## Measured Cost on a Non-Packable Workload

- Trace: `traces/mlebench_v100_100jobs.jsonl`, pair slowdown 4.0, serial makespan 15230.9 s

| `colocation_trial_epochs` | Makespan (s) | vs serial | Rejections |
|---|---|---|---|
| ~0 | 15255.4 | 0.998x | 99 |
| 0.5 | 27251.7 | 0.559x | 49 |
| 1 | 30013.8 | 0.507x | 10 |
| 2 (current default) | 30406.0 | 0.501x | 0 |
| 3 | 30406.0 | 0.501x | 0 |

- Longer trials produce fewer rejections, which is the opposite of the intended behavior

- At a near-zero trial the scheduler rejects all 99 bad packs and lands within 0.2 % of serial, which is the correct outcome for a workload that cannot pack

- At the default it rejects nothing and finishes at half the serial rate

## Why Longer Trials Reject Less

- Jobs in these traces have `planned_epochs` in {1, 2, 3}, at or below the 2-epoch trial length

- When a member finishes inside the trial window, `run_trial` returns `"completed"`, and when the pre-existing members all finish it returns `"accepted"`; both exits bypass the `colocation_min_gain` comparison

- A job shorter than the trial length can therefore never be judged, so raising the trial length converts decisions into unjudged admissions

## Conclusion

- The probe is not merely slow, it is currently counterproductive at its default setting in both regimes: 31 % of makespan wasted where packing helps, and all rejections suppressed where packing hurts

- A near-zero trial length is the best measured setting in both regimes, 4.350x on the packable workload and 0.998x on the non-packable one, which indicates the evidence needed to decide is available almost immediately and that requiring whole epochs is what destroys it

## Suggested Changes

- Measure over a short wall-clock slice rather than a whole number of epochs

- Gate the trial on the candidate's own progress instead of every member's

- Evaluate gain from partial evidence on the `"completed"` and membership-change exits rather than returning unjudged

- Remove the 300 s minimum evidence window

## Reproduction

```bash
python3 -c "import sys,dataclasses; sys.path.insert(0,'.'); from localml_scheduler.scheduler import trace_simulator as ts; from scheduler_benchmark_test.test_trace_policies import trace_to_problem, load_trace, TRACES_DIR; raw=load_trace(TRACES_DIR/'mlebench_tabular_v100_100jobs.jsonl'); base=trace_to_problem(raw, pair_slowdown=1.017); [print(te, ts.simulate_recursive_time_aware(dataclasses.replace(base, colocation_trial_epochs=te)).makespan_seconds) for te in [0.001,1,2,5]]"
```
