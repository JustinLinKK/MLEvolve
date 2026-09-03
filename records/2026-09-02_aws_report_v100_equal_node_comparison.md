# AWS report V100 equal-node comparison update

## Scope

The AWS GPU request report replaces its final two figures with the two original
one-V100 Cassava Gantt artifacts supplied for this revision:

- DeepSeek trace: 42 executions, 18 non-buggy completed nodes.
- Sonnet 5 trace: 60 executions, 15 non-buggy completed nodes.

The differing total node counts make total trace spans unsuitable for a direct
timing comparison.

## Matched comparison

For each JSONL trace, sort non-buggy nodes with an execution duration of at
least one second by `exec_complete_at`.  Measure elapsed time from the earliest
`gen_start_at` in that trace to the completion of the fifteenth such node.

| Trace | Common completed-node count | Time to common count |
| --- | ---: | ---: |
| DeepSeek | 15 | 258.8 min |
| Sonnet 5 | 15 | 455.3 min |

The matched-count completion ratio is `455.3 / 258.8 = 1.76x`.  The PDF labels
this explicitly as an equal-node comparison and does not compare the raw
42-execution and 60-execution trace spans.

## Evidence

- `traces/mlevolve_cassava_v100_deepseek.jsonl`
- `traces/mlevolve_cassava_sonnet5_p8.jsonl`
- Original supplied Gantt images embedded directly in the revised PDF.

## Claim boundary revision

The prior report front page displayed 1.53x and 1.58x results from offline
trace-replay comparisons.  These are not same-hardware, live end-to-end
baseline/Scheduler experiments.  They are therefore removed from the executive
claim and from the primary report page.  The revised report makes only the
following primary claims:

- The real PetFinder baseline reached 50 effective nodes in 17.70 hours.
- Real profile-guided RTX 5090 runs completed their stated workloads with zero
  Out-of-Memory failures.
- The same-hardware 50-node Scheduler comparison has only two effective nodes
  and is explicitly not a speedup result.

The report subsequently removes all throughput, speedup, label-count, and
calibration-count headline figures.  The two agent-specific V100 traces remain
as auditable artifacts only, with no causal or speedup claim.

The final revision also removes large headline values for baseline duration,
node count, RMSE, concurrency, profile-run counts, and scheduler execution
time. It uses qualitative, evidence-bounded statements instead.

## Correct paired scheduler result

The earlier 1.53x and 1.58x figures used a serial counterfactual, and their
old harness had `PARALLEL_CAP = 5`. They are not used. The corrected benchmark
uses no fixed parallel cap. Both policies process the same 39 schedulable jobs
from `mlevolve_cassava_v100_deepseek.jsonl`, with the same release times,
deconvolved solo durations, profile-derived memory footprints, 31 GB memory
budget, and pair slowdown measured from V100 power samples.

| Metric | Original `recursive_time_aware` | Current profile occupancy-density | Change |
| --- | ---: | ---: | ---: |
| Makespan | 314.17 min | 304.65 min | -3.03% (-9.51 min) |
| Mean flow time | 22.79 min | 22.40 min | -1.70% |
| p95 flow time | 62.12 min | 48.75 min | -21.52% |
| Maximum wait | 33.68 min | 53.92 min | worse |
| Starvation count | 1 | 4 | worse |

This is a deterministic paired trace replay, not a live end-to-end run. The
report names both the benefit and the fairness tradeoff, and requests matched
GPU capacity for live validation.
