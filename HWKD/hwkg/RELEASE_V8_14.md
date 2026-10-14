# HWKD v8.14 scoped fused optimizer experiment

Adds a measured RTX5080 runtime/recipe-specific graph entry and compact advice
for one change: fused=True on an existing CUDA AdamW. Eligibility is restricted
to the tested R50V2/BF16/channels_last/b64/128px/AdamW .0003/.05/CE smoothing.1
primitive, matching runtime/backend flags and positive conditional cost. Parent
execution must have succeeded; source scope, memory, stage and cost guards apply.
FP16 is excluded after an observed scale skip. No batch, dtype, LR, data pipeline,
training budget, eval/checkpoint or provider request changes are forced.

The primary 5-repeat/200-update windows showed ~11%faster normal updates,
conditional six-epoch gross saving5.827s, net .827s after a declared5s reserve,
fast-reference/slow-target spread .110s. This is a thin signal; earlier shorter
factorial spread was negative. The spread is not a CI, marginal LLM cost and
quality/full-pipeline benefit remain unknown. Prior batch/cache and failed FP16
studies are retained rather than omitted. Other models/recipes do not borrow it.

Component routing audits suggested profile and source fused option separately
from actual execution/measurement. Exact whole-source AST except a literal
AdamW fused flag can qualify warning comparisons; this is not semantic or causal
proof and never mutates native nodes/rewards. Additional recipe changes remain
confounded. Original SDK/API16K/parse/search/score/executor/hook behavior preserved.

369 component tests plus4subtests,43 targeted cases and12native compatibility
checks passed. Historical graph/data/models and prior calibration bundles are
preserved. New scored native performance/accuracy validation is pending; the
planned fresh5+5 is exploratory,not a claim of beating baseline or significance.
