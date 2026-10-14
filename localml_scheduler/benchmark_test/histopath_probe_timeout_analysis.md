# Histopath Scheduler Probe Timeout Analysis

Source run:
`runs/profile_scheduler_compare_histopathologic-cancer-detection_20260704_212842`

The mid-run cancel commands were caused by the executor-side model-family probe
wait limit, not by the scheduler independently deciding to abandon jobs.

Evidence from the scheduler SQLite DB:

| cancel command | job prefix | model family | age at cancel | final status | started? | reason |
| --- | --- | --- | ---: | --- | --- | --- |
| 6 | `f31eff98` | `siglip2_so400m_p16_256_feature_extractor_v1` | 300.9s | `CANCELLED` | no | cancelled while queued |
| 10 | `94b398ab` | `siglip2-so400m-patch16-256_feature_extractor` | 300.0s | `CANCELLED` | no | cancelled while queued |
| 20 | `4a3a8516` | `timm/mobilenetv3_small_100` | 300.2s | `FAILED` | yes | batch probe failed after starting |
| 31 | `77d51834` | `convnext_femto` | 300.5s | `CANCELLED` | no | cancelled while queued |
| 35 | `9dbc96f1` | `convnext_base` | 300.7s | `CANCELLED` | no | cancelled while queued |
| 37 | `59770c07` | `resnet18` | 300.9s | `FAILED` | yes | batch probe failed after starting |

The 300-second alignment matches
`gpu_scheduler.model_family_probe_timeout_seconds: 300`.
The executor submits a model-family probe, polls for the profile, and calls
`scheduler_client.cancel(probe_job_id)` if no profile or terminal result arrives
before the deadline.

Four probe jobs never reached `RUNNING`; they were waiting behind other GPU work
and were cancelled while queued. Two reached `RUNNING` near the timeout boundary,
then failed during batch probing, so their final status is `FAILED` even though a
cancel command exists in the command stream.

The later cancel burst at `2026-07-05T03:28:19Z` is different. The run received
SIGTERM and exited with code `124`, then cleanup cancelled outstanding scheduler
jobs before writing the hardware report.
