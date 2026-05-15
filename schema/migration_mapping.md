# Migration Mapping From SQLite To Graph

This file maps the current `localml_scheduler` SQLite model into the proposed graph schema.

## ID Rules For Synthetic Graph Nodes

Some current tables already have stable keys and can be reused directly. Others need one synthetic graph key:

- `solo_profile_id = "solo::" + hardware_key + "::" + signature`
- `packet_profile_id = "pair::" + hardware_key + "::" + pair_key` for `pair_profiles`
- `packet_profile_id = "group::" + combination_key` for `combination_profiles`
- `cache_key = "cache::" + model_id`
- `checkpoint_id = "checkpoint::" + checkpoint_id`
- `event_id = "event::" + event_id`
- `command_id = "command::" + command_id`
- `run_profile_id` should be materialized per observed execution or probe episode, for example `run::<job_id>::<run_kind>::<sequence>`

## Table Mapping

| SQLite source | Graph target | Key | Notes |
| --- | --- | --- | --- |
| `jobs` | `Job` | `job_id` | Parse `payload_json` into first-class job properties. |
| `jobs.payload_json.workflow_id` | `Workflow` | `workflow_id` | Create when present and connect with `HAS_JOB`. |
| `jobs.baseline_model_id`, `payload_json.batch_probe.model_key` | `Model` | `model_key` | Prefer explicit `batch_probe.model_key`, else `baseline_model_id`. |
| `jobs.payload_json.packing.signature` | `WorkloadSignature` | `signature` | Preserve exact scheduler signature used by runtime and packing logic. |
| `commands` | `Command` | `command_id` | Link to `Job` with `EXECUTED_COMMAND`. |
| `events` | `Event` | `event_id` | Link to `Job` with `EMITTED_EVENT`. |
| `checkpoints` | `Checkpoint` | `checkpoint_id` | Link to `Job` with `PRODUCED_CHECKPOINT`. |
| `cache_entries` | `CacheEntry` | `cache_key` | Link to `Model` with `CACHES_MODEL`. |
| `solo_profiles` | `SoloProfile` | `solo_profile_id` | Link to `WorkloadSignature` and `Hardware`. |
| `pair_profiles` | `PacketProfile` | `packet_profile_id` | Set `profile_scope = "pair"`. Link to member `Model` and `WorkloadSignature` nodes. |
| `batch_probe_profiles` | `BatchProbeProfile` | `probe_key` | Link to `Model`, `BatchShape`, and `Accelerator`. |
| `batch_size_observations` | `BatchSizeObservation` | `observation_key` | Link to `Model`, `BatchShape`, `Hardware`, and `Backend`. |
| `combination_profiles` | `PacketProfile` | `packet_profile_id` | Set `profile_scope = "group"`. Link to `Hardware`, `Backend`, and member signatures. |
| `runtime_profiles` | `RuntimeProfile` | `profile_key` | Link to `WorkloadSignature`, `Hardware`, and `Backend`. |

## Materializing `RunProfile`

`RunProfile` does not exist as a single current table, but it should become the central graph fact node because your request explicitly wants the running profile of every job run and probing run.

Build it from these sources:

1. `jobs`
   - base identity, lifecycle timestamps, configured batch size, `max_epochs`, `max_steps`, status
2. `events`
   - `job_started`, `job_resumed`, `job_completed`, `job_failed`, `batch_probe_started`, `batch_probe_trial`, `batch_probe_resolved`, `checkpoint_saved`, `packed_group_fallback`
3. `runtime_profiles`
   - startup time, epoch-1 time, average step time, estimated total runtime
4. `batch_probe_profiles`
   - resolved batch size, memory target, observation count
5. `batch_size_observations`
   - step timing and utilization at a concrete batch size
6. completion payloads or sidecar artifacts
   - any custom metrics returned by the runner such as elapsed time, final loss, peak VRAM, parameter count, or throughput

Telemetry mapping note:

- current repo `avg_gpu_utilization` can be copied into `avg_sm_utilization` when you want one normalized utilization field for GraphRAG, but it is not a true SM hardware counter
- current repo `avg_memory_utilization` is GPU memory utilization, not host RAM usage
- keep `avg_ram_utilization` null unless you add explicit host RAM telemetry

Recommended `run_kind` values:

- `training`
- `training_resume`
- `batch_probe`
- `runtime_probe`
- `packed_training`
- `recovery_replay`

Recommended `probe_kind` values:

- `batch`
- `runtime`

## Hardware Mapping

The repo currently uses two hardware scopes:

- exact hardware scope via `hardware_key`
- loose device-class scope via `device_type`

Map them separately:

- `Hardware`
  - exact environment keyed by `hardware_key`
  - populated from `localml_scheduler.hardware.HardwareProfile`
  - includes OS, GPU name, VRAM, compute capability, toolkit version, and torch version
- `Accelerator`
  - reusable class keyed from normalized device name plus hardware characteristics
  - used by `BatchProbeProfile` because batch-probe reuse is scoped to `device_type`, not full `hardware_key`

## Toolkit Mapping

Current code exposes `torch.version.cuda` directly. The target graph should normalize toolkit identity into:

- `toolkit_name`
  - `cuda`
  - `rocm`
  - `unknown`
- `toolkit_version`
  - runtime version string such as `12.4`

For current data:

- if `cuda_runtime` is populated, set `toolkit_name = "cuda"`
- if future ROCm support is added, map its runtime version to `toolkit_name = "rocm"`

## Packet Profile Mapping

To satisfy your requested packet node while staying faithful to the repo:

- `pair_profiles` become `PacketProfile` nodes with `profile_scope = "pair"`
- `combination_profiles` become `PacketProfile` nodes with `profile_scope = "group"`

Each `PacketProfile` should connect to:

- `Hardware`
- `Backend`
- involved `WorkloadSignature` nodes
- involved `Model` nodes

Relationship properties on `INVOLVES_MODEL` or `INVOLVES_SIGNATURE` should carry:

- `position`
- `batch_size`
- `signature`
- optional role labels such as `left`, `right`, or `member`

## Recommended Load Order

1. Load `Backend`, `Toolkit`, `Accelerator`, `Hardware`.
2. Load `Model`, `WorkloadSignature`, `BatchShape`, `Workflow`.
3. Load `Job`, `Command`, `Event`, `Checkpoint`, `CacheEntry`.
4. Materialize `RunProfile`.
5. Load `BatchProbeProfile`, `BatchSizeObservation`, `RuntimeProfile`, `SoloProfile`, `PacketProfile`.
6. Backfill `summary_text` on all fact nodes for GraphRAG.

## Recommended `summary_text` Templates

Use short factual text, not long narratives:

- `RunProfile`
  - `"Model {model_name} ran on {gpu_name} with batch size {resolved_batch_size}, {epochs} epochs, avg SM {avg_sm_utilization:.1%}, avg RAM {avg_ram_utilization:.1%}, epoch time {epoch_time_seconds}s."`
- `RuntimeProfile`
  - `"Runtime profile for signature {signature} on {gpu_name} with backend {backend_name}: startup {startup_seconds}s, epoch 1 {epoch_1_seconds}s, estimated total {estimated_total_runtime_seconds}s."`
- `BatchProbeProfile`
  - `"Batch probe for {model_name} on {device_type} resolved batch size {resolved_batch_size} under target budget {target_budget_mb} MB."`
- `PacketProfile`
  - `"Packed profile for {member_models} on {gpu_name} using {backend_name}: compatible={compatible}, peak VRAM {peak_vram_mb} MB, avg SM {avg_sm_utilization:.1%}."`
