# MLEvolve — Hardware-Aware Agent and Scheduler

This repository combines the MLEvolve search-agent framework with the hardware-aware training-job scheduler:

- **Agentic search runtime** (`agents/`, `engine/`, `llm/`, `run.py`) — plans, generates, reviews, selectively repairs, executes, and evolves ML solutions.
- **Trace replay benchmark** (`scheduler_benchmark_test/`) — replays recorded MLEvolve candidate timelines against the scheduler and a multiprocess baseline, plus stress-test fixtures.
- **Lesson Profile Database** (`lesson_profile_database/`) — independent SQLite/Qdrant/Redis memory built only from final-validated search nodes.
- **Scheduler** (`localml_scheduler/`) — local machine-learning-job scheduler with profiling, prediction, early stopping, checkpointing, packing, and observability.

## Layout

| Path | Purpose |
|------|---------|
| `agents/`, `engine/`, `llm/`, `run.py` | MLEvolve planning, generation, stage-aware review/repair, MCTS search, and execution runtime |
| `scheduler_benchmark_test/` | Trace replay entry points (`run_histopath_*.sh`), timeline extraction, stress-test data |
| `replay_model_sources/` | Archived model-source traces used by the replay fixtures |
| `localml_scheduler/` | Scheduler package (`client`, `scheduler/`, `execution/`, `prediction/`, `profiling/`, `storage/`, CLI) |
| `nn-model-preflight-checker/` | Pinned CPU model-preflight checker submodule used before GPU admission |
| `hardware_knowledge_graph/` | HWKB client/store code (Neo4j + vector DB) |
| `lesson_profile_database/` | Family/hardware observations, immutable revisions, role-scoped lessons, worker, CLI, and read-only MCP API |
| `hardware_graph_scripts/` | HWKB database setup, ingest, query, and verification scripts |
| `schema/` | Graph/vector DB schema YAMLs and `schema-guidance.md` |
| `engine/script_introspection.py` | Static introspection of training scripts (batch size, framework, script signature) used by the scheduler adapter |
| `utils/` | `candidate_timing.py` (phase instrumentation), `plot_hardware_awareness_comparison.py` |
| `tests/` | Scheduler and HWKB unit tests |
| `reports/` | Experiment records (stress-test benchmark) |
| `docker-compose.local.yml`, `docker_host_databases.sh`, `bootstrap.sh` | Local Neo4j, Qdrant, Redis, and database bootstrap infrastructure |

## Agent Framework

- [MLEvolve Agent Workflow](docs/mlevolve_agent_workflow.md): step-by-step walkthrough of how the search controller, stage agents, executor, and evaluation loop work together during a run.
- [Hardware-Aware Optimization](docs/mlevolve_hardware_aware_optimization.md): design notes for feeding scheduler profile evidence and code-knowledge retrieval back into MLEvolve stage agents.
- [Pipeline Stage Prompt Contract](docs/pipeline_stage_prompt_contract.md): configurable ordered `model-design -> datatype/quantization -> training` prompt contract and persisted decision trace.

## Timeline

- **2026-03-23** — Now supports OpenAI-compatible APIs (GPT, Qwen, DeepSeek, etc.). Models with function calling support are recommended for best performance.
- **2026-02-14** — MLEvolve codebase is now open-source.
- **2026-02-14** — MLEvolve achieves **#1 on MLE-bench** (12-hour budget).


## MLE-bench Results

Performance on the [MLE-bench](https://github.com/openai/mle-bench) leaderboard (Any Medal %, mean ± SEM):

| Rank | Agent | LLM | Low (%) | Medium (%) | High (%) | All (%) | Time (h) |
|------|-------|-----|---------|------------|----------|---------|----------|
| 1 | **MLEvolve (Ours)** | Gemini-3-Pro-Preview | **80.30 ± 1.52** | 57.89 ± 1.52 | **42.22 ± 2.22** | **61.33 ± 1.33** | 12 |
| 2 | PiEvolve | Gemini-3-Pro-Preview | **80.30 ± 1.52** | **58.77 ± 0.88** | 40.00 ± 0.00 | **61.33 ± 0.77** | 24 |
| 3 | Famou-Agent 2.0 | Gemini-2.5-Pro | 75.76 ± 1.52 | 57.89 ± 1.52 | 40.00 ± 0.00 | 59.56 ± 0.89 | 24 |
| 4 | ML-Master 2.0 | Deepseek-V3.2-Speciale | 75.76 ± 1.51 | 50.88 ± 3.51 | **42.22 ± 2.22** | 56.44 ± 2.47 | 24 |
| 5 | PiEvolve | Gemini-3-Pro-Preview | 74.24 ± 3.03 | 45.61 ± 0.88 | 35.55 ± 2.22 | 52.00 ± 0.77 | 12 |

## Coding Module in AI-Scientist

MLEvolve powers the **coding and algorithm optimization** module within the [InternAgent](https://github.com/InternScience/InternAgent) system. Built on MLEvolve's refinement engine, [InternAgent 1.5](https://arxiv.org/abs/2602.08990) **further enables autonomous algorithm design and end-to-end scientific discovery**.

## Key Technical Contributions

**Multi-Mode Planning & Code Generation** — Supports base (single-shot) and memory-enhanced (two-stage retrieval-augmented) planning, paired with three code generation strategies: single-pass, stepwise multi-agent pipeline, and incremental SEARCH/REPLACE diff patching. Different modes are dispatched adaptively based on search state.

**Experience-Driven Memory** — A global memory layer records plan, code, metrics, and success/failure labels for every node. Retrieval combines BM25 + FAISS allowing the planner to reinforce proven strategies and avoid known pitfalls from its own search history. Different agents query memory in different ways to encourage novel approaches.

**Progressive MCGS with Cross-Branch Fusion** — The search graph extends vanilla UCT with piecewise exploration decay, time-aware explore-exploit switching, and automatic stagnation detection. Multiple solution branches evolve in parallel; when progress stalls, the system performs cross-branch fusion — merging insights from top-performing nodes across different branches into new solution candidates — and trajectory-aware evolution that leverages each branch's full improvement history to propose informed next steps.

## Hardware-Aware Agent Integration

This branch extends the original MLEvolve structure with a scheduler-backed hardware and profile feedback loop. In the original flow, stage agents reasoned mainly from task text, data preview, search-tree memory, parent code, execution logs, and optional global memory. The new flow keeps that search architecture intact, but adds a compact hardware/profile context before each major reasoning and code-generation step.

What changed structurally:

- **Scheduler context reaches the agent layer**: `run.py` now attaches the in-process `SchedulerClient` to `AgentSearch`, so stage agents can ask for read-only optimization context without going through a separate MCP process.
- **Shared hardware prompt layer**: `agents/hardware_context.py` builds a candidate description, calls `get_optimization_context(...)`, compacts the graph/vector response, and formats a `Hardware/Profile Optimization Context` prompt section.
- **Lightweight generated-code introspection**: `engine/script_introspection.py` extracts batch size, epoch count, model/backbone hints, framework, AMP usage, GPU requirement, model family, and stable script signatures from generated Python. The executor reuses the same signature and batch-probe logic, so prompt-time reasoning and scheduler submission stay aligned.
- **All stage agents receive profile evidence**: draft, improve, debug, evolution, fusion, aggregation, planner, diff-generation, stepwise generation, merge, and code-review paths now receive the same compact context when the scheduler is enabled.
- **Training-script generation is more hardware-aware**: the stepwise `training_evaluation` agent is explicitly guided to choose physical batch size, precision, gradient accumulation, dataloader settings, checkpointing, and timeout/OOM fallbacks based on profile evidence.
- **Code review includes hardware-critical checks**: the reviewer still prioritizes data leakage and correctness, but can now flag concrete high-confidence hardware risks such as fixed oversized batch sizes, missing OOM fallback, or timeout-prone training budgets. It is still forbidden from replacing the model/backbone just for hardware convenience.
- **Search nodes retain compact evidence**: `SearchNode` now stores compact hardware context, graph evidence, derived diagnosis, vector evidence, scheduler risk flags, confidence, evidence refs, resolved batch size, runtime estimate, peak VRAM, and backend name. This makes hardware-aware decisions visible in journals without persisting raw graph/vector payloads.

The key behavior rule is: hardware recommendations are evidence, not law. Agents should follow high-confidence profile guidance by default, but may override it for leaderboard reasons if they explain the tradeoff and include a fallback such as smaller batch size, AMP, gradient accumulation, reduced resolution, fewer epochs, or checkpointing.

In practice, this improves MLEvolve in three places:

- **Drafting** starts from hardware-compatible defaults instead of blindly proposing memory-heavy first attempts.
- **Improvement and debugging** can react to OOMs, low SM utilization, timeout risk, precision inefficiency, and dataloader bottlenecks using empirical evidence from previous scheduler runs.
- **Evolution, fusion, and aggregation** can compare not only validation score, but also runtime, VRAM pressure, packing compatibility, and hardware risk when selecting which ideas to combine.

## Setup

```bash
# For a new checkout, use: git clone --recurse-submodules <MLEvolve repository URL>
# For an existing checkout, including after switching to this branch:
git submodule update --init --recursive

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
./install_dependencies.sh
python -c "import jsonschema, model_preflight; print('model preflight ready')"
cp config.example.yaml config.yaml   # fill in hardware_knowledge, lesson_profiles, and scheduler settings
bash docker_host_databases.sh        # local Neo4j + Qdrant + persistent Redis
bash bootstrap.sh                    # checks services; initializes lesson SQLite/Qdrant indexes
```

## Reducing End-to-End Run Time

MLEvolve has separate optimizations for LLM requests and GPU training jobs.
Enable and measure them separately: prompt caching primarily reduces repeated
LLM-prefix processing, while the scheduler reduces training time and queue drain
time. Start from `config.example.yaml`; the fragments below are intended as
edits to that complete configuration, not as standalone configuration files.

### Enable local context and provider prompt caching

Use a deterministic local pack cache and allow supported providers to reuse the
stable prompt prefix shared across agent calls:

```yaml
context_cache:
  enabled: true
  local_pack_cache_enabled: true
  provider_prompt_cache_enabled: true
  # Empty lists mean all configured roles and models. Populate either list for
  # a gradual rollout, for example: [reviewer].
  provider_prompt_cache_roles: []
  provider_prompt_cache_models: []
  cache_dir: var/context-cache
  policy: auto
  # This prepares local packs. The first real provider request warms its cache
  # family; MLEvolve does not send a synthetic, billable warm-up request.
  prewarm: true
  telemetry: true
  capture_prompts: false
  openrouter_sticky_routing: true
  openrouter_routing_shards: 1
```

Provider prompt-cache integration is implemented for OpenRouter, OpenAI,
DeepSeek, and vLLM. Direct Gemini and Anthropic integrations currently retain
the uncached request path. OpenRouter users should keep sticky routing enabled
and use one routing shard for the highest reuse; additional shards trade cache
hit rate for distribution. Provider misses and unsupported cache fields are
non-fatal.

For a self-hosted vLLM stage, enable automatic prefix caching in the vLLM
deployment and provide a private salt of at least 32 bytes. Do not commit it to
YAML:

```bash
export MLEVOLVE_VLLM_CACHE_SALT="$(openssl rand -base64 32)"
```

```yaml
vllm_client:
  cache_salt_env: MLEVOLVE_VLLM_CACHE_SALT
  require_cache_salt: true
  session_affinity: true
```

After a representative run, confirm real cache-read tokens rather than
inferring a hit from latency alone:

```bash
python -m context_cache export --format jsonl --output observations.jsonl
python -m benchmarks.context_cache_bench observations.jsonl \
  --output-dir benchmark-results/context-cache
```

See [Context Caching](docs/context-cache.md) for provider behavior, role/model
allowlists, environment-variable equivalents, staged rollout, and rollback.

### Enable time-aware GPU scheduling

The following settings let the scheduler reuse runtime profiles, probe
quality-safe batch choices, and admit concurrent jobs only when measured drain
time improves:

```yaml
experiment:
  mode: hardware_aware

scheduler:
  enabled: true
  start_service: true
  settings:
    prediction:
      mode: branch_profile
    gpu_scheduler:
      enabled: true
      mode: parallel_time_aware
      packing_backend: cuda_process
      parallel_job_cap: null
      batch_probe_enabled: true
      profiling:
        reuse_profile_if_confidence_ge: 0.8
      source_trial_ranking:
        source_analysis:
          cache_enabled: true
      submission_defaults:
        packing_eligible: true
        backend_allowlist: [cuda_process]
        batch_probe_enabled: true
        runtime_probe_enabled: true
```

Use `mps_process` in both backend fields only when the host is configured for
NVIDIA MPS; otherwise use `cuda_process`. A `null` parallel cap does not mean
unconditional concurrency: the scheduler incrementally applies backend,
quality, live-memory, compatibility, and measured drain-time gates before
admitting each additional job. Set a numeric cap if the machine has a stricter
operational limit.

Batch probing follows the quality contract: the agent approves
`QUALITY_SAFE_PHYSICAL_BATCH_SIZES` and the learning-rate scaling policy, then
the scheduler chooses the fastest currently feasible physical batch inside
that envelope. Do not increase batch size or reduce epochs mechanically;
validation quality and convergence evidence determine the safe envelope.

Scheduler-level early stopping requires a runner that emits epoch safe points.
Raw generated scripts are instead required by code review to implement
validation-based patience, save and restore the best checkpoint, and emit
`MLEVOLVE_EPOCH_METRIC` JSON after each validation epoch. Configure the
scheduler's `early_stopping.metric_name` to match the runner (`loss` in the
included examples, or preferably `val_loss` when available). Planned,
completed, and best epochs are recorded separately.

If sibling jobs share one immutable source or model artifact, configure
`scheduler.preload_source_model_path`, `scheduler.preload_source_model_id`, and
`scheduler.preload_source_loader_target` to avoid repeated loading. See
[localml_scheduler](localml_scheduler/README.md) for preload,
safe-point, profile-reuse, early-stopping, and backend details.

### Use bounded agent parallelism

Independent search branches and compact code patches can reduce wall-clock LLM
time:

```yaml
agent:
  use_diff_mode: true
  search:
    parallel_search_num: 2
```

Increase `parallel_search_num` only while provider rate limits, CPU capacity,
and scheduler/GPU admission remain healthy. Excess concurrency can increase
total time through throttling or GPU contention; two branches are a reasonable
starting point, not a universal optimum.

`install_dependencies.sh` initializes the checker submodule automatically and verifies both
`jsonschema` and `model_preflight` imports. See [CPU Model-Preflight Admission](docs/model_preflight_admission.md)
for policy, adapter, profile, and artifact details.

MLEvolve keeps two logical knowledge domains. Hardware knowledge is curated capability and optimization guidance in the existing Neo4j/Qdrant stores. Lesson profiles are run-derived experience in their own SQLite authority and the dedicated Qdrant collection `lesson_profile_records_v1`; Redis database 1 is only a cache. Records are never mixed across these domains. See [Lesson Profile Database](docs/lesson_profile_database.md).

## Running the Trace Benchmark

```bash
# scheduler replay of the recorded histopathologic-cancer-detection timeline
bash scheduler_benchmark_test/run_histopath_scheduler_replay.sh

# multiprocess baseline for comparison
bash scheduler_benchmark_test/run_histopath_multiprocess_baseline.sh

# trace performance summary
bash scheduler_benchmark_test/run_histopath_trace_performance.sh
```

Results and comparison plots are written under the benchmark output roots; stress-test records live in `reports/stress_test/`.

## Tests

```bash
pytest tests/ localml_scheduler/tests/
```

# Current Status

- Task: petfinder-pawpularity-score
