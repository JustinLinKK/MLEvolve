# Context-cache baseline and call-path inventory

Captured on 2026-08-22 before enabling the new path. The global context-cache
switch defaults to `false`; the existing request construction remains the
behavioral baseline.

## Actual repository map

The plan's conceptual `agents/base.py` is implemented here by the shared
`llm/__init__.py` dispatcher and the `llm/openai.py` / `llm/gemini.py`
backends. This repository does not use LangGraph agent classes with the names in
the design document. The role mapping used for cache telemetry is:

| Cache role | Current callers | Shared call |
|---|---|---|
| `model_generator` | draft, improve, evolution, fusion, aggregation, planner, and stepwise coder paths | `llm.generate` |
| `analysis` | debug/failure analysis | `llm.generate` |
| `result_parser` | metric-direction and execution-result parsing | `llm.query` |
| `reviewer` | code review, stage repair, data-leakage review, and validation | `llm.query` / `llm.generate` |
| `supervisor` | hardware-aware pipeline decision | `llm.generate` |

OpenRouter, OpenAI direct, DeepSeek direct, and other OpenAI-compatible endpoints
currently use OpenAI Chat Completions. Text generation streams; structured
function calls are non-streaming. Gemini uses its native SDK and remains an
extension path for provider caching.

## Existing prompt and retrieval behavior

- `agents/hardware_context.py` obtains candidate-specific optimization and
  model-design context through `SchedulerClient`. That path may reach Neo4j,
  Qdrant-backed knowledge, prediction data, or current scheduler state. It is
  dynamic and remains after the cache boundary.
- `agents/lesson_context.py` calls `LessonProfileClient.profile_for_agent` for
  family/hardware evidence. Candidate-specific results remain dynamic.
- `AgentSearch.attach_scheduler` prewarms the current hardware graph
  neighborhood independently of LLM prompt caching.
- Stable knowledge-pack compilation supports injected Qdrant and Neo4j source
  resolvers, but the checked-in migration manifests contain empty sections.
  This preserves current prompts until maintainers deliberately publish stable
  knowledge under a new immutable version.

Sanitized logical snapshots for all five mapped roles live in
`tests/context_cache/snapshots/role_prompts.json`. Dynamic task, prediction,
trace, queue, and output markers are all after the stable prefix.

## Timing boundaries

The shared telemetry uses a monotonic clock:

- `t0`: context-cache preparation starts.
- `t1`: local packs and the deterministic prefix are ready.
- `t2`: the provider call starts.
- `t3`: the first non-empty text delta arrives, or the complete non-streaming
  response/tool call arrives.
- `t4`: the response finishes or fails.

The normalized event stores `t3-t2` as TTFT, `t4-t2` as provider request time,
and `t4-t0` as end-to-end time. Missing provider usage remains `null`.

## Baseline run status

The pre-change regression slice passed 65 tests. No provider credentials or API
budget were supplied, so no billable 20-call baseline was run and no latency
numbers are claimed. The machine-readable record is
`benchmarks/context_cache_baseline.json`; its latency fields are intentionally
`null`. Use the benchmark procedure in `docs/context-cache.md` to collect live,
paired measurements.

