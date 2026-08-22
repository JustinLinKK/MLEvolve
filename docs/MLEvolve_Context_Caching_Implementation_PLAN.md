# MLEvolve Context-Caching Implementation Plan

**Status:** Ready for implementation  
**Primary objective:** Reduce repeated database work and LLM time-to-first-token (TTFT) for MLEvolve's multi-agent calls without changing agent behavior.  
**Primary provider:** OpenRouter  
**Also supported:** OpenAI and DeepSeek; leave clean extension points for Anthropic and Gemini.  
**Last reviewed:** 2026-08-21

---

## 1. Instructions for the coding agent

Implement this plan incrementally. Before changing code, inspect the repository and map the conceptual paths in this document to the actual package layout. Preserve existing public APIs unless a phase explicitly introduces a replacement.

For every phase:

1. Add or update tests with the implementation.
2. Run the narrow tests first, then the full project test suite.
3. Record any repository-specific deviation in the final implementation report.
4. Do not continue if a change alters uncached agent behavior or prompt content unexpectedly.
5. Keep all caching features behind configuration flags until the rollout phase.

Do not add Redis online version in the initial implementation, instead, I want you to implement redis on local RAM layer. MLEvolve is local-first and must work across local subprocesses using immutable files plus SQLite metadata.

---

## 2. Problem statement

MLEvolve repeatedly retrieves and serializes nearly identical knowledge from Qdrant and Neo4j for each agent request. It then sends the same long prefix—system instructions, tool definitions, hardware knowledge, evaluation rules, and role knowledge—to an LLM provider.

Total request time can be decomposed as:

```text
T_agent = T_DB + T_pack + T_upload + T_queue + T_prefill + T_decode
```

The implementation must address two separate costs:

- **Local knowledge-pack cache:** reduce `T_DB` and `T_pack` by avoiding repeated retrieval, sorting, rendering, and serialization.
- **Provider prompt/KV cache:** reduce `T_prefill` by keeping the repeated portion of the request as an exact, stable prefix.

Neither layer should cache semantically different model responses.

---

## 3. Scope and non-goals

### In scope

- Versioned, content-addressed knowledge packs for each agent role.
- Deterministic prompt construction with stable and dynamic regions.
- Provider-neutral cache policy and provider-specific request adapters.
- OpenRouter-first implementation, followed by OpenAI and DeepSeek.
- Provider usage normalization, TTFT measurement, cache telemetry, and benchmarks.
- Safe prewarming where a provider supports it.
- Multi-process-safe local pack compilation and reads.
- Feature flags, shadow verification, rollout, and rollback.

### Non-goals

- No semantic response cache across different tasks or inputs.
- No neural network or learned policy for cache placement or scheduling.
- No scheduler authority in Neo4j or Qdrant.
- No live queue, trace, prediction, or task state in a long-lived cached prefix.
- No Redis or additional always-on service in the first version.
- No provider-specific request logic inside individual agent classes.
- No assumption that all providers expose identical cache behavior.
- No exact-response caching for tool calls with side effects.

---

## 4. Current architecture assumptions

Confirm these paths and names before implementation:

- LangGraph construction: `engine/graph.py`
- Workflow orchestration: `engine/supervisor.py`
- Agent construction: `agents/factory.py`
- Shared model-call path: `agents/base.py`
- Agent roles:
  - `ModelGeneratorAgent`
  - `AnalysisAgent`
  - `ResultParseAgent`
  - `ReviewerAgent`
  - `SupervisorAgent`
- SQLite is the local control plane.
- Neo4j stores measured profile evidence.
- Qdrant stores code and hardware knowledge.
- PerfSeer-like CPU predictions and live execution traces are dynamic request data.

If the repository differs, preserve the same separation of responsibilities rather than forcing these exact paths.

---

## 5. Target architecture

```text
Qdrant + Neo4j
      |
      v
KnowledgePackCompiler -----> immutable content-addressed files
      |                                  |
      +---------------------------> SQLite registry
                                         |
Agent role + run snapshot                 v
      |                          KnowledgePackStore
      v                                  |
DeterministicPromptAssembler <-----------+
      |
      +-- stable tools
      +-- common knowledge pack
      +-- role knowledge pack
      +-- cache breakpoint
      +-- task retrieval / predictions / traces / history
      |
      v
ProviderCacheAdapter
      |
      +-- OpenRouter
      +-- OpenAI
      +-- DeepSeek
      +-- future Anthropic / Gemini
      |
      v
Normalized telemetry + benchmark results
```

The local cache and provider cache must remain independently switchable so their benefits can be measured separately.

---

## 6. Stable-prefix contract

Construct every provider request in this logical order:

```text
1. Stable tool definitions
2. Stable global/system instructions
3. Common knowledge pack
4. Role-specific knowledge pack
5. Explicit cache breakpoint, when supported
6. Job-specific retrieval
7. PerfSeer/hardware prediction for the current candidate
8. Current execution trace and conversation history
9. Current task/user input
```

Rules:

- Items 1–4 must be byte-stable for a cache family.
- Items 6–9 are dynamic and must not enter a long-lived prefix.
- A cache key is a routing hint; the exact prefix content is the real cache identity.
- Do not insert timestamps, random identifiers, run IDs, volatile similarity scores, or request metadata into items 1–4.
- Keep tool definitions in a stable order with stable JSON serialization.
- If provider rendering places tools before instructions, different tool sets will split the cache. Either give roles the same stable tool schema or deliberately use separate cache families.
- Freeze the selected knowledge versions for the duration of an MLEvolve run. Publish a new version between runs rather than mutating a pack in place.

### Role pack contents

| Agent role | Stable role knowledge | Dynamic suffix examples |
|---|---|---|
| Model generator | Model-design rules, supported operations, target hardware facts, batch-size guidance | Candidate request, current search state, recent failures |
| Analysis | Measurement interpretation, failure taxonomy, profiler guidance | Latest measurements and traces |
| Result parser | Output schema, log grammar, normalization rules | Raw logs and current output |
| Reviewer | CPU preflight rules, CUDA compatibility, correctness and repair rules | Candidate code, compiler output, current error |
| Supervisor | Workflow invariants and stop/continue policy | Queue state, budgets, run progress |

The reviewer pack and CPU preflight compiler must use the same frozen knowledge version.

---

## 7. Proposed module and data layout

Adapt the root package name to the repository:

```text
mlevolve/
  context_cache/
    __init__.py
    config.py
    models.py
    canonicalize.py
    compiler.py
    store.py
    assembler.py
    coordinator.py
    telemetry.py
    providers/
      __init__.py
      base.py
      openrouter.py
      openai.py
      deepseek.py
      anthropic.py       # optional stub or later implementation
      gemini.py          # optional stub or later implementation
  knowledge/
    manifests/           # source manifests, if repository-managed
  benchmarks/
    context_cache_bench.py
    scenarios.py
tests/
  context_cache/
  integration/
knowledge-packs/
  common/hardware-core-k17.json
  generator/model-design-k17.json
  analysis/measurement-k17.json
  reviewer/preflight-k17.json
  supervisor/scheduling-k17.json
  parser/log-schema-k17.json
var/
  context-cache/
    objects/<sha256>.json
    cache-registry.sqlite3
```

Do not commit runtime `var/context-cache` contents or secrets. Whether compiled packs under `knowledge-packs` are committed should be an explicit repository decision.

---

## 8. Core data contracts

Use typed dataclasses, Pydantic models, or the repository's existing model convention.

```python
from dataclasses import dataclass
from typing import Literal, Mapping

CacheMode = Literal["auto", "explicit", "none"]

@dataclass(frozen=True)
class CachePolicy:
    mode: CacheMode = "auto"
    ttl: str | None = None
    scope: str = "role"
    prewarm: bool = False

@dataclass(frozen=True)
class CacheCapabilities:
    explicit_breakpoints: bool
    ttl_values: tuple[str, ...]
    supports_prewarm: bool
    metrics_mapping: Mapping[str, str]

@dataclass(frozen=True)
class KnowledgePackRef:
    role: str
    schema_version: str
    knowledge_version: str
    content_sha256: str
    path: str

@dataclass(frozen=True)
class CacheFamily:
    provider: str
    model: str
    common_pack_hash: str
    role_pack_hash: str
    tool_schema_hash: str
    reasoning_config_hash: str

@dataclass(frozen=True)
class NormalizedCacheUsage:
    prompt_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cache_miss_tokens: int | None
    output_tokens: int | None
```

Calculate the cache-family ID as:

```text
sha256(
  provider + model + common_pack_hash + role_pack_hash
  + tool_schema_hash + reasoning_config_hash
)
```

Also include API family/version in the hash if Chat Completions and Responses serialize the same logical prompt differently.

### Knowledge-pack envelope

```json
{
  "schema_version": "1",
  "knowledge_version": "k17",
  "role": "reviewer",
  "content_sha256": "<sha256-of-canonical-content>",
  "compiled_at": "2026-08-21T00:00:00Z",
  "sources": [
    {"kind": "qdrant", "snapshot": "hardware-k17"},
    {"kind": "neo4j", "snapshot": "profiles-k17"}
  ],
  "content": {
    "sections": []
  }
}
```

`compiled_at` and source diagnostics belong in the envelope and must not be rendered into the stable prompt content.

---

## 9. Configuration and feature flags

Add configuration using the project's existing settings mechanism. Suggested environment variables:

```text
MLEVOLVE_CONTEXT_CACHE_ENABLED=false
MLEVOLVE_LOCAL_PACK_CACHE_ENABLED=true
MLEVOLVE_PROVIDER_PROMPT_CACHE_ENABLED=false
MLEVOLVE_CONTEXT_CACHE_DIR=var/context-cache
MLEVOLVE_CONTEXT_CACHE_POLICY=auto
MLEVOLVE_CONTEXT_CACHE_TTL=
MLEVOLVE_CONTEXT_CACHE_PREWARM=false
MLEVOLVE_CONTEXT_CACHE_TELEMETRY=true
MLEVOLVE_CONTEXT_CACHE_VERIFY_PREFIX=false
MLEVOLVE_OPENROUTER_STICKY_ROUTING=true
MLEVOLVE_OPENROUTER_ROUTING_SHARDS=1
```

Requirements:

- A global kill switch must restore the existing path.
- Local pack caching and provider prompt caching must be independently controllable.
- Unsupported policy values must fail configuration validation at startup.
- Unsupported provider capabilities must degrade to an uncached request, not fail an agent call.
- Never log provider credentials or full sensitive dynamic prompts.

---

## 10. Implementation phases

### Phase 0 — Inventory and baseline

**Goal:** Establish behavior and performance before changing request construction.

Tasks:

- Trace every model call from each agent through the shared client path.
- Record provider, API family, model, tools, instruction ordering, reasoning settings, streaming behavior, and usage fields.
- Identify every Qdrant and Neo4j query used to build recurring context.
- Capture sanitized prompt snapshots for all five agent roles.
- Add baseline timing at these boundaries:
  - `t0`: knowledge retrieval begins
  - `t1`: knowledge/prompt pack is ready
  - `t2`: provider request starts
  - `t3`: first meaningful text or tool-call delta arrives
  - `t4`: response completes
- Run at least 20 representative uncached calls per primary model, if API budget permits.

Deliverables:

- `docs/context-cache-baseline.md`
- Sanitized prompt snapshots in tests.
- Machine-readable baseline JSON or CSV.
- A map from agent roles to model-call and retrieval code paths.

Exit criteria:

- Existing prompt text and output behavior are covered by regression tests.
- Baseline includes median and p95 retrieval time, TTFT, total request time, and end-to-end time.

### Phase 1 — Models, configuration, and disabled-path integration

**Goal:** Add the abstraction without activating caching.

Tasks:

- Implement the core data contracts from Section 8.
- Add capability discovery and the feature flags from Section 9.
- Add a provider-adapter registry selected in the shared model-call layer.
- Pass cache policy through the shared agent base/factory, not through individual role logic.
- Implement a no-op adapter that produces the exact existing request.

Tests:

- Invalid configuration is rejected.
- Unknown providers use the no-op adapter.
- With all flags disabled, serialized requests match baseline snapshots byte-for-byte after redaction.

Exit criteria:

- Full test suite passes.
- The disabled path is behaviorally identical to the current system.

### Phase 2 — Deterministic knowledge-pack compiler and local store

**Goal:** Eliminate repeated database retrieval and rendering for stable knowledge.

Tasks:

- Implement a manifest for the common pack and each role pack.
- Query Qdrant and Neo4j only during pack compilation.
- Canonicalize all retrieved content:
  - sort records by stable identifiers;
  - sort object keys;
  - use stable section ordering;
  - normalize line endings and Unicode;
  - strip timestamps, volatile scores, database IDs that are not semantic, and transient metadata;
  - use deterministic float and JSON formatting;
  - reject non-finite numbers.
- Hash canonical semantic content, not filesystem metadata.
- Write immutable objects to `objects/<sha256>.json` using temp-file plus atomic rename.
- Store aliases, versions, paths, source snapshots, and access metadata in SQLite.
- Add a file lock or SQLite lock so only one subprocess compiles a missing pack.
- Allow concurrent readers without a writer lock.
- Freeze selected pack references in the MLEvolve run record.
- Provide CLI commands equivalent to:

```text
mlevolve context-cache compile --version k17
mlevolve context-cache inspect --role reviewer --version k17
mlevolve context-cache verify --version k17
mlevolve context-cache list
```

Tests:

- Identical inputs produce identical bytes and SHA-256 across repeated runs.
- Reordered database results produce the same hash.
- A semantic one-byte change produces a different hash.
- Volatile metadata changes do not change semantic content or its hash.
- Concurrent compilation creates one valid object and no partial files.
- Missing/corrupt local objects are rebuilt safely.
- A repeated pack load performs no Qdrant or Neo4j query.

Exit criteria:

- Repeated role-context construction reads immutable local objects only.
- Local cache failure falls back to compilation or the existing retrieval path according to configuration.

### Phase 3 — Deterministic prompt assembler

**Goal:** Create stable provider prefixes while preserving prompt semantics.

Tasks:

- Implement `DeterministicPromptAssembler` as the sole builder of the provider-facing prompt structure.
- Separate `stable_prefix` from `dynamic_suffix` in the internal representation.
- Canonicalize tool schemas and keep their order stable.
- Render common and role packs using fixed templates with no timestamps or run IDs.
- Place dynamic retrieval, PerfSeer predictions, current traces, history, and task after the breakpoint.
- Compute and log hashes for the stable prefix and each component.
- Add an optional verification mode that reconstructs the prefix twice and fails tests if hashes differ.
- Snapshot the assembled logical prompt for each role.

Tests:

- Same role, versions, tools, model, and reasoning config produce the same prefix hash.
- Dynamic task changes do not change the prefix hash.
- Tool, pack, model, API-family, or reasoning-config changes produce a new cache-family ID.
- No forbidden volatile field appears before the breakpoint.
- Review and CPU preflight use the same knowledge version.

Exit criteria:

- Prompt snapshots show unchanged semantics.
- Stable prefix remains identical across representative tasks for each cache family.

### Phase 4 — Provider adapters

**Goal:** Translate the semantic policy into current provider-specific request fields.

Implement providers in this order:

#### 4A. OpenRouter adapter

- Preserve identical prefix bytes and message/tool ordering.
- Use a stable session identifier for sticky routing when supported:

```text
mlevolve:{model}:{common_pack_hash}:{routing_shard}
```

- Default `routing_shard` to one for the expected 3–4 concurrent agents. Make bounded sharding configurable.
- Do not create a unique session ID per request.
- Surface the selected upstream provider in telemetry.
- Normalize upstream cache metrics returned by OpenRouter.
- Support immutable OpenRouter preset names if the deployment uses presets, for example `@preset/mlevolve-reviewer-k17`.
- Do not silently update a preset in place. Publish a new versioned slug.
- For controlled benchmarks, support pinning the upstream and disabling fallbacks so routing changes do not hide cache behavior.
- Keep full-response caching off by default. If later enabled, restrict it to explicitly idempotent, identical retries with no side-effecting tools.

#### 4B. OpenAI adapter

- Prefer the Responses API when it matches the existing client architecture.
- Set a stable `prompt_cache_key` derived from the cache family.
- For models supporting current explicit controls, map policy to `prompt_cache_options` and mark the content boundary with `prompt_cache_breakpoint`.
- As of 2026-08-21, the Responses API documents explicit controls for GPT-5.6 and later, one implicit breakpoint by default, up to four explicit writes per request, and `30m` as the supported `prompt_cache_options.ttl` value. Treat this as a capability table, not a universal assumption.
- Normalize `usage.input_tokens_details.cached_tokens` to cache reads and `cache_write_tokens` to cache writes when present.
- Do not assume a documented cache-only prewarm request exists.
- Fall back to automatic prefix caching on models without explicit controls.

#### 4C. DeepSeek adapter

- Send the exact common prefix at the start of each request; caching is automatic and best effort.
- Do not invent a cache key, breakpoint, TTL, or prewarm control.
- Normalize `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`.
- Benchmark at least three repeated calls because a newly constructed prefix may not be observable as a hit immediately.
- Treat cache eviction and retention as provider-managed.

#### 4D. Optional extension adapters

- **Anthropic:** map explicit `cache_control` breakpoints and supported 5-minute/1-hour TTLs; support cache-only warming only after an integration test confirms current `max_tokens=0` behavior.
- **Gemini:** map implicit caching and optionally explicit cache resources; normalize cached-token usage.

Shared adapter requirements:

- `capabilities(model)`
- `apply_cache_policy(logical_request, family, policy)`
- `extract_cache_usage(raw_response)`
- `extract_upstream_provider(raw_response)` when applicable
- `supports_prewarm(model, policy)`
- No cache-specific conditionals in agent classes.

Tests:

- Golden request fixtures for each provider.
- Usage-mapping fixtures for missing, zero, and populated metrics.
- Unsupported modes fall back safely.
- Adapter failure never corrupts or drops dynamic context.
- Real provider integration tests are opt-in and guarded by environment variables/markers.

Exit criteria:

- OpenRouter, OpenAI, and DeepSeek can run through the same logical request interface.
- Each adapter reports explicit capabilities and normalized metrics.

### Phase 5 — Telemetry and trace persistence

**Goal:** Make local and provider-cache effects observable and comparable.

Record one event per agent request with at least:

```text
timestamp
run_id
request_id
provider
upstream_provider
api_family
model
agent_role
cache_family_id
common_pack_hash
role_pack_hash
tool_schema_hash
reasoning_config_hash
local_pack_cache_hit
db_retrieval_ms
pack_build_ms
request_prepare_ms
ttft_ms
total_request_ms
end_to_end_ms
prompt_tokens
cache_read_tokens
cache_write_tokens
cache_miss_tokens
output_tokens
cache_hit_ratio
cost_usd
finish_reason
error_type
```

Definitions:

```text
ttft_ms = t3 - t2
total_request_ms = t4 - t2
end_to_end_ms = t4 - t0
cache_hit_ratio = cache_read_tokens / expected_stable_prefix_tokens
```

Implementation requirements:

- For streaming, `t3` is the first meaningful text delta or complete tool-call delta, not merely a transport event.
- Store normalized events in SQLite using the existing analytics/control-plane pattern.
- Export JSON Lines and CSV for benchmarks.
- Do not store hidden reasoning or credentials.
- Make full prompt capture opt-in, sanitized, and disabled by default.
- Preserve raw provider usage in a bounded JSON field for debugging schema changes.

Tests:

- Timing math uses a monotonic clock.
- Missing provider usage produces `null`, not invented zeroes.
- Cache hit ratio handles missing and zero denominators.
- Streaming and non-streaming paths both emit one final event.

Exit criteria:

- The same query can compare providers, models, roles, cache families, and cold/warm calls.

### Phase 6 — Warmup coordinator and concurrency

**Goal:** Avoid a fanout of simultaneous cold requests and preserve cache locality.

Tasks:

- Add a run-start coordinator after the run's pack versions and models are frozen.
- Compile/load all required local packs before agent fanout.
- Warm only providers/models that explicitly support a safe warmup strategy.
- If no cache-only warmup exists, optionally designate the first real request as the leader; do not create a billable dummy request by default.
- Add singleflight per `(provider, upstream, model, cache_family_id)` within one process.
- Use stable OpenRouter sticky-routing IDs and bounded shards across subprocesses.
- Ensure cache warmup failure is logged and normal execution continues.

Tests:

- Four simultaneous local pack requests compile once.
- Warm-first then fanout produces the expected provider cache reads where supported.
- A warmup timeout does not block the run indefinitely.
- Provider fallback changes are visible in telemetry.

Exit criteria:

- Local compilation completes before the first parallel agent wave.
- No cache failure can fail an otherwise valid MLEvolve task.

### Phase 7 — Benchmark harness

**Goal:** Quantify latency improvements and detect false cache-hit assumptions.

Implement four primary modes:

1. Baseline: database retrieval + provider cold path.
2. Local-only: local knowledge pack enabled, provider cache isolated/cold.
3. Provider-only: database work retained, repeated provider prefix warmed.
4. Both: local pack and provider prompt caching enabled.

Benchmark matrix:

| Dimension | Required cases |
|---|---|
| Context length | Actual role context, approximately 4K, 16K, and 64K tokens where model limits permit |
| Reuse | Same-agent warm reuse; cross-agent common-root reuse |
| Invalidation | One semantic byte; pack version; tool schema; model; reasoning config |
| Concurrency | 1 and 4 calls |
| Idle gap | Immediate, 6, 12, 31, and 61 minutes where cost/time permits |
| OpenRouter routing | Pinned upstream with fallbacks off; normal production routing |
| Calls per condition | At least 20 measured trials after warmup when budget permits |

Use paired trials. Add a unique but semantically neutral trial marker before the would-be stable prefix when forcing a cold condition; keep it out of production prompts. Record the exact method used to produce cold and warm conditions.

Report:

```text
TTFT speedup = median(cold TTFT) / median(warm TTFT)
End-to-end speedup = median(cold end_to_end_ms) / median(warm end_to_end_ms)
Local retrieval saved = median(baseline DB+pack) - median(local-only DB+pack)
```

For every condition, report p50 and p95 for TTFT and end-to-end time, token counts, normalized cache-read/write/miss tokens, hit ratio, selected upstream, errors, and cost.

Benchmark outputs:

- Timestamped JSON Lines with raw normalized observations.
- CSV summary.
- Markdown report with environment, model IDs, provider routes, commit SHA, configuration, and limitations.

Exit criteria:

- Results distinguish local-cache savings from provider-prefill savings.
- A provider cache hit is claimed only when supported by usage metrics or a clearly labeled latency inference.

### Phase 8 — Shadow verification and rollout

**Goal:** Enable caching without an accuracy regression.

Rollout sequence:

1. Enable local pack compilation only; continue using the existing prompt path.
2. In shadow mode, assemble both prompts and compare sanitized structures/hashes without making a second provider call.
3. Enable the deterministic assembler with provider caching disabled.
4. Enable provider caching for one low-risk role and one model.
5. Expand role by role on OpenRouter.
6. Enable OpenAI and DeepSeek adapters after their integration benchmarks pass.
7. Make caching the default only after the observation window meets the acceptance criteria.

Quality gates:

- Compare task success, generated-code validity, CPU preflight pass rate, GPU execution success, reviewer disagreement, retries, and final objective quality against baseline.
- Abort rollout if error rate or correctness metrics regress beyond the project's existing tolerance.
- Keep the global kill switch operational and documented.

Rollback:

- Disable provider caching first while retaining immutable local packs.
- If prompt behavior differs, disable the deterministic assembler and return to the no-op path.
- Never delete packs during rollback; mark aliases inactive so artifacts remain auditable.

Exit criteria:

- At least one full representative MLEvolve workload completes with stable quality and improved latency.
- Rollback is exercised in staging or a local integration run.

### Phase 9 — Documentation and maintenance

Tasks:

- Document how to compile, inspect, verify, publish, and retire knowledge versions.
- Document per-provider capabilities and the date they were verified.
- Add a maintainer checklist for provider SDK or API upgrades.
- Document how tools and reasoning settings affect cache-family identity.
- Add a troubleshooting guide for low hit ratios, route drift, simultaneous cold starts, and stale version selection.
- Add retention cleanup for unreferenced local objects. Cleanup must be a separate explicit command with dry-run support.

Exit criteria:

- A new agent role or provider can be added without modifying existing agent classes.
- An operator can disable, inspect, benchmark, and roll back caching from documented commands/configuration.

---

## 11. Provider capability matrix

Treat this table as configuration that must be verified against current official documentation and SDK behavior during implementation.

| Provider path | Cache placement | Explicit control | Useful metrics | MLEvolve strategy |
|---|---|---|---|---|
| OpenRouter | Depends on routed upstream | Normalized/provider-specific; sticky routing matters | Cached read/write tokens where upstream exposes them; upstream identity | Primary path; stable session ID; benchmark pinned and normal routing |
| OpenAI direct | Automatic prefix caching; explicit controls on supported newer models | `prompt_cache_key`, `prompt_cache_options`, content breakpoint on supported models | Cached and cache-write input tokens | Use family-derived key; explicit boundary where supported; otherwise stable-prefix automatic caching |
| DeepSeek direct | Automatic disk prefix cache | No application key/breakpoint/TTL | Cache-hit and cache-miss prompt tokens | Exact prefix; run 3+ repetitions; treat retention as best effort |
| Anthropic direct | Automatic/explicit prompt caching | `cache_control`, supported TTL values, limited breakpoints | Cache reads and creation tokens | Optional adapter; role/common boundaries; validate cache-only warmup |
| Gemini direct | Implicit and explicit cache-resource modes | Model/API dependent | Cached token count | Optional adapter; explicit resource for sufficiently reused long context |

Provider cache misses, metric omissions, evictions, and route changes are normal operational outcomes—not fatal errors.

---

## 12. Required tests

### Unit tests

- Canonicalization is deterministic.
- Hashing is stable and sensitive only to semantic changes.
- Volatile fields are excluded.
- Tool order and JSON serialization are stable.
- Cache-family invalidation covers provider, upstream constraints, model, API family, packs, tools, and reasoning configuration.
- Provider capability and usage mappings handle absent fields.
- Dynamic state cannot be rendered before the breakpoint.

### Integration tests

- Fake provider simulates cold write, warm read, miss, eviction, and missing metrics.
- Each real provider test is opt-in and skips cleanly without credentials.
- OpenRouter test records actual upstream identity.
- OpenAI test validates request schema for both explicit-capable and automatic-only models.
- DeepSeek test performs sufficient repetitions to observe available metrics.

### Concurrency tests

- Multiple subprocesses request the same missing pack.
- Exactly one complete content-addressed object is published.
- Readers never observe partial content.
- Warmup leader failure releases waiters.
- Four-agent fanout remains functional with caching disabled, enabled, and partially supported.

### Regression tests

- Disabled mode matches baseline request snapshots.
- Cached and uncached runs satisfy the same agent-output contracts.
- Model generation, analysis, parsing, review, and supervision each retain role-specific instructions.
- CPU preflight and reviewer pack versions cannot diverge within a run.

---

## 13. Security, privacy, and correctness controls

- Never put secrets, access tokens, user identifiers, or database credentials in a pack.
- Classify each pack's contents and confirm provider data-handling policy before sending it externally.
- Prefer hashes or internal opaque identifiers in telemetry.
- Redact dynamic prompts and tool results from default logs.
- Validate pack schema before use.
- Enforce a maximum pack size and provider/model context limit before request submission.
- Treat local cache contents as derived artifacts; source-of-truth updates require a new knowledge version.
- Never use stale cached scheduler state, current workload state, or PerfSeer output.
- Exact-response cache entries, if ever added, must exclude side-effecting tool calls and nondeterministic tasks.

---

## 14. Definition of done

The project is complete when all of the following are true:

- [ ] All five current agent roles use the shared deterministic assembler.
- [ ] Repeated calls for a frozen knowledge version cause no repeated Qdrant/Neo4j retrieval for stable packs.
- [ ] OpenRouter, OpenAI, and DeepSeek adapters share one provider-neutral interface.
- [ ] Stable-prefix and cache-family hashes are recorded for each request.
- [ ] Cache read/write/miss metrics are normalized when the provider exposes them.
- [ ] TTFT and end-to-end latency are measured correctly for streaming and non-streaming calls.
- [ ] Pack, tool, model, API-family, and reasoning changes invalidate the appropriate cache family.
- [ ] Dynamic state is absent from long-lived cached prefixes.
- [ ] Multi-process compilation is atomic and corruption-safe.
- [ ] Cache misses and adapter failures fall back without failing valid agent calls.
- [ ] `MLEVOLVE_CONTEXT_CACHE_ENABLED=false` restores baseline behavior.
- [ ] Benchmarks produce JSON Lines, CSV, and a human-readable report with p50/p95 results.
- [ ] Quality metrics show no material accuracy regression on representative workloads.
- [ ] Rollout and rollback procedures have been exercised.
- [ ] Provider capability assumptions and verification dates are documented.

---

## 15. Recommended first implementation slice

For the smallest useful pull request, implement only:

1. Phase 0 baseline instrumentation and prompt snapshots.
2. Phase 1 models, flags, and no-op adapter.
3. Phase 2 common + reviewer knowledge packs with deterministic hashing and SQLite registry.
4. Phase 3 deterministic prompt assembly for `ReviewerAgent` only.
5. OpenRouter adapter telemetry without enabling provider caching by default.

This slice proves the architecture on the role most closely tied to the CPU preflight knowledge version. It also limits behavioral risk before expanding to generation and supervision.

Suggested pull-request sequence after that:

```text
PR 1: baseline + abstractions + disabled-path parity
PR 2: local immutable knowledge packs + compiler/store
PR 3: reviewer deterministic prompt + OpenRouter adapter
PR 4: remaining roles + normalized telemetry
PR 5: OpenAI and DeepSeek adapters
PR 6: warmup coordinator + benchmark harness
PR 7: controlled rollout + documentation
```

---

## 16. Handoff prompt for a coding agent

Copy the following together with this file when starting implementation:

```text
Implement the MLEvolve context-caching plan in
MLEvolve_Context_Caching_Implementation_PLAN.md.

Start with Phase 0 and the recommended first implementation slice. Inspect the
repository before editing and map the conceptual module paths to the actual
layout. Preserve existing behavior with caching disabled. Add tests in every
phase, run narrow tests and then the full suite, and stop if prompt snapshots or
agent behavior change unexpectedly. Do not add Redis, do not put dynamic run
state before a cache breakpoint, and do not add provider-specific logic to agent
classes. At the end, report changed files, test commands/results, benchmark
results available so far, deviations from the plan, and the next recommended PR.
```

---

## 17. References to verify during implementation

- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
- [OpenAI Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [OpenRouter prompt caching](https://openrouter.ai/docs/features/prompt-caching)
- [OpenRouter provider routing](https://openrouter.ai/docs/features/provider-routing)
- [OpenRouter presets](https://openrouter.ai/docs/features/presets)
- [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache)
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching)

Provider APIs change. Pin SDK versions, save golden fixtures, and update the capability table only after verification against current official documentation and a live integration test.
