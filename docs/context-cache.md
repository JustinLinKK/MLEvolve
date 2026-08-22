# MLEvolve context caching

MLEvolve has two independent cache layers:

1. A local-first knowledge-pack store. Canonical semantic content is written to
   immutable `objects/<sha256>.json` files; SQLite stores aliases, run-frozen
   references, access metadata, and telemetry. A bounded process-local RAM
   layer accelerates reads. No Redis service is used.
2. Provider prompt caching. One deterministic logical prefix and cache-family
   identity are translated by OpenRouter, OpenAI, or DeepSeek adapters. Provider
   misses, evictions, omitted metrics, and adapter failures are non-fatal.

Both layers are off behind the global kill switch by default.

## Configuration

The complete YAML example is in `config.example.yaml`. Environment equivalents
are:

```text
MLEVOLVE_CONTEXT_CACHE_ENABLED=false
MLEVOLVE_LOCAL_PACK_CACHE_ENABLED=true
MLEVOLVE_PROVIDER_PROMPT_CACHE_ENABLED=false
MLEVOLVE_PROVIDER_PROMPT_CACHE_ROLES=
MLEVOLVE_PROVIDER_PROMPT_CACHE_MODELS=
MLEVOLVE_CONTEXT_CACHE_DIR=var/context-cache
MLEVOLVE_CONTEXT_CACHE_POLICY=auto
MLEVOLVE_CONTEXT_CACHE_TTL=
MLEVOLVE_CONTEXT_CACHE_PREWARM=false
MLEVOLVE_CONTEXT_CACHE_TELEMETRY=true
MLEVOLVE_CONTEXT_CACHE_VERIFY_PREFIX=false
MLEVOLVE_CONTEXT_CACHE_CAPTURE_PROMPTS=false
MLEVOLVE_CONTEXT_CACHE_SHADOW=false
MLEVOLVE_CONTEXT_CACHE_KNOWLEDGE_VERSION=k1
MLEVOLVE_OPENROUTER_STICKY_ROUTING=true
MLEVOLVE_OPENROUTER_ROUTING_SHARDS=1
MLEVOLVE_OPENROUTER_UPSTREAM=
MLEVOLVE_OPENROUTER_ALLOW_FALLBACKS=true
```

`MLEVOLVE_CONTEXT_CACHE_ENABLED=false` bypasses pack I/O, deterministic
assembly, adapter fields, stream-usage options, and cache telemetry. Local and
provider caching can then be isolated by their two subordinate flags.
Unsupported policy, Boolean, shard-count, TTL/capability, and pack-size inputs
either fail configuration at startup or safely produce an uncached provider
request. `preflight.knowledge_version` is frozen to the same version as the
reviewer context pack.

Full prompt capture is disabled by default. When explicitly enabled it stores a
bounded snapshot after redacting credential-shaped fields and text. Treat those
snapshots as sensitive operational data even after sanitation. Default
telemetry stores only hashes and bounded provider usage.

## Pack lifecycle

The repository's command entry point is `python -m context_cache`:

```bash
python -m context_cache compile --version k17
python -m context_cache inspect --role reviewer --version k17
python -m context_cache verify --version k17
python -m context_cache list
python -m context_cache retire --role reviewer --version k17
python -m context_cache cleanup                 # dry-run
python -m context_cache cleanup --execute       # explicit deletion
```

The command differs from the plan's illustrative `mlevolve context-cache`
syntax because this repository has no packaged top-level `mlevolve` CLI.

Source manifests are in `knowledge/manifests`. The checked-in `k1` manifests
are intentionally empty migration packs: enabling caching does not add new
instructions to existing prompts. To publish real stable knowledge:

1. Add deterministic sections or register a compiler source resolver for a
   frozen Qdrant/Neo4j snapshot.
2. Use stable record identifiers and keep task retrieval, predictions, traces,
   history, queue state, and scores out of the pack.
3. Choose a new version; an existing `(role, version)` alias cannot be changed
   in place.
4. Compile, inspect, verify, run snapshots/tests, then enable shadow mode.

Canonicalization normalizes Unicode and line endings, sorts keys and explicitly
set-like record/source collections, rejects non-finite numbers, removes known
volatile metadata, and rejects credential-bearing fields. Publication uses a
temporary file, fsync, and atomic rename while `BEGIN IMMEDIATE` serializes
missing-pack compilation across subprocesses. Readers need no writer lock.

Run references are frozen in SQLite. Retiring an alias does not delete its
object or a run reference. Cleanup removes only objects with no alias and no
run reference, and only with `--execute`.

## Stable-prefix contract

The deterministic assembler hashes stable tools, optional stable system
instructions, the common pack, and the role pack. Original task messages stay
in their original order after that prefix. Tools are canonicalized by function
name. Cache-family identity includes provider, model, API family, both pack
hashes, tool schema, reasoning settings, and upstream-routing constraints.

Callers can supply `context_cache_stable_prefix` only for content proven stable.
The five role labels use the shared LLM boundary; provider-specific fields never
appear in agent classes. Dynamic hardware/profile evidence remains in original
messages, which is the safe initial rollout behavior.

## Provider capabilities

Verified against official documentation on 2026-08-22:

| Path | Implemented behavior | Metrics |
|---|---|---|
| OpenRouter | Stable `session_id`, bounded shards, optional pinned upstream/fallback control, automatic caching, and explicit text-block breakpoints on supported routed models. Versioned preset names such as `@preset/...` can be used as the configured model and therefore enter family identity. | `prompt_tokens_details.cached_tokens` and `cache_write_tokens`; upstream provider when returned. |
| OpenAI direct | Stable `prompt_cache_key`; automatic caching on older models; Chat Completions `prompt_cache_breakpoint` plus `prompt_cache_options` for GPT-5.6+; supported TTL is `30m`. | Chat `prompt_tokens_details` and Responses-shaped `input_tokens_details`. |
| DeepSeek direct | Exact-prefix automatic caching only. No invented key, breakpoint, TTL, or prewarm request. | `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`. |
| Anthropic / Gemini direct | Registry stubs only. No direct-SDK cache fields are emitted until opt-in integration fixtures exist. | None yet. |

Sources: [OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching),
[OpenAI Responses prompt-cache fields](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache).

OpenAI remains on Chat Completions to preserve the current backend and output
contracts; migration to Responses is a separate change. No implemented provider
documents a safe cache-only warm request, so run preparation compiles/fetches
local packs only and never sends a billable dummy request.

When upgrading an SDK or provider API, rerun golden adapter fixtures, check
accepted request fields, validate missing/zero/populated usage mappings, and run
an opt-in cold/warm integration before changing this table.

## Telemetry and benchmarks

Each enabled request writes one terminal row to `context_cache_events` in the
cache registry. The row includes component/family hashes, local hit state,
retrieval/build/preparation time, TTFT, total and end-to-end time, normalized
tokens, hit ratio, cost when exposed, finish reason, upstream provider, and
error type. Export with:

```bash
python -m context_cache export --format jsonl --output observations.jsonl
python -m context_cache export --format csv --output observations.csv
python -m benchmarks.context_cache_bench observations.jsonl \
  --output-dir benchmark-results/context-cache
```

The harness supports the four required modes (`baseline`, `local-only`,
`provider-only`, `both`), p50/p95 summaries, paired TTFT/end-to-end speedups,
local retrieval savings, context-size/reuse/concurrency/idle-gap/routing
dimensions, error/cost reporting, and JSONL/CSV/Markdown output. Live runners
must record at least 20 post-warmup trials per condition when budget permits.
DeepSeek conditions need at least three repetitions.

Cold benchmark trials may use `benchmarks.scenarios.inject_cold_marker`, which
puts a unique neutral marker before the would-be prefix. That helper is isolated
from production assembly. A cache hit is confirmed only by positive provider
cache-read metrics; latency without metrics is labeled an inference.

## Rollout and rollback

Recommended rollout:

1. Enable the global switch and local packs only.
2. Set `shadow: true` to compute structures/hashes while sending original
   requests.
3. Disable shadow with provider caching still off.
4. Enable provider caching for one reviewer model, then expand role by role.
5. Run paired quality/latency workloads before making any flag a default.

Track task success, code validity, CPU preflight admission, GPU execution,
reviewer disagreement, retries, and objective quality. To roll back, first set
`provider_prompt_cache_enabled: false`; then use `shadow: true` or disable the
global switch. Do not delete packs. The test suite exercises kill-switch parity,
partial support, immutable rollback artifacts, and corrupted-object recovery;
a live representative workload still requires operator credentials and budget.

## Troubleshooting

- Low hit ratio: compare `stable_prefix_hash`, tool/reasoning hashes, model/API
  family, prompt length, and cache-read metrics. Check idle expiry.
- OpenRouter route drift: inspect `upstream_provider`; keep `session_id` stable.
  For controlled measurements set one upstream and disable fallbacks.
- Simultaneous cold starts: verify all processes use the same cache directory
  and knowledge version. SQLite locking should yield one source compilation.
- Repeated database work: verify the alias/object, local-hit flag, and run-frozen
  reference. Candidate-specific retrieval correctly remains dynamic.
- Corrupt or missing object: the next compilation rebuilds it atomically. Run
  `verify` to audit all aliases.
- Stale version: publish a new version and start a new run; never mutate an
  active alias.

To add a role, add one manifest and pass its role label at the shared LLM call.
To add a provider, implement the adapter interface (`capabilities`, policy
translation, usage extraction, optional upstream extraction) and register it;
existing agent classes do not change.
