# vLLM server handoff for MLEvolve

This document covers the server-side work for the cache-aware `provider: vllm`
client. The repository itself remains client-only. MLEvolve uses Chat
Completions, sends a deterministic stable prefix, adds a private `cache_salt`,
and sends `X-Session-ID: mlevolve:<cache-family-id>` for router affinity.

Automatic prefix caching (APC) skips repeated prefill computation. It improves
time to first token (TTFT), but it does not make decoding faster; long answers
will still be dominated by inter-token latency. See the official
[APC behavior and limits](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html).

## Required client/server contract

- Serve the exact public model name `qwen3.8-27b-int8-w8a16`. It must match
  `agent.code.model` or `agent.feedback.model` exactly.
- Expose the OpenAI-compatible endpoint at `http(s)://HOST:PORT/v1`.
- Enable APC and prompt-token details. The client treats missing token details
  as unknown, not as a cache miss.
- Keep the tokenizer/chat template identical across every replica. A template
  change changes tokenization and invalidates practical prefix reuse even if
  the text looks the same.
- Accept `cache_salt` in Chat Completion request bodies and preserve the
  `X-Session-ID` header through any proxy/router.
- Named tool choice must work. MLEvolve sends a specific function name, not
  `tool_choice: auto`; named function calling does not require auto-tool-choice
  flags. Only add an auto tool parser if other clients need automatic tool
  selection and the exact checkpoint/parser combination has been tested.

## vLLM 0.17.0 / Qwen3.8 / three-V100 baseline

This is the cache-enabled form of the repository's existing tested launch in
`benchmarks/qwen38_v100_int8/vllm_exact_int8_3gpu.sh`. Preserve the three-stage
pipeline-parallel layout, Mamba alignment, and MTP settings first; tune only
after collecting a baseline.

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.8-27B-INT8-W8A16-MTP
export VLLM_API_KEY='replace-with-a-random-service-key'

CUDA_VISIBLE_DEVICES=0,1,2 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --served-model-name qwen3.8-27b-int8-w8a16 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 3 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --generation-config vllm
```

vLLM 0.17 documents `--enable-prefix-caching`, Mamba cache modes,
`--enable-prompt-tokens-details`, `--served-model-name`, and API-key support in
its [0.17 serve reference](https://docs.vllm.ai/en/v0.17.0/cli/serve/). The
`--generation-config vllm` guard prevents an upstream model generation config
from imposing a hidden server-wide `max_new_tokens` value.

Do not add `--enable-auto-tool-choice` merely for MLEvolve. If the checkpoint's
embedded `tokenizer_config.json` lacks a tool-aware chat template, supply a
reviewed file with `--chat-template /absolute/path/template.jinja`, then rerun
the smoke test below. Do not accept request-supplied templates.

## Modern server profile

After validating the exact model and hardware on a newer vLLM release, retain
the baseline flags and add:

```bash
  --enable-per-request-metrics \
  --enable-request-id-headers
```

The client reads optional response metrics named `time_to_first_token_ms`,
`queue_time_ms`, `generation_time_ms`, `mean_itl_ms`, and
`tokens_per_second`. These fields are described in the official
[per-request metrics guide](https://docs.vllm.ai/en/latest/features/per_request_metrics/).
Older 0.17 servers omit them safely. Upgrade the isolated vLLM environment and
rerun correctness, tool, and cache smoke tests before changing production; do
not assume a modern wheel still supports V100, this quantization, the same
Mamba layout, or the same MTP implementation.

## Salt generation and distribution

Generate one 256-bit application-group secret:

```bash
openssl rand -base64 32 | tr -d '\n'
```

Store it in the application secret manager as `MLEVOLVE_VLLM_CACHE_SALT` and
mount the same value into every trusted MLEvolve client. Do not put it in YAML,
container images, shell history, URLs, access logs, traces, or metrics. It is a
request value, not a vLLM launch argument. vLLM recommends an unpredictable
256-bit (about 43 base64 characters) salt to prevent cross-tenant prefix-cache
probing; see the 0.17
[Chat request schema](https://docs.vllm.ai/en/v0.17.0/serving/openai_compatible_server/).

All requests intentionally share one salt because MLEvolve is one trusted
cache-sharing group. Use different salts for unrelated tenants. Rotating the
salt immediately creates a new cache namespace and therefore causes cold
prefills; coordinate rotation across clients.

## API key and reverse-proxy hardening

Bind vLLM to loopback or a private interface and put TLS/mTLS at the ingress.
Use `--api-key` even on a private network and configure that value as the
stage's `api_key`. A blank client key becomes `EMPTY`, which is suitable only
when the server is deliberately unauthenticated inside a protected test
network.

At the proxy:

- Allow only the MLEvolve network/service identity to reach `/v1/*`.
- Keep `/metrics`, `/docs`, model files, health internals, and admin endpoints
  private. Disable public FastAPI docs where appropriate.
- Preserve `Authorization`, streaming/SSE behavior, and `X-Session-ID`; cap
  request-body and header sizes and set long upstream read timeouts.
- Never log request bodies. Redact `Authorization`, `cache_salt`, prompts, and
  tool arguments. Reject client-supplied forwarding headers and normalize the
  trusted source address.
- Rate-limit by service identity and bound concurrent requests so the queue
  cannot consume all memory. Use readiness probes before routing traffic.

## Smoke tests

First verify the served name and basic text path:

```bash
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: mlevolve:smoke' \
  -d "{\"model\":\"qwen3.8-27b-int8-w8a16\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}],\"max_tokens\":8,\"cache_salt\":\"$MLEVOLVE_VLLM_CACHE_SALT\"}"
```

Then verify the exact named-tool contract used by MLEvolve:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3.8-27b-int8-w8a16\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Report score 7\"}],
    \"tools\": [{\"type\":\"function\",\"function\":{
      \"name\":\"report_score\",\"description\":\"Report a score\",
      \"parameters\":{\"type\":\"object\",\"properties\":{
        \"score\":{\"type\":\"integer\"}},\"required\":[\"score\"]}}}],
    \"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"report_score\"}},
    \"max_tokens\":32,
    \"cache_salt\":\"$MLEVOLVE_VLLM_CACHE_SALT\"
  }"
```

Require `choices[0].message.tool_calls[0].function.name == "report_score"` and
JSON arguments containing `score`. Finally, issue two requests with a long,
byte-identical prefix and different trailing user text. A hit is proven by a
positive `usage.prompt_tokens_details.cached_tokens` value on the repeated
request. Lower TTFT alone is supporting evidence, not proof.

The repository includes an opt-in live test:

```bash
MLEVOLVE_RUN_VLLM_CACHE_INTEGRATION=1 \
MLEVOLVE_VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
MLEVOLVE_VLLM_INTEGRATION_MODEL=qwen3.8-27b-int8-w8a16 \
MLEVOLVE_VLLM_CACHE_SALT="$MLEVOLVE_VLLM_CACHE_SALT" \
pytest -q tests/integration/test_context_cache_providers.py -k vllm
```

## Fleet routing

Start with one server. For replicas, use vLLM Production Stack load-aware
routing rather than round-robin. The router weighs prefix-cache benefit against
live load, so a warm but saturated replica does not become a queue hotspot; the
official [load-aware routing guide](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)
documents the scoring and `--loadaware-beta` tradeoff.

Preserve the client's `X-Session-ID` and keep model/template/build configuration
identical across replicas. Begin with the default beta, then benchmark 0.25,
0.5, 1, and 2 against both cache-hit ratio and queue time. Lower beta favors
locality; higher beta spills to idle replicas sooner. Test one-server and fleet
conditions separately because an aggregate fleet hit counter cannot prove that
the router selected the warm replica for a particular request.

## Optional LMCache/offloading

LMCache can extend reuse beyond one process or offload KV blocks to CPU, but it
is a second rollout, not a prerequisite for APC. The current multi-process
connector runs a standalone LMCache server and may require
`--disable-hybrid-kv-cache-manager`; see the official
[LMCache MP integration example](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/).

Compatibility-test it with this Qwen/Mamba checkpoint, pipeline parallelism,
prefix caching, speculative MTP, V100/CUDA version, failure recovery, and mixed
prompt lengths. Do not copy `kv_transfer_params` into MLEvolve client requests;
the client intentionally sends no raw KV tensors or experimental transfer
controls. Measure CPU capacity, PCIe/NVLink transfer cost, cache correctness,
and restart behavior before enabling external KV storage in production.

## Benchmark-driven tuning checklist

Use the restored matrix in `benchmarks.scenarios.required_vllm_scenarios()`:
cold/warm/disabled, sequential/concurrent, and single-server/fleet. Run at
least 20 post-warmup trials per condition. Record prompt/output lengths,
`cached_tokens`, `created_cache_tokens` when available, client and server TTFT,
queue time, generation time, mean ITL, output tokens/second, errors, and replica.

Tune in this order:

1. **GPU KV capacity.** Raise `--gpu-memory-utilization` only while leaving
   measured headroom for CUDA graphs, temporary buffers, MTP, and workload
   spikes. Watch `vllm:kv_cache_usage_perc`, prefix hit/query counters,
   preemptions, OOMs, and block eviction age/pressure. Current metric names are
   catalogued in [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/).
2. **Chunked prefill.** Sweep `--max-num-batched-tokens`. Smaller budgets tend
   to protect ITL; larger budgets tend to improve TTFT/throughput. The official
   [optimization guide](https://docs.vllm.ai/en/stable/configuration/optimization/)
   recommends measuring the tradeoff rather than assuming one universal value.
3. **Concurrency and queueing.** Set admission limits from queue-time p95 and
   TTFT p95, not only throughput. Cache hits can still queue behind decode-heavy
   requests.
4. **MTP/speculative decoding.** Compare MTP on/off and 1/2/3 speculative
   tokens. APC targets prefill; MTP targets decode/ITL. Track acceptance rate,
   ITL, throughput, memory, and quality. Model-based speculation helps primarily
   under medium/low-QPS memory-bound workloads according to the
   [speculative decoding guide](https://docs.vllm.ai/en/stable/features/speculative_decoding/).
5. **Eviction pressure.** Run repeated-prefix trials across realistic idle gaps
   and competing prefixes. Alert on falling hit/query ratio alongside rising KV
   utilization, preemptions, eviction frequency, queue time, and TTFT.

Promote a setting only when cache-token counters prove reuse, TTFT improves for
prefill-heavy calls, ITL/quality do not regress, and the fleet remains stable
under the target concurrency.
