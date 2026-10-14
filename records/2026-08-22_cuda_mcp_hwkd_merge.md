# Merging NVIDIA CUDA MCP with the hardware knowledge database

- Date: 2026-08-22

- Code: `localml_scheduler/cuda_mcp_bridge.py`

## What CUDA MCP is

- NVIDIA-hosted MCP server giving an agent a search tool over indexed, current CUDA documentation and code samples curated by NVIDIA engineers

- Endpoint: `https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs`, HTTP transport

- Authentication is an NVIDIA Developer account login on first connection, after which the client reuses credentials

- Registration for Claude Code: `claude mcp add --scope user --transport http nvidia-cuda-docs https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs`

## Why a merge is worth doing, measured rather than assumed

- Bug taxonomy from this repo's own traces, classified from journal `term_out`

| cause | cassava n=112 | leaf n=46 |
|---|---|---|
| syntax / indentation | 32.1% | 23.9% |
| timeout (exec or scheduler wait) | 31.2% | 45.7% |
| CUDA out of memory | 11.6% | 0% |
| CUDA API error (also out of memory) | 3.6% | 0% |
| import / environment | 7.1% | 23.9% |
| shape or dtype | 0.9% | 6.5% |

- CUDA documentation is addressable for the memory rows only, which is 15.2% of cassava failures and 0% of leaf failures

- Leaf peaks at 388 MiB and never pressures a 32 GB card, so it cannot exercise this integration; a GPU-heavy task is required to test it

## Why neither source is sufficient alone

- HWKD knows what happened on this machine: measured peak VRAM per script signature, the scheduler's committed budget, compute capability, CUDA and torch versions. It cannot say how to fix anything

- CUDA MCP knows what NVIDIA documents. It does not know which GPU is installed, how much VRAM the scheduler will commit, or what this model already measured

- A documentation answer cannot size a batch against a 31 GB budget, and a measured VRAM number cannot tell an agent to enable AMP

## The second reason: HWKD's curated store is hand-written

- `localml_scheduler/hardware_features/seed_records.yaml` is maintained by hand and stamped `retrieved_or_verified_date: "2026-05-15"`

- `SchedulerClient.search_hardware_features` is a deprecated wrapper over `search_code_knowledge` with record types `code_doc_chunks` and `optimization_recipe_chunks`

- `CodeKnowledgeStore` already exposes `ingest_records(...)` and `ingest_source(...)`, so an ingestion path exists and is empty

- CUDA MCP is the authoritative source that can keep those records current, with `source_refs` pointing at real NVIDIA pages instead of recollection

## Design

- `cuda_mcp_bridge.py` composes the two sources in four steps

- `topic_for_error(error_text)` maps a failure to a documentation topic, and returns `None` for syntax and import failures so no useless query is issued

- `facts_from_knowledge_base(client, signature)` reads GPU name, compute capability, total VRAM, scheduler budget, toolkit versions, and the measured peak for that signature out of HWKD

- `build_query(topic, facts)` puts the measured constraints inside the query rather than applying them to the answer afterwards

- `to_records(...)` shapes the answer into `optimization_recipe_chunk_v1` records tagged with this card's compute capability, then `ingest(...)` writes them through `CodeKnowledgeStore`

- After ingestion HWKD's existing MCP tools serve NVIDIA guidance joined with measured profiles, so the agent reaches both through one query

## Hardware gating

- Records are filtered at ingestion by minimum compute capability, so advice the installed card cannot run never reaches the agent

| technique | minimum capability | available on V100 SM 7.0 |
|---|---|---|
| TF32 | 8.0 | no |
| FP8 / float8 / Transformer Engine | 8.9 | no |
| Flash Attention 3 / TMA / thread block cluster | 9.0 | no |
| AMP fp16, gradient checkpointing, channels_last | none | yes |

- `torch.cuda.is_bf16_supported()` returns `True` on this V100 even though SM 7.0 has no native bf16 tensor-core path, so bf16 is deliberately not treated as an exclusion

## Verification

- Run against the live HWKD schema on Nautilus

- Composed query for a cassava-style OOM

```
reduce peak GPU memory during training. Target GPU is Tesla V100-SXM2-32GB
(compute capability 7.0). Toolkit: CUDA 12.4, PyTorch 2.4.0. The scheduler
commits at most 30965 MB of VRAM per job. This workload has measured a peak of
8432 MB across 7841 observations on this hardware. Exclude techniques
unavailable on this card: flash attention 3, float8, fp8, tf32, thread block
cluster, tma, transformer engine. Answer with concrete PyTorch changes and cite
the CUDA documentation section.
```

- Routing: both OOM strings map to the memory topic, while `SyntaxError` and `ModuleNotFoundError` map to `None`

- Record validated by `validate_code_knowledge_record` as `recipe_id=nvidia.cuda_mcp.9c757559e823`, `optimization_targets=['gpu_memory']`, `compute_capabilities=['7.0']`

- A TF32 recommendation present in the sample answer was dropped by the gate, leaving three applicable recommendations and one prohibition

## Not yet done

- The bridge does not call the MCP server; the caller supplies the answer text, because the server requires an interactive NVIDIA Developer login that a headless run cannot perform

- Registering `nvidia-cuda-docs` with the `claude` CLI on Nautilus makes the tool available to the MLEvolve generation agent, since MLEvolve uses `claude_cli` as its backend; that registration needs the same interactive login

- `fastmcp` is not installed on Nautilus, so HWKD's own MCP server cannot be served there yet

- No end-to-end effect on bug rate has been measured; that requires a GPU-heavy task, because leaf produces zero CUDA failures
