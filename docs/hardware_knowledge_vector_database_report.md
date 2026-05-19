# Hardware Knowledge Vector Database Construction Report

## Purpose of This Report

This report summarizes the current two-database design in MLEvolve and explains why the hardware knowledge vector database requires manual construction. It is written as a professor-facing request for labour assistance: the implementation exists, but the usefulness of the system depends on carefully collected, source-backed hardware and software optimization knowledge.

The main assistance needed is not ordinary data entry. Helpers must read vendor documents, framework documents, API references, architecture notes, benchmark reports, and MLEvolve internal findings, then convert them into structured vector-database records with stable metadata.

## 1. Current System Design

The current design follows `mlevolve_two_database_design_plan.md` and the implementation in `localml_scheduler`. MLEvolve uses two complementary memory layers:

```text
Profile Graph DB = what happened.
Vector DB        = how to change code.
Agent            = why this change should be tried now.
```

### 1.1 Profile Graph Database

The graph database stores measured evidence from actual runs. It should answer questions such as:

- Which hardware was used?
- Which model, workload, batch size, precision mode, backend, or packing strategy was used?
- Did the job succeed, fail, time out, or run out of memory?
- What was the measured runtime, throughput, VRAM use, RAM use, GPU/SM utilization, and validation metric?
- Did a previous optimization help this model family on similar hardware?
- Did packed or concurrent training improve throughput or cause slowdown?

The graph database is an empirical evidence store, not a programming manual. It records `SingleJob`, `PackedJob`, `PackedJobMember`, `Hardware`, `Model`, `TrainingConfig`, and `Technology` evidence. SQLite remains the live scheduler control-plane source of truth for current jobs, commands, checkpoints, and events.

### 1.2 Code-Knowledge Vector Database

The vector database stores coding knowledge, not profile facts. It answers questions such as:

- How should PyTorch AMP be applied correctly?
- Which APIs enable BF16, FP16, TF32, `torch.compile`, CUDA Graphs, dataloader optimization, checkpointing, or CUDA process packing?
- Which hardware capabilities make an optimization relevant?
- Which code patterns are recommended or risky for a given workload?
- What exact API syntax or example code should an agent use?

The current implementation uses Qdrant and the following collections:

```text
code_doc_chunks
optimization_recipe_chunks
api_symbol_chunks
hardware_feature_knowledge
```

The first three are the current code-knowledge collections. `hardware_feature_knowledge` remains as a compatibility collection for curated hardware feature records. The preferred path is now `code_knowledge`, but hardware feature seed records are still useful because they can be converted automatically into code-knowledge documents and recipes.

### 1.3 Agent Retrieval Flow

The intended agent loop is:

```text
1. Read current or historical job evidence from the graph DB.
2. Convert graph metrics into diagnosis labels.
3. Use diagnosis labels as vector metadata filters.
4. Retrieve relevant code documents, recipes, and API records from Qdrant.
5. Generate a code mutation.
6. Validate the mutation by running the job or probe.
7. Write the new empirical result back to the graph DB.
8. Optionally summarize repeated successful patterns as new vector recipes.
```

For example, a graph record may show FP32 transformer training on a Tensor Core GPU with low SM utilization and moderate VRAM usage. The agent derives labels such as `low_sm_utilization`, `precision_not_optimized`, and `tensor_core_not_used`, then retrieves vector records about `pytorch_amp`, `bf16_autocast`, `tf32_matmul`, and `torch.amp.autocast`.

## 2. Why Manual Labour Is Needed

The hardware vector database cannot be produced reliably by automatic scraping alone. Hardware optimization knowledge is version-sensitive, source-sensitive, and often conditional:

- Vendor marketing pages, architecture PDFs, framework docs, and API references contain different levels of detail.
- Some features are hardware capabilities, while others are framework-supported programming paths.
- Low-precision modes such as BF16, FP16, TF32, INT8, or FP4 do not have identical training semantics.
- A recommendation may be safe for inference but risky or unsupported for training.
- The same API can have different behavior across PyTorch, CUDA, driver, and GPU architecture versions.
- MLEvolve needs concise records that agents can retrieve and use safely, not unstructured document dumps.

Therefore, helpers should manually collect, verify, classify, and write structured records. Each record must cite sources, state its confidence, include applicability tags, and separate recommended patterns from avoid patterns.

## 3. Shared Metadata and Ontology Rules

All vector records should use shared ontology keys from `schema/ontology/`. These keys connect graph evidence to vector retrieval filters.

Important ontology files:

```text
schema/ontology/technology_keys.yaml
schema/ontology/hardware_feature_keys.yaml
schema/ontology/model_families.yaml
schema/ontology/workload_types.yaml
schema/ontology/profile_symptoms.yaml
schema/ontology/optimization_targets.yaml
schema/ontology/api_symbols.yaml
```

Common current keys include:

```yaml
technology_keys:
  - pytorch_amp
  - bf16_autocast
  - fp16_autocast
  - tf32_matmul
  - torch_compile
  - cuda_graphs
  - triton_kernel
  - nvidia_mps
  - cuda_streams
  - cuda_process

hardware_feature_keys:
  - tensor_core
  - bf16_support
  - fp16_support
  - tf32_support
  - high_bandwidth_memory
  - cuda_capability_80
  - cuda_capability_90
  - cuda_capability_120
  - mps
  - cuda_graphs

model_families:
  - cnn
  - resnet
  - convnext
  - vit
  - transformer
  - diffusion
  - tabular
  - custom

workload_types:
  - vision_training
  - transformer_training
  - conv_heavy
  - attention_heavy
  - matmul_heavy
  - memory_bound
  - activation_memory
  - data_loading_bound
  - small_batch
  - large_batch
  - launch_overhead_bound
  - packed_training
  - unknown

profile_symptoms:
  - low_sm_utilization
  - high_vram_pressure
  - oom
  - timeout
  - launch_overhead_bound
  - memory_bandwidth_bound
  - dataloader_bottleneck
  - precision_not_optimized
  - tensor_core_not_used
  - slow_packed_training

optimization_targets:
  - improve_sm_utilization
  - improve_throughput
  - reduce_step_time
  - reduce_vram
  - avoid_oom
  - enable_tensor_core
  - improve_precision_efficiency
  - reduce_kernel_launch_overhead
  - improve_memory_efficiency
  - improve_dataloader
  - improve_packing
```

Construction rule: use the most general correct key. Prefer `hardware_feature_keys: [tensor_core, bf16_support]` when a document applies to a capability class, and use `hardware_keys` only when the record is device-specific.

## 4. YAML Corpus Format

The ingestion code accepts either a top-level list of records or an object with a `records` list:

```yaml
records:
  - schema_version: code_doc_chunk_v1
    ...
  - schema_version: optimization_recipe_chunk_v1
    ...
```

Each record must have one of these schema versions:

```text
hardware_feature_record_v1
code_doc_chunk_v1
optimization_recipe_chunk_v1
api_symbol_chunk_v1
```

The `confidence` field must be a number between `0.0` and `1.0`. Use higher confidence for official vendor/framework docs and lower confidence for blog posts, inferred patterns, or early MLEvolve observations.

Suggested confidence scale:

```text
0.90-1.00 official API or vendor reference, directly verified
0.75-0.89 official guide, architecture document, or repeated internal result
0.60-0.74 credible tutorial, benchmark note, or single internal observation
0.40-0.59 plausible but conditional guidance
below 0.40 do not ingest unless clearly labelled as experimental
```

## 5. Schema A: `hardware_feature_record_v1`

### Purpose

Use this format for manually curated hardware capability and hardware-aware programming guidance. These records are useful when the information is organized around a GPU architecture, accelerator, compute capability, or generic hardware feature.

Current implementation detail: `hardware_feature_record_v1` records can be ingested into the compatibility `hardware_feature_knowledge` collection. They can also be converted automatically into:

- one `code_doc_chunk_v1` record
- one `optimization_recipe_chunk_v1` record when `recommended_patterns` is non-empty

This conversion is helpful, but manual source quality still matters.

### Required Fields

```yaml
schema_version: hardware_feature_record_v1
record_id: string
title: string
summary_text: string
detail_text: string
vendor: string
architectures: list[string]
accelerator_names: list[string]
compute_capabilities: list[string]
toolkits: list[string]
frameworks: list[string]
workload_types: list[string]
features: list[string]
recommended_patterns: list[string]
avoid_patterns: list[string]
source_refs: list[object]
confidence: float
tags: list[string]
last_verified: string
```

`last_verified` is optional in the validator. If omitted, it is inferred from the latest `source_refs[].retrieved_or_verified_date`. Include it explicitly for professor-facing review.

`accelerator_names`, `compute_capabilities`, and `avoid_patterns` may be empty lists. Other required list fields should contain at least one useful value.

Each `source_refs` item must contain:

```yaml
title: string
url: string
source_type: string
retrieved_or_verified_date: "YYYY-MM-DD"
```

### Construction Instructions

- Use a stable `record_id`, for example `nvidia.blackwell.tensor_cores.precision_policy`.
- Keep `summary_text` short: one or two sentences.
- Use `detail_text` for conditions, caveats, version constraints, and why the feature matters.
- Use `features` for hardware or optimization tags that can become `hardware_feature_keys` and `technology_keys` during conversion.
- Write `recommended_patterns` as agent-actionable coding guidance.
- Write `avoid_patterns` as specific warnings, not vague cautions.
- Include at least one official source whenever possible.
- Avoid claiming that a hardware feature is usable in training unless framework support is confirmed.

### Template

```yaml
records:
  - schema_version: hardware_feature_record_v1
    record_id: vendor.architecture.feature.short_name
    title: Short descriptive title
    summary_text: One or two sentences explaining the hardware feature and why it matters.
    detail_text: Longer guidance with applicability, framework support, limits, and caveats.
    vendor: nvidia
    architectures: [blackwell]
    accelerator_names: [rtx_5090, geforce_rtx_5090]
    compute_capabilities: ["12.0"]
    toolkits: [cuda]
    frameworks: [pytorch]
    workload_types: [vision_training, transformer_training]
    features: [tensor_core, bf16_support, fp16_support]
    recommended_patterns:
      - Use torch.amp.autocast with an explicit dtype selected from hardware and framework context.
      - Log resolved precision, batch size, throughput, and peak VRAM for graph feedback.
    avoid_patterns:
      - Do not use inference-only low-precision paths in generic training code unless the framework explicitly supports them.
    source_refs:
      - title: Official source title
        url: https://example.com/source
        source_type: vendor_reference
        retrieved_or_verified_date: "2026-05-17"
    confidence: 0.85
    last_verified: "2026-05-17"
    tags: [precision, tensor_core, training]
```

## 6. Schema B: `code_doc_chunk_v1`

### Purpose

Use this format for general documentation chunks, internal notes, hardware-aware explanations, framework guides, architecture summaries, and source-backed concept notes.

This schema is stored in Qdrant collection:

```text
code_doc_chunks
```

### Required by Current Validator

The current validator requires:

```yaml
schema_version: code_doc_chunk_v1
chunk_id: string
title: string
text: string
```

If `text_hash` is omitted, the validator computes it from `text`.

Most metadata fields are optional in code, but should be included for retrieval quality:

```yaml
source_id: string
source_type: string
source_title: string
source_url: string
source_version: string
framework: string
framework_version: string
technology_keys: list[string]
hardware_keys: list[string]
hardware_feature_keys: list[string]
model_keys: list[string]
model_families: list[string]
workload_types: list[string]
optimization_targets: list[string]
profile_symptoms: list[string]
api_symbols: list[string]
precision_modes: list[string]
risk_level: low | medium | high
confidence: float
deprecated: boolean
```

### Construction Instructions

- One chunk should explain one coherent concept, API family, or hardware-aware technique.
- Do not paste entire web pages. Remove navigation, unrelated examples, marketing text, and duplicated boilerplate.
- Keep `text` rich enough to answer "how and when should the agent use this?"
- Add `api_symbols` when the text discusses exact APIs.
- Add `profile_symptoms` and `optimization_targets` when the chunk is useful for diagnosis-driven retrieval.
- Mark old or version-incompatible content with `deprecated: true`.

### Template

```yaml
records:
  - schema_version: code_doc_chunk_v1
    chunk_id: pytorch.amp.autocast.training.doc.001
    title: PyTorch AMP autocast for mixed-precision training
    text: >
      Explain when autocast should wrap forward and loss computation, which dtype choices are commonly used,
      and what validation should be performed after enabling mixed precision.
    source_id: pytorch.amp.docs
    source_type: official_doc
    source_title: PyTorch Automatic Mixed Precision Documentation
    source_url: https://pytorch.org/docs/stable/amp.html
    source_version: pytorch-2.x
    framework: pytorch
    framework_version: "2.x"
    technology_keys: [pytorch_amp, bf16_autocast, fp16_autocast]
    hardware_keys: []
    hardware_feature_keys: [tensor_core, bf16_support, fp16_support]
    model_keys: []
    model_families: [transformer, cnn, vit]
    workload_types: [vision_training, transformer_training, matmul_heavy]
    optimization_targets: [improve_throughput, enable_tensor_core, reduce_vram]
    profile_symptoms: [low_sm_utilization, precision_not_optimized, tensor_core_not_used]
    api_symbols: [torch.amp.autocast, torch.amp.GradScaler]
    precision_modes: [bf16, fp16, mixed]
    risk_level: medium
    confidence: 0.9
    deprecated: false
```

## 7. Schema C: `optimization_recipe_chunk_v1`

### Purpose

Use this format for actionable recipes that MLEvolve can directly turn into a code mutation proposal.

This schema is stored in Qdrant collection:

```text
optimization_recipe_chunks
```

### Required by Current Validator

The current validator requires:

```yaml
schema_version: optimization_recipe_chunk_v1
recipe_id: string
title: string
text: string
optimization_targets: list[string]
```

The validator also accepts `solution_summary` instead of `text`, but for clarity include both.

Strongly recommended fields:

```yaml
problem_statement: string
solution_summary: string
recommended_patterns: list[string]
avoid_patterns: list[string]
source_chunk_ids: list[string]
source_job_ids: list[string]
technology_keys: list[string]
hardware_feature_keys: list[string]
model_families: list[string]
workload_types: list[string]
profile_symptoms: list[string]
api_symbols: list[string]
precision_modes: list[string]
framework: string
framework_version: string
risk_level: low | medium | high
confidence: float
deprecated: boolean
```

### Construction Instructions

- Write the recipe as a conditional action: "When symptom X appears under hardware feature Y, try code change Z."
- Include `problem_statement` so retrieval can match the observed graph diagnosis.
- Include `recommended_patterns` as practical implementation steps.
- Include `avoid_patterns` for failure modes, incorrect API usage, or numerical risk.
- Use `source_chunk_ids` to link to documentation records.
- Use `source_job_ids` only when graph job evidence supports this recipe.
- Keep recipes specific enough to be implemented, but not so specific that they only apply to one experiment.

### Template

```yaml
records:
  - schema_version: optimization_recipe_chunk_v1
    recipe_id: pytorch.amp.tensor_core.low_sm.recipe.001
    title: Use PyTorch AMP to improve Tensor Core utilization
    problem_statement: >
      A PyTorch training job on Tensor Core-capable NVIDIA hardware uses FP32 and shows low SM utilization
      with enough VRAM headroom.
    solution_summary: >
      Try BF16 or FP16 autocast around forward and loss computation, keep a full-precision fallback,
      and validate loss behavior, throughput, and peak VRAM.
    text: >
      Add a precision configuration option. Wrap forward and loss computation in torch.amp.autocast.
      Prefer BF16 when hardware and model support it. Use GradScaler for FP16 where needed.
    recommended_patterns:
      - Keep precision mode configurable rather than hard-coded.
      - Log selected dtype, throughput, peak VRAM, and validation metric.
      - Compare the first validation metric against the FP32 baseline.
    avoid_patterns:
      - Do not enable multiple low-precision paths at once without ablation.
      - Do not use inference-only FP4 guidance as generic training guidance.
    source_chunk_ids: [pytorch.amp.autocast.training.doc.001]
    source_job_ids: []
    framework: pytorch
    framework_version: "2.x"
    technology_keys: [pytorch_amp, bf16_autocast, fp16_autocast]
    hardware_feature_keys: [tensor_core, bf16_support, fp16_support]
    model_families: [transformer, cnn, vit]
    workload_types: [vision_training, transformer_training, matmul_heavy]
    optimization_targets: [improve_throughput, enable_tensor_core, reduce_vram]
    profile_symptoms: [low_sm_utilization, precision_not_optimized, tensor_core_not_used]
    api_symbols: [torch.amp.autocast, torch.amp.GradScaler]
    precision_modes: [bf16, fp16, mixed]
    risk_level: medium
    confidence: 0.85
    deprecated: false
```

## 8. Schema D: `api_symbol_chunk_v1`

### Purpose

Use this format for exact API usage records. These are important when the agent already knows what optimization to try and needs precise syntax, parameter behavior, examples, or version constraints.

This schema is stored in Qdrant collection:

```text
api_symbol_chunks
```

### Required by Current Validator

The current validator requires:

```yaml
schema_version: api_symbol_chunk_v1
api_symbol_id: string
api_symbol: string
usage_summary: string
```

Because `api_symbol` can act as the title and `usage_summary` can act as text, those three fields are the minimum practical record. Include the metadata below for useful retrieval.

Recommended fields:

```yaml
signature: string
parameters_json: string
example_code: string
source_id: string
source_type: string
source_title: string
source_url: string
source_version: string
framework: string
framework_version: string
technology_keys: list[string]
hardware_feature_keys: list[string]
optimization_targets: list[string]
profile_symptoms: list[string]
api_symbols: list[string]
precision_modes: list[string]
risk_level: low | medium | high
confidence: float
deprecated: boolean
```

### Construction Instructions

- One record should describe one function, class, method, or configuration API.
- Include the full import path in `api_symbol`.
- Include `signature` when the API signature is stable enough to quote.
- Keep `parameters_json` machine-readable when possible.
- Include `example_code` only when it is short and correct.
- Include version and deprecation notes in `usage_summary`.

### Template

```yaml
records:
  - schema_version: api_symbol_chunk_v1
    api_symbol_id: pytorch.torch_amp_autocast.api.001
    api_symbol: torch.amp.autocast
    signature: "torch.amp.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)"
    usage_summary: >
      Context manager for running selected operations in mixed precision. For CUDA training,
      it is commonly used around forward and loss computation. Choose dtype based on hardware,
      framework support, and numerical validation.
    parameters_json: >
      {"device_type": "Usually 'cuda' for GPU training", "dtype": "torch.bfloat16 or torch.float16 when supported", "enabled": "Allows runtime fallback"}
    example_code: |
      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
          outputs = model(inputs)
          loss = criterion(outputs, targets)
    source_id: pytorch.amp.docs
    source_type: api_reference
    source_title: PyTorch Automatic Mixed Precision Documentation
    source_url: https://pytorch.org/docs/stable/amp.html
    source_version: pytorch-2.x
    framework: pytorch
    framework_version: "2.x"
    technology_keys: [pytorch_amp, bf16_autocast, fp16_autocast]
    hardware_feature_keys: [tensor_core, bf16_support, fp16_support]
    optimization_targets: [improve_throughput, enable_tensor_core, reduce_vram]
    profile_symptoms: [precision_not_optimized, tensor_core_not_used]
    api_symbols: [torch.amp.autocast]
    precision_modes: [bf16, fp16, mixed]
    risk_level: medium
    confidence: 0.9
    deprecated: false
```

## 9. What Information Should Be Collected

Helpers should collect information in the following categories.

### 9.1 Hardware and Architecture Sources

Collect:

- GPU architecture names and generation names.
- Accelerator product names.
- CUDA compute capability.
- VRAM size and memory type when relevant.
- Tensor Core generation and supported precision modes.
- BF16, FP16, TF32, FP8, FP4, INT8, and related constraints.
- MPS, MIG, CUDA Graphs, CUDA streams, or process-packing support.
- Driver, CUDA, and toolkit compatibility notes.

Preferred source types:

- vendor product page
- vendor architecture PDF
- vendor compute capability table
- vendor technical blog
- CUDA documentation

### 9.2 Framework and API Sources

Collect:

- PyTorch AMP usage and caveats.
- `torch.amp.autocast`, `torch.amp.GradScaler`, `torch.set_float32_matmul_precision`, `torch.compile`, `torch.cuda.CUDAGraph`, and dataloader APIs.
- Version-specific behavior.
- Deprecated APIs or replacement APIs.
- Short correct examples.

Preferred source types:

- official framework documentation
- official API reference
- official tutorials
- release notes when behavior changed

### 9.3 Optimization Recipes

Collect or write recipes for:

- Low SM utilization.
- Precision not optimized.
- Tensor Core not used.
- High VRAM pressure.
- CUDA OOM fallback.
- Dataloader bottleneck.
- Kernel launch overhead.
- Memory bandwidth bound workloads.
- Slow packed training.
- Checkpoint/resume robustness for long jobs.

Each recipe must say:

- When to apply it.
- What code change to try.
- Which APIs are involved.
- Which hardware features make it relevant.
- What metrics should improve.
- What risks or contraindications exist.
- How to validate correctness and performance.

### 9.4 MLEvolve Internal Findings

When MLEvolve produces repeated successful outcomes, helpers can convert them into internal `optimization_recipe_chunk_v1` records. These records should link back to graph evidence through `source_job_ids` where available.

Do not store raw profiling facts in the vector DB. Store only reusable coding lessons. The graph DB remains the place for measured truth.

## 10. Suggested Labour Division

The following division would make the work parallel and reviewable.

### Role A: Hardware and Source Collection

Responsibilities:

- Find official NVIDIA, CUDA, and hardware architecture sources.
- Record product names, architecture names, compute capability, and feature support.
- Draft `hardware_feature_record_v1` entries.
- Ensure every record has at least one high-quality `source_refs` entry.

Expected output:

```text
hardware_feature_records.yaml
```

### Role B: PyTorch and CUDA API Extraction

Responsibilities:

- Read official PyTorch and CUDA API docs.
- Draft `api_symbol_chunk_v1` records.
- Include signatures, parameters, short examples, version notes, and deprecation notes.
- Tag each API with matching `technology_keys`, `optimization_targets`, and `profile_symptoms`.

Expected output:

```text
api_symbol_records.yaml
```

### Role C: Optimization Recipe Writing

Responsibilities:

- Convert source-backed guidance into actionable MLEvolve recipes.
- Write `optimization_recipe_chunk_v1` records for common symptoms.
- Include recommended patterns, avoid patterns, risk level, validation method, and source links.

Expected output:

```text
optimization_recipe_records.yaml
```

### Role D: Metadata and Ontology Validation

Responsibilities:

- Check that every record uses ontology-aligned labels.
- Add missing ontology keys only when necessary and after review.
- Check stable IDs, confidence scores, source dates, and YAML syntax.
- Run dry-run ingestion commands before Qdrant writes.

Expected output:

```text
validated_records/
```

### Role E: Retrieval Test Query Creation

Responsibilities:

- Write test queries that represent real MLEvolve agent needs.
- Check whether the top retrieved records are useful.
- Suggest missing records when retrieval fails.

Example test queries:

```text
PyTorch transformer training has low SM utilization on Tensor Core GPU and uses FP32.
CUDA OOM during vision training, preserve effective batch size if possible.
Dataloader bottleneck with low GPU utilization during image training.
Packed training slowdown with CUDA process backend.
Need exact torch.amp.autocast usage for BF16 training.
```

## 11. Review and Validation Process

### 11.1 Human Review Checklist

Before ingestion, each record should pass this checklist:

- The record has a stable ID.
- The record has at least one credible source.
- The source was verified recently.
- The confidence score is justified.
- The guidance is not copied blindly from unrelated hardware.
- The record separates recommended patterns from avoid patterns.
- The record uses ontology labels consistently.
- The record is clear enough for an agent to use without reading the original source.
- Training-specific claims are not inferred from inference-only documentation.
- Deprecated or version-specific behavior is marked.

### 11.2 Dry-Run Validation Commands

Validate code-knowledge records without writing to Qdrant:

```bash
python -m localml_scheduler.cli code-knowledge ingest \
  --settings localml_scheduler/configs/scheduler.example.yaml \
  --source path/to/code_knowledge_records.yaml \
  --dry-run
```

Validate hardware-feature records without writing to Qdrant:

```bash
python -m localml_scheduler.cli hardware-features ingest \
  --settings localml_scheduler/configs/scheduler.example.yaml \
  --source path/to/hardware_feature_records.yaml \
  --dry-run
```

Validate the repo seed corpus:

```bash
python -m localml_scheduler.cli code-knowledge ingest \
  --settings localml_scheduler/configs/scheduler.example.yaml \
  --dry-run
```

```bash
python -m localml_scheduler.cli hardware-features ingest \
  --settings localml_scheduler/configs/scheduler.example.yaml \
  --dry-run
```

### 11.3 Qdrant Ingestion Commands

After review, ingest code knowledge:

```bash
python -m localml_scheduler.cli code-knowledge ingest \
  --settings localml_scheduler/configs/scheduler.example.yaml \
  --source path/to/code_knowledge_records.yaml
```

After review, ingest hardware feature compatibility records:

```bash
python -m localml_scheduler.cli hardware-features ingest \
  --settings localml_scheduler/configs/scheduler.example.yaml \
  --source path/to/hardware_feature_records.yaml
```

Use `--recreate` only when intentionally rebuilding the collection.

## 12. Important Implementation Notes

The current implementation is authoritative where it differs from the original design plan.

Current validator files:

```text
localml_scheduler/hardware_features/records.py
localml_scheduler/code_knowledge/records.py
localml_scheduler/code_knowledge/store.py
```

Current configured Qdrant collections:

```text
code_doc_chunks
optimization_recipe_chunks
api_symbol_chunks
hardware_feature_knowledge
```

Current embedding configuration in `localml_scheduler/configs/scheduler.example.yaml`:

```yaml
hardware_feature_db:
  enabled: true
  provider: qdrant
  url: http://127.0.0.1:6333
  collection_name: hardware_feature_knowledge
  code_doc_collection_name: code_doc_chunks
  optimization_recipe_collection_name: optimization_recipe_chunks
  api_symbol_collection_name: api_symbol_chunks
  embedding_model_type: local
  embedding_model_name: BAAI/bge-base-en-v1.5
  embedding_device: cpu
  distance: Cosine
```

Important current behavior:

- Code-knowledge ingestion creates three physical Qdrant collections.
- Hardware-feature seed records can be converted to code docs and recipes.
- Record payloads are embedded using selected text fields plus metadata labels.
- Search filters currently use exact metadata matches. Consistent ontology spelling is therefore important.
- Compatibility wrappers such as `search_hardware_features` still exist, but new work should prefer `search_code_knowledge` and `get_code_optimization_context`.

## 13. Requested Assistance

I request labour assistance to construct the initial high-quality hardware knowledge corpus for MLEvolve. The code infrastructure for ingestion and retrieval is present, but the database requires careful manual curation to be useful and safe for optimization agents.

The desired outcome is a reviewed YAML corpus covering:

- NVIDIA hardware architecture and compute capability guidance.
- Tensor Core and precision-mode guidance.
- PyTorch AMP, TF32, `torch.compile`, CUDA Graphs, dataloader, checkpointing, and OOM fallback APIs.
- Common MLEvolve optimization recipes linked to graph-derived symptoms.
- Retrieval test queries and coverage notes.

This work will make MLEvolve's hardware-aware agent behavior more reliable because future code mutations will be grounded in both measured graph evidence and curated implementation knowledge.
