# MLEvolve Two-Database Design Plan

## Purpose

This document describes a two-database memory architecture for the MLEvolve framework.

The design separates empirical profiling memory from code-optimization knowledge:

```text
Profile Graph DB
  Stores measured facts from training/profiling runs.

Code-Docs Vector DB
  Stores programming documentation, API usage, optimization recipes, and hardware-aware coding guidance.

MLEvolve Optimization Agent
  Retrieves from both databases, reasons across the evidence, proposes code mutations, validates them, and writes new empirical results back to the graph.
```

Core rule:

```text
Graph DB  = what happened.
Vector DB = how to change code.
Agent     = why this change should be tried now.
```

---

# 1. High-Level Architecture

```text
                         ┌──────────────────────────────┐
                         │ MLEvolve Optimization Agent   │
                         └───────────────┬──────────────┘
                                         │
           ┌─────────────────────────────┴─────────────────────────────┐
           │                                                           │
           ▼                                                           ▼
┌──────────────────────────────┐                       ┌──────────────────────────────┐
│ Profile Graph DB             │                       │ Code-Docs Vector DB           │
│ Neo4j or similar graph store  │                       │ Qdrant / Milvus / Pinecone /  │
│                              │                       │ FAISS / pgvector / etc.       │
│ Major nodes:                 │                       │                              │
│   (:Job)                     │                       │ Collections:                 │
│   (:Hardware)                │                       │   CodeDocChunk                │
│   (:Model)                   │                       │   OptimizationRecipeChunk     │
│   (:Technology)              │                       │   ApiSymbolChunk              │
│                              │                       │                              │
│ Stores:                      │                       │ Stores:                      │
│   batch size                 │                       │   API usage                   │
│   VRAM usage                 │                       │   Tensor Core programming     │
│   SM utilization             │                       │   AMP / BF16 / TF32 guidance  │
│   runtime                    │                       │   optimization recipes        │
│   failures                   │                       │   code patterns               │
│   packing results            │                       │   MLEvolve mutation examples  │
└──────────────────────────────┘                       └──────────────────────────────┘
```

The graph database answers questions like:

```text
What happened before?
What batch size fit?
What was peak VRAM?
Was SM utilization low?
Did MPS packing help?
Which technology stack worked on similar jobs?
```

The vector database answers questions like:

```text
How do I use Tensor Cores in PyTorch?
How should I apply AMP, BF16, TF32, torch.compile, CUDA graphs, or Triton?
What APIs should the MLEvolve mutation agent use?
What code pattern can fix low SM utilization or high VRAM pressure?
```

---

# 2. Main Design Principle

The vector database schema should be derived from the graph database schema, but it should not duplicate graph records.

The graph has four major node concepts:

```text
Job
Hardware
Model
Technology
```

The vector DB should use the same concepts as metadata filters:

```text
hardware_key
hardware_feature_keys
model_key
model_family
technology_keys
optimization_targets
profile_symptoms
api_symbols
framework
framework_version
```

Example:

A graph profile says:

```yaml
Job:
  model_family: transformer
  hardware_key: nvidia_a100_80gb
  technology_set_key: pytorch_fp32
  avg_sm_utilization_pct: 25
  peak_vram_utilization_of_cap_pct: 40
  precision: fp32
```

The agent converts that into vector-search intent:

```yaml
profile_symptoms:
  - low_sm_utilization
  - precision_not_optimized
  - tensor_core_not_used

optimization_targets:
  - improve_sm_utilization
  - improve_throughput
  - enable_tensor_core

hardware_feature_keys:
  - tensor_core
  - bf16_support
  - tf32_support

candidate_technology_keys:
  - pytorch_amp
  - bf16_autocast
  - tf32_matmul
  - torch_compile
```

This lets the vector DB retrieve code documents and recipes that are relevant to the actual empirical profile.

---

# 3. Profile Graph DB Design

Keep the graph DB clean and focused on empirical memory.

## 3.1 Major Nodes

```yaml
graph_major_nodes:
  Job:
    purpose: >
      Empirical evidence from training, profiling, probing, or packed training.
      Stores configuration, memory use, runtime, utilization, and outcome.

  Hardware:
    purpose: >
      Hardware identity and capabilities, such as GPU model, VRAM, architecture,
      compute capability, Tensor Core support, BF16 support, and TF32 support.

  Model:
    purpose: >
      Model identity, model family, workload shape, and training characteristics.

  Technology:
    purpose: >
      Technology used or proposed, such as AMP, BF16, TF32, MPS, torch.compile,
      CUDA graphs, Tensor Core, Triton, or a batch-size optimizer.
```

## 3.2 Core Relationships

```yaml
graph_relationships:
  RAN_ON:
    from: Job
    to: Hardware
    description: Job was executed on this hardware.

  TRAINS_MODEL:
    from: Job
    to: Model
    description: Job trained or profiled this model.

  USES_TECHNOLOGY:
    from: Job
    to: Technology
    description: Job used this technology during execution.

  COMPATIBLE_WITH:
    from: Hardware
    to: Technology
    description: >
      Optional compatibility edge, e.g. A100 supports Tensor Core, BF16,
      TF32, CUDA graphs, etc.

  RELATED_TO:
    from: Technology
    to: Technology
    description: >
      Optional relationship between technologies, e.g. Tensor Core is related
      to AMP, BF16, FP16, TF32, matmul, convolution, and torch.compile.
```

## 3.3 Example Technology Nodes

```yaml
Technology:
  - technology_key: pytorch_amp
    name: PyTorch Automatic Mixed Precision
    category: precision_optimization

  - technology_key: bf16_autocast
    name: BF16 Autocast
    category: precision_optimization

  - technology_key: fp16_autocast
    name: FP16 Autocast
    category: precision_optimization

  - technology_key: tf32_matmul
    name: TF32 Matrix Multiplication
    category: tensor_core_acceleration

  - technology_key: torch_compile
    name: torch.compile
    category: compiler_optimization

  - technology_key: cuda_graphs
    name: CUDA Graphs
    category: launch_overhead_reduction

  - technology_key: triton_kernel
    name: Triton Kernel
    category: custom_kernel_optimization

  - technology_key: nvidia_mps
    name: NVIDIA MPS
    category: packing_or_concurrency

  - technology_key: power_of_two_batch_optimizer
    name: Power-of-two batch-size optimizer
    category: batch_size_search
```

---

# 4. Vector DB Design

The vector DB stores code knowledge, not profile facts.

Recommended collections:

```text
code_doc_chunks
optimization_recipe_chunks
api_symbol_chunks
```

You can also implement these as one collection with a `record_type` field, but separate collections are usually easier to maintain and search.

---

## 4.1 `CodeDocChunk` Collection

This is the main documentation chunk record.

```yaml
vector_collections:
  code_doc_chunks:
    description: >
      Embedded chunks from framework docs, CUDA docs, Tensor Core notes,
      PyTorch docs, Triton docs, MLEvolve internal notes, and optimization references.

    fields:
      chunk_id:
        type: string
        required: true
        unique: true
        description: Stable ID for this chunk.

      vector:
        type: float_vector
        required: true
        description: Embedding of the chunk text.

      text:
        type: string
        required: true
        description: Original chunk text.

      text_hash:
        type: string
        required: true
        description: Hash of normalized text for deduplication and update detection.

      source_id:
        type: string
        required: true
        description: Stable source document ID.

      source_type:
        type: string
        required: true
        allowed_values:
          - official_doc
          - api_reference
          - tutorial
          - paper
          - benchmark_note
          - internal_note
          - mlevolve_recipe
          - code_example
          - other

      source_title:
        type: string
        required: false

      source_url:
        type: string
        required: false

      source_version:
        type: string
        required: false
        description: Framework/library/documentation version.

      published_at:
        type: datetime
        required: false

      indexed_at:
        type: datetime
        required: true

      updated_at:
        type: datetime
        required: false

      deprecated:
        type: boolean
        required: false
        default: false

      # Graph-aligned metadata
      technology_keys:
        type: list[string]
        required: false
        description: >
          Technology keys matching graph Technology nodes.
          Examples: pytorch_amp, bf16_autocast, tf32_matmul, tensor_core,
          torch_compile, cuda_graphs, triton_kernel.

      hardware_keys:
        type: list[string]
        required: false
        description: >
          Specific graph Hardware keys if the doc is device-specific.
          Usually optional.

      hardware_feature_keys:
        type: list[string]
        required: false
        description: >
          Hardware capability tags. Examples: tensor_core, bf16_support,
          fp16_support, tf32_support, high_bandwidth_memory, cuda_capability_80.

      model_keys:
        type: list[string]
        required: false
        description: Specific graph Model keys if the doc is model-specific.

      model_families:
        type: list[string]
        required: false
        description: >
          Model categories. Examples: transformer, llm, cnn, vit, diffusion,
          recommendation, mlp, graph_neural_network.

      workload_types:
        type: list[string]
        required: false
        description: >
          Workload types. Examples: matmul_heavy, conv_heavy, attention_heavy,
          memory_bound, data_loading_bound, small_batch, large_batch.

      # Agent reasoning metadata
      optimization_targets:
        type: list[string]
        required: false
        examples:
          - reduce_vram
          - improve_sm_utilization
          - improve_throughput
          - reduce_step_time
          - enable_tensor_core
          - reduce_kernel_launch_overhead
          - improve_packing
          - improve_precision_efficiency
          - reduce_fragmentation
          - improve_dataloader

      profile_symptoms:
        type: list[string]
        required: false
        examples:
          - low_sm_utilization
          - high_vram_pressure
          - oom
          - low_vram_usage_low_sm
          - high_step_time
          - slow_packed_training
          - launch_overhead_bound
          - memory_bandwidth_bound
          - dataloader_bottleneck
          - precision_not_optimized
          - tensor_core_not_used

      api_symbols:
        type: list[string]
        required: false
        examples:
          - torch.amp.autocast
          - torch.amp.GradScaler
          - torch.set_float32_matmul_precision
          - torch.compile
          - torch.cuda.CUDAGraph
          - torch.backends.cuda.matmul.allow_tf32

      code_language:
        type: string
        required: false
        allowed_values:
          - python
          - cuda_cpp
          - triton
          - cython
          - shell
          - yaml
          - other

      framework:
        type: string
        required: false
        examples:
          - pytorch
          - cuda
          - triton
          - deepspeed
          - transformers
          - xformers

      framework_version:
        type: string
        required: false

      min_cuda_compute_capability:
        type: string
        required: false
        description: Optional hardware constraint, e.g. 8.0, 8.6, 8.9, 9.0.

      precision_modes:
        type: list[string]
        required: false
        examples:
          - fp32
          - fp16
          - bf16
          - tf32
          - int8
          - mixed

      risk_level:
        type: string
        required: false
        allowed_values:
          - low
          - medium
          - high

      applicability_notes:
        type: string
        required: false

      contraindications:
        type: list[string]
        required: false

      confidence:
        type: float
        required: false
        description: Confidence score from 0.0 to 1.0.
```

---

## 4.2 `OptimizationRecipeChunk` Collection

This collection stores actionable optimization recipes that the MLEvolve agent can directly use to propose code mutations.

```yaml
vector_collections:
  optimization_recipe_chunks:
    description: >
      Embedded optimization recipes for MLEvolve code mutations.

    fields:
      recipe_id:
        type: string
        required: true
        unique: true

      vector:
        type: float_vector
        required: true

      title:
        type: string
        required: true

      problem_statement:
        type: string
        required: true
        description: The optimization problem this recipe addresses.

      solution_summary:
        type: string
        required: true
        description: Short explanation of the proposed optimization.

      before_code:
        type: string
        required: false

      after_code:
        type: string
        required: false

      patch_template:
        type: string
        required: false
        description: Optional patch template the agent can use to mutate training code.

      validation_method:
        type: string
        required: false
        description: >
          How to validate the optimization, e.g. compare loss, throughput,
          VRAM, numerical stability, and convergence.

      technology_keys:
        type: list[string]
        required: true

      hardware_feature_keys:
        type: list[string]
        required: false

      model_families:
        type: list[string]
        required: false

      workload_types:
        type: list[string]
        required: false

      optimization_targets:
        type: list[string]
        required: true

      profile_symptoms:
        type: list[string]
        required: true

      api_symbols:
        type: list[string]
        required: false

      precision_modes:
        type: list[string]
        required: false

      framework:
        type: string
        required: false

      framework_version:
        type: string
        required: false

      source_chunk_ids:
        type: list[string]
        required: false
        description: Links back to CodeDocChunk records that justify this recipe.

      source_job_ids:
        type: list[string]
        required: false
        description: Optional graph Job IDs that empirically support this recipe.

      risk_level:
        type: string
        required: true
        allowed_values:
          - low
          - medium
          - high

      expected_metric_effects:
        type: map
        required: false
        description: >
          Expected effect on metrics, for example:
          {sm_utilization: increase, vram: decrease, step_time: decrease}

      contraindications:
        type: list[string]
        required: false

      confidence:
        type: float
        required: false
```

Example recipe:

```yaml
recipe_id: pytorch_amp_tensor_core_recipe_001
title: Use PyTorch AMP to enable Tensor Core-friendly mixed precision
technology_keys:
  - pytorch_amp
  - tensor_core
  - bf16_autocast
optimization_targets:
  - improve_sm_utilization
  - improve_throughput
  - reduce_vram
profile_symptoms:
  - low_sm_utilization
  - precision_not_optimized
  - tensor_core_not_used
api_symbols:
  - torch.amp.autocast
  - torch.amp.GradScaler
model_families:
  - transformer
  - cnn
  - vit
hardware_feature_keys:
  - tensor_core
  - bf16_support
risk_level: medium
expected_metric_effects:
  sm_utilization: increase
  throughput: increase
  vram: decrease_or_no_increase
  step_time: decrease
```

---

## 4.3 `ApiSymbolChunk` Collection

This collection stores exact API usage details. It is useful when the agent needs precise implementation guidance.

```yaml
vector_collections:
  api_symbol_chunks:
    description: >
      API-level documentation records for exact function, class, or method usage.

    fields:
      api_symbol_id:
        type: string
        required: true
        unique: true

      vector:
        type: float_vector
        required: true

      api_symbol:
        type: string
        required: true
        indexed: true
        examples:
          - torch.amp.autocast
          - torch.amp.GradScaler
          - torch.compile
          - torch.cuda.CUDAGraph

      framework:
        type: string
        required: true

      framework_version:
        type: string
        required: false

      signature:
        type: string
        required: false

      usage_summary:
        type: string
        required: true

      parameters_json:
        type: string
        required: false

      example_code:
        type: string
        required: false

      technology_keys:
        type: list[string]
        required: false

      optimization_targets:
        type: list[string]
        required: false

      profile_symptoms:
        type: list[string]
        required: false

      source_chunk_ids:
        type: list[string]
        required: false

      risk_level:
        type: string
        required: false
```

Two-stage retrieval example:

```text
1. Retrieve recipe:
   "Use AMP for Tensor Core mixed precision."

2. Retrieve exact API:
   "How do I correctly call torch.amp.autocast and GradScaler?"
```

---

# 5. Shared Ontology

The shared ontology is the most important integration mechanism.

Create one controlled vocabulary used by:

```text
graph ingestion
vector ingestion
agent retrieval
agent reasoning
agent write-back
```

Example:

```yaml
shared_ontology:
  technology_keys:
    pytorch_amp:
      name: PyTorch AMP
      categories:
        - precision_optimization
        - tensor_core_enablement
      related_api_symbols:
        - torch.amp.autocast
        - torch.amp.GradScaler

    bf16_autocast:
      name: BF16 autocast
      categories:
        - precision_optimization
      related_features:
        - bf16_support
        - tensor_core

    fp16_autocast:
      name: FP16 autocast
      categories:
        - precision_optimization
      related_features:
        - fp16_support
        - tensor_core

    tf32_matmul:
      name: TF32 matmul
      categories:
        - tensor_core_enablement
      related_api_symbols:
        - torch.set_float32_matmul_precision
        - torch.backends.cuda.matmul.allow_tf32

    torch_compile:
      name: torch.compile
      categories:
        - compiler_optimization
        - graph_capture

    cuda_graphs:
      name: CUDA Graphs
      categories:
        - launch_overhead_reduction

    triton_kernel:
      name: Triton custom kernel
      categories:
        - custom_kernel_optimization

    nvidia_mps:
      name: NVIDIA MPS
      categories:
        - concurrency
        - packing

  hardware_feature_keys:
    tensor_core:
      description: NVIDIA Tensor Core support.

    bf16_support:
      description: BF16 acceleration support.

    fp16_support:
      description: FP16 acceleration support.

    tf32_support:
      description: TF32 acceleration support.

    high_bandwidth_memory:
      description: High-bandwidth device memory.

    cuda_capability_80:
      description: CUDA compute capability 8.0.

  model_families:
    transformer:
      common_bottlenecks:
        - attention_heavy
        - matmul_heavy
        - activation_memory

    cnn:
      common_bottlenecks:
        - convolution_heavy
        - tensor_core_candidate

    diffusion:
      common_bottlenecks:
        - unet_memory
        - attention_memory
        - mixed_precision_candidate

  profile_symptoms:
    low_sm_utilization:
      description: GPU compute units are underused.

    high_vram_pressure:
      description: Job is close to memory cap or physical VRAM limit.

    oom:
      description: Job failed due to out-of-memory.

    launch_overhead_bound:
      description: Many small kernels or repeated launch overhead.

    memory_bandwidth_bound:
      description: Runtime likely limited by memory bandwidth.

    precision_not_optimized:
      description: Workload is likely still using inefficient precision.

    tensor_core_not_used:
      description: Workload likely does not use Tensor Core-friendly paths.

    dataloader_bottleneck:
      description: GPU may be waiting on CPU-side data loading or preprocessing.

  optimization_targets:
    improve_sm_utilization:
      description: Increase useful GPU compute utilization.

    reduce_vram:
      description: Reduce peak or average device memory usage.

    improve_throughput:
      description: Increase samples/sec or tokens/sec.

    reduce_step_time:
      description: Lower average training step time.

    improve_packing:
      description: Improve concurrent multi-model training behavior.
```

---

# 6. Integration Contract Between the Two Databases

The two DBs should not rely on each other's internal IDs. They should connect through stable semantic keys.

## 6.1 Shared Keys

```yaml
integration_keys:
  hardware_key:
    used_by:
      - Graph Hardware node
      - Vector metadata hardware_keys

  hardware_feature_key:
    used_by:
      - Graph Hardware capability list
      - Vector metadata hardware_feature_keys

  model_key:
    used_by:
      - Graph Model node
      - Vector metadata model_keys

  model_family:
    used_by:
      - Graph Model.model_family
      - Vector metadata model_families

  technology_key:
    used_by:
      - Graph Technology node
      - Vector metadata technology_keys

  profile_symptom:
    used_by:
      - Derived from Job metrics
      - Vector metadata profile_symptoms

  optimization_target:
    used_by:
      - Agent query planning
      - Vector metadata optimization_targets

  api_symbol:
    used_by:
      - Vector API docs
      - Agent code-generation plan
```

## 6.2 Avoid Internal Graph IDs

Avoid:

```text
neo4j_node_id = 12345
```

Prefer:

```text
technology_key = pytorch_amp
hardware_key = nvidia_a100_80gb
model_family = transformer
```

Internal graph IDs can change. Stable keys should not.

---

# 7. Generating Vector Metadata from the Graph Schema

## 7.1 From `Hardware`

Graph example:

```yaml
Hardware:
  hardware_key: nvidia_a100_80gb
  vendor: nvidia
  architecture: ampere
  total_vram_mb: 81920
  cuda_compute_capability: "8.0"
  hardware_feature_keys:
    - tensor_core
    - bf16_support
    - fp16_support
    - tf32_support
    - high_bandwidth_memory
```

Vector metadata should use:

```yaml
hardware_feature_keys:
  - tensor_core
  - bf16_support
  - fp16_support
  - tf32_support
```

Most code docs should be feature-based, not device-specific.

Good:

```yaml
hardware_feature_keys:
  - tensor_core
  - bf16_support
```

Less general:

```yaml
hardware_keys:
  - nvidia_a100_80gb
```

Use `hardware_key` only when documentation is device-specific.

---

## 7.2 From `Model`

Graph example:

```yaml
Model:
  model_key: resnet50
  model_family: cnn
  workload_types:
    - conv_heavy
    - tensor_core_candidate
```

Vector metadata should use:

```yaml
model_families:
  - cnn

workload_types:
  - conv_heavy
  - tensor_core_candidate
```

This lets the agent search for optimization docs relevant to CNNs even if no doc mentions `resnet50` directly.

---

## 7.3 From `Technology`

Graph example:

```yaml
Technology:
  technology_key: pytorch_amp
  category: precision_optimization
  api_symbols:
    - torch.amp.autocast
    - torch.amp.GradScaler
```

Vector metadata should use:

```yaml
technology_keys:
  - pytorch_amp

api_symbols:
  - torch.amp.autocast
  - torch.amp.GradScaler
```

This is the strongest bridge between the graph and vector DB.

---

## 7.4 From `Job`

Do not put `job_id` into most vector chunks. Instead, convert Job metrics into symptoms.

Example graph record:

```yaml
Job:
  avg_sm_utilization_pct: 22.0
  peak_vram_utilization_of_cap_pct: 38.0
  avg_step_time_ms: 420.0
  precision: fp32
  status: succeeded
```

Agent-derived symptoms:

```yaml
profile_symptoms:
  - low_sm_utilization
  - precision_not_optimized
  - tensor_core_not_used

optimization_targets:
  - improve_sm_utilization
  - improve_throughput
  - enable_tensor_core
```

These symptoms become vector DB filters.

---

# 8. Retrieval and Reasoning Workflow

Use a multi-stage retrieval pipeline.

## Stage 1: Normalize the Optimization Request

Input:

```text
Optimize this training code for A100.
The previous job used fp32, batch size 16, peak VRAM 32 GB under an 80 GB cap,
but SM utilization was only 25%.
```

The agent extracts:

```yaml
task_context:
  hardware_key: nvidia_a100_80gb
  hardware_feature_keys:
    - tensor_core
    - bf16_support
    - tf32_support

  model_family: transformer

  current_technologies:
    - pytorch_fp32

  observed_metrics:
    avg_sm_utilization_pct: 25
    peak_vram_utilization_of_cap_pct: 40

  derived_symptoms:
    - low_sm_utilization
    - precision_not_optimized
    - tensor_core_not_used

  optimization_targets:
    - improve_sm_utilization
    - improve_throughput
    - enable_tensor_core
```

---

## Stage 2: Query the Profile Graph DB

The agent asks:

```text
Have we seen similar jobs before?
```

Example Cypher:

```cypher
MATCH (j:Job)-[:RAN_ON]->(h:Hardware)
MATCH (j)-[:TRAINS_MODEL]->(m:Model)
WHERE h.hardware_key = $hardware_key
  AND m.model_family = $model_family
  AND j.status = "succeeded"
  AND j.reusable = true
RETURN
  j.job_id,
  j.profile_key,
  j.precision,
  j.batch_size,
  j.effective_available_vram_mb,
  j.peak_vram_mb,
  j.cap_headroom_mb,
  j.avg_sm_utilization_pct,
  j.avg_step_time_ms,
  j.throughput_samples_per_sec,
  j.technology_set_key,
  j.finished_at
ORDER BY j.finished_at DESC
LIMIT 20;
```

Graph result:

```yaml
empirical_context:
  similar_profiles:
    - job_id: job_001
      technology_set_key: pytorch_amp+bf16
      avg_sm_utilization_pct: 64
      peak_vram_mb: 41000
      avg_step_time_ms: 180

    - job_id: job_002
      technology_set_key: pytorch_fp32
      avg_sm_utilization_pct: 26
      peak_vram_mb: 36000
      avg_step_time_ms: 390
```

Interpretation:

```text
BF16 AMP worked well on similar jobs.
FP32 was slower and underused the GPU.
```

---

## Stage 3: Convert Graph Results into Vector Search Filters

Example filter:

```yaml
vector_filter:
  framework: pytorch
  hardware_feature_keys:
    any:
      - tensor_core
      - bf16_support
      - tf32_support
  technology_keys:
    any:
      - pytorch_amp
      - bf16_autocast
      - tf32_matmul
      - tensor_core
      - torch_compile
  model_families:
    any:
      - transformer
  profile_symptoms:
    any:
      - low_sm_utilization
      - precision_not_optimized
      - tensor_core_not_used
  optimization_targets:
    any:
      - improve_sm_utilization
      - improve_throughput
      - enable_tensor_core
```

Metadata filtering is important because pure vector similarity may retrieve semantically related but irrelevant chunks.

---

## Stage 4: Query the Vector DB

Query text should be generated from graph symptoms, not only from the user's wording.

Example vector query:

```text
PyTorch training has low SM utilization on NVIDIA Tensor Core GPU.
Current code uses FP32. Need code-level optimization to enable Tensor Core,
BF16 or FP16 AMP, TF32 matmul, and improve throughput without increasing VRAM too much.
```

Search collections:

```text
1. optimization_recipe_chunks
2. code_doc_chunks
3. api_symbol_chunks
```

Example retrieved context:

```yaml
retrieved_code_context:
  recipes:
    - recipe_id: pytorch_amp_tensor_core_recipe_001
      technology_keys:
        - pytorch_amp
        - bf16_autocast
        - tensor_core
      api_symbols:
        - torch.amp.autocast
      risk_level: medium

  docs:
    - chunk_id: pytorch_amp_doc_023
      api_symbols:
        - torch.amp.autocast
        - torch.amp.GradScaler

  api_symbols:
    - api_symbol: torch.amp.autocast
    - api_symbol: torch.set_float32_matmul_precision
```

---

## Stage 5: Cross-Rank Graph and Vector Evidence

The agent should not blindly trust the top vector result. It should rank candidates using both empirical graph evidence and retrieved code knowledge.

Recommended scoring:

```yaml
candidate_score:
  vector_similarity: 0.35
  metadata_match: 0.25
  graph_empirical_support: 0.25
  source_quality: 0.10
  risk_penalty: 0.05
```

Example:

```text
Candidate A: Use BF16 AMP
  vector similarity: high
  metadata match: high
  graph support: high, because previous similar jobs improved SM utilization
  risk: medium
  final rank: 1

Candidate B: Write custom Triton kernel
  vector similarity: medium
  metadata match: medium
  graph support: unknown
  risk: high
  final rank: 3
```

---

## Stage 6: Produce an Evidence Packet for the Reasoning Agent

Before code generation, assemble a structured evidence packet.

```yaml
agent_evidence_packet:
  task:
    goal:
      - improve_sm_utilization
      - improve_throughput
    constraints:
      - preserve training correctness
      - avoid OOM
      - maintain same model semantics

  graph_evidence:
    current_profile:
      avg_sm_utilization_pct: 25
      peak_vram_utilization_of_cap_pct: 40
      precision: fp32

    similar_successful_profiles:
      - technology_set_key: pytorch_amp+bf16
        avg_sm_utilization_pct: 64
        avg_step_time_ms: 180

  vector_evidence:
    recommended_recipes:
      - recipe_id: pytorch_amp_tensor_core_recipe_001
        summary: Use PyTorch autocast with BF16 or FP16 to enable Tensor Core-friendly operations.
        api_symbols:
          - torch.amp.autocast
          - torch.amp.GradScaler

    relevant_api_docs:
      - api_symbol: torch.amp.autocast
      - api_symbol: torch.set_float32_matmul_precision

  proposed_mutations:
    - add_bf16_autocast
    - enable_tf32_matmul
    - optionally_try_torch_compile

  validation_plan:
    metrics:
      - loss equivalence
      - avg_sm_utilization_pct
      - peak_vram_mb
      - avg_step_time_ms
      - throughput_samples_per_sec
```

---

# 9. Agent-Side Reasoning Policy

The agent should follow a fixed loop:

```text
1. Observe
   Read current profile from the graph DB.

2. Diagnose
   Convert metrics into symptoms.

3. Retrieve
   Query graph for similar empirical profiles.
   Query vector DB for code/API/recipe knowledge.

4. Rank
   Prefer optimizations supported by both empirical profiles and code docs.

5. Propose
   Generate one or more code mutations.

6. Validate
   Run benchmark/training probe.

7. Record
   Write the new result back to the graph DB as a new Job node.

8. Learn
   Optionally create or update an internal OptimizationRecipeChunk if the result
   becomes a reliable MLEvolve-specific recipe.
```

---

# 10. Example End-to-End Reasoning Flow

## Input Profile

```yaml
Job:
  job_id: job_fp32_001
  hardware_key: nvidia_a100_80gb
  model_family: transformer
  precision: fp32
  batch_size: 16
  avg_sm_utilization_pct: 24
  peak_vram_utilization_of_cap_pct: 45
  avg_step_time_ms: 410
  status: succeeded
```

## Diagnosis

```yaml
derived_symptoms:
  - low_sm_utilization
  - precision_not_optimized
  - tensor_core_not_used

optimization_targets:
  - improve_sm_utilization
  - improve_throughput
  - enable_tensor_core
```

## Graph Retrieval Result

```yaml
similar_jobs:
  - precision: bf16
    technologies:
      - pytorch_amp
      - bf16_autocast
    avg_sm_utilization_pct: 67
    avg_step_time_ms: 190
    peak_vram_utilization_of_cap_pct: 39

  - precision: fp32
    technologies:
      - none
    avg_sm_utilization_pct: 27
    avg_step_time_ms: 400
```

## Vector Retrieval Result

```yaml
top_recipes:
  - title: Use PyTorch AMP BF16 autocast
    api_symbols:
      - torch.amp.autocast

  - title: Enable TF32 matmul for FP32-heavy models
    api_symbols:
      - torch.set_float32_matmul_precision

  - title: Try torch.compile for repeated training step
    api_symbols:
      - torch.compile
```

## Agent Proposal

```yaml
proposed_plan:
  primary_mutation:
    - wrap forward/loss computation with torch.amp.autocast using bf16

  secondary_mutation:
    - enable high or medium float32 matmul precision for TF32

  optional_mutation:
    - test torch.compile on the model or training step

  validation:
    - compare loss curve against baseline
    - measure avg_step_time_ms
    - measure avg_sm_utilization_pct
    - measure peak_vram_mb
```

---

# 11. Write-Back Design After Each Experiment

After MLEvolve tests a code mutation, write empirical results back to the graph.

Example new graph node:

```yaml
Job:
  job_id: job_bf16_amp_002
  job_kind: single_profile
  purpose: real_training
  status: succeeded
  reusable: true
  model_set_key: transformer_model_x
  hardware_set_key: nvidia_a100_80gb
  technology_set_key: pytorch_amp+bf16_autocast+tf32_matmul
  precision: bf16
  batch_size: 16
  avg_sm_utilization_pct: 68
  peak_vram_mb: 35000
  peak_vram_utilization_of_cap_pct: 43
  avg_step_time_ms: 185
  throughput_samples_per_sec: 2.2x_baseline
```

Relationships:

```cypher
(:Job)-[:RAN_ON]->(:Hardware {hardware_key: "nvidia_a100_80gb"})
(:Job)-[:TRAINS_MODEL]->(:Model {model_key: "transformer_model_x"})
(:Job)-[:USES_TECHNOLOGY]->(:Technology {technology_key: "pytorch_amp"})
(:Job)-[:USES_TECHNOLOGY]->(:Technology {technology_key: "bf16_autocast"})
(:Job)-[:USES_TECHNOLOGY]->(:Technology {technology_key: "tf32_matmul"})
```

This closes the loop:

```text
vector DB suggests code optimization
→ agent mutates code
→ benchmark runs
→ graph DB records outcome
→ future agents use empirical evidence
```

---

# 12. Optional: Add Optimization Outcome Summaries to Vector DB

Most experimental results should go to the graph. But if a repeated pattern becomes a reusable coding rule, summarize it into the vector DB as an internal recipe.

Example:

```yaml
OptimizationRecipeChunk:
  recipe_id: mlevolve_bf16_amp_a100_transformer_001
  source_type: mlevolve_recipe
  title: BF16 AMP improves Transformer training on A100 for low-SM FP32 profiles
  problem_statement: >
    FP32 Transformer training on A100 shows low SM utilization and moderate VRAM usage.
  solution_summary: >
    Try BF16 autocast and TF32 matmul before more invasive kernel rewrites.
  technology_keys:
    - bf16_autocast
    - pytorch_amp
    - tf32_matmul
  hardware_feature_keys:
    - tensor_core
    - bf16_support
    - tf32_support
  model_families:
    - transformer
  profile_symptoms:
    - low_sm_utilization
    - precision_not_optimized
  optimization_targets:
    - improve_sm_utilization
    - improve_throughput
  source_job_ids:
    - job_bf16_amp_002
    - job_bf16_amp_014
  confidence: 0.86
```

This creates MLEvolve-specific knowledge without polluting the graph.

---

# 13. Retrieval Router Design

The agent should choose retrieval mode based on the task.

```yaml
retrieval_router:
  graph_only:
    when:
      - asking for previous batch size
      - asking whether a model fits in memory
      - asking for previous peak VRAM
      - asking for packing compatibility
      - asking for empirical runtime

  vector_only:
    when:
      - asking how to use a framework API
      - asking how to write Tensor Core-friendly PyTorch code
      - asking for CUDA/Triton implementation details
      - asking for code examples

  graph_then_vector:
    when:
      - asking why a job is slow
      - asking how to optimize based on profile metrics
      - asking how to improve SM utilization
      - asking how to reduce VRAM
      - asking how to improve packed training

  vector_then_graph:
    when:
      - asking whether a proposed optimization has worked before
      - asking whether torch.compile helped similar models
      - asking whether AMP caused instability in previous jobs
```

For your main MLEvolve use case, `graph_then_vector` is the most important route.

---

# 14. Retrieval Modes

## 14.1 Exact Graph Retrieval

Use for direct memory reuse:

```text
same model + same hardware + same memory cap + same technology stack
```

Example:

```cypher
MATCH (j:Job)-[:RAN_ON]->(h:Hardware)
MATCH (j)-[:TRAINS_MODEL]->(m:Model)
WHERE h.hardware_key = $hardware_key
  AND m.model_key = $model_key
  AND j.effective_available_vram_mb >= $required_cap_mb
  AND j.status = "succeeded"
  AND j.reusable = true
RETURN j
ORDER BY j.finished_at DESC
LIMIT 5;
```

---

## 14.2 Similar Graph Retrieval

Use when exact match is missing:

```text
same model family + similar hardware features + related technology
```

Example:

```cypher
MATCH (j:Job)-[:RAN_ON]->(h:Hardware)
MATCH (j)-[:TRAINS_MODEL]->(m:Model)
WHERE m.model_family = $model_family
  AND h.vendor = $vendor
  AND h.architecture = $architecture
  AND j.status = "succeeded"
RETURN j, h, m
ORDER BY j.finished_at DESC
LIMIT 20;
```

---

## 14.3 Filtered Vector Retrieval

Use when symptoms and hardware capabilities are known.

Example:

```yaml
filter:
  framework: pytorch
  hardware_feature_keys:
    any:
      - tensor_core
      - bf16_support
  profile_symptoms:
    any:
      - low_sm_utilization
      - precision_not_optimized
  optimization_targets:
    any:
      - improve_sm_utilization
      - improve_throughput
```

---

## 14.4 Graph-Augmented Vector Retrieval

With two separate databases, emulate graph-augmented vector retrieval as:

```text
1. Graph query returns hardware/model/technology/profile symptoms.
2. Agent converts them into vector metadata filters.
3. Vector DB returns docs/recipes/API chunks.
4. Agent combines both contexts.
```

---

# 15. Recommended Agent Context Format

The agent should not receive raw database dumps. It should receive a structured context object.

```yaml
optimization_context:
  current_job:
    job_id: job_fp32_001
    model_key: model_x
    model_family: transformer
    hardware_key: nvidia_a100_80gb
    technology_set_key: pytorch_fp32
    precision: fp32
    batch_size: 16
    avg_sm_utilization_pct: 24
    peak_vram_utilization_of_cap_pct: 45
    avg_step_time_ms: 410

  derived_diagnosis:
    profile_symptoms:
      - low_sm_utilization
      - precision_not_optimized
      - tensor_core_not_used
    optimization_targets:
      - improve_sm_utilization
      - improve_throughput
      - enable_tensor_core

  graph_evidence:
    exact_profiles: []
    similar_profiles:
      - job_id: job_bf16_014
        technology_set_key: pytorch_amp+bf16_autocast
        avg_sm_utilization_pct: 67
        avg_step_time_ms: 190
        peak_vram_utilization_of_cap_pct: 39

  vector_evidence:
    recipes:
      - recipe_id: pytorch_amp_tensor_core_recipe_001
        title: Use BF16 AMP to enable Tensor Core-friendly execution
        technology_keys:
          - pytorch_amp
          - bf16_autocast
        risk_level: medium

    api_symbols:
      - api_symbol: torch.amp.autocast
        usage_summary: Use autocast around forward and loss computation.

  constraints:
    preserve_accuracy: true
    avoid_oom: true
    max_allowed_vram_mb: 80000
    allowed_risk_level: medium
```

The reasoning agent then produces:

```yaml
optimization_decision:
  chosen_mutations:
    - enable_bf16_autocast
    - enable_tf32_matmul

  rejected_mutations:
    - custom_triton_kernel

  rejection_reasons:
    custom_triton_kernel: >
      Higher implementation risk; graph evidence already supports AMP/BF16.

  validation_plan:
    - run short benchmark
    - compare loss numerics
    - record avg_sm_utilization_pct
    - record peak_vram_mb
    - record avg_step_time_ms
```

---

# 16. Vector DB Ingestion Pipeline

Recommended ingestion process:

```text
1. Collect source documents
   PyTorch docs, CUDA docs, Tensor Core docs, Triton docs,
   MLEvolve internal optimization notes.

2. Normalize text
   Remove navigation, boilerplate, and unrelated examples.

3. Chunk semantically
   Prefer chunks around APIs, recipes, or optimization concepts.
   Avoid arbitrary fixed-size chunks when possible.

4. Extract metadata
   Use rules plus LLM annotation:
     technology_keys
     api_symbols
     hardware_feature_keys
     model_families
     optimization_targets
     profile_symptoms

5. Validate metadata
   Check all keys exist in the shared ontology.

6. Embed text
   Store vector + text + metadata.

7. Upsert into vector DB
   Use stable chunk_id and text_hash.

8. Run retrieval tests
   Query known scenarios:
     low SM utilization
     OOM
     Tensor Core not used
     packed training slowdown
     dataloader bottleneck

9. Periodically refresh
   Re-index docs when source versions change.
```

---

# 17. Versioning and Freshness

Code docs are version-sensitive.

Store these fields in vector records:

```yaml
versioning_fields:
  source_version: string
  framework_version: string
  cuda_version: string
  indexed_at: datetime
  updated_at: datetime
  valid_from: datetime
  valid_until: datetime
  deprecated: boolean
```

Example vector metadata:

```yaml
framework: pytorch
framework_version: "2.x"
deprecated: false
```

The graph-side `Job` should also store runtime versions:

```yaml
Job:
  framework: pytorch
  framework_version: "2.5.1"
  cuda_version: "12.4"
  driver_version: "..."
```

Then vector search can filter by compatible versions.

---

# 18. Handling Contradictions

Sometimes graph evidence and documentation disagree.

Example:

```text
Docs say AMP should improve throughput.
Graph says AMP caused instability for this model family.
```

Recommended policy:

```yaml
conflict_policy:
  graph_empirical_evidence:
    priority: high
    reason: >
      It reflects your actual hardware/model/code environment.

  official_docs:
    priority: high
    reason: >
      They define correct API usage.

  internal_recipes:
    priority: medium_to_high
    reason: >
      Useful if backed by multiple successful Job records.

  generic_blog_or_tutorial:
    priority: low
    reason: >
      May not match your environment.
```

Practical rule:

```text
Use docs for how to implement.
Use graph for whether to try it.
```

---

# 19. Minimal Implementation Plan

## Phase 1: Clean Graph DB

Create only the major nodes:

```text
Job
Hardware
Model
Technology
```

Add relationships:

```text
Job → Hardware
Job → Model
Job → Technology
```

Store profiling outcomes in `Job`.

---

## Phase 2: Define Shared Ontology

Create controlled vocabulary files:

```text
technology_keys.yaml
hardware_feature_keys.yaml
model_families.yaml
workload_types.yaml
profile_symptoms.yaml
optimization_targets.yaml
api_symbols.yaml
```

This ontology is used by graph ingestion, vector ingestion, and agent reasoning.

---

## Phase 3: Build Vector DB Schema

Start with two collections:

```text
code_doc_chunks
optimization_recipe_chunks
```

Add later:

```text
api_symbol_chunks
```

---

## Phase 4: Build Retrieval Router

Implement four routes:

```text
graph_only
vector_only
graph_then_vector
vector_then_graph
```

For MLEvolve hardware-aware coding optimization, prioritize `graph_then_vector`.

---

## Phase 5: Add Feedback Loop

Every optimization run produces:

```text
new Job node in graph DB
```

Successful repeated patterns can produce:

```text
new OptimizationRecipeChunk in vector DB
```

---

# 20. Final Recommended Design

```text
Profile Graph DB:
  Stores measured truth.

  Nodes:
    Job
    Hardware
    Model
    Technology

  Main purpose:
    empirical memory, profile reuse, optimization history.

Code-Docs Vector DB:
  Stores coding knowledge.

  Collections:
    CodeDocChunk
    OptimizationRecipeChunk
    ApiSymbolChunk

  Main purpose:
    API usage, code mutation ideas, hardware-aware programming guidance.

Shared Ontology:
  technology_key
  hardware_feature_key
  model_family
  workload_type
  profile_symptom
  optimization_target
  api_symbol

Agent:
  Converts Job metrics into symptoms.
  Retrieves empirical evidence from graph.
  Retrieves coding knowledge from vector DB.
  Ranks candidate optimizations.
  Generates code mutations.
  Runs validation.
  Writes new result back to graph.
```

The central principle:

```text
Do not merge the code-doc vector DB into the profile graph as ordinary graph nodes.
Keep them separate, but make them interoperable through shared ontology keys and agent-side retrieval logic.
```
