# V100 Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy and benchmark a V100-compatible tensor-parallel local Qwen3.8-27B service.

**Architecture:** Keep the current Transformers service live during feasibility work. Probe vLLM first because it natively supports Qwen3.5 architecture, continuous batching, prefix cache, and CUDA Graphs. If vLLM cannot execute V100 kernels, stop that path and evaluate llama.cpp CUDA separately rather than mixing engines.

**Tech Stack:** vLLM, PyTorch 2.6.0 CUDA 11.8, NVIDIA V100, NVLink, FastAPI-compatible HTTP, Python benchmark client.

**Spec:** `docs/superpowers/specs/2026-08-23-v100-inference-engine-design.md`

## Global Constraints

- Use username `downeyflyfan` for Nautilus V100.
- Selected V100 memory maximum: 30 GiB; scheduler memory upper bound: 31 GiB.
- Use GPU 0 and GPU 1 for first tensor-parallel candidate; GPU 3 remains free.
- Use `nohup` for remote installations and benchmarks.
- Store results, records, and Gantt-plus-metric-node image in `records`.

---

### Task 1: vLLM V100 feasibility probe

**Files:**
- Create: `benchmarks/qwen38_v100_int8/vllm_config.py`
- Create: `benchmarks/qwen38_v100_int8/vllm_probe.sh`
- Create: `benchmarks/qwen38_v100_int8/test_vllm_probe_config.py`
- Test: `benchmarks/qwen38_v100_int8/test_vllm_probe_config.py`

**Interfaces:**
- Consumes: `QWEN_MODEL_PATH`, `CUDA_VISIBLE_DEVICES`, `VLLM_VENV`.
- Produces: `results/vllm_v100_probe.json` with package, device capability, and
  executable status.

- [ ] **Step 1: Write failing test**

```python
def test_vllm_probe_requires_two_nvlink_v100_devices() -> None:
    assert build_probe_command("/models/Qwen3.8-27B") == [
        "vllm", "serve", "/models/Qwen3.8-27B", "--tensor-parallel-size", "2"
    ]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest benchmarks/qwen38_v100_int8/test_vllm_probe_config.py::test_vllm_probe_requires_two_nvlink_v100_devices -v`

Expected: FAIL because `build_probe_command` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_probe_command(model_path: str) -> list[str]:
    return ["vllm", "serve", model_path, "--tensor-parallel-size", "2"]
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest benchmarks/qwen38_v100_int8/test_vllm_probe_config.py::test_vllm_probe_requires_two_nvlink_v100_devices -v`

Expected: PASS.

- [ ] **Step 5: Run remote package and kernel probe**

Run under `nohup`; record exact vLLM, PyTorch, CUDA, and V100 compatibility.
Do not alter the live Transformers service.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/qwen38_v100_int8/vllm_config.py benchmarks/qwen38_v100_int8/vllm_probe.sh benchmarks/qwen38_v100_int8/test_vllm_probe_config.py records/
git commit -m "probe vllm on v100"
```

### Task 2: Tensor-parallel deployment or V100-compatible fallback

**Files:**
- Create: `benchmarks/qwen38_v100_int8/vllm_server.sh`
- Create: `benchmarks/qwen38_v100_int8/test_vllm_server_config.py`
- Modify: `benchmarks/qwen38_v100_int8/vllm_config.py`
- Modify: `benchmarks/qwen38_v100_int8/benchmark.py`
- Create: `records/2026-08-23_v100_tensor_parallel.md`

**Interfaces:**
- Consumes: Task 1 executable result and model path.
- Produces: a health-checked service on GPU 0 and GPU 1 and benchmark JSON/PNG.

- [ ] **Step 1: Write failing test**

```python
def test_server_command_reserves_gpu_three_and_limits_memory() -> None:
    command = build_server_command("/models/Qwen3.8-27B")
    assert "CUDA_VISIBLE_DEVICES=0,1" in command
    assert "--tensor-parallel-size 2" in command
    assert "--gpu-memory-utilization 0.92" in command
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest benchmarks/qwen38_v100_int8/test_vllm_server_config.py::test_server_command_reserves_gpu_three_and_limits_memory -v`

Expected: FAIL because `build_server_command` does not exist.

- [ ] **Step 3: Implement one selected engine only**

Use vLLM only if Task 1 proves V100 inference works. Otherwise record the
failure and create a separate llama.cpp plan; do not silently switch engine.

- [ ] **Step 4: Run health, memory, and benchmark checks**

Use one warm-up plus three streamed prompts. Collect Time to First Token,
decode tokens per second, per-GPU memory, Gantt, and metric-node graph.

- [ ] **Step 5: Commit and push**

```bash
git add benchmarks/qwen38_v100_int8 records/
git commit -m "bench v100 tensor parallel inference"
git push origin HEAD:refs/heads/codex/scheduler-runtime-estimate-v100
```
