<p align="center">
  <img src="assets/logo.svg" alt="MLEvolve" width="400"/>
</p>

🌐 **Project Page**: https://internscience.github.io/MLEvolve/

An agentic MLE (Machine Learning Engineering) system that automatically solves Kaggle-style ML competitions through Monte Carlo Graph Search (MCGS) with multi-agent collaboration. This is an advanced version based on [AutoMLGen](https://arxiv.org/abs/2510.08511). MLEvolve achieves **#1 on the [MLE-bench](https://github.com/openai/mle-bench) leaderboard** with **only 12 hours** of runtime.

## Documentation

- [MLEvolve Agent Workflow](docs/mlevolve_agent_workflow.md): step-by-step walkthrough of how the search controller, stage agents, executor, and evaluation loop work together during a run.
- [Hardware-Aware Optimization](docs/mlevolve_hardware_aware_optimization.md): design notes for feeding scheduler profile evidence and code-knowledge retrieval back into MLEvolve stage agents.

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

**1. Prepare mle-bench** — Install [mle-bench](https://github.com/openai/mle-bench) and download the dataset following its instructions.

**2. Install MLEvolve dependencies**

```bash
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt  
```

**3. Configure** — Copy `config.example.yaml` to the ignored local `config.yaml`, then fill in the fields you need:

```yaml
dataset_dir: "/path/to/mle-bench/data"

agent:
  code:
    provider: "gemini"
    base_url: "https://your-gemini-endpoint"
    api_key: "your-api-key"
  feedback:
    provider: "gemini"
    base_url: "https://your-gemini-endpoint"
    api_key: "your-api-key"
```

To use OpenRouter instead, select an OpenRouter model slug and the OpenRouter
OpenAI-compatible endpoint:

```yaml
agent:
  code:
    provider: "openrouter"
    model: "google/gemini-2.5-pro"
    base_url: "https://openrouter.ai/api/v1"
    api_key: "your-openrouter-api-key"
  feedback:
    provider: "openrouter"
    model: "google/gemini-2.5-pro"
    base_url: "https://openrouter.ai/api/v1"
    api_key: "your-openrouter-api-key"
```

OpenRouter embedding ingestion is configured in the same local file under `scheduler.settings.hardware_feature_db` with `embedding_model_type`, `embedding_model_name`, `embedding_dimension`, and `embedding_api_key_env`. Other tunable fields (`agent.steps`, `agent.time_limit`, etc.) have sensible defaults in `config.example.yaml`.

If `agent.use_global_memory: True`, you must also set `agent.memory_embedding_model_path` to a valid HuggingFace embedding model name or local model path. Set `agent.memory_embedding_device` to `cpu` if CUDA is unavailable.

### Cold-Start Models (optional)

Cold-start recommends pretrained models per task category based on `engine/coldstart/models_guidance_classified.json`. Most models auto-download from HuggingFace; for models requiring local weights, set `torch_hub_dir` in `config.yaml`. To disable cold-start entirely, set `coldstart.use_coldstart: False`.

## Quick Start

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]

# Example
bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

Results are written to `./runs/<timestamp>_<exp_id>/` including search tree logs, best solution code, and top-K candidate submissions.

## Acknowledgments

We thank [AIDE](https://github.com/WecoAI/aideml) and [ML-Master](https://github.com/sjtu-sai-agents/ML-Master) for their contributions to the development of the MCTS in MLE, and [InternAgent 1.5](https://github.com/InternScience/InternAgent) for its contributions to the development of the agentic memory mechanism. We sincerely thank all teams for their open-source contributions to the community.

## Citation

If you find this repo useful, you can also cite our earlier work.

```bibtex
@article{du2025automlgen,
  title={AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents},
  author={Du, Shangheng and Yan, Xiangchao and Jiang, Dengyang and Yuan, Jiakang and Hu, Yusong and Li, Xin and He, Liang and Zhang, Bo and Bai, Lei},
  journal={arXiv preprint arXiv:2510.08511},
  year={2025}
}

@article{feng2026internagent,
  title={InternAgent-1.5: A Unified Agentic Framework for Long-Horizon Autonomous Scientific Discovery},
  author={Shiyang Feng and Runmin Ma and Xiangchao Yan and Yue Fan and Yusong Hu and Songtao Huang and Shuaiyu Zhang and Zongsheng Cao and Tianshuo Peng and Jiakang Yuan and Zijie Guo and Zhijie Zhong and Shangheng Du and Weida Wang and Jinxin Shi and Yuhao Zhou and Xiaohan He and Zhiyin Yu and Fangchen Yu and Bihao Zhan and Qihao Zheng and Jiamin Wu and Mianxin Liu and Chi Zhang and Shaowei Hou and Shuya Li and Yankai Jiang and Wenjie Lou and Lilong Wang and Zifu Wang and Jiong Wang and Wanghan Xu and Yue Deng and Dongrui Liu and Yiheng Wang and Wenlong Zhang and Fenghua Ling and Shufei Zhang and Xiaosong Wang and Shuangjia Zheng and Xun Huang and Siqi Sun and Shuyue Hu and Peng Ye and Chunfeng Song and Bin Wang and Conghui He and Yihao Liu and Xin Li and Qibin Hou and Tao Chen and Xiangyu Yue and Bin Wang and Liang He and Dahua Lin and Bowen Zhou and Bo Zhang and Lei Bai},
  journal={arXiv preprint arXiv:2602.08990},
  year={2026}
}
```
