# MLEvolve — Hardware-Aware Scheduler Test Bed

This repository contains the trace-driven test bed for the hardware-aware training-job scheduler:

- **Trace replay benchmark** (`scheduler_benchmark_test/`) — replays recorded MLEvolve candidate timelines against the scheduler and a multiprocess baseline, plus stress-test fixtures.
- **Hardware Knowledge Base (HWKB)** (`hardware_knowledge_graph/`, `schema/`, `hardware_graph_scripts/`) — Neo4j graph + vector store of hardware capability knowledge queried by the scheduler.
- **Scheduler** (`localml_scheduler/`) — local machine-learning-job scheduler with profiling, prediction, early stopping, checkpointing, packing, and observability.

## Layout

| Path | Purpose |
|------|---------|
| `scheduler_benchmark_test/` | Trace replay entry points (`run_histopath_*.sh`), timeline extraction, stress-test data |
| `replay_model_sources/` | Archived model-source traces used by the replay fixtures |
| `localml_scheduler/` | Scheduler package (`client`, `scheduler/`, `execution/`, `prediction/`, `profiling/`, `storage/`, CLI) |
| `hardware_knowledge_graph/` | HWKB client/store code (Neo4j + vector DB) |
| `hardware_graph_scripts/` | HWKB database setup, ingest, query, and verification scripts |
| `schema/` | Graph/vector DB schema YAMLs and `schema-guidance.md` |
| `engine/script_introspection.py` | Static introspection of training scripts (batch size, framework, script signature) used by the scheduler adapter |
| `utils/` | `candidate_timing.py` (phase instrumentation), `plot_hardware_awareness_comparison.py` |
| `tests/` | Scheduler and HWKB unit tests |
| `reports/` | Experiment records (stress-test benchmark) |
| `docker-compose.local.yml`, `docker_host_databases.sh`, `bootstrap.sh` | Local Neo4j/database infrastructure |

## Setup

```bash
pip install --no-deps -r requirements_base.txt
cp config.example.yaml config.yaml   # fill in hardware_knowledge + scheduler settings
bash docker_host_databases.sh        # local Neo4j / databases
bash bootstrap.sh                    # HWKB checks + optional knowledge ingest (MLEVOLVE_INGEST_KNOWLEDGE=1)
```

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

## Acknowledgments

We thank [AIDE](https://github.com/WecoAI/aideml) and [ML-Master](https://github.com/sjtu-sai-agents/ML-Master) for their contributions to the development of the MCTS in MLE, and [InternAgent 1.5](https://github.com/InternScience/InternAgent) for its contributions to the development of the agentic memory mechanism.

## Citation

```bibtex
@article{du2025automlgen,
  title={AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents},
  author={Du, Shangheng and Yan, Xiangchao and Jiang, Dengyang and Yuan, Jiakang and Hu, Yusong and Li, Xin and He, Liang and Zhang, Bo and Bai, Lei},
  journal={arXiv preprint arXiv:2510.08511},
  year={2025}
}
```
