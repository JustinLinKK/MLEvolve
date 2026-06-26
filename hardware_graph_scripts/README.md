# Hardware Graph Pre-Integration Scripts

These scripts load and verify the static hardware knowledge graph before it is
integrated into the MLEvolve agent loop.

Default target:

- Hardware graph DB outside containers: `bolt://127.0.0.1:7688`
- Hardware graph DB inside devcontainers: `bolt://host.docker.internal:7688`
- User: `neo4j`
- Password env: `LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD`
- Default password: `test12345`
- Default hardware for exact checks: `GeForce RTX 5090`

## One-Command Smoke

```bash
./hardware_graph_scripts/run_hardware_graph_preintegration.sh --start-databases --recreate
```

This starts/reuses the local Docker hardware Neo4j service, loads
`schema/hardware_knowledge_graph.json`, checks Neo4j edges directly, then runs
the three-stage agent query simulation.

## Individual Steps

Load the hardware graph:

```bash
./hardware_graph_scripts/setup_hardware_graph_db.sh --recreate
```

Query the stage-specific static query tool:

```bash
./hardware_graph_scripts/query_hardware_graph.sh stage-context "GeForce RTX 5090" model_design 16
./hardware_graph_scripts/query_hardware_graph.sh features "GeForce RTX 5090" training_evaluation 16
```

Verify loaded Neo4j records:

```bash
./hardware_graph_scripts/verify_hardware_graph_db.sh "GeForce RTX 5090"
```

Simulate the pre-integration agent flow:

```bash
./hardware_graph_scripts/simulate_3_stage_hardware_agent.sh "GeForce RTX 5090" --db-check
```

The three simulated stages use the same composite stage names as the active
stepwise pipeline:

1. `model_design`: data loading, feature engineering, model/loss/interface, and hardware-compatible model shape choices.
2. `datatype_precision`: numeric dtype, AMP/autocast, TF32, Transformer Engine recipes, precision-required adapters, and fallbacks.
3. `training_evaluation`: optimizer, scheduler, batch policy, dataloader policy, memory/runtime policy, validation, and submission.

Legacy low-level filters (`datatype`, `model`, `optimizer`, `tuning`) remain
accepted by `query_hardware_graph.sh` for manual inspection, but the simulation
asserts only the three composite pipeline stages.
