# Novelty / Positioning

> Schedule the heterogeneous, LLM-generated training jobs that the search tree produces, where job utility = expected search value per GPU-hour (Value-of-Information), and the scheduler co-designs bidirectionally with the MCGS search controller.

# Others

- Agent Generation Time also has to be considered in scheduler

- A model size predictor (PerfSeer)

- Parallel Parameter Tune

- [ ] Optimize root initial draft flow: the inherited root draft phase is intentionally sequential and uses `execute_immediately=False` to defer training, but this delays GPU execution and scheduler/profile evidence. Parallelize root draft/code-review generation with reservation-safe child slots, keep `AgentSearch.step(... execute_immediately=True)` as the method default, preserve `execute_immediately=False` only for explicit deferred batch/scheduler collection, and consider overlapping first draft execution with remaining draft generation for profiling-heavy comparisons.

- SWE Bench Output results for fine-tuning GNN

- [ ] Define model-structure features for prediction: backbone name, parameter count if available, framework, training mode, feature extractor vs end-to-end fine-tune, input shape, batch size, optimizer, epochs, folds, augmentation cost, and precision mode.
- [ ] Build a lightweight predictor from accumulated scheduler runs and batch-size observations.
- [ ] Track predictor confidence and trigger selective actual probing only when confidence is low or the estimated memory margin is unsafe.
- [ ] Design predictor-based scheduler mode. Intentionally undecided: do not add predictor planner behavior, APIs, schema, or config until the mode contract is specified.
