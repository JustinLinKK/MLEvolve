# Minimal Reproducer Notes

Fresh stress run:

```bash
MLEVOLVE_CONFIG=/workspaces/MLEvolve/config.example.yaml CUDA_VISIBLE_DEVICES=0 timeout --foreground --signal=TERM --kill-after=10s 21600s python /workspaces/MLEvolve/run.py exp_id=dogs-vs-cats-redux-kernels-edition dataset_dir=/workspaces/MLEvolve/data/mle-bench data_dir=/workspaces/MLEvolve/data/mle-bench/dogs-vs-cats-redux-kernels-edition/prepared/public desc_file=/workspaces/MLEvolve/data/mle-bench/dogs-vs-cats-redux-kernels-edition/prepared/public/description.md exp_name=dogs-vs-cats-redux-kernels-edition_current_stress_20260720_214920 log_dir=/workspaces/MLEvolve/reports/stress_test/20260720_214920/full_stress/runs workspace_dir=/workspaces/MLEvolve/reports/stress_test/20260720_214920/full_stress/runs experiment.mode=hardware_aware hardware_knowledge.enabled=true hardware_knowledge.include_profile_evidence=true hardware_knowledge.settings.graph.enabled=false scheduler.enabled=true agent.steps=20 agent.initial_drafts=3 agent.seed=5220 agent.time_limit=172800 scheduler.runtime_root=/workspaces/MLEvolve/reports/stress_test/20260720_214920/full_stress/scheduler_runtime scheduler.wait_timeout_seconds=150 exec.timeout=120 agent.use_global_memory=false agent.code.provider=codex agent.feedback.provider=codex agent.code.model=gpt-5.5 agent.feedback.model=gpt-5.5 'agent.code.base_url=""' 'agent.feedback.base_url=""' 'agent.code.api_key=""' 'agent.feedback.api_key=""' agent.code.executable=/home/vscode/.local/bin/codex agent.feedback.executable=/home/vscode/.local/bin/codex agent.code.reasoning_effort=low agent.feedback.reasoning_effort=low agent.code.timeout_seconds=300 agent.feedback.timeout_seconds=300 agent.code.ephemeral=true agent.feedback.ephemeral=true agent.code.ignore_user_config=true agent.feedback.ignore_user_config=true agent.code.isolated_home=true agent.feedback.isolated_home=true scheduler.settings.gpu_scheduler.mode=auto scheduler.settings.gpu_scheduler.backend_priority=[stream,cuda_process,exclusive] scheduler.settings.gpu_scheduler.concurrent_backend_allowlist=[stream] scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist=[stream,cuda_process] scheduler.settings.gpu_scheduler.stream.enabled=false scheduler.settings.gpu_scheduler.cuda_process.enabled=true scheduler.settings.gpu_scheduler.mps.enabled=false scheduler.settings.gpu_scheduler.model_family_probe_enabled=false scheduler.settings.gpu_scheduler.startpoint_probe_enabled=false scheduler.settings.gpu_scheduler.batch_probe_max_batch_size=32 scheduler.settings.gpu_scheduler.batch_probe_max_search_rounds=4 scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_startup_timeout_seconds=90 scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_step_timeout_seconds=30 scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_probe_timeout_seconds=45
```

Observed deterministic blocker:

1. Keep the validation server at `http://127.0.0.1:5005` unavailable, as in this run.
2. Produce or replay a node whose worker result succeeds but whose submission validation is unavailable.
3. Generate the next draft. `agents/draft_agent.py:94` calls `agent.virtual_root.fetch_child_memory()`.
4. `engine/search_node.py:344` raises `ValueError: max() iterable argument is empty` when every non-buggy child has `metric.value is None`.
5. In live scheduler mode, `run.py:474-493` retries generation immediately while a scheduler job is outstanding.

Saved artifacts:

- Run: `reports/stress_test/20260720_214920/full_stress/runs/20260720_214957_dogs-vs-cats-redux-kernels-edition_current_stress_20260720_214920`
- Journal: `reports/stress_test/20260720_214920/full_stress/runs/20260720_214957_dogs-vs-cats-redux-kernels-edition_current_stress_20260720_214920/logs/journal.json`
- Scheduler events: `reports/stress_test/20260720_214920/full_stress/scheduler_runtime/logs/events.jsonl`
- Scheduler DB: `reports/stress_test/20260720_214920/full_stress/scheduler_runtime/db/scheduler.sqlite3`
