python -m hardware_knowledge_graph.cli ingest --config "$PWD/config.example.yaml" --schema-root "$PWD/schema" --dry-run

# Optional if HARDWARE_KNOWLEDGE_NEO4J_PASSWORD is set:
python -m hardware_knowledge_graph.cli ingest --config "$PWD/config.example.yaml" --schema-root "$PWD/schema" --recreate

RUN_ROOT="$PWD/runs/profile_scheduler_on_histopathologic-cancer-detection_codex_$(date +%Y%m%d_%H%M%S)"

bash compare_profile_scheduler.sh histopathologic-cancer-detection \
  --dataset-root "$PWD/data/mle-bench" \
  --config "$PWD/config.example.yaml" \
  --skip-prepare \
  --scheduler-on-only \
  --disable-max-packing-limit \
  --steps 500 \
  --initial-drafts 4 \
  --seed 42 \
  --agent-time-limit 43200 \
  --timeout-seconds 43200 \
  --memory-index 0 \
  --server-id 121 \
  --run-root "$RUN_ROOT" \
  --plot-output-dir "$RUN_ROOT/comparison_plots" \
  -- \
  agent.code.provider=codex agent.feedback.provider=codex \
  agent.code.model=gpt-5.5 agent.feedback.model=gpt-5.5 \
  'agent.code.base_url=""' 'agent.feedback.base_url=""' \
  'agent.code.api_key=""' 'agent.feedback.api_key=""' \
  agent.code.executable=/home/vscode/.local/bin/codex \
  agent.feedback.executable=/home/vscode/.local/bin/codex \
  agent.code.reasoning_effort=low agent.feedback.reasoning_effort=low \
  agent.code.timeout_seconds=1200 agent.feedback.timeout_seconds=1200 \
  agent.code.ephemeral=true agent.feedback.ephemeral=true \
  agent.code.ignore_user_config=true agent.feedback.ignore_user_config=true \
  agent.code.isolated_home=true agent.feedback.isolated_home=true \
  scheduler.wait_timeout_seconds=300 exec.timeout=1200 \
  hardware_knowledge.enabled=true hardware_knowledge.include_profile_evidence=true \
  hardware_knowledge.settings.graph.enabled=true \
  'scheduler.settings.gpu_scheduler.backend_priority=[stream_mps,stream,cuda_process,mps,exclusive]' \
  'scheduler.settings.gpu_scheduler.concurrent_backend_allowlist=[stream_mps,stream]' \
  'scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist=[]' \
  scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_startup_timeout_seconds=90 \
  scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_step_timeout_seconds=30 \
  scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_probe_timeout_seconds=45