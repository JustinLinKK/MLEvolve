QDRANT_URL=http://qdrant:6333 \
GRAPH_DB_URI=bolt://neo4j:7687 \
bash compare_hardware_awareness.sh histopathologic-cancer-detection \
  --dataset-root data/mle-bench \
  --config config.yaml \
  --steps 5 \
  --initial-drafts 3 \
  --timeout-seconds 3600 \
  --memory-index 0 \
  --skip-prepare