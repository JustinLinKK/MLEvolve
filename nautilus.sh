kubectl create secret generic scheduler-db-env \
  --from-literal=LOCALML_SCHEDULER_NEO4J_PASSWORD=change-me \
  --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/nautilus/qdrant.yaml
kubectl apply -f k8s/nautilus/neo4j.yaml

kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=neo4j --timeout=300s

kubectl apply -f k8s/nautilus/knowledge-ingest-job.yaml
kubectl wait --for=condition=complete job/mlevolve-knowledge-ingest --timeout=1800s

kubectl apply -f k8s/nautilus/mlevolve-job.yaml