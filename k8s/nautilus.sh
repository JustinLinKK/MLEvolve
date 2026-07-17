kubectl create secret generic hardware-knowledge-env \
  --from-literal=HARDWARE_KNOWLEDGE_NEO4J_PASSWORD=change-me \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/nautilus/neo4j.yaml

kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=neo4j --timeout=300s

kubectl apply -f k8s/nautilus/knowledge-ingest-job.yaml
kubectl wait --for=condition=complete job/mlevolve-knowledge-ingest --timeout=1800s

kubectl apply -f k8s/nautilus/mlevolve-job.yaml
