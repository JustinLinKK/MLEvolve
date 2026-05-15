// 1. Find the best resolved batch size for a model on a given accelerator class.
MATCH (m:Model {model_name: $model_name})<-[:OBSERVES_MODEL]-(p:BatchProbeProfile)-[:FOR_ACCELERATOR]->(a:Accelerator {accelerator_name: $accelerator_name})
RETURN m.model_name, a.accelerator_name, p.resolved_batch_size, p.target_budget_mb, p.peak_vram_mb, p.observations
ORDER BY p.observations DESC, p.updated_at DESC;

// 2. Compare runtime profiles for one workload signature across hardware targets.
MATCH (s:WorkloadSignature {signature: $signature})<-[:RUNTIME_FOR_SIGNATURE]-(r:RuntimeProfile)-[:RUNTIME_ON_HARDWARE]->(h:Hardware)
RETURN s.signature, h.gpu_name, h.toolkit_version, r.backend_name, r.resolved_batch_size, r.epoch_1_seconds, r.avg_step_time_ms, r.estimated_total_runtime_seconds, r.confidence
ORDER BY r.confidence DESC, r.updated_at DESC;

// 3. Retrieve successful packet profiles linking two models.
MATCH (m1:Model {model_name: $left_model})<-[:INVOLVES_MODEL]-(p:PacketProfile {compatible: true})-[:INVOLVES_MODEL]->(m2:Model {model_name: $right_model})
MATCH (p)-[:PACKET_ON_HARDWARE]->(h:Hardware)
RETURN p.profile_scope, h.gpu_name, p.backend_name, p.peak_vram_mb, p.avg_gpu_utilization, p.avg_memory_utilization, p.slowdown_ratio, p.observations
ORDER BY p.observations DESC, p.updated_at DESC;

// 4. Walk from one job to all of its evidence and reusable profiles.
MATCH (j:Job {job_id: $job_id})-[:HAS_RUN_PROFILE]->(run:RunProfile)
OPTIONAL MATCH (run)-[:AGGREGATED_INTO_BATCH_PROBE]->(bp:BatchProbeProfile)
OPTIONAL MATCH (run)-[:AGGREGATED_INTO_RUNTIME_PROFILE]->(rp:RuntimeProfile)
OPTIONAL MATCH (run)-[:AGGREGATED_INTO_SOLO_PROFILE]->(sp:SoloProfile)
OPTIONAL MATCH (run)-[:AGGREGATED_INTO_PACKET_PROFILE]->(pp:PacketProfile)
RETURN j.job_id, run.run_kind, run.status, bp.probe_key, rp.profile_key, sp.solo_profile_id, pp.packet_profile_id;

// 5. GraphRAG-oriented retrieval for factual profile summaries.
CALL db.index.fulltext.queryNodes("graphrag_fact_summary", $query) YIELD node, score
RETURN labels(node) AS labels, node.uid AS uid, node.summary_text AS summary_text, score
ORDER BY score DESC
LIMIT 20;
