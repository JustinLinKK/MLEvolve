// 1. Find successful single-job evidence for a model on current hardware.
MATCH (j:Job:SingleJob)-[:SINGLE_TRAINS_MODEL]->(m:Model)
MATCH (j)-[:JOB_USED_HARDWARE]->(h:Hardware)
WHERE m.model_key = $model_key
  AND h.hardware_key = $hardware_key
  AND j.status = "succeeded"
RETURN j.job_id, j.purpose, j.resolved_batch_size, j.peak_vram_mb,
       j.observed_avg_step_time_ms, j.primary_metric_value, j.finished_at
ORDER BY j.finished_at DESC
LIMIT 10;

// 2. Reuse batch-size probe evidence.
MATCH (j:Job:SingleJob)-[:SINGLE_TRAINS_MODEL]->(m:Model)
MATCH (j)-[:JOB_USED_HARDWARE]->(h:Hardware)
WHERE j.purpose = "batch_size_probe"
  AND m.model_key = $model_key
  AND h.hardware_key = $hardware_key
  AND j.max_safe_batch_size IS NOT NULL
RETURN j.max_safe_batch_size, j.oom_batch_size, j.peak_vram_mb, j.confidence
ORDER BY j.finished_at DESC
LIMIT 5;

// 3. Find packed combinations that worked for a model family.
MATCH (p:Job:PackedJob)-[:HAS_PACKED_MEMBER]->(member:PackedJobMember)
MATCH (member)-[:MEMBER_TRAINS_MODEL]->(m:Model)
MATCH (p)-[:JOB_USED_HARDWARE]->(h:Hardware)
WHERE m.model_family = $model_family
  AND h.hardware_key = $hardware_key
  AND p.compatible = true
RETURN p.job_id, p.packing_group_key, p.packing_strategy,
       p.slowdown_ratio, p.throughput_efficiency, collect(m.model_key) AS members
ORDER BY p.finished_at DESC
LIMIT 10;

// 4. Convert graph evidence into vector-search keys.
MATCH (j:Job:SingleJob)-[:SINGLE_TRAINS_MODEL]->(m:Model)
MATCH (j)-[:JOB_USED_HARDWARE]->(h:Hardware)
OPTIONAL MATCH (j)-[:JOB_USES_TECHNOLOGY]->(t:Technology)
WHERE j.job_id = $job_id
RETURN h.hardware_key AS hardware_key,
       h.technology_keys AS hardware_feature_keys,
       m.model_family AS model_family,
       collect(t.technology_key) + coalesce(j.technology_keys, []) AS technology_keys,
       j.avg_sm_utilization_pct AS avg_sm_utilization_pct,
       j.peak_vram_mb AS peak_vram_mb,
       j.error_message AS error_message;
