CREATE CONSTRAINT workflow_id_unique IF NOT EXISTS
FOR (n:Workflow) REQUIRE n.workflow_id IS UNIQUE;

CREATE CONSTRAINT job_id_unique IF NOT EXISTS
FOR (n:Job) REQUIRE n.job_id IS UNIQUE;

CREATE CONSTRAINT model_key_unique IF NOT EXISTS
FOR (n:Model) REQUIRE n.model_key IS UNIQUE;

CREATE CONSTRAINT workload_signature_unique IF NOT EXISTS
FOR (n:WorkloadSignature) REQUIRE n.signature IS UNIQUE;

CREATE CONSTRAINT backend_name_unique IF NOT EXISTS
FOR (n:Backend) REQUIRE n.backend_name IS UNIQUE;

CREATE CONSTRAINT toolkit_key_unique IF NOT EXISTS
FOR (n:Toolkit) REQUIRE n.toolkit_key IS UNIQUE;

CREATE CONSTRAINT accelerator_key_unique IF NOT EXISTS
FOR (n:Accelerator) REQUIRE n.accelerator_key IS UNIQUE;

CREATE CONSTRAINT hardware_key_unique IF NOT EXISTS
FOR (n:Hardware) REQUIRE n.hardware_key IS UNIQUE;

CREATE CONSTRAINT batch_shape_signature_unique IF NOT EXISTS
FOR (n:BatchShape) REQUIRE n.shape_signature IS UNIQUE;

CREATE CONSTRAINT run_profile_id_unique IF NOT EXISTS
FOR (n:RunProfile) REQUIRE n.run_profile_id IS UNIQUE;

CREATE CONSTRAINT batch_probe_key_unique IF NOT EXISTS
FOR (n:BatchProbeProfile) REQUIRE n.probe_key IS UNIQUE;

CREATE CONSTRAINT batch_observation_key_unique IF NOT EXISTS
FOR (n:BatchSizeObservation) REQUIRE n.observation_key IS UNIQUE;

CREATE CONSTRAINT runtime_profile_key_unique IF NOT EXISTS
FOR (n:RuntimeProfile) REQUIRE n.profile_key IS UNIQUE;

CREATE CONSTRAINT solo_profile_id_unique IF NOT EXISTS
FOR (n:SoloProfile) REQUIRE n.solo_profile_id IS UNIQUE;

CREATE CONSTRAINT packet_profile_id_unique IF NOT EXISTS
FOR (n:PacketProfile) REQUIRE n.packet_profile_id IS UNIQUE;

CREATE CONSTRAINT checkpoint_id_unique IF NOT EXISTS
FOR (n:Checkpoint) REQUIRE n.checkpoint_id IS UNIQUE;

CREATE CONSTRAINT event_id_unique IF NOT EXISTS
FOR (n:Event) REQUIRE n.event_id IS UNIQUE;

CREATE CONSTRAINT command_id_unique IF NOT EXISTS
FOR (n:Command) REQUIRE n.command_id IS UNIQUE;

CREATE CONSTRAINT cache_key_unique IF NOT EXISTS
FOR (n:CacheEntry) REQUIRE n.cache_key IS UNIQUE;

CREATE INDEX job_status_priority_lookup IF NOT EXISTS
FOR (n:Job) ON (n.status, n.priority, n.queue_sequence);

CREATE INDEX run_profile_lookup IF NOT EXISTS
FOR (n:RunProfile) ON (n.model_key, n.backend_name, n.resolved_batch_size, n.run_kind);

CREATE INDEX runtime_profile_lookup IF NOT EXISTS
FOR (n:RuntimeProfile) ON (n.signature, n.hardware_key, n.backend_name, n.resolved_batch_size);

CREATE INDEX batch_probe_lookup IF NOT EXISTS
FOR (n:BatchProbeProfile) ON (n.model_key, n.shape_signature, n.resolved_batch_size);

CREATE INDEX batch_observation_lookup IF NOT EXISTS
FOR (n:BatchSizeObservation) ON (n.model_key, n.hardware_key, n.backend_name, n.batch_size);

CREATE INDEX solo_profile_lookup IF NOT EXISTS
FOR (n:SoloProfile) ON (n.signature, n.hardware_key, n.family);

CREATE INDEX packet_profile_lookup IF NOT EXISTS
FOR (n:PacketProfile) ON (n.profile_scope, n.hardware_key, n.backend_name, n.compatible);

CREATE FULLTEXT INDEX graphrag_fact_summary IF NOT EXISTS
FOR (n:RunProfile|BatchProbeProfile|BatchSizeObservation|RuntimeProfile|SoloProfile|PacketProfile|Event)
ON EACH [n.summary_text, n.metadata_json];

CREATE FULLTEXT INDEX graphrag_entity_summary IF NOT EXISTS
FOR (n:Model|WorkloadSignature|Hardware|Accelerator|Toolkit)
ON EACH [n.summary_text, n.metadata_json];
