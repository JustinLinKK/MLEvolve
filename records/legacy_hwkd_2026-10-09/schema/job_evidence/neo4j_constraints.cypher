CREATE CONSTRAINT job_id_unique IF NOT EXISTS
FOR (n:Job) REQUIRE n.job_id IS UNIQUE;

CREATE CONSTRAINT hardware_key_unique IF NOT EXISTS
FOR (n:Hardware) REQUIRE n.hardware_key IS UNIQUE;

CREATE CONSTRAINT model_key_unique IF NOT EXISTS
FOR (n:Model) REQUIRE n.model_key IS UNIQUE;

CREATE CONSTRAINT config_key_unique IF NOT EXISTS
FOR (n:TrainingConfig) REQUIRE n.config_key IS UNIQUE;

CREATE CONSTRAINT technology_key_unique IF NOT EXISTS
FOR (n:Technology) REQUIRE n.technology_key IS UNIQUE;

CREATE CONSTRAINT packed_member_id_unique IF NOT EXISTS
FOR (n:PackedJobMember) REQUIRE n.member_id IS UNIQUE;

CREATE INDEX job_profile_lookup IF NOT EXISTS
FOR (n:Job) ON (n.profile_key, n.purpose, n.status, n.hardware_set_key);

CREATE INDEX single_job_lookup IF NOT EXISTS
FOR (n:SingleJob) ON (n.model_key, n.config_key, n.resolved_batch_size, n.completed_full_training);

CREATE INDEX packed_job_lookup IF NOT EXISTS
FOR (n:PackedJob) ON (n.packing_group_key, n.packing_strategy, n.compatible);

CREATE INDEX packed_member_lookup IF NOT EXISTS
FOR (n:PackedJobMember) ON (n.model_key, n.config_key, n.status, n.resolved_batch_size);

CREATE INDEX hardware_lookup IF NOT EXISTS
FOR (n:Hardware) ON (n.vendor, n.product_name, n.architecture, n.compute_capability);

CREATE INDEX model_lookup IF NOT EXISTS
FOR (n:Model) ON (n.model_name, n.model_family, n.architecture_type, n.task_type);

CREATE INDEX training_config_lookup IF NOT EXISTS
FOR (n:TrainingConfig) ON (n.input_signature, n.batch_size, n.precision, n.optimizer);

CREATE INDEX technology_lookup IF NOT EXISTS
FOR (n:Technology) ON (n.category, n.name);
