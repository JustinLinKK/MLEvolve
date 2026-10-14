"""Hand-written minimal fixture covering all node and relationship types
defined in schema/graph_schema.yaml. Pure Python — no I/O, no network.

All primary keys are prefixed with `mgd:` (mini-graph-demo) so MERGE
cannot collide with user-authored data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

KEY_PREFIX = "mgd:"


def _k(suffix: str) -> str:
    return KEY_PREFIX + suffix


@dataclass
class Hardware:
    hardware_key: str
    vendor: str
    product_name: str
    architecture: str
    compute_capability: str
    total_vram_mb: int
    vram_type: str
    technology_keys: list[str] = field(default_factory=list)

    def primary_key(self) -> str:
        return self.hardware_key


@dataclass
class Model:
    model_key: str
    model_name: str
    framework: str
    model_family: str
    architecture_type: str
    parameter_count: int

    def primary_key(self) -> str:
        return self.model_key


@dataclass
class Technology:
    technology_key: str
    name: str
    category: str

    def primary_key(self) -> str:
        return self.technology_key


@dataclass
class TrainingConfig:
    config_key: str
    input_signature: str
    batch_size: int
    precision: str
    optimizer: str
    learning_rate: float

    def primary_key(self) -> str:
        return self.config_key


@dataclass
class SingleJob:
    job_id: str
    profile_key: str
    purpose: str
    status: str
    hardware_set_key: str
    run_scope: str
    model_key: str
    config_key: str
    peak_vram_mb: int
    observed_avg_step_time_ms: float
    resolved_batch_size: int
    max_safe_batch_size: int | None = None
    confidence: float = 1.0

    def primary_key(self) -> str:
        return self.job_id


@dataclass
class PackedJob:
    job_id: str
    profile_key: str
    purpose: str
    status: str
    hardware_set_key: str
    run_scope: str
    packing_group_key: str
    packing_strategy: str
    compatible: bool
    slowdown_ratio: float
    throughput_efficiency: float
    peak_vram_mb: int

    def primary_key(self) -> str:
        return self.job_id


@dataclass
class PackedJobMember:
    member_id: str
    model_key: str
    config_key: str
    status: str
    resolved_batch_size: int
    observed_avg_step_time_ms: float
    slowdown_vs_single: float

    def primary_key(self) -> str:
        return self.member_id


@dataclass
class JobUsedHardwareEdge:
    job_id: str
    hardware_key: str
    resource_role: str
    allocation_scope: str
    memory_cap_mb: int


@dataclass
class JobUsesTechnologyEdge:
    job_id: str
    technology_key: str
    role: str


@dataclass
class MiniGraphFixture:
    hardware: list[Hardware]
    models: list[Model]
    technologies: list[Technology]
    training_configs: list[TrainingConfig]
    single_jobs: list[SingleJob]
    packed_jobs: list[PackedJob]
    packed_members: list[PackedJobMember]
    single_trains_model: list[tuple[str, str]]
    single_uses_config: list[tuple[str, str]]
    job_used_hardware: list[JobUsedHardwareEdge]
    job_uses_technology: list[JobUsesTechnologyEdge]
    has_packed_member: list[tuple[str, str]]
    member_trains_model: list[tuple[str, str]]
    member_uses_config: list[tuple[str, str]]
    member_uses_technology: list[tuple[str, str]]
    member_baseline_single_job: list[tuple[str, str]]


def build_fixture() -> MiniGraphFixture:
    hw_rtx = Hardware(
        hardware_key=_k("nvidia.rtx_5090.cu128"),
        vendor="nvidia",
        product_name="GeForce RTX 5090",
        architecture="blackwell",
        compute_capability="12.0",
        total_vram_mb=32768,
        vram_type="gddr7",
        technology_keys=[
            "tensor_cores_5gen", "fp4", "fp8", "fp8_e4m3",
            "fp8_e5m2", "sm_120", "pcie5_x16",
        ],
    )
    hw_h100 = Hardware(
        hardware_key=_k("nvidia.h100_sxm.cu128"),
        vendor="nvidia",
        product_name="H100 SXM",
        architecture="hopper",
        compute_capability="9.0",
        total_vram_mb=81920,
        vram_type="hbm3",
        technology_keys=[
            "tensor_cores_4gen", "fp8", "fp8_e4m3", "fp8_e5m2",
            "sm_90", "nvlink",
        ],
    )

    m_rn50 = Model(
        model_key=_k("resnet50.timm"),
        model_name="resnet50",
        framework="pytorch",
        model_family="resnet",
        architecture_type="cnn",
        parameter_count=25_557_032,
    )
    m_vit = Model(
        model_key=_k("vit_base_patch16_224.timm"),
        model_name="vit_base_patch16_224",
        framework="pytorch",
        model_family="vit",
        architecture_type="transformer",
        parameter_count=86_567_656,
    )
    m_llama = Model(
        model_key=_k("llama2-7b.hf"),
        model_name="llama-2-7b",
        framework="pytorch",
        model_family="llama",
        architecture_type="transformer",
        parameter_count=6_738_415_616,
    )

    t_amp = Technology(_k("pytorch_amp"), "torch.amp", "precision_optimization")
    t_bf16 = Technology(_k("bf16_autocast"), "bf16 autocast", "precision_optimization")
    t_tf32 = Technology(_k("tf32_matmul"), "TF32 matmul", "tensor_core_enablement")
    t_cg = Technology(_k("cuda_graphs"), "CUDA Graphs", "launch_overhead_reduction")

    cfg_rn50 = TrainingConfig(
        config_key=_k("cfg-rn50-bs256-bf16"),
        input_signature="image=224x224",
        batch_size=256,
        precision="bf16",
        optimizer="sgd",
        learning_rate=0.1,
    )
    cfg_vit = TrainingConfig(
        config_key=_k("cfg-vit-bs128-bf16"),
        input_signature="image=224x224",
        batch_size=128,
        precision="bf16",
        optimizer="adamw",
        learning_rate=1e-4,
    )
    cfg_llama = TrainingConfig(
        config_key=_k("cfg-llama-bs4-bf16"),
        input_signature="seq_len=2048",
        batch_size=4,
        precision="bf16",
        optimizer="adamw",
        learning_rate=2e-5,
    )

    job_001 = SingleJob(
        job_id=_k("job-001"),
        profile_key=_k("profile-rn50-rtx5090-baseline"),
        purpose="baseline_benchmark",
        status="succeeded",
        hardware_set_key=hw_rtx.hardware_key,
        run_scope="full_epoch",
        model_key=m_rn50.model_key,
        config_key=cfg_rn50.config_key,
        peak_vram_mb=22000,
        observed_avg_step_time_ms=145.0,
        resolved_batch_size=256,
        max_safe_batch_size=256,
        confidence=0.95,
    )
    job_002 = SingleJob(
        job_id=_k("job-002"),
        profile_key=_k("profile-vit-rtx5090-bsprobe"),
        purpose="batch_size_probe",
        status="succeeded",
        hardware_set_key=hw_rtx.hardware_key,
        run_scope="fixed_steps",
        model_key=m_vit.model_key,
        config_key=cfg_vit.config_key,
        peak_vram_mb=28000,
        observed_avg_step_time_ms=210.0,
        resolved_batch_size=128,
        max_safe_batch_size=192,
        confidence=0.85,
    )

    job_003 = PackedJob(
        job_id=_k("job-003"),
        profile_key=_k("profile-rn50-vit-rtx5090-mps"),
        purpose="packed_benchmark",
        status="succeeded",
        hardware_set_key=hw_rtx.hardware_key,
        run_scope="fixed_steps",
        packing_group_key=_k("pack-rn50+vit"),
        packing_strategy="mps",
        compatible=True,
        slowdown_ratio=1.18,
        throughput_efficiency=0.85,
        peak_vram_mb=30500,
    )

    member_a = PackedJobMember(
        member_id=_k("member-003-a"),
        model_key=m_rn50.model_key,
        config_key=cfg_rn50.config_key,
        status="succeeded",
        resolved_batch_size=256,
        observed_avg_step_time_ms=170.0,
        slowdown_vs_single=1.17,
    )
    member_b = PackedJobMember(
        member_id=_k("member-003-b"),
        model_key=m_vit.model_key,
        config_key=cfg_vit.config_key,
        status="succeeded",
        resolved_batch_size=128,
        observed_avg_step_time_ms=248.0,
        slowdown_vs_single=1.18,
    )

    return MiniGraphFixture(
        hardware=[hw_rtx, hw_h100],
        models=[m_rn50, m_vit, m_llama],
        technologies=[t_amp, t_bf16, t_tf32, t_cg],
        training_configs=[cfg_rn50, cfg_vit, cfg_llama],
        single_jobs=[job_001, job_002],
        packed_jobs=[job_003],
        packed_members=[member_a, member_b],
        single_trains_model=[
            (job_001.job_id, m_rn50.model_key),
            (job_002.job_id, m_vit.model_key),
        ],
        single_uses_config=[
            (job_001.job_id, cfg_rn50.config_key),
            (job_002.job_id, cfg_vit.config_key),
        ],
        job_used_hardware=[
            JobUsedHardwareEdge(job_001.job_id, hw_rtx.hardware_key,
                                "primary_accelerator", "whole_device", 31744),
            JobUsedHardwareEdge(job_002.job_id, hw_rtx.hardware_key,
                                "primary_accelerator", "whole_device", 31744),
            JobUsedHardwareEdge(job_003.job_id, hw_rtx.hardware_key,
                                "primary_accelerator", "shared_device", 31744),
        ],
        job_uses_technology=[
            JobUsesTechnologyEdge(job_001.job_id, t_amp.technology_key, "precision"),
            JobUsesTechnologyEdge(job_001.job_id, t_bf16.technology_key, "precision"),
            JobUsesTechnologyEdge(job_001.job_id, t_tf32.technology_key, "optimization"),
            JobUsesTechnologyEdge(job_001.job_id, t_cg.technology_key, "optimization"),
        ],
        has_packed_member=[
            (job_003.job_id, member_a.member_id),
            (job_003.job_id, member_b.member_id),
        ],
        member_trains_model=[
            (member_a.member_id, m_rn50.model_key),
            (member_b.member_id, m_vit.model_key),
        ],
        member_uses_config=[
            (member_a.member_id, cfg_rn50.config_key),
            (member_b.member_id, cfg_vit.config_key),
        ],
        member_uses_technology=[
            (member_a.member_id, t_amp.technology_key),
            (member_b.member_id, t_amp.technology_key),
        ],
        member_baseline_single_job=[
            (member_a.member_id, job_001.job_id),
            (member_b.member_id, job_002.job_id),
        ],
    )
