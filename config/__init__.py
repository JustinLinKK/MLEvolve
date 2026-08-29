"""configuration and setup utils"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Hashable, cast
import datetime
import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

# Lazy import to avoid circular dependency with engine.search_node
# Journal and filter_journal are imported where needed via _get_journal_classes()
def _get_journal_classes():
    from engine.search_node import Journal, filter_journal
    return Journal, filter_journal

from utils import copytree, preproc_data, serialize
from utils.precision_policy import normalize_precision_optimization_mode
from context_cache.config import (
    ContextCacheSettings,
    environment_overrides as context_cache_environment_overrides,
)

shutup.mute_warnings()
logger = logging.getLogger("MLEvolve")
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ROOT_CONFIG_EXAMPLE_PATH = REPO_ROOT / "config.example.yaml"
CONFIG_ENV_VAR = "MLEVOLVE_CONFIG"

EXPERIMENT_MODE_ORIGIN = "origin"
EXPERIMENT_MODE_BASELINE = "baseline"
EXPERIMENT_MODE_HARDWARE_AWARE = "hardware_aware"
_EXPERIMENT_MODES = {EXPERIMENT_MODE_ORIGIN, EXPERIMENT_MODE_BASELINE, EXPERIMENT_MODE_HARDWARE_AWARE}


def normalize_experiment_mode(value: str | None) -> str:
    mode = str(value or EXPERIMENT_MODE_HARDWARE_AWARE).strip().lower().replace("-", "_")
    if mode not in _EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment.mode: {value}. Expected one of {sorted(_EXPERIMENT_MODES)}")
    return mode


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str
    provider: str = ""


@dataclass
class VLLMClientConfig:
    """Client-only controls for an OpenAI-compatible vLLM deployment."""

    cache_salt_env: str = "MLEVOLVE_VLLM_CACHE_SALT"
    require_cache_salt: bool = True
    session_affinity: bool = True

@dataclass
class DecayConfig:
    exploration_constant: float
    lower_bound: float
    alpha: float
    phase_ratios: list


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    topk_max_improves: int
    max_improve_failure: int
    parallel_search_num: int
    branch_stagnation_threshold: int
    topk_stagnation_threshold: int
    top_candidates_size: int
    stagnation_window: int
    num_gpus: int
    explore_switch_start: float
    explore_switch_end: float
    min_exploration_weight: float
    topk_early_k: int
    topk_early_max_per_branch: int
    topk_late_k: int
    topk_late_max_per_branch: int
    force_backprop_late_threshold: float
    force_backprop_late_prob: float
    force_backprop_mid_threshold: float
    force_backprop_mid_modulo: int
    recent_best_window: int
    fusion_min_time_hours: float
    fusion_max_time_hours: float
    fusion_min_successful_nodes: int
    fusion_min_branches: int


@dataclass
class ReviewConfig:
    enabled: bool = True
    max_repair_rounds: int = 2
    classifier_retries: int = 3
    repair_retries: int = 2
    reject_unresolved_critical: bool = True
    fail_open_on_unavailable: bool = True
    parallel_training_repairs: bool = True


@dataclass
class CudaDocsConfig:
    """Role-gated local-first NVIDIA CUDA documentation enrichment."""

    enabled: bool = False
    rollout_mode: str = "off"
    endpoint: str = "https://api.copilot.nsight.ngc.nvidia.com/mcp/cuda-docs"
    auth_token_env: str = "NVIDIA_CUDA_MCP_TOKEN"
    remote_roles: list[str] = field(default_factory=lambda: ["debug"])
    blocking_roles: list[str] = field(default_factory=lambda: ["debug"])
    local_roles: list[str] = field(
        default_factory=lambda: [
            "draft", "improve", "debug", "code_review",
            "evolution", "fusion", "aggregation",
        ]
    )
    soft_timeout_seconds: float = 6.0
    hard_timeout_seconds: float = 8.0
    total_enrichment_deadline_seconds: float = 10.0
    max_remote_calls_per_action: int = 1
    prompt_max_chars: int = 2000
    prompt_max_chunks: int = 3
    ram_cache_max_entries: int = 512
    ram_cache_ttl_seconds: int = 21600
    positive_ttl_seconds: int = 604800
    stale_ttl_seconds: int = 2592000
    negative_ttl_seconds: int = 600
    transient_failure_ttl_seconds: int = 60
    auth_failure_ttl_seconds: int = 600
    ttl_jitter_fraction: float = 0.1
    async_prewarm: bool = True
    prewarm_concurrency: int = 2
    persist_raw_chunks: bool = True
    synthesize_recipes_async: bool = True
    send_source_code: bool = False
    remote_rate_per_minute: float = 12.0
    remote_burst: int = 2
    circuit_failure_threshold: int = 3
    circuit_window_seconds: int = 60
    circuit_cooldown_seconds: int = 60
    singleflight_wait_seconds: float = 0.25
    redis_namespace_capacity: int = 512
    raw_response_max_chars: int = 32000
    normalized_chunk_max_chars: int = 4000


@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    initial_drafts: int
    seed: int
    data_preview: bool
    code: StageConfig
    feedback: StageConfig
    check_data_leakage: bool
    fusion_vs_evolution_prob: float
    branch_fusion_trigger_prob: float
    max_fusion_drafts: int
    use_global_memory: bool
    memory_similarity_threshold: float
    memory_embedding_device: str
    memory_embedding_model_path: str
    search: SearchConfig
    decay: DecayConfig
    use_diff_mode: bool = True
    hardware_context_enabled: bool = True
    pipeline_decision_enabled: bool = True
    hardware_context_limit: int = 8
    hardware_context_max_prompt_chars: int = 3500
    precision_optimization_mode: str = "normal"
    review: ReviewConfig = field(default_factory=ReviewConfig)
    cuda_docs: CudaDocsConfig = field(default_factory=CudaDocsConfig)


@dataclass
class ExecConfig:
    timeout: int | None
    agent_file_name: str


@dataclass
class SchedulerBridgeConfig:
    enabled: bool = False
    settings: dict | None = None
    runtime_root: str | None = None
    start_service: bool = True
    wait_poll_interval_seconds: float = 1.0
    wait_timeout_seconds: int | None = None
    preload_source_model_id: str | None = None
    preload_source_model_path: str | None = None
    preload_source_loader_target: str | None = None


@dataclass
class ExperimentConfig:
    mode: str = EXPERIMENT_MODE_HARDWARE_AWARE


@dataclass
class ColdstartConfig:
    use_coldstart: bool
    task_json_path: str
    model_json_path: str
    description: str


@dataclass
class MonitorConfig:
    enabled: bool = True
    interval_seconds: float = 5
    gpu_idle_util_threshold: float = 10
    gpu_idle_memory_threshold_mb: float = 1024
    cpu_idle_util_threshold: float = 20
    adaptive_compression: bool = True
    max_csv_rows: int = 1000
    compress_to_rows: int = 500


@dataclass
class InitSolutionConfig:
    use: bool = False


@dataclass
class Config(Hashable):
    data_dir: Path
    dataset_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str
    exp_id: str

    torch_hub_dir: str
    pretrain_model_dir: str

    exec: ExecConfig
    scheduler: SchedulerBridgeConfig
    experiment: ExperimentConfig
    agent: AgentConfig
    start_cpu_id: str
    cpu_number: str

    coldstart: ColdstartConfig

    context_cache: ContextCacheSettings = field(default_factory=ContextCacheSettings)
    vllm_client: VLLMClientConfig = field(default_factory=VLLMClientConfig)
    # Retain the independent hardware-knowledge mapping in the unified config.
    # HardwareKnowledgeClient validates its nested settings at its own boundary.
    hardware_knowledge: dict = field(default_factory=dict)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    use_grading_server: bool = True
    init_solution: InitSolutionConfig = field(default_factory=InitSolutionConfig)
    resume_journal: Path | None = None


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve the unified MLEvolve config path.

    Precedence: explicit path, MLEVOLVE_CONFIG, root config.yaml, then the
    root config.example.yaml.
    """
    if path is not None:
        return Path(path).expanduser().resolve()

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    if ROOT_CONFIG_PATH.exists():
        return ROOT_CONFIG_PATH.resolve()

    if ROOT_CONFIG_EXAMPLE_PATH.exists():
        logger.warning(
            "Local config.yaml not found; using sanitized config.example.yaml. "
            "Copy it to config.yaml for local API keys and machine-specific settings."
        )
        return ROOT_CONFIG_EXAMPLE_PATH.resolve()

    raise FileNotFoundError("No MLEvolve config found. Expected config.yaml or config.example.yaml at repo root.")


def _load_cfg(path: str | Path | None = None, use_cli_args=True) -> Config:
    cfg = OmegaConf.load(resolve_config_path(path))
    env_mode = os.getenv("MLEVOLVE_EXPERIMENT_MODE")
    if env_mode:
        cfg = OmegaConf.merge(cfg, {"experiment": {"mode": env_mode}})
    cache_overrides = context_cache_environment_overrides()
    if cache_overrides:
        cfg = OmegaConf.merge(cfg, {"context_cache": cache_overrides})
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg

def load_cfg(path: str | Path | None = None) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if str(cfg.data_dir).startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    resume_journal_value = getattr(cfg, "resume_journal", None)
    resume_journal = (
        Path(resume_journal_value).expanduser().resolve()
        if resume_journal_value is not None
        else None
    )
    if resume_journal is not None:
        if resume_journal.name != "journal.json" or not resume_journal.is_file():
            raise ValueError("resume_journal must be an existing logs/journal.json file.")
        if resume_journal.parent.name != "logs":
            raise ValueError("resume_journal must be located directly in a run logs directory.")
        resumed_run_root = resume_journal.parent.parent
        resumed_workspace = resumed_run_root / "workspace"
        if not resumed_workspace.is_dir():
            raise ValueError("The resumed run workspace directory is missing.")

    top_log_dir = Path(cfg.log_dir).resolve()
    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    # generate experiment name and prefix with consecutive index
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.exp_name = f"{timestamp}_{cfg.exp_name or coolname.generate_slug(3)}"

    if resume_journal is not None:
        cfg.resume_journal = resume_journal
        cfg.exp_name = resumed_run_root.name
        cfg.log_dir = resume_journal.parent
        cfg.workspace_dir = resumed_workspace.resolve()
    # If log_dir and workspace_dir point to the same path, treat it as a unified
    # "runs" root and place logs/workspace under the per-run directory
    elif top_log_dir == top_workspace_dir:
        runs_root = top_log_dir
        runs_root.mkdir(parents=True, exist_ok=True)
        per_run_root = (runs_root / cfg.exp_name).resolve()
        cfg.log_dir = (per_run_root / "logs").resolve()
        cfg.workspace_dir = (per_run_root / "workspace").resolve()
    else:
        top_log_dir.mkdir(parents=True, exist_ok=True)
        top_workspace_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
        cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    validated_context_cache = ContextCacheSettings.from_mapping(cfg.context_cache)
    for field_name in ContextCacheSettings.__dataclass_fields__:
        setattr(
            cfg.context_cache,
            field_name,
            getattr(validated_context_cache, field_name),
        )
    vllm_stages = [
        stage
        for stage in (cfg.agent.code, cfg.agent.feedback)
        if str(getattr(stage, "provider", "") or "").strip().lower() == "vllm"
    ]
    if vllm_stages:
        for stage in vllm_stages:
            if not str(getattr(stage, "base_url", "") or "").strip():
                raise ValueError("A vLLM stage requires agent.<stage>.base_url")
        salt_env = str(cfg.vllm_client.cache_salt_env or "").strip()
        if cfg.vllm_client.require_cache_salt:
            if not salt_env:
                raise ValueError("vllm_client.cache_salt_env must not be empty")
            salt = os.getenv(salt_env, "")
            if len(salt.encode("utf-8")) < 32:
                raise ValueError(
                    f"{salt_env} must contain at least 32 bytes of private cache salt"
                )
    cfg.experiment.mode = normalize_experiment_mode(cfg.experiment.mode)
    cfg.agent.precision_optimization_mode = normalize_precision_optimization_mode(
        cfg.agent.precision_optimization_mode
    )
    if cfg.experiment.mode in {EXPERIMENT_MODE_ORIGIN, EXPERIMENT_MODE_BASELINE}:
        cfg.agent.hardware_context_enabled = False
    if cfg.experiment.mode == EXPERIMENT_MODE_ORIGIN:
        cfg.scheduler.enabled = False

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    Journal, filter_journal = _get_journal_classes()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")

    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
