from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import logging
import sys

import yaml

import config as mle_config
from localml_scheduler.config import SchedulerConfig
from run import _scheduler_settings_from_cfg


def _write_config(path: Path, *, marker: str, scheduler_runtime: str = "./runtime") -> None:
    payload = {
        "data_dir": "/tmp/data",
        "dataset_dir": "",
        "desc_file": None,
        "goal": "test goal",
        "eval": None,
        "exp_name": "test",
        "exp_id": "",
        "log_dir": "./runs",
        "log_level": "INFO",
        "workspace_dir": "./runs",
        "preprocess_data": False,
        "copy_data": False,
        "experiment": {"mode": "hardware_aware"},
        "start_cpu_id": "0",
        "cpu_number": "2",
        "torch_hub_dir": "",
        "pretrain_model_dir": "",
        "exec": {"timeout": 10, "agent_file_name": "runfile.py"},
        "scheduler": {
            "enabled": True,
            "runtime_root": None,
            "start_service": False,
            "wait_poll_interval_seconds": 1,
            "wait_timeout_seconds": None,
            "preload_source_model_id": None,
            "preload_source_model_path": None,
            "preload_source_loader_target": None,
            "settings": {
                "runtime_root": scheduler_runtime,
                "cache_socket_name": f"{marker}.sock",
                "graph_db": {"enabled": False, "mode": "off"},
            },
        },
        "agent": {
            "steps": 1,
            "time_limit": 60,
            "initial_drafts": 0,
            "seed": 1,
            "data_preview": False,
            "code": {"model": "openai/test", "temp": 1, "provider": "openrouter", "base_url": "", "api_key": ""},
            "feedback": {"model": "openai/test", "temp": 1, "provider": "openrouter", "base_url": "", "api_key": ""},
            "check_data_leakage": False,
            "fusion_vs_evolution_prob": 0.0,
            "branch_fusion_trigger_prob": 0.0,
            "max_fusion_drafts": 0,
            "use_global_memory": False,
            "memory_similarity_threshold": 0.7,
            "memory_embedding_device": "cpu",
            "memory_embedding_model_path": "BAAI/bge-base-en-v1.5",
            "search": {
                "max_debug_depth": 1,
                "debug_prob": 0.0,
                "num_drafts": 1,
                "metric_improvement_threshold": 0.0,
                "back_debug_depth": 1,
                "num_bugs": 1,
                "num_improves": 1,
                "topk_max_improves": 1,
                "max_improve_failure": 1,
                "parallel_search_num": 1,
                "branch_stagnation_threshold": 1,
                "topk_stagnation_threshold": 1,
                "top_candidates_size": 1,
                "stagnation_window": 1,
                "num_gpus": 1,
                "explore_switch_start": 0.5,
                "explore_switch_end": 0.7,
                "min_exploration_weight": 0.2,
                "topk_early_k": 1,
                "topk_early_max_per_branch": 1,
                "topk_late_k": 1,
                "topk_late_max_per_branch": 1,
                "force_backprop_late_threshold": 0.8,
                "force_backprop_late_prob": 0.0,
                "force_backprop_mid_threshold": 0.4,
                "force_backprop_mid_modulo": 2,
                "recent_best_window": 1,
                "fusion_min_time_hours": 1,
                "fusion_max_time_hours": 2,
                "fusion_min_successful_nodes": 1,
                "fusion_min_branches": 1,
            },
            "decay": {"exploration_constant": 1.0, "lower_bound": 0.5, "alpha": 0.01, "phase_ratios": [0.3, 0.7]},
        },
        "coldstart": {"use_coldstart": False, "task_json_path": "", "model_json_path": "", "description": ""},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_resolution_precedence(monkeypatch, tmp_path: Path, caplog) -> None:
    root = tmp_path / "config.yaml"
    legacy = tmp_path / "legacy" / "config.yaml"
    example = tmp_path / "config.example.yaml"
    explicit = tmp_path / "explicit.yaml"
    legacy.parent.mkdir()
    _write_config(root, marker="root")
    _write_config(legacy, marker="legacy")
    _write_config(example, marker="example")
    _write_config(explicit, marker="explicit")
    monkeypatch.setattr(mle_config, "ROOT_CONFIG_PATH", root)
    monkeypatch.setattr(mle_config, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.setattr(mle_config, "ROOT_CONFIG_EXAMPLE_PATH", example)
    monkeypatch.delenv("MLEVOLVE_CONFIG", raising=False)

    assert mle_config.resolve_config_path() == root.resolve()
    assert mle_config.resolve_config_path(explicit) == explicit.resolve()

    root.unlink()
    monkeypatch.setenv("MLEVOLVE_CONFIG", str(explicit))
    assert mle_config.resolve_config_path() == explicit.resolve()

    monkeypatch.delenv("MLEVOLVE_CONFIG")
    with caplog.at_level(logging.WARNING):
        assert mle_config.resolve_config_path() == legacy.resolve()
    assert "legacy config path" in caplog.text

    legacy.unlink()
    assert mle_config.resolve_config_path() == example.resolve()


def test_env_and_cli_overrides_still_win(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _write_config(path, marker="root")
    monkeypatch.setenv("MLEVOLVE_EXPERIMENT_MODE", "baseline")
    monkeypatch.setattr(sys, "argv", ["prog", "agent.steps=7"])

    cfg = mle_config.prep_cfg(mle_config._load_cfg(path, use_cli_args=True))

    assert cfg.experiment.mode == "baseline"
    assert cfg.agent.hardware_context_enabled is False
    assert cfg.agent.steps == 7


def test_scheduler_settings_prefer_nested_config(tmp_path: Path) -> None:
    cfg = SimpleNamespace(workspace_dir=tmp_path / "workspace")
    scheduler_cfg = SimpleNamespace(
        runtime_root=str(tmp_path / "scheduler-runtime"),
        settings={
            "runtime_root": "./ignored-by-bridge",
            "cache_socket_name": "nested.sock",
            "graph_db": {"enabled": False, "mode": "off"},
        },
        settings_path=None,
    )

    settings = _scheduler_settings_from_cfg(cfg, scheduler_cfg)

    assert settings.cache_socket_name == "nested.sock"
    assert settings.runtime_root == (tmp_path / "scheduler-runtime").resolve()


def test_scheduler_config_from_dict_matches_file(tmp_path: Path) -> None:
    payload = {"runtime_root": str(tmp_path / "runtime"), "cache_socket_name": "from-dict.sock"}
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert SchedulerConfig.from_dict(payload).cache_socket_name == SchedulerConfig.from_file(path).cache_socket_name
