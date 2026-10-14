"""CLI for the standalone hardware knowledge graph package."""

from __future__ import annotations

from pathlib import Path
import json
import os

import typer
import yaml

from .client import HardwareKnowledgeClient
from .config import HardwareKnowledgeSettings


app = typer.Typer(help="Hardware knowledge graph commands")

_REPO_ROOT = Path(__file__).resolve().parents[1]


@app.callback()
def main() -> None:
    """Manage the standalone hardware knowledge graph."""


def _resolve_unified_config_path(config_path: str | None = None) -> Path | None:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("MLEVOLVE_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    root_config = _REPO_ROOT / "config.yaml"
    if root_config.exists():
        return root_config
    root_example = _REPO_ROOT / "config.example.yaml"
    if root_example.exists():
        return root_example
    return None


def _hardware_settings_from_unified_config(config_path: str | None = None) -> HardwareKnowledgeSettings | None:
    path = _resolve_unified_config_path(config_path)
    if path is None:
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    hardware = payload.get("hardware_knowledge") if isinstance(payload, dict) else None
    if not isinstance(hardware, dict):
        return None
    settings_payload = hardware.get("settings") if isinstance(hardware.get("settings"), dict) else hardware
    return HardwareKnowledgeSettings.from_dict(settings_payload)


def _build_hardware_client(config_path: str | None = None) -> HardwareKnowledgeClient:
    settings = _hardware_settings_from_unified_config(config_path) or HardwareKnowledgeSettings()
    return HardwareKnowledgeClient(settings)


@app.command("ingest")
def ingest(
    config_path: str | None = typer.Option(None, "--config", help="Path to root MLEvolve config.yaml"),
    schema_root: Path = typer.Option(Path("schema"), "--schema-root", help="Schema root containing hardware_knowledge_graph.json"),
    recreate: bool = typer.Option(False, "--recreate/--no-recreate", help="Recreate hardware graph nodes before ingesting"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Validate and summarize records without writing to Neo4j"),
) -> None:
    client = _build_hardware_client(config_path)
    result = client.ingest_hardware_knowledge_graph(schema_root=schema_root, recreate=recreate, dry_run=dry_run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
