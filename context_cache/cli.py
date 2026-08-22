"""Operator CLI for local context-cache artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .compiler import KnowledgePackCompiler, load_default_manifest
from .coordinator import ROLE_NAMES
from .store import KnowledgePackStore
from .telemetry import CacheTelemetryStore

app = typer.Typer(
    help="Compile, inspect, verify, and maintain MLEvolve knowledge packs."
)


def _store(
    cache_dir: Path, max_pack_bytes: int = 16 * 1024 * 1024
) -> KnowledgePackStore:
    return KnowledgePackStore(cache_dir, max_pack_bytes=max_pack_bytes)


@app.command("compile")
def compile_packs(
    version: str = typer.Option(
        ..., "--version", help="Immutable knowledge version to publish."
    ),
    role: Optional[str] = typer.Option(
        None, "--role", help="Compile one role; defaults to common and all roles."
    ),
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
) -> None:
    roles = [role] if role else ["common", *ROLE_NAMES]
    store = _store(cache_dir)
    compiler = KnowledgePackCompiler(store)
    rows = []
    for role_name in roles:
        result = compiler.compile(load_default_manifest(role_name, version))
        rows.append(
            {
                "role": role_name,
                "version": version,
                "sha256": result.ref.content_sha256,
                "path": result.ref.path,
                "cache_hit": result.cache_hit,
            }
        )
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@app.command("inspect")
def inspect_pack(
    role: str = typer.Option(..., "--role"),
    version: str = typer.Option(..., "--version"),
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
) -> None:
    store = _store(cache_dir)
    ref = store.resolve(role, version, active_only=False)
    if ref is None:
        raise typer.BadParameter(f"no pack for role={role!r}, version={version!r}")
    typer.echo(
        json.dumps(
            store.load_object(ref.content_sha256),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@app.command("verify")
def verify_packs(
    version: Optional[str] = typer.Option(None, "--version"),
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
) -> None:
    results = _store(cache_dir).verify(knowledge_version=version)
    typer.echo(json.dumps(results, indent=2, sort_keys=True))
    if any(not row["valid"] for row in results):
        raise typer.Exit(code=1)


@app.command("list")
def list_packs(
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
    all_versions: bool = typer.Option(False, "--all", help="Include retired aliases."),
) -> None:
    typer.echo(
        json.dumps(
            _store(cache_dir).list(include_inactive=all_versions),
            indent=2,
            sort_keys=True,
        )
    )


@app.command("retire")
def retire_pack(
    role: str = typer.Option(..., "--role"),
    version: str = typer.Option(..., "--version"),
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
) -> None:
    changed = _store(cache_dir).retire(role, version)
    typer.echo(
        json.dumps(
            {"retired": changed, "role": role, "version": version}, sort_keys=True
        )
    )


@app.command("cleanup")
def cleanup_packs(
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
    execute: bool = typer.Option(
        False, "--execute", help="Delete unreferenced objects; default is dry-run."
    ),
) -> None:
    removed = _store(cache_dir).cleanup(dry_run=not execute)
    typer.echo(
        json.dumps(
            {"dry_run": not execute, "objects": removed}, indent=2, sort_keys=True
        )
    )


@app.command("export")
def export_events(
    output: Path = typer.Option(..., "--output"),
    format: str = typer.Option("jsonl", "--format"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    cache_dir: Path = typer.Option(Path("var/context-cache"), "--cache-dir"),
) -> None:
    telemetry = CacheTelemetryStore(_store(cache_dir).registry_path)
    normalized = format.lower()
    if normalized == "jsonl":
        destination = telemetry.export_jsonl(output, run_id=run_id)
    elif normalized == "csv":
        destination = telemetry.export_csv(output, run_id=run_id)
    else:
        raise typer.BadParameter("format must be jsonl or csv")
    typer.echo(str(destination))


def main() -> None:
    app()
