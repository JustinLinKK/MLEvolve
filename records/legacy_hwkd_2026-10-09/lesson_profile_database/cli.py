"""Operational CLI for the lesson profile database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

from config import resolve_config_path
from engine.search_node import Journal
from utils.pipeline_logging import PipelineActionLogger
from utils.serialize import load_json

from .client import LessonProfileClient
from .config import LessonProfileSettings


def _settings(config_path: str | None, runtime_root: str | None, sqlite_path: str | None) -> tuple[LessonProfileSettings, Any | None]:
    cfg = None
    payload: Any = None
    if config_path:
        cfg = OmegaConf.load(resolve_config_path(config_path))
        payload = getattr(cfg, "lesson_profiles", None)
    settings = LessonProfileSettings.from_mapping(payload)
    if runtime_root:
        settings.runtime_root = runtime_root
    if sqlite_path:
        settings.sqlite_path = sqlite_path
    return settings, cfg


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLEvolve lesson profile database")
    parser.add_argument("--config", help="MLEvolve YAML config")
    parser.add_argument("--runtime-root")
    parser.add_argument("--sqlite-path")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Initialize SQLite and Qdrant indexes")
    init.add_argument("--sqlite-only", action="store_true")
    init.add_argument("--recreate-qdrant", action="store_true")

    worker = commands.add_parser("worker", help="Run durable builder in the foreground")
    worker.add_argument("--once", action="store_true")

    commands.add_parser("status", help="Show record, job, and outbox counts")

    query = commands.add_parser("query", help="Get one exact profile by key")
    query.add_argument("profile_key")
    query.add_argument("--role", default="draft")
    query.add_argument("--query", default="")

    search = commands.add_parser("search", help="Semantic similar-only lesson search")
    search.add_argument("query")
    search.add_argument("--role", default="improve")
    search.add_argument("--limit", type=int, default=3)

    replay = commands.add_parser("replay", help="Replay final validation events from completed run directories")
    replay.add_argument("paths", nargs="+", type=Path)

    commands.add_parser("retry", help="Retry all failed builder jobs")

    revisions = commands.add_parser("revisions", help="Inspect immutable profile revisions")
    revisions.add_argument("profile_key")

    conflicts = commands.add_parser("conflicts", help="Inspect preserved contradictions")
    conflicts.add_argument("--profile-key")

    rollback = commands.add_parser("rollback", help="Publish a new revision copied from an older one")
    rollback.add_argument("profile_key")
    rollback.add_argument("revision", type=int)

    benchmark = commands.add_parser("benchmark", help="Report cold and warm retrieval p50/p95")
    benchmark.add_argument("profile_key")
    benchmark.add_argument("--role", default="improve")
    benchmark.add_argument("--iterations", type=int, default=20)

    return parser


def _validation_outcomes(path: Path, run_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT node_id, payload_json FROM pipeline_events
            WHERE run_id=? AND event_type='validation_completed'
            ORDER BY event_id
            """,
            (run_id,),
        ).fetchall()
    for node_id, raw in rows:
        try:
            result[str(node_id)] = str(json.loads(raw).get("outcome") or "")
        except Exception:
            continue
    return result


def _replay_run(client: LessonProfileClient, run_path: Path) -> dict[str, Any]:
    run_path = run_path.resolve()
    log_dir = run_path / "logs" if (run_path / "logs").is_dir() else run_path
    journal_path = log_dir / "journal.json"
    pipeline_path = log_dir / "pipeline.sqlite3"
    config_path = log_dir / "config.yaml"
    if not journal_path.exists() or not pipeline_path.exists():
        return {"path": str(run_path), "ok": False, "reason": "journal.json or pipeline.sqlite3 missing"}
    cfg_data = OmegaConf.load(config_path) if config_path.exists() else OmegaConf.create({})
    run_id = str(getattr(cfg_data, "exp_name", None) or run_path.name)
    outcomes = _validation_outcomes(pipeline_path, run_id)
    journal = load_json(journal_path, Journal)
    historical_scheduler = bool(getattr(getattr(cfg_data, "scheduler", None), "enabled", False))
    task_description = str(getattr(cfg_data, "goal", "") or "")
    desc_file = getattr(cfg_data, "desc_file", None)
    if not task_description and desc_file:
        try:
            task_description = Path(str(desc_file)).expanduser().read_text(encoding="utf-8")
        except Exception:
            task_description = ""
    replay_cfg = SimpleNamespace(
        exp_name=run_id,
        scheduler=SimpleNamespace(enabled=historical_scheduler),
    )
    logger = PipelineActionLogger(
        pipeline_path,
        run_id=run_id,
        mode=str(getattr(getattr(cfg_data, "experiment", None), "mode", "replay")),
    )
    agent = SimpleNamespace(
        cfg=replay_cfg,
        task_desc=task_description,
        scheduler_client=None,
        pipeline_logger=logger,
    )
    queued = 0
    skipped = 0
    try:
        for node in journal.nodes:
            outcome = outcomes.get(str(node.id))
            if not outcome:
                continue
            result = client.enqueue_validated_node(agent, node, outcome=outcome)
            if result.get("ok"):
                queued += int(bool(result.get("inserted")))
            else:
                skipped += 1
    finally:
        logger.close()
    return {"path": str(run_path), "ok": True, "queued": queued, "skipped": skipped, "validated_events": len(outcomes)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings, cfg = _settings(args.config, args.runtime_root, args.sqlite_path)
    client = LessonProfileClient(settings, cfg=cfg)
    client.registry.initialize()

    if args.command == "init":
        result = client.initialize(initialize_qdrant=not args.sqlite_only)
        if args.recreate_qdrant and not args.sqlite_only:
            result["qdrant"] = client.vector_store.ensure_collection(recreate=True)
        _print(result)
    elif args.command == "worker":
        try:
            client.worker.run_foreground(once=args.once)
        except KeyboardInterrupt:
            pass
    elif args.command == "status":
        _print(client.status())
    elif args.command == "query":
        profile = client.registry.profile(args.profile_key)
        if profile is None:
            _print({"family_hardware_profile": {"match_level": "none"}})
            return 1
        _print(client.get_family_hardware_profile(
            agent_role=args.role,
            identity=profile["identity"],
            code=args.query,
        ))
    elif args.command == "search":
        _print(client.search_lesson_profiles(query=args.query, agent_role=args.role, limit=args.limit))
    elif args.command == "replay":
        _print([_replay_run(client, path) for path in args.paths])
    elif args.command == "retry":
        _print({"retried": client.registry.retry_failed_jobs()})
    elif args.command == "revisions":
        _print(client.registry.list_revisions(args.profile_key))
    elif args.command == "conflicts":
        _print(client.registry.list_conflicts(args.profile_key))
    elif args.command == "rollback":
        publication = client.registry.rollback(args.profile_key, args.revision)
        client.vector_store.upsert_publication(publication["payload"])
        activated = client.registry.activate_publication(publication["outbox_id"])
        client.registry.complete_publication_for_observation(publication["source_observation_id"])
        client.invalidate_profile(args.profile_key)
        _print(activated)
    elif args.command == "benchmark":
        from .benchmark import benchmark_retrieval

        profile = client.registry.profile(args.profile_key)
        if profile is None:
            _print({"ok": False, "reason": "profile not found"})
            return 1
        _print(benchmark_retrieval(
            client,
            identity=profile["identity"],
            agent_role=args.role,
            iterations=args.iterations,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
