import atexit
import logging
import os
import signal
import sys
import shutil
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description, collect_startpoint_model_specs
from utils.logging_config import setup_logging
from utils.hardware_monitor import HardwareMonitor
from utils.experiment_metrics import build_comparison_metrics, write_comparison_metrics
from utils.pipeline_logging import PipelineActionLogger, set_process_pipeline_logger
import torch
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.hardware_client import HardwareKnowledgeClient
from localml_scheduler.hardware_knowledge import HardwareKnowledgeSettings


class SignalShutdown(BaseException):
    """Controlled shutdown requested by an external process signal."""

    def __init__(self, signum: int):
        self.signum = signum
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = f"signal {signum}"
        super().__init__(self.signal_name)


def _scheduler_settings_from_cfg(cfg, scheduler_cfg) -> SchedulerSettings:
    scheduler_runtime_root = getattr(scheduler_cfg, "runtime_root", None) or str(cfg.workspace_dir / "scheduler_runtime")
    nested_settings = getattr(scheduler_cfg, "settings", None)
    if nested_settings:
        payload = OmegaConf.to_container(nested_settings, resolve=True) if not isinstance(nested_settings, dict) else nested_settings
        return SchedulerSettings.from_dict(payload, runtime_root=scheduler_runtime_root)

    scheduler_settings_path = getattr(scheduler_cfg, "settings_path", None)
    if scheduler_settings_path:
        logger = logging.getLogger("MLEvolve")
        logger.warning(
            "scheduler.settings_path is deprecated; move scheduler settings under scheduler.settings in root config.yaml."
        )
        return SchedulerSettings.from_file(
            scheduler_settings_path,
            runtime_root=scheduler_runtime_root,
        )

    return SchedulerSettings(runtime_root=scheduler_runtime_root)


def _hardware_knowledge_settings_from_cfg(cfg) -> HardwareKnowledgeSettings:
    hardware_cfg = getattr(cfg, "hardware_knowledge", None)
    nested_settings = getattr(hardware_cfg, "settings", None) if hardware_cfg is not None else None
    scheduler_cfg = getattr(cfg, "scheduler", None)
    scheduler_runtime_root = getattr(scheduler_cfg, "runtime_root", None) or str(cfg.workspace_dir / "scheduler_runtime")
    branch_profile_db_path = str(Path(scheduler_runtime_root).expanduser().resolve() / "db" / "branch_profile.sqlite3")
    if nested_settings:
        payload = OmegaConf.to_container(nested_settings, resolve=True) if not isinstance(nested_settings, dict) else dict(nested_settings)
        if not payload.get("runtime_root"):
            payload["runtime_root"] = str(cfg.workspace_dir / "hardware_knowledge_runtime")
        if not payload.get("branch_profile_db_path"):
            payload["branch_profile_db_path"] = branch_profile_db_path
        return HardwareKnowledgeSettings.from_dict(payload)

    payload = {
        "runtime_root": str(cfg.workspace_dir / "hardware_knowledge_runtime"),
        "branch_profile_db_path": branch_profile_db_path,
    }
    if hardware_cfg is not None:
        for key in ("graph", "redis_cache", "device_index", "sqlite_busy_timeout_ms"):
            value = getattr(hardware_cfg, key, None)
            if value is not None:
                payload[key] = OmegaConf.to_container(value, resolve=True) if not isinstance(value, (str, int, float, bool, dict)) else value
    return HardwareKnowledgeSettings.from_dict(payload)


def _submit_startpoint_probe_jobs(
    *,
    scheduler_client,
    scheduler_settings: SchedulerSettings,
    cfg,
    startpoint_specs: list[dict],
    pipeline_logger: PipelineActionLogger,
    logger: logging.Logger,
) -> list[str]:
    gpu_settings = scheduler_settings.gpu_scheduler
    if not bool(getattr(gpu_settings, "startpoint_probe_enabled", True)):
        return []
    if not startpoint_specs:
        return []
    from localml_scheduler.adapters.mlevolve import build_startpoint_probe_job

    max_models = getattr(gpu_settings, "startpoint_probe_max_models", None)
    selected_specs = list(startpoint_specs)
    if max_models is not None:
        selected_specs = selected_specs[: max(0, int(max_models))]
    if not selected_specs:
        return []
    jobs = [
        build_startpoint_probe_job(
            workflow_id=str(getattr(cfg, "exp_name", "mlevolve")),
            startpoint=spec,
            priority=100,
        )
        for spec in selected_specs
    ]
    submitted = scheduler_client.submit_many(jobs) if hasattr(scheduler_client, "submit_many") else [scheduler_client.submit(job) for job in jobs]
    job_ids = [job.job_id for job in submitted]
    logger.info("🧭 Submitted %s startpoint probe job(s): %s", len(job_ids), ", ".join(job_ids))
    pipeline_logger.emit(
        "scheduler_startpoint_probes_submitted",
        stage="scheduler",
        payload={
            "job_ids": job_ids,
            "startpoints": [
                {
                    "model_key": spec.get("model_key"),
                    "display_name": spec.get("display_name"),
                    "modality": spec.get("modality"),
                    "rank": spec.get("rank"),
                }
                for spec in selected_specs
            ],
        },
    )
    return job_ids


def run():
    run_started_at = time.time()
    cfg = load_cfg()
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')
    hardware_monitor = HardwareMonitor(cfg, logger)
    hardware_monitor.start()
    scheduler_service = None
    scheduler_client = None
    hardware_knowledge_client = None
    pipeline_logger = PipelineActionLogger(
        cfg.log_dir / "pipeline.sqlite3",
        run_id=cfg.exp_name,
        mode=cfg.experiment.mode,
    )
    set_process_pipeline_logger(pipeline_logger)
    pipeline_logger.emit(
        "run_started",
        payload={
            "exp_name": cfg.exp_name,
            "exp_id": cfg.exp_id,
            "mode": cfg.experiment.mode,
            "scheduler_enabled": bool(getattr(cfg.scheduler, "enabled", False)),
        },
    )
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    shutdown_exit_code = None
    shutdown_handled = False

    def handle_sigterm(signum, frame):
        del frame
        shutdown = SignalShutdown(signum)
        logger.warning("%s received; stopping run so hardware report can be written.", shutdown.signal_name)
        raise shutdown

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        task_desc = load_task_desc(cfg)

        startpoint_specs = []
        if cfg.coldstart.use_coldstart:
            logger.info("Loading guidance from knowledge base")
            cfg.coldstart.description = build_guidance_description(cfg)
            startpoint_specs = collect_startpoint_model_specs(cfg)
            logger.info(f"Guidance description: {cfg.coldstart.description}")

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)

        global_step = 0

        def cleanup():
            if global_step == 0:
                shutil.rmtree(cfg.workspace_dir)

        atexit.register(cleanup)

        journal = Journal()
        agent = Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            pipeline_logger=pipeline_logger,
        )

        interpreter = Interpreter(
            cfg.workspace_dir,
            **OmegaConf.to_container(cfg.exec),
            cfg=cfg,  # type: ignore
            pipeline_logger=pipeline_logger,
        )
        hardware_knowledge_cfg = getattr(cfg, "hardware_knowledge", None)
        hardware_knowledge_enabled = (
            str(getattr(getattr(cfg, "experiment", None), "mode", "")).strip().lower().replace("-", "_") == "hardware_aware"
            and bool(getattr(cfg.agent, "hardware_context_enabled", True))
            and (hardware_knowledge_cfg is None or bool(getattr(hardware_knowledge_cfg, "enabled", True)))
        )
        if hardware_knowledge_enabled:
            hardware_settings = _hardware_knowledge_settings_from_cfg(cfg)
            hardware_knowledge_client = HardwareKnowledgeClient(
                hardware_settings,
                include_profile_evidence=(
                    bool(getattr(hardware_knowledge_cfg, "include_profile_evidence", True))
                    if hardware_knowledge_cfg is not None
                    else True
                ),
                probe_timeout_seconds=(
                    float(getattr(hardware_knowledge_cfg, "probe_timeout_seconds", 10.0))
                    if hardware_knowledge_cfg is not None
                    else 10.0
                ),
            )
            probe_status = hardware_knowledge_client.probe_current_hardware()
            if probe_status.get("ok"):
                profile = probe_status.get("hardware_profile") or {}
                logger.info(
                    "🧭 Hardware knowledge enabled via %s for %s.",
                    probe_status.get("source"),
                    profile.get("gpu_name") or profile.get("hardware_key") or "current hardware",
                )
            else:
                logger.warning("🧭 Hardware probe failed; hardware knowledge will use explicit missing context: %s", probe_status.get("reason"))
            agent.attach_hardware_knowledge(hardware_knowledge_client)
            pipeline_logger.emit(
                "hardware_knowledge_attached",
                payload={
                    "settings_runtime_root": str(hardware_settings.runtime_root),
                    "include_profile_evidence": hardware_knowledge_client.include_profile_evidence,
                    "probe_status": probe_status,
                },
            )
        scheduler_cfg = getattr(cfg, "scheduler", None)
        if scheduler_cfg is not None and bool(getattr(scheduler_cfg, "enabled", False)):
            scheduler_settings = _scheduler_settings_from_cfg(cfg, scheduler_cfg)
            scheduler_client = SchedulerClient(
                scheduler_settings,
                hardware_knowledge_client=hardware_knowledge_client,
            )
            if hardware_knowledge_client is not None:
                hardware_knowledge_client.attach_scheduler_client(scheduler_client)
            try:
                prewarm_source = hardware_knowledge_client or scheduler_client
                prewarm_result = prewarm_source.prewarm_current_hardware_neighborhood("current")
                if prewarm_result.get("ok"):
                    logger.info(
                        "🧭 Prewarmed hardware graph neighborhood for %s (%s features).",
                        prewarm_result.get("hardware_name") or prewarm_result.get("hardware_id") or "current hardware",
                        prewarm_result.get("feature_count", 0),
                    )
            except Exception as exc:
                logger.debug("Hardware graph neighborhood prewarm skipped: %s", exc)
            if bool(getattr(scheduler_cfg, "start_service", True)):
                scheduler_service = scheduler_client.create_service().start(background=True)
                logger.info(f"🧭 localml_scheduler service started at {scheduler_settings.runtime_root}")
            else:
                if scheduler_client.scheduler_service_active():
                    logger.info(f"🧭 localml_scheduler bridge enabled using external service at {scheduler_settings.runtime_root}")
                else:
                    scheduler_service = scheduler_client.create_service().start(background=True)
                    logger.warning(
                        "🧭 No active external localml_scheduler service detected at %s; started an in-process fallback service instead.",
                        scheduler_settings.runtime_root,
                    )
            interpreter.attach_scheduler(scheduler_client, scheduler_cfg)
            agent.attach_scheduler(scheduler_client)
            pipeline_logger.emit(
                "scheduler_attached",
                payload={
                    "settings_runtime_root": str(scheduler_settings.runtime_root),
                    "scheduler_mode": scheduler_settings.gpu_scheduler.mode,
                    "start_service": bool(getattr(scheduler_cfg, "start_service", True)),
                },
            )

        global_step = len(journal)
        status = Status("[green]Generating code...")

        def exec_callback(*args, **kwargs):
            if scheduler_client is not None:
                status.update("[magenta]Submitting scheduler job...")
                code = args[0]
                node_id = args[1]
                node_context = kwargs.get("node_context")
                handles = interpreter.submit_scheduler(
                    [
                        {
                            "code": code,
                            "id": node_id,
                            "node": node_context,
                            "branch_id": getattr(node_context, "branch_id", None),
                            "branch_name": getattr(node_context, "branch_name", None) or getattr(node_context, "model_family", None),
                            "model_family": getattr(node_context, "model_family", None),
                            "branch_profile_key": getattr(node_context, "branch_profile_key", None) or getattr(node_context, "active_profile_key", None),
                            "active_profile_key": getattr(node_context, "active_profile_key", None),
                            "parent_branch_name": getattr(getattr(node_context, "parent", None), "branch_name", None) or getattr(getattr(node_context, "parent", None), "model_family", None),
                            "parent_model_family": getattr(getattr(node_context, "parent", None), "model_family", None),
                        }
                    ],
                    working_dir=kwargs.get("working_dir"),
                    submission_label="stream",
                )
                status.update("[green]Generating code...")
                return handles.get(str(node_id)) or next(iter(handles.values()))
            status.update("[magenta]Executing code...")
            res = interpreter.run(*args, **kwargs)
            status.update("[green]Generating code...")
            return res

        def exec_many_callback(items):
            status.update("[magenta]Executing scheduler round...")
            res = interpreter.run_many(items)
            status.update("[green]Generating code...")
            return res

        def step_task(node=None):
            if node:
                logger.info(f"[step_task] Processing node: {node.id}")
            else:
                logger.info(f"[step_task] Processing virtual root node.")
            return agent.step(exec_callback=exec_callback, node=node)

        max_workers = interpreter.max_parallel_run
        total_steps = cfg.agent.steps
        initial_draft_count = cfg.agent.initial_drafts
        scheduler_enabled = scheduler_client is not None
        if scheduler_enabled:
            logger.info(
                "🚀 Scheduler execution enabled; MLEvolve will submit whole runnable rounds and let the scheduler choose parallelism."
            )
        else:
            logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (local subprocess capacity)")
        logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

        lock = threading.Lock()
        completed = 0

        pending_draft_nodes = []
        if not scheduler_enabled and initial_draft_count > 0 and total_steps > 0:
            logger.info(f"📝 Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

            def step_task_generate_only():
                logger.info(f"[step_task_generate_only] Generating draft from virtual root")
                return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)

            for draft_idx in range(min(initial_draft_count, total_steps)):
                try:
                    logger.info(f"🔨 Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)")
                    cur_node = step_task_generate_only()
                    if cur_node is None:
                        logger.warning(f"⚠️  Draft {draft_idx + 1} generation produced no runnable node")
                        if not agent.has_selectable_work():
                            logger.warning("No selectable work remains during initial draft generation; stopping draft phase early.")
                            break
                        continue
                    pending_draft_nodes.append(cur_node)
                    logger.info(f"✅ Draft {draft_idx + 1} code generated: node.id={cur_node.id}, added to virtual_root.children")

                except Exception as e:
                    logger.exception(f"❌ Exception during draft {draft_idx + 1} generation: {e}")

            logger.info(f"✅ Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

        if scheduler_client is not None and completed < total_steps:
            logger.info("🚀 Phase 2: Live scheduler streaming execution")
            candidate_window = int(getattr(scheduler_settings.gpu_scheduler, "candidate_window_size", 2))
            scheduler_queue_limit = min(max(1, total_steps), max(1, candidate_window))
            poll_interval = max(0.1, float(getattr(scheduler_cfg, "wait_poll_interval_seconds", 1.0)))
            draft_attempts_target = min(initial_draft_count, total_steps)
            draft_attempts = 0
            parent_hint = None

            def collect_scheduler_results():
                nonlocal completed, parent_hint
                status.update("[magenta]Collecting scheduler results...")
                results = interpreter.collect_scheduler(wait=False)
                status.update("[green]Generating code...")
                if not results:
                    return []
                executed_nodes = agent.finalize_scheduler_results(results)
                if executed_nodes:
                    parent_hint = executed_nodes[-1]
                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1
                        if completed >= total_steps:
                            logger.info(journal_to_string_tree(journal))
                    logger.info(
                        "📊 Progress: %s/%s steps completed, %s scheduler job(s) outstanding",
                        completed,
                        total_steps,
                        interpreter.scheduler_outstanding_count(),
                    )
                return executed_nodes

            def submit_next_scheduler_node():
                nonlocal draft_attempts, parent_hint
                force_root_draft = draft_attempts < draft_attempts_target
                node_arg = None
                if force_root_draft:
                    draft_attempts += 1
                    logger.info("🔨 Generating and submitting draft %s/%s", draft_attempts, draft_attempts_target)
                elif parent_hint is not None and parent_hint in journal.nodes and not parent_hint.is_terminal:
                    node_arg = parent_hint
                if not force_root_draft and not agent.has_selectable_work():
                    return None
                try:
                    submitted_node = agent.step(exec_callback=exec_callback, node=node_arg)
                except Exception as exc:
                    logger.exception("Exception while generating live scheduler node: %s", exc)
                    return None
                parent_hint = None
                if submitted_node is not None and getattr(submitted_node, "scheduler_submitted", False):
                    logger.info(
                        "📤 Submitted live scheduler node %s as job %s (%s outstanding)",
                        submitted_node.id,
                        getattr(submitted_node, "scheduler_job_id", None),
                        interpreter.scheduler_outstanding_count(),
                    )
                return submitted_node

            try:
                empty_generation_attempts = 0
                while completed < total_steps:
                    collect_scheduler_results()
                    if completed >= total_steps:
                        break

                    outstanding = interpreter.scheduler_outstanding_count()
                    if completed + outstanding >= total_steps:
                        if outstanding == 0:
                            break
                        time.sleep(poll_interval)
                        continue

                    if outstanding >= scheduler_queue_limit:
                        time.sleep(poll_interval)
                        continue

                    if not agent.has_selectable_work() and outstanding > 0:
                        time.sleep(poll_interval)
                        continue
                    if not agent.has_selectable_work() and outstanding == 0:
                        logger.warning(
                            "No selectable work and no scheduler jobs remain; stopping early at %s/%s completed steps.",
                            completed,
                            total_steps,
                        )
                        break

                    submitted_node = submit_next_scheduler_node()
                    collect_scheduler_results()
                    if submitted_node is None:
                        empty_generation_attempts += 1
                        if empty_generation_attempts >= 3 and interpreter.scheduler_outstanding_count() == 0:
                            logger.warning(
                                "Live scheduler generation produced no node repeatedly; stopping at %s/%s completed steps.",
                                completed,
                                total_steps,
                            )
                            break
                    else:
                        empty_generation_attempts = 0

                    logger.info(
                        "📊 Progress: %s/%s steps completed, %s scheduler job(s) outstanding",
                        completed,
                        total_steps,
                        interpreter.scheduler_outstanding_count(),
                    )

                while completed < total_steps and interpreter.scheduler_outstanding_count() > 0:
                    executed_nodes = collect_scheduler_results()
                    if not executed_nodes:
                        time.sleep(poll_interval)
            except SignalShutdown as exc:
                shutdown_exit_code = 128 + exc.signum
                shutdown_handled = True
                logger.info("%s received, terminating subprocesses and shutting down...", exc.signal_name)
                interpreter.terminate_all_subprocesses()
                raise
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
                interpreter.terminate_all_subprocesses()
                raise
        elif pending_draft_nodes or completed < total_steps:
            logger.info(f"🚀 Phase 2: Pipelined parallel execution")
            logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
            logger.info(f"   - Remaining steps: {total_steps - completed}")

            def execute_draft_node(node):
                try:
                    executed_node = agent.execute_deferred_node(node, exec_callback)
                    logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                    return executed_node
                except Exception as e:
                    logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                    return None

            executor = ThreadPoolExecutor(max_workers=max_workers)
            interrupted = False
            try:
                futures = set()
                for i, node in enumerate(pending_draft_nodes):
                    futures.add(executor.submit(execute_draft_node, node))
                    logger.info(f"📤 Submitted draft execution: {node.id}")
                    if i < len(pending_draft_nodes) - 1:
                        time.sleep(10)
                        logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

                initial_step_tasks = min(max_workers, total_steps - completed) - len(pending_draft_nodes)
                if initial_step_tasks > 0:
                    for _ in range(initial_step_tasks):
                        if not agent.has_selectable_work():
                            logger.warning("No selectable work available to fill the thread pool.")
                            break
                        futures.add(executor.submit(step_task))
                        logger.info(f"📤 Submitted initial step_task to fill thread pool")

                while completed < total_steps:
                    if not futures:
                        if agent.has_selectable_work():
                            futures.add(executor.submit(step_task))
                            logger.info("📤 Submitted root step_task after worker pool drained")
                        else:
                            logger.warning(
                                "No futures remain and no selectable work is available; stopping early at %s/%s completed steps.",
                                completed,
                                total_steps,
                            )
                            break

                    done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                    if not done:
                        continue  # timeout, no completed futures, retry (allows SIGINT handling)

                    for fut in done:
                        futures.remove(fut)
                        try:
                            cur_node = fut.result()
                            if cur_node:
                                logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                            else:
                                logger.warning(f"⚠️  Task returned None (execution failed)")
                        except Exception as e:
                            logger.exception(f"❌ Exception during task execution: {e}")
                            cur_node = None

                        with lock:
                            save_run(cfg, journal)
                            completed = len(journal) - 1  # Exclude virtual node
                            if completed == total_steps:
                                logger.info(journal_to_string_tree(journal))

                        if completed + len(futures) < total_steps:
                            if not agent.has_selectable_work():
                                logger.warning(
                                    "No selectable work available after task completion; not submitting a replacement task."
                                )
                                logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
                                continue
                            futures.add(executor.submit(step_task, cur_node))
                            logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                        logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
            except SignalShutdown as exc:
                interrupted = True
                shutdown_exit_code = 128 + exc.signum
                shutdown_handled = True
                logger.info("%s received, terminating subprocesses and shutting down...", exc.signal_name)
                interpreter.terminate_all_subprocesses()
                executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
                raise
            except KeyboardInterrupt:
                interrupted = True
                logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
                interpreter.terminate_all_subprocesses()
                executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
                raise
            finally:
                if not interrupted:
                    executor.shutdown(wait=True)
        else:
            logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

        interpreter.cleanup_session(-1)
    except SignalShutdown as exc:
        if shutdown_exit_code is None:
            shutdown_exit_code = 128 + exc.signum
        if not shutdown_handled and "interpreter" in locals():
            logger.info("%s received, terminating subprocesses and shutting down...", exc.signal_name)
            interpreter.terminate_all_subprocesses()
    finally:
        if "journal" in locals():
            try:
                metrics = build_comparison_metrics(
                    cfg,
                    journal,
                    started_at=run_started_at,
                    finished_at=time.time(),
                    scheduler_client=scheduler_client,
                    hardware_knowledge_client=hardware_knowledge_client,
                    metric_maximize=getattr(locals().get("agent", None), "metric_maximize", None),
                    pipeline_logger=pipeline_logger,
                )
                write_comparison_metrics(metrics, cfg.log_dir)
                pipeline_logger.record_run_metrics(metrics)
                pipeline_logger.emit("run_finished", payload=metrics)
            except Exception as exc:
                logger.warning("Failed to write comparison metrics: %s", exc)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if "interpreter" in locals():
            interpreter.cleanup_session(-1)
        if scheduler_service is not None:
            scheduler_service.stop()
        hardware_monitor.stop()
        set_process_pipeline_logger(None)
        pipeline_logger.close()
    if shutdown_exit_code is not None:
        logging.shutdown()
        os._exit(shutdown_exit_code)


if __name__ == "__main__":    
    run()
