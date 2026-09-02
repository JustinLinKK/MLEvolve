import atexit
import logging
import os
import signal
import sys
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.node_accounting import count_budget_nodes
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
from utils.pipeline_logging import PipelineActionLogger
from utils.serialize import load_json
import torch
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from lesson_profile_database import LessonProfileClient


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

    return SchedulerSettings(runtime_root=scheduler_runtime_root)


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
    # The current time-aware scheduler probes each real candidate natively.
    # Synthetic cold-start probes are an explicit opt-in.
    if not bool(getattr(gpu_settings, "startpoint_probe_enabled", False)):
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


def _run_scheduler_rounds(
    *,
    agent,
    interpreter,
    cfg,
    journal,
    logger: logging.Logger,
    exec_callback=None,
    save_callback=save_run,
) -> int:
    """Submit ready candidates immediately while generation continues."""
    execute_one = exec_callback or interpreter.run
    total_steps = int(cfg.agent.steps)
    completed = count_budget_nodes(journal.nodes)

    # These workers only submit jobs and wait for their terminal results. They
    # do not limit GPU admission: every remaining experiment node has a waiter,
    # while localml_scheduler alone decides how many jobs may run concurrently.
    submission_workers = max(1, total_steps - completed)
    inflight = {}
    with ThreadPoolExecutor(
        max_workers=submission_workers,
        thread_name_prefix="scheduler-submission",
    ) as executor:
        while completed < total_steps or inflight:
            finished = {future for future in inflight if future.done()}
            for future in finished:
                node_id = inflight.pop(future)
                try:
                    future.result()
                except Exception:
                    logger.exception("Scheduler execution failed for ready candidate %s", node_id)
                save_callback(cfg, journal)
                completed = count_budget_nodes(journal.nodes)
                logger.info(
                    "Scheduler-controlled progress: %s/%s budget-counted nodes.",
                    completed,
                    total_steps,
                )

            if completed >= total_steps and not inflight:
                break

            admitted = completed + len(inflight)
            if admitted < total_steps and agent.has_selectable_work():
                candidate = agent.step(
                    exec_callback=execute_one,
                    node=None,
                    execute_immediately=False,
                )
                if candidate is None:
                    continue
                logger.info(
                    "Submitting ready candidate %s to the scheduler immediately.",
                    candidate.id,
                )
                inflight[
                    executor.submit(
                        agent.execute_deferred_nodes,
                        [candidate],
                        interpreter.run_many,
                    )
                ] = candidate.id
                continue

            if inflight:
                wait(set(inflight), return_when=FIRST_COMPLETED)
                continue

            logger.warning(
                "No scheduler candidate remains; stopping at %s/%s budget-counted nodes.",
                completed,
                total_steps,
            )
            break

    return completed


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
    lesson_profile_client = None
    pipeline_logger = PipelineActionLogger(
        cfg.log_dir / "pipeline.sqlite3",
        run_id=cfg.exp_name,
        mode=cfg.experiment.mode,
    )
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

        if cfg.resume_journal is None:
            with Status("Preparing agent workspace (copying and extracting files) ..."):
                prep_agent_workspace(cfg)
        else:
            logger.info("Resuming from %s; preserving its existing workspace and scheduler state.", cfg.resume_journal)

        global_step = 0

        def cleanup():
            if global_step == 0 and cfg.resume_journal is None:
                shutil.rmtree(cfg.workspace_dir)

        atexit.register(cleanup)

        journal = load_json(cfg.resume_journal, Journal) if cfg.resume_journal is not None else Journal()
        agent = Agent(
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            pipeline_logger=pipeline_logger,
        )
        try:
            from context_cache.coordinator import prepare_run_context_cache

            frozen_context_packs = prepare_run_context_cache(cfg)
            if frozen_context_packs:
                logger.info(
                    "Prepared and froze %s context-cache pack(s) for run %s.",
                    len(frozen_context_packs),
                    cfg.exp_name,
                )
        except Exception as exc:
            logger.warning(
                "Context-cache run preparation failed; continuing uncached: %s", exc
            )

        interpreter = Interpreter(
            cfg.workspace_dir,
            **OmegaConf.to_container(cfg.exec),
            cfg=cfg,  # type: ignore
            pipeline_logger=pipeline_logger,
        )
        scheduler_cfg = getattr(cfg, "scheduler", None)
        if scheduler_cfg is not None and bool(getattr(scheduler_cfg, "enabled", False)):
            scheduler_settings = _scheduler_settings_from_cfg(cfg, scheduler_cfg)
            scheduler_client = SchedulerClient(scheduler_settings)
            try:
                prewarm_result = scheduler_client.prewarm_current_hardware_neighborhood("current")
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
            interpreter.set_metric_direction(agent.metric_maximize)
            agent.attach_scheduler(scheduler_client)
            pipeline_logger.emit(
                "scheduler_attached",
                payload={
                    "settings_runtime_root": str(scheduler_settings.runtime_root),
                    "scheduler_mode": scheduler_settings.gpu_scheduler.mode,
                    "start_service": bool(getattr(scheduler_cfg, "start_service", True)),
                },
            )

        lesson_profile_client = LessonProfileClient.from_config(cfg)
        agent.attach_lesson_profiles(lesson_profile_client)
        if bool(getattr(cfg.lesson_profiles, "enabled", True)):
            # SQLite initialization is local and fast. Qdrant/embedding setup is
            # deferred to the durable daemon so prompt generation never waits.
            lesson_profile_client.initialize(initialize_qdrant=False)
            lesson_profile_client.start_worker()
            pipeline_logger.emit(
                "lesson_profiles_attached",
                payload={
                    "read_enabled": bool(cfg.lesson_profiles.read_enabled),
                    "write_enabled": bool(cfg.lesson_profiles.write_enabled),
                    "sqlite_path": str(lesson_profile_client.registry.path),
                    "qdrant_collection": cfg.lesson_profiles.qdrant.collection_name,
                },
            )

        global_step = len(journal)
        status = Status("[green]Generating code...")

        def exec_callback(*args, **kwargs):
            status.update("[magenta]Executing code...")
            res = interpreter.run(*args, **kwargs)
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
            # Scheduler-backed execution is pipelined below. Do not accumulate
            # a code-only initial group before the first submission.
            initial_draft_count = 0
        if scheduler_enabled:
            logger.info(
                "🚀 Scheduler execution enabled; MLEvolve will submit each node as soon as it is ready and let the scheduler choose parallelism."
            )
        else:
            logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (local subprocess capacity)")
        logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

        lock = threading.Lock()
        completed = count_budget_nodes(journal.nodes)

        pending_draft_nodes = []
        if cfg.resume_journal is None and initial_draft_count > 0 and total_steps > 0:
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

        if scheduler_enabled:
            completed = _run_scheduler_rounds(
                agent=agent,
                interpreter=interpreter,
                cfg=cfg,
                journal=journal,
                logger=logger,
                exec_callback=exec_callback,
            )
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
                            completed = count_budget_nodes(journal.nodes)
                            retained_attempts = max(0, len(journal) - 1)
                            excluded_quick_failures = retained_attempts - completed
                            if excluded_quick_failures:
                                logger.info(
                                    "Budget progress excludes %s quickly detected failed node(s)",
                                    excluded_quick_failures,
                                )
                            if completed >= total_steps:
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
        if "agent" in locals():
            agent.close_cuda_docs()
        if "journal" in locals():
            try:
                metrics = build_comparison_metrics(
                    cfg,
                    journal,
                    started_at=run_started_at,
                    finished_at=time.time(),
                    scheduler_client=scheduler_client,
                    metric_maximize=getattr(locals().get("agent", None), "metric_maximize", None),
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
        if lesson_profile_client is not None:
            lesson_profile_client.stop_worker()
        hardware_monitor.stop()
        pipeline_logger.close()
    if shutdown_exit_code is not None:
        logging.shutdown()
        os._exit(shutdown_exit_code)


if __name__ == "__main__":
    run()
