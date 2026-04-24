"""
Python interpreter for executing code snippets via subprocess.
- Executes code in a separate Python process (avoids CUDA/fork issues).
- Captures stdout/stderr, exceptions and stack traces, execution time limit.
- Supports multiple parallel slots (max_parallel_run) with CPU pinning.
"""

import logging
import os
import signal
import sys
import threading
import time
import traceback
import subprocess
import json
import uuid
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path
from typing import Any

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("MLEvolve")

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None



class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 3,
        cfg=None,
        **kwargs,
    ):
        """
        Executes Python code in subprocess(es). No fork/multiprocessing.Process to avoid CUDA issues.

        Args:
            working_dir: working directory of the agent
            timeout: timeout per code execution (seconds)
            agent_file_name: base name for runfile (actual names are runfile_0.py, ...)
            max_parallel_run: max concurrent execution slots
            cfg: config (start_cpu_id, cpu_number, agent.search.parallel_search_num)
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.cfg = cfg
        self.timeout = timeout
        self.max_parallel_run = (
            cfg.agent.search.parallel_search_num if (cfg and getattr(cfg.agent.search, "parallel_search_num", None)) else max_parallel_run
        )
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.current_parallel_run = 0
        self.status_map = [0] * self.max_parallel_run
        self.start_cpu_id = int(cfg.start_cpu_id) if cfg else 0
        self.cpu_number = int(cfg.cpu_number) if cfg else 1
        if self.cpu_number < self.max_parallel_run:
            raise ValueError(
                "The maximum level of parallelism exceeds the number of allocated CPU cores; "
                "ensure that each process has at least one CPU core."
            )
        self.lock = Lock()
        self._procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        self.scheduler_api = None
        self.scheduler_cfg = None
        self._scheduler_job_ids: set[str] = set()
        self._scheduler_jobs_lock = threading.Lock()

    def attach_scheduler(self, api: Any, scheduler_cfg: Any) -> None:
        """Route future executions through localml_scheduler."""
        self.scheduler_api = api
        self.scheduler_cfg = scheduler_cfg

    def terminate_all_subprocesses(self) -> None:
        """Terminate all active subprocesses (for graceful Ctrl+C exit)."""
        with self._procs_lock:
            procs = list(self._active_procs.items())
            self._active_procs.clear()
        for slot_id, proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Error terminating subprocess slot {slot_id}: {e}")
        if self.scheduler_api is not None:
            with self._scheduler_jobs_lock:
                job_ids = list(self._scheduler_job_ids)
            for job_id in job_ids:
                try:
                    self.scheduler_api.cancel_job(job_id)
                except Exception as e:
                    logger.warning(f"Error cancelling scheduler job {job_id}: {e}")

    def check_current_status(self):
        """Check current parallel run number."""
        return self.current_parallel_run < self.max_parallel_run

    def isolate_submission_path(self, code: str, _id) -> str:
        """Per-process submission filename to avoid write conflicts."""
        target = f"submission_{_id}.csv"

        code = code.replace("submission/submission.csv", f"submission/{target}")
        code = code.replace("/submission.csv", f"/{target}")

        for quote in ("'", '"'):
            code = code.replace(
                f"to_csv({quote}submission.csv",
                f"to_csv({quote}submission/{target}",
            )

        for quote in ("'", '"'):
            code = code.replace(f"{quote}submission.csv{quote}", f"{quote}{target}{quote}")

        return code
    
    def isolate_model_path(self, code, _id):
        """Replace generic model filenames in code to avoid multi-process conflicts."""
        if '.pth' not in code and '.bin' not in code and '.pt' not in code:
            return code

        modified_code = code

        generic_model_names = [
            "best_model.pth",
            "best_model.bin",
            "best_model.pt",
            "model_best.pth",
            "model_best.bin",
            "model_best.pt",
            "model.pth",
            "model.pt",
            "model.bin",
            "checkpoint.pth",
            "checkpoint.pt",
            "checkpoint.bin",
        ]

        generic_model_names.sort(key=len, reverse=True)

        for filename in generic_model_names:
            if filename not in modified_code:
                continue

            name, ext = filename.rsplit('.', 1)
            new_filename = f"{name}_{_id}.{ext}"

            modified_code = modified_code.replace(f"/{filename}", f"/{new_filename}")
            modified_code = modified_code.replace(f'"{filename}"', f'"{new_filename}"')
            modified_code = modified_code.replace(f"'{filename}'", f"'{new_filename}'")

        return modified_code
    
    def cleanup_session(self, process_id: int = -1) -> None:
        """Clean up resources for the given process slot."""
        pass

    def run(self, code: str, id, reset_session=True, working_dir: str | None = None):
        """
        Execute the provided Python command in a subprocess and return its output.

        Parameters:
            code: Python code to execute.
            reset_session: Reserved for future use.
            working_dir: Optional per-run working directory.

        Returns:
            ExecutionResult: output, exec_time, exc_type, exc_info, exc_stack.
        """
        if self.scheduler_api is not None:
            return self._run_scheduler_job(code=code, id=id, working_dir=working_dir)
        return self._run_subprocess(code=code, id=id, working_dir=working_dir)

    def _run_scheduler_job(self, code: str, id, working_dir: str | None = None) -> ExecutionResult:
        """Submit generated code as a localml_scheduler job and wait for its result."""
        from localml_scheduler.adapters.mlevolve import build_mlevolve_job
        from localml_scheduler.schemas import ResourceRequirements

        if self.scheduler_api is None:
            return self._run_subprocess(code=code, id=id, working_dir=working_dir)

        logger.info("REPL is submitting code to localml_scheduler")
        process_id = None
        job_id = None
        runfile_path = None
        start_time = time.time()

        try:
            with self.lock:
                self.current_parallel_run += 1
                for idx in range(self.max_parallel_run):
                    if self.status_map[idx] == 0:
                        self.status_map[idx] = 1
                        process_id = idx
                        logger.info(f"Assigned scheduler submission slot: {process_id}")
                        break
                    elif idx == self.max_parallel_run - 1:
                        logger.info("reach max process parallel number")
                        raise ValueError("reach max process parallel number")

            cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
            avail_cpus = sorted(os.sched_getaffinity(0))
            start = process_id * cpu_number_per_session
            cpu_set = set(avail_cpus[start:start + cpu_number_per_session])
            if not cpu_set:
                cpu_set = set(avail_cpus)
            pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)

            code = self.isolate_submission_path(code=code, _id=id)
            code = self.isolate_model_path(code=code, _id=id)
            code = pre_code + code

            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            run_wd.mkdir(parents=True, exist_ok=True)
            runfile_path = run_wd / self.agent_file_name[process_id]
            runfile_path.write_text(code, encoding="utf-8")

            result_dir = run_wd / "working" / "scheduler_results"
            result_path = result_dir / f"result_{id}_{process_id}_{uuid.uuid4().hex}.json"
            scheduler_cfg = self.scheduler_cfg
            runner_kwargs = {
                "script_path": str(runfile_path),
                "working_dir": str(run_wd),
                "result_path": str(result_path),
                "timeout": self.timeout,
            }
            resource_requirements = ResourceRequirements(
                requires_gpu=bool(getattr(scheduler_cfg, "requires_gpu", True)),
                estimated_vram_mb=getattr(scheduler_cfg, "estimated_vram_mb", None),
                estimated_ram_mb=getattr(scheduler_cfg, "estimated_ram_mb", None),
            )
            job = build_mlevolve_job(
                workflow_id=str(getattr(self.cfg, "exp_name", "mlevolve")),
                baseline_model_id=f"mlevolve-script-{id}",
                baseline_model_path=str(runfile_path),
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs=runner_kwargs,
                task_type="mlevolve_script",
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                resource_requirements=resource_requirements,
                packing_family=getattr(scheduler_cfg, "packing_family", "mlevolve_script"),
                packing_eligible=bool(getattr(scheduler_cfg, "packing_eligible", False)),
                packing_max_slowdown_ratio=getattr(scheduler_cfg, "packing_max_slowdown_ratio", None),
                metadata={"mlevolve_node_id": str(id), "submission_slot": process_id},
            )
            submitted = self.scheduler_api.submit_job(job)
            job_id = submitted.job_id
            with self._scheduler_jobs_lock:
                self._scheduler_job_ids.add(job_id)
            logger.info(f"Submitted scheduler job {job_id} for node {id}")

            wait_timeout = getattr(scheduler_cfg, "wait_timeout_seconds", None)
            if wait_timeout is None:
                wait_timeout = self.timeout * max(1, self.max_parallel_run) + 60
            poll_interval = max(0.1, float(getattr(scheduler_cfg, "wait_poll_interval_seconds", 1.0)))
            deadline = time.time() + float(wait_timeout)
            final_job = None
            while time.time() < deadline:
                final_job = self.scheduler_api.get_job(job_id)
                if final_job is not None and final_job.status.is_terminal:
                    break
                time.sleep(poll_interval)
            else:
                self.scheduler_api.cancel_job(job_id)
                exec_time = time.time() - start_time
                return ExecutionResult(
                    term_out=[
                        f"Execution time: TimeoutError: Scheduler job {job_id} exceeded wait limit of {humanize.naturaldelta(wait_timeout)}"
                    ],
                    exec_time=exec_time,
                    exc_type="TimeoutError",
                    exc_info={"message": "scheduler wait timeout", "job_id": job_id},
                    exc_stack=[],
                )

            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                return ExecutionResult(
                    term_out=payload.get("term_out", [""]),
                    exec_time=float(payload.get("exec_time", time.time() - start_time)),
                    exc_type=payload.get("exc_type"),
                    exc_info=payload.get("exc_info") or {},
                    exc_stack=payload.get("exc_stack") or [],
                )

            reason = final_job.status_reason if final_job is not None else "scheduler job finished without result"
            return ExecutionResult(
                term_out=[f"Scheduler job {job_id} finished without an execution result: {reason}\n"],
                exec_time=time.time() - start_time,
                exc_type="RuntimeError",
                exc_info={"message": reason, "job_id": job_id},
                exc_stack=[],
            )
        except Exception as e:
            logger.error(f"Error in _run_scheduler_job: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            return ExecutionResult(
                term_out=[f"Scheduler execution error: {str(e)}", error_trace],
                exec_time=time.time() - start_time,
                exc_type="RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
        finally:
            if job_id is not None:
                with self._scheduler_jobs_lock:
                    self._scheduler_job_ids.discard(job_id)
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove scheduler runfile after execution: {e}")
            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1

    def _run_subprocess(self, code: str, id, working_dir: str | None = None):
        """
        Execute code via subprocess (avoids CUDA fork issues).
        Aligned with multiprocessing mode for consistency.
        """
        logger.info("REPL is executing code via subprocess")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None

        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    logger.info(f"Assigned process_id: {process_id}")
                    break
                elif idx == self.max_parallel_run - 1:
                    logger.info("reach max process parallel number")
                    raise ValueError("reach max process parallel number")

        start_time = time.time()
        runfile_path = None
        proc = None
        
        try:
            cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
            avail_cpus = sorted(os.sched_getaffinity(0))
            start = process_id * cpu_number_per_session
            cpu_set = set(avail_cpus[start:start + cpu_number_per_session])
            if not cpu_set:
                cpu_set = set(avail_cpus)
            logger.info(f"has set process_id:{process_id} to use cpu: {cpu_set}")
            pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)

            code = self.isolate_submission_path(code=code, _id=id)
            code = self.isolate_model_path(code=code, _id=id)
            code = pre_code + code

            # decide runfile location and cwd
            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            runfile_path = run_wd / self.agent_file_name[process_id]
            run_wd.mkdir(parents=True, exist_ok=True)
            with open(runfile_path, "w") as f:
                f.write(code)

            cmd = [sys.executable, str(runfile_path)]
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_wd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            with self._procs_lock:
                self._active_procs[process_id] = proc

            child_in_overtime = False
            exc_type = None
            exc_info = {}
            exc_stack = []
            
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                exec_time = time.time() - start_time
                
                if proc.returncode != 0:
                    exc_type = "RuntimeError"
                    exc_info = {}
                    exc_stack = []
                    
                    if stderr:
                        stderr_text = stderr
                        exc_patterns = [
                            ("KeyboardInterrupt", "KeyboardInterrupt"),
                            ("TimeoutError", "TimeoutError"),
                            ("CUDA", "RuntimeError"),
                            ("cuda", "RuntimeError"),
                            ("ValueError", "ValueError"),
                            ("TypeError", "TypeError"),
                            ("AttributeError", "AttributeError"),
                            ("KeyError", "KeyError"),
                            ("IndexError", "IndexError"),
                            ("FileNotFoundError", "FileNotFoundError"),
                            ("ImportError", "ImportError"),
                            ("AssertionError", "AssertionError"),
                            ("NameError", "NameError"),
                            ("RuntimeError", "RuntimeError"),
                        ]
                        
                        for pattern, exc_name in exc_patterns:
                            if pattern in stderr_text:
                                exc_type = exc_name
                                break
                        
                        stderr_lines = stderr_text.splitlines()
                        for line in stderr_lines:
                            if 'File "' in line and 'line' in line:
                                try:
                                    file_start = line.find('File "') + 6
                                    file_end = line.find('"', file_start)
                                    if file_end > file_start:
                                        filename = line[file_start:file_end]
                                        line_start = line.find('line ') + 5
                                        line_end = line.find(',', line_start)
                                        if line_end == -1:
                                            line_end = line.find('\n', line_start)
                                        if line_end == -1:
                                            line_end = len(line)
                                        line_num_str = line[line_start:line_end].strip()
                                        if line_num_str.isdigit():
                                            line_num = int(line_num_str)
                                            func_name = ""
                                            if 'in ' in line:
                                                func_start = line.find('in ') + 3
                                                func_end = line.find('\n', func_start)
                                                if func_end == -1:
                                                    func_end = len(line)
                                                func_name = line[func_start:func_end].strip()
                                            
                                            filename_short = filename.replace(str(self.working_dir / self.agent_file_name[process_id]), self.agent_file_name[process_id])
                                            filename_short = os.path.basename(filename_short)
                                            exc_stack.append((filename_short, line_num, func_name, ""))
                                except Exception:
                                    pass
                        
                        for line in reversed(stderr_lines):
                            line = line.strip()
                            if line and not line.startswith("File") and not line.startswith("Traceback"):
                                if ":" in line:
                                    parts = line.split(":", 1)
                                    if len(parts) == 2:
                                        exc_info["message"] = parts[1].strip()
                                    break
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess timeout, sending SIGINT...")
                try:
                    proc.send_signal(signal.SIGINT)
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("Subprocess failed to terminate after SIGINT, killing...")
                    proc.kill()
                    stdout, stderr = proc.communicate()
                
                exec_time = time.time() - start_time
                exc_type = "TimeoutError"
                exc_info = {}
                exc_stack = []
            
            output: list[str] = []
            if stdout:
                output.extend(stdout.splitlines(keepends=True))
            if stderr:
                output.extend(stderr.splitlines(keepends=True))
            if not output:
                output = [""]
            if output and output[-1] and not output[-1].endswith("\n"):
                output.append("\n")

            if exc_type == "TimeoutError":
                output.append(
                    f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
                )
            
            return ExecutionResult(output, exec_time, exc_type, exc_info, exc_stack)
            
        except Exception as e:
            logger.error(f"Error in _run_subprocess: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            exec_time = time.time() - start_time if start_time else 0
            return ExecutionResult(
                term_out=[f"Subprocess execution error: {str(e)}", error_trace],
                exec_time=exec_time,
                exc_type="RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
        finally:
            if process_id is not None:
                with self._procs_lock:
                    self._active_procs.pop(process_id, None)
            if proc is not None:
                try:
                    if proc.poll() is None:
                        logger.warning(f"Subprocess {process_id} still running, terminating...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Subprocess {process_id} failed to terminate, killing...")
                            proc.kill()
                            proc.wait()
                except Exception as e:
                    logger.warning(f"Error cleaning up subprocess {process_id}: {e}")
            
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove runfile after subprocess execution: {e}")
            
            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1
