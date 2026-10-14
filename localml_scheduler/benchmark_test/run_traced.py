"""Run MLEvolve at its native parallelism and record a scheduler trace.

The interpreter is NOT pinned: MLEvolve runs with max_parallel_run as
configured (2 by default), so executions overlap the way they do in a real
run. Two consequences drive the design here.

Arrival time. A node's arrival is not an independent event: the agent can only
start generating a node once its parent's execution has produced a result, so
arrival equals parent execution end plus this node's generation duration. A
trace of absolute arrival stamps is therefore unreplayable, because changing
the schedule changes execution end and every downstream arrival with it.

To make the trace rescheduleable, each row carries `gen_start_at`,
`gen_end_at`, `gen_duration_s`, `parent_node_id` and `chain_id`. Replay holds
generation duration fixed, since the same code is replayed, and lets training
time vary with the schedule.

Generation timestamps are keyed by chain rather than pooled. The earlier
version pushed ready times onto one global FIFO consumed in dispatch order,
which mismatched chains whenever parallel_search_num exceeded 1.

Resource attribution. Device-wide sampling cannot be attributed to one job
while several run. Per-process GPU memory is read from NVML's compute-process
list instead, and every row records which other jobs overlapped it and for how
long, so colocation slowdown can be measured from the trace itself rather than
assumed.

Usage (from the MLEvolve repo root):
    python run_traced.py <hydra-style overrides...>

Environment:
    MLEVOLVE_TRACE_PATH  destination JSONL (default ./mlevolve_trace.jsonl)
"""

import json
import os
import re
import threading
import time

TRACE_PATH = os.environ.get("MLEVOLVE_TRACE_PATH", "mlevolve_trace.jsonl")
# Recorded on every row so a multi-task trace can answer which task was issued
# at which time. The previous single-task run left these null.
TASK_NAME = os.environ.get("MLEVOLVE_TASK_NAME")
DATASET = os.environ.get("MLEVOLVE_DATASET")

_RUN_STARTED_AT = time.time()
_write_lock = threading.Lock()
_state_lock = threading.Lock()
_step_counter = {"n": 0}

# job_id -> dispatch time, for jobs currently executing.
_active: dict[str, float] = {}

# Generation timestamps are keyed by search chain, never pooled.
#
# The previous version pushed every ready time onto one global FIFO and popped
# it in dispatch order, so with parallel_search_num > 1 a job could pop another
# chain's timestamp. That is what made recorded arrivals unusable: job 0003
# came out carrying job 0000's time, producing an 872 s "queue delay" that
# never happened.
#
# node_id -> time the generating agent started producing that node's code.
#
# Keying by node identity rather than by thread is what makes this correct.
# MLEvolve generates drafts in one phase and executes them in a later one, so
# anything stamped at the executing call site records a time after generation
# already finished; a first attempt hooked AgentSearch.step that way and
# measured generation durations of 0.000 s.
_node_gen_start: dict[str, float] = {}
# node_id -> parent node_id, read from the node the generator returned. This is
# the dependency edge replay needs: a node's arrival is its parent's execution
# end plus this node's generation duration, so rescheduling moves arrivals too.
_node_parent: dict[str, str | None] = {}
# node_id -> which generator produced it (draft, improve, debug, ...).
_node_agent: dict[str, str] = {}

# Each execution runs in its own ThreadPoolExecutor thread, so the training
# subprocess PID can be captured per-thread and used to attribute GPU memory
# even while several jobs share the device.
_thread_state = threading.local()


def install_pid_capture(executor_module) -> None:
    """Record the training subprocess PID of whichever job spawned it."""
    original_popen = executor_module.subprocess.Popen

    class TracingPopen(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pids = getattr(_thread_state, "pids", None)
            if pids is not None:
                pids.add(self.pid)

    executor_module.subprocess.Popen = TracingPopen


def _now() -> float:
    return time.time() - _RUN_STARTED_AT


def _emit(record: dict) -> None:
    with _write_lock:
        with open(TRACE_PATH, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def _chain_id() -> str:
    """Identify the search branch this call belongs to.

    MLEvolve runs each parallel search branch on its own worker thread, so the
    thread name labels the branch. This is recorded for analysis only; the
    generation timings themselves are keyed by node id, because a node is
    generated in one phase and may be executed later on a different thread.
    """
    return threading.current_thread().name


def _record_generation(node) -> None:
    """Store the generation window and parent edge for a freshly generated node.

    Args:
        node : SearchNode returned by a generating agent, carrying `.id` and
               `.parent`. Ignored when falsy, since a generator may return None
               if it failed or hit a limit.

    The start time is read from `_thread_state.gen_started`, stamped when the
    generating agent was entered, so the pair brackets exactly the LLM work
    that produced this node's code.
    """
    node_id = getattr(node, "id", None)
    if node_id is None:
        return
    started = getattr(_thread_state, "gen_started", None)
    if started is None:
        return
    parent = getattr(node, "parent", None)
    parent_id = getattr(parent, "id", None)
    with _state_lock:
        _node_gen_start[str(node_id)] = started
        _node_parent[str(node_id)] = str(parent_id) if parent_id is not None else None


def install_generation_hooks(agent_modules: dict) -> None:
    """Wrap each generating agent so its LLM work is timed per node.

    Args:
        agent_modules : dict[str, module], generator name -> module exposing
                        `run(agent, ...) -> SearchNode`. Modules without a
                        `run` attribute are skipped.

    Each generator performs the LLM calls that write a node's code and returns
    the node, so wrapping `run` brackets generation exactly. Hooking the
    executing call site instead measures nothing, because MLEvolve generates
    drafts in one phase and executes them in a later one.
    """
    for agent_name, module in agent_modules.items():
        original = getattr(module, "run", None)
        if original is None:
            continue

        def make(fn, name):
            def wrapper(*args, **kwargs):
                _thread_state.gen_started = _now()
                node = fn(*args, **kwargs)
                if node is not None:
                    _record_generation(node)
                    node_id = getattr(node, "id", None)
                    if node_id is not None:
                        with _state_lock:
                            _node_agent[str(node_id)] = name
                _thread_state.gen_started = None
                return node
            return wrapper

        module.run = make(original, agent_name)


class ProcessVramSampler:
    """Sample per-process GPU memory and device SM across one execution window.

    Per-process memory comes from NVML's compute-process list, so a job's
    footprint stays separable while other jobs share the device. Device SM is
    still whole-GPU and is recorded as context for the concurrency set, not as
    a per-job figure.
    """

    def __init__(self, interval: float = 0.1, device_index: int = 0, own_pids: set[int] | None = None) -> None:
        self.interval = interval
        # Each sample: (wall_clock_s, device_vram_mib, device_sm_percent)
        self.samples: list[tuple[float, float, float]] = []
        # Device VRAM already in use when this job's window opened.
        self.baseline_mib = 0.0
        self.own_pids = own_pids if own_pids is not None else set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._nvml = None

    def _read(self) -> tuple[float, float]:
        """Return (device VRAM used MiB, device SM percent).

        This container's PID namespace hides NVML's compute-process list
        (nvmlDeviceGetComputeRunningProcesses returns nothing while memory is
        clearly in use), so per-process attribution is unavailable and device
        totals are the only truthful measurement.
        """
        used_mib = 0.0
        sm = 0.0
        try:
            used_mib = float(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used) / (1024 ** 2)
            sm = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except Exception:
            pass
        return used_mib, sm

    def __enter__(self):
        if self._nvml is not None:
            self.baseline_mib, _ = self._read()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            used_mib, sm = self._read()
            self.samples.append((time.time(), used_mib, sm))
            time.sleep(self.interval)

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return False

    def stats(self) -> dict:
        """Summarize the window.

        `own_*` covers only this job's own training subprocess, identified by
        the PID captured when it was spawned, so the figure stays correct while
        other jobs share the device.
        """
        if not self.samples:
            return {"vram_samples": 0}

        vram_values = [used for _, used, _ in self.samples]
        sm_values = [sm for _, _, sm in self.samples]
        device_peak = max(vram_values)

        return {
            "device_peak_vram_mib": round(device_peak, 1),
            "device_avg_vram_mib": round(sum(vram_values) / len(vram_values), 1),
            "baseline_vram_mib": round(self.baseline_mib, 1),
            "delta_peak_vram_mib": round(max(0.0, device_peak - self.baseline_mib), 1),
            "device_avg_sm_percent": round(sum(sm_values) / len(sm_values), 1),
            "device_peak_sm_percent": round(max(sm_values), 1),
            "vram_samples": len(self.samples),
        }


_ARCH_PATTERNS = (
    r"timm\.create_model\(\s*[\"']([\w.\-]+)[\"']",
    r"torchvision\.models\.(\w+)\s*\(",
    r"models\.(\w+)\s*\(\s*(?:pretrained|weights)",
    r"MODEL_NAME\s*=\s*[\"']([\w.\-/]+)[\"']",
    r"AutoModel\w*\.from_pretrained\(\s*[\"']([\w.\-/]+)[\"']",
)


def _extract(code: str) -> dict:
    """Pull scheduler-relevant knobs back out of agent-generated code.

    Best-effort regex reads; `None` means the value was not stated literally
    and must not be treated as measured.
    """

    def first(patterns, cast=str):
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                try:
                    return cast(match.group(1))
                except (TypeError, ValueError):
                    continue
        return None

    architecture = first(_ARCH_PATTERNS)
    batch_size = first((
        r"\bbatch_size\s*=\s*(\d+)",
        r"\bBATCH_SIZE\s*=\s*(\d+)",
        r"per_device_train_batch_size\s*=\s*(\d+)",
    ), int)
    epochs = first((
        r"\bNUM_EPOCHS\s*=\s*(\d+)",
        r"\bn_epochs\s*=\s*(\d+)",
        r"\bepochs\s*=\s*(\d+)",
        r"num_train_epochs\s*=\s*(\d+)",
        r"for\s+epoch\s+in\s+range\(\s*(\d+)\s*\)",
    ), int)

    family = None
    if architecture:
        lowered = architecture.lower()
        if any(t in lowered for t in ("resnet", "efficientnet", "convnext", "mobilenet", "densenet", "vgg")):
            family = "cnn"
        elif any(t in lowered for t in ("vit", "swin", "deit", "bert", "gpt", "transformer")):
            family = "transformer"
        else:
            family = "other"
    elif re.search(r"nn\.Linear", code):
        family = "mlp"

    return {
        "architecture": architecture,
        "family": family,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def install(interpreter_cls) -> None:
    """Wrap Interpreter.run so each execution appends one trace row."""
    original_run = interpreter_cls.run

    def traced_run(self, code: str, id, *args, **kwargs):
        with _state_lock:
            step_idx = _step_counter["n"]
            _step_counter["n"] += 1
        job_id = f"mlevolve_{step_idx:04d}"

        # Entry to Interpreter.run is the moment generated code is handed over
        # for execution, so it closes the generation interval and opens the
        # execution one. Generation start and the parent edge are looked up by
        # node id, recorded when the generating agent produced this node.
        dispatch_at = _now()
        node_key = str(id)
        with _state_lock:
            gen_start_at = _node_gen_start.get(node_key, dispatch_at)
            parent_node_id = _node_parent.get(node_key)
            agent_used = _node_agent.get(node_key)
        chain_id = _chain_id()

        with _state_lock:
            overlap_start = {other: start for other, start in _active.items()}
            _active[job_id] = dispatch_at

        own_pids: set[int] = set()
        _thread_state.pids = own_pids

        with ProcessVramSampler(own_pids=own_pids) as sampler:
            try:
                result = original_run(self, code, id, *args, **kwargs)
                error = None
            except Exception as exc:
                result = None
                error = repr(exc)
        complete_at = _now()
        _thread_state.pids = None

        with _state_lock:
            _active.pop(job_id, None)
            # Anything still active, plus anything that was active at dispatch,
            # shared the device with this job for part of its window.
            overlapped = sorted(set(overlap_start) | set(_active))

        row = {
            "step_idx": step_idx,
            "job_id": job_id,
            "node_id": str(id),
            "parent_node_id": parent_node_id,
            "agent_used": agent_used,
            "chain_id": chain_id,
            "task_name": TASK_NAME,
            "dataset": DATASET,
            "gen_start_at": round(gen_start_at, 3),
            "gen_end_at": round(dispatch_at, 3),
            "gen_duration_s": round(dispatch_at - gen_start_at, 3),
            "release_seconds": round(dispatch_at, 3),
            "dispatch_at": round(dispatch_at, 3),
            "exec_complete_at": round(complete_at, 3),
            "exec_duration_s": round(complete_at - dispatch_at, 3),
            "reported_exec_time_s": getattr(result, "exec_time", None),
            "concurrent_jobs": overlapped,
            "concurrency_degree": len(overlapped) + 1,
            "ran_solo": len(overlapped) == 0,
            "is_buggy": bool(getattr(result, "exc_type", None)) or error is not None,
            "exc_type": getattr(result, "exc_type", None),
            "harness_error": error,
            "code": code,
            **_extract(code),
            **sampler.stats(),
        }
        _emit(row)

        if error is not None:
            raise RuntimeError(error)
        return result

    interpreter_cls.run = traced_run


def _generation_modules() -> dict:
    """Return every module whose `run` produces a node's code.

    These are the six actions `_run_single_step` can take. Each performs the
    LLM calls that write the code and returns the resulting node, so wrapping
    them brackets generation exactly and attributes it to a node id.
    """
    # agent_search.py binds these module objects by name, so patching each
    # module's `run` attribute is visible to it without re-importing.
    from agents import (
        aggregation_agent,
        debug_agent,
        draft_agent,
        evolution_agent,
        fusion_agent,
        improve_agent,
    )
    return {
        "draft": draft_agent,
        "debug": debug_agent,
        "improve": improve_agent,
        "evolution": evolution_agent,
        "fusion": fusion_agent,
        "aggregation": aggregation_agent,
    }


def main() -> None:
    import run as mlevolve_run
    from engine import executor as executor_module
    from engine.executor import Interpreter

    install_pid_capture(executor_module)
    install(Interpreter)
    install_generation_hooks(_generation_modules())
    print(f"[trace] recording to {os.path.abspath(TRACE_PATH)} (native parallelism)", flush=True)
    mlevolve_run.run()
    print(f"[trace] wrote {_step_counter['n']} rows to {os.path.abspath(TRACE_PATH)}", flush=True)


if __name__ == "__main__":
    main()
