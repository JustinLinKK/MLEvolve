"""Dependency-aware replay: arrivals move when the schedule moves.

Holding `release_seconds` fixed makes the cassava replay arrival-limited.
Measured on that trace, at most five jobs are ever ready at once and the
median gap between arrivals is 275 s, so every admission policy starves and
they all land within a few minutes of each other regardless of how cleverly
they pack.

That fixed-arrival replay is also wrong. A node cannot be generated until its
parent has produced a result, so an arrival is a consequence of the schedule,
exactly as Trace_Generation.md states:

    t_arrive(n) = t_exec_end(parent(n)) + gen_duration(n)

Under this model finishing a parent earlier pulls its child forward, and the
saving compounds down each branch. A root node, whose parent lies outside the
recorded window, keeps its recorded arrival.

Generation is treated as fixed work: the same code is replayed, so the LLM
does the same job regardless of when it is asked. Training time is what varies
with the schedule.

Shape symbols: none, this module handles scalar job records rather than
tensors.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from scheduler_benchmark_test.occupancy_scheduler import Result, _slowdown


@dataclass(frozen=True, slots=True)
class DagJob:
    """One recorded execution together with its place in the search tree.

    Args:
        job_id      : str, unique identifier
        node_id     : str, MLEvolve search-node id
        parent_id   : str | None, the node whose result this one was built from
        gen_seconds : float, LLM time to produce this node's code, held fixed
        solo        : float, seconds to train with the device to itself
        memory_mb   : float, peak VRAM this job alone adds
        root_release: float, recorded arrival, used only when the parent lies
                      outside the recorded window
    """
    job_id: str
    node_id: str
    parent_id: str | None
    gen_seconds: float
    solo: float
    memory_mb: float
    root_release: float


def simulate_dag(
    jobs: list[DagJob],
    *,
    pair_slowdown: float,
    memory_budget_mb: float,
    parallel_cap: int,
    scorer: Callable[[DagJob], tuple],
    starvation_timeout: float = 1800.0,
    policy_name: str = "dag",
    guard: Callable[[list[DagJob], DagJob], bool] | None = None,
) -> Result:
    """Replay `jobs` with arrivals derived from the schedule itself.

    Args:
        jobs             : list[DagJob], the workload and its dependency edges
        pair_slowdown    : float, slowdown contributed per co-runner
        memory_budget_mb : float, hard VRAM ceiling on the running set
        parallel_cap     : int, maximum simultaneous jobs
        scorer           : callable(DagJob) -> sort key, admission preference
        starvation_timeout : float, seconds before a waiting job is forced in
        policy_name      : str, label carried into the Result

    Returns:
        Result. `makespan` is the last completion minus the earliest arrival,
        matching the definition used by the repo's trace_metrics.

    A job becomes ready once its parent has finished and its generation time
    has elapsed. Roots are ready at their recorded arrival. The event loop
    advances to the earliest of the next completion or the next arrival,
    charges each running job the work it did over that interval at the current
    slowdown, frees finished slots, and refills.
    """
    by_node = {j.node_id: j for j in jobs}
    known = set(by_node)

    arrival: dict[str, float] = {}
    for j in jobs:
        if j.parent_id is None or j.parent_id not in known:
            arrival[j.job_id] = j.root_release

    children: dict[str, list[DagJob]] = {}
    for j in jobs:
        if j.job_id not in arrival:
            children.setdefault(j.parent_id, []).append(j)

    pending: dict[str, DagJob] = {j.job_id: j for j in jobs}
    running: dict[str, DagJob] = {}
    remaining: dict[str, float] = {}
    start_at: dict[str, float] = {}
    finish_at: dict[str, float] = {}
    slowdown_samples: list[float] = []
    conc_time: dict[int, float] = {}
    timeline: list[tuple[str, float, float]] = []

    now = min(arrival.values()) if arrival else 0.0

    def used_memory() -> float:
        return sum(j.memory_mb for j in running.values())

    def ready_jobs() -> list[DagJob]:
        return [
            j for jid, j in pending.items()
            if jid in arrival and arrival[jid] <= now + 1e-9
        ]

    def admit() -> None:
        while len(running) < parallel_cap:
            free = memory_budget_mb - used_memory()
            fits = [j for j in ready_jobs() if j.memory_mb <= free + 1e-9]
            if not fits:
                return
            starving = [j for j in fits if now - arrival[j.job_id] >= starvation_timeout]
            if not starving and guard is not None:
                # A job already running may sit on the critical path, and every
                # extra co-runner slows it, stretching the chain that sets the
                # makespan. The guard vetoes admissions whose contention costs
                # the DAG more than the extra parallelism gains.
                fits = [j for j in fits if guard(list(running.values()), j)]
                if not fits:
                    return
            pick = (
                min(starving, key=lambda j: (arrival[j.job_id], j.job_id))
                if starving else min(fits, key=scorer)
            )
            running[pick.job_id] = pick
            remaining[pick.job_id] = pick.solo
            start_at[pick.job_id] = now
            pending.pop(pick.job_id)

    admit()

    iterations = 0
    while running or any(jid in arrival for jid in pending):
        iterations += 1
        if iterations > 100000:
            raise RuntimeError("dag simulation failed to converge")

        if not running:
            future = [arrival[jid] for jid in pending if jid in arrival and arrival[jid] > now]
            if not future:
                break
            now = min(future)
            admit()
            continue

        n = len(running)
        sd = _slowdown(n, pair_slowdown)
        t_finish = min(remaining[jid] * sd for jid in running)
        future = [arrival[jid] - now for jid in pending if jid in arrival and arrival[jid] > now]
        dt = min([t_finish] + [f for f in future if f > 0])

        for jid in list(running):
            remaining[jid] -= dt / sd
        conc_time[n] = conc_time.get(n, 0.0) + dt
        slowdown_samples.extend([sd] * n)
        now += dt

        for jid in [j for j, left in list(remaining.items()) if left <= 1e-6]:
            job = running.pop(jid)
            remaining.pop(jid)
            finish_at[jid] = now
            timeline.append((jid, start_at[jid], now))
            # Releasing a parent makes its children generatable.
            for child in children.get(job.node_id, []):
                arrival[child.job_id] = now + child.gen_seconds

        admit()

    flows = [finish_at[j] - arrival[j] for j in finish_at]
    waits = [start_at[j] - arrival[j] for j in finish_at]
    flows_sorted = sorted(flows)
    p95_index = max(0, int(0.95 * len(flows_sorted)) - 1)
    total_time = sum(conc_time.values()) or 1.0

    return Result(
        policy=policy_name,
        makespan=max(finish_at.values()) - min(arrival[j] for j in finish_at),
        mean_flow=statistics.fmean(flows),
        p95_flow=flows_sorted[p95_index],
        max_wait=max(waits),
        average_slowdown=statistics.fmean(slowdown_samples) if slowdown_samples else 1.0,
        starvation_count=sum(1 for w in waits if w >= starvation_timeout),
        peak_concurrency=max(conc_time) if conc_time else 0,
        mean_concurrency=sum(k * v for k, v in conc_time.items()) / total_time,
        timeline=timeline,
    )
