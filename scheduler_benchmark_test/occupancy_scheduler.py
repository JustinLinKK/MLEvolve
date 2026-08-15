"""An occupancy-maximising scheduler, and a harness to compare it fairly.

Why a new scheduler. Replaying the cassava trace showed this repo's
`simulate_recursive_time_aware` reaching an effective aggregate throughput of
only 1.48x serial, while the recorded memory footprints allow roughly three
concurrent jobs on a 31 GB budget. Always running two would already finish in
278 min and always three in 216 min, against the 314 min it actually achieved,
so the repo scheduler is leaving throughput unused rather than being blocked
by the device.

Two causes, both addressed here:

    unit drain      `simulate_policy` advances the clock by a pack's
                    drain_seconds, so every slot stays blocked until the
                    slowest member of a pack finishes. A one-minute job packed
                    behind a sixty-minute job wastes a slot for fifty-nine
                    minutes. This simulator instead lets each job finish and
                    free its slot independently.

    idle capacity   the repo scheduler admits candidates one at a time behind
                    a two-epoch colocation trial. This one refills every free
                    slot the instant a job completes or arrives, using the
                    profile estimate directly, because with the measured
                    slowdown curve aggregate throughput rises monotonically
                    with concurrency and there is nothing to trial.

Throughput model, identical to the repo simulator so the comparison is fair:
a member sharing the device with (N-1) others runs at

    slowdown = 1 + (N - 1) * (pair_slowdown - 1)

so N concurrent jobs deliver an aggregate of N / slowdown, which for the
measured pair value of 1.198 gives 1.67 at N=2 rising to 2.79 at N=5.

Metrics are computed with the same definitions the repo uses in
trace_metrics: flow is completion minus release, wait is first dispatch minus
release, and makespan is the last completion minus the first release.

Shape symbols: none, this module handles scalar job records rather than
tensors.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True, slots=True)
class Job:
    """One recorded execution, expressed as schedulable work.

    Args:
        job_id    : str, unique identifier
        release   : float, seconds from run start when the job became ready
        solo      : float, seconds the job would take with the device to itself
        memory_mb : float, peak VRAM this job alone adds to the device
    """
    job_id: str
    release: float
    solo: float
    memory_mb: float


@dataclass
class Result:
    """Outcome of one simulated policy."""
    policy: str
    makespan: float
    mean_flow: float
    p95_flow: float
    max_wait: float
    average_slowdown: float
    starvation_count: int
    peak_concurrency: int
    mean_concurrency: float
    timeline: list[tuple[str, float, float]] = field(default_factory=list)


def _slowdown(n: int, pair: float) -> float:
    """Per-member slowdown when n jobs share the device."""
    return 1.0 + max(0, n - 1) * (pair - 1.0)


# Candidate scorers. Each returns a sort key over ready jobs; lower sorts first.

def score_lpt(job: Job) -> tuple:
    """Longest processing time first.

    Starting long jobs early keeps them from forming a tail that runs alone
    after everything else has drained, which is the classic makespan argument.
    """
    return (-job.solo, job.release, job.job_id)


def score_density(job: Job) -> tuple:
    """Most work per megabyte first.

    Memory is the binding constraint here, so admitting the job that converts
    the scarcest resource into the most work is the greedy knapsack choice.
    """
    return (-(job.solo / max(job.memory_mb, 1.0)), job.release, job.job_id)


def score_small_memory(job: Job) -> tuple:
    """Smallest footprint first, maximising how many jobs fit at once."""
    return (job.memory_mb, -job.solo, job.job_id)


def simulate(
    jobs: list[Job],
    *,
    pair_slowdown: float,
    memory_budget_mb: float,
    parallel_cap: int,
    scorer: Callable[[Job], tuple] = score_lpt,
    starvation_timeout: float = 1800.0,
    policy_name: str = "occupancy",
) -> Result:
    """Run the occupancy-maximising policy over `jobs`.

    Args:
        jobs               : list[Job], the workload
        pair_slowdown      : float, measured slowdown contributed per co-runner
        memory_budget_mb   : float, hard VRAM ceiling for the running set
        parallel_cap       : int, maximum simultaneous jobs
        scorer             : callable(Job) -> sort key, admission preference
        starvation_timeout : float, seconds after which a waiting job is
                             force-admitted ahead of the scorer's preference
        policy_name        : str, label carried into the Result

    Returns:
        Result with makespan, flow and concurrency statistics.

    The loop is event driven. At each step the clock advances to the earliest
    of the next completion or the next arrival, every running job is charged
    the work it performed over that interval at the current slowdown, finished
    jobs release their slot and memory, and every free slot is then refilled.
    Because the slowdown depends on how many jobs are running, remaining work
    is tracked in solo-seconds and burned at a rate of 1/slowdown.
    """
    pending = sorted(jobs, key=lambda j: (j.release, j.job_id))
    remaining: dict[str, float] = {}   # job_id -> solo-seconds still to do
    running: dict[str, Job] = {}
    start_at: dict[str, float] = {}
    finish_at: dict[str, float] = {}
    timeline: list[tuple[str, float, float]] = []
    slowdown_samples: list[float] = []
    conc_time: dict[int, float] = {}

    now = pending[0].release if pending else 0.0

    def used_memory() -> float:
        return sum(j.memory_mb for j in running.values())

    def admit() -> None:
        """Fill every free slot that memory allows, honouring starvation first.

        Jobs are removed from `pending` when admitted, so the ready set is
        recomputed from the current list each pass rather than tracked by
        index.
        """
        while len(running) < parallel_cap:
            free = memory_budget_mb - used_memory()
            fits = [
                j for j in pending
                if j.release <= now + 1e-9 and j.memory_mb <= free + 1e-9
            ]
            if not fits:
                return
            starving = [j for j in fits if now - j.release >= starvation_timeout]
            pick = (
                min(starving, key=lambda j: (j.release, j.job_id))
                if starving else min(fits, key=scorer)
            )
            running[pick.job_id] = pick
            remaining[pick.job_id] = pick.solo
            start_at[pick.job_id] = now
            pending.remove(pick)

    admit()

    while running or pending:
        if not running:
            now = min(j.release for j in pending)
            admit()
            continue

        n = len(running)
        sd = _slowdown(n, pair_slowdown)
        # Wall-clock until the first running job finishes at this slowdown.
        t_finish = min(remaining[jid] * sd for jid in running)
        t_arrival = min((j.release for j in pending if j.release > now), default=float("inf"))
        dt = min(t_finish, max(0.0, t_arrival - now))
        if dt <= 0:
            dt = t_finish

        for jid in list(running):
            remaining[jid] -= dt / sd
        conc_time[n] = conc_time.get(n, 0.0) + dt
        slowdown_samples.extend([sd] * n)
        now += dt

        for jid in [j for j, left in remaining.items() if j in running and left <= 1e-6]:
            finish_at[jid] = now
            timeline.append((jid, start_at[jid], now))
            running.pop(jid)
            remaining.pop(jid)

        admit()

    by_id = {j.job_id: j for j in jobs}
    flows = [finish_at[j] - by_id[j].release for j in finish_at]
    waits = [start_at[j] - by_id[j].release for j in finish_at]
    flows_sorted = sorted(flows)
    p95_index = max(0, int(0.95 * len(flows_sorted)) - 1)
    total_time = sum(conc_time.values()) or 1.0

    return Result(
        policy=policy_name,
        makespan=max(finish_at.values()) - min(j.release for j in jobs),
        mean_flow=statistics.fmean(flows),
        p95_flow=flows_sorted[p95_index],
        max_wait=max(waits),
        average_slowdown=statistics.fmean(slowdown_samples) if slowdown_samples else 1.0,
        starvation_count=sum(1 for w in waits if w >= starvation_timeout),
        peak_concurrency=max(conc_time) if conc_time else 0,
        mean_concurrency=sum(k * v for k, v in conc_time.items()) / total_time,
        timeline=timeline,
    )
