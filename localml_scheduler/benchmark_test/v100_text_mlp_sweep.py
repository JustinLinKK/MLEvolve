"""Test whether a FLOP-bound text MLP packs 4-5 jobs on a V100.

Everything measured so far failed for one of two reasons. Tabular MLPs at 54
input features ran ~2 ms/step with throughput independent of hidden width, so
they were CUDA-launch bound and their 1.00x slowdown meant an idle GPU.
resnet18 at r=32, 48 and 64 ran 12.9, 13.0 and 15.3 ms/step respectively
despite r=32 having a quarter of r=64's convolution work, so those were
dominated by fixed per-layer overhead across ~20 conv and norm layers rather
than by convolution FLOPs. Both ceiling near an aggregate of 3.

A text MLP is the remaining shape that can be genuinely FLOP-bound without
many kernel launches. MLEBench-Lite text tasks such as
jigsaw-toxic-comment-classification-challenge and
spooky-author-identification are normally attacked with a TF-IDF vector of a
few thousand dimensions fed to a shallow MLP, so the first Linear is one large
GEMM of shape (b, f) x (f, w). Few layers, large kernels, which is the inverse
of every configuration that has failed.

The grid walks width and batch to find a config whose solo demand is near a
fifth of the device, since that is what 4-5 concurrent jobs need.

Criterion matches v100_confirm_winner.py: real GPU work per step, per-job
slowdown within the gate at N=4-5, aggregate peaking at N=4-5, and solo
repeatability good enough to resolve the gate at all.

Shape symbols:
    b = batch size, f = input features (TF-IDF vocabulary size)
    w = hidden width, d = hidden layer count, c = class count

Loss and optimizer are the plain ones, not special: CrossEntropyLoss and SGD
at lr = 1e-3, matching the other sweeps so slowdowns stay comparable.
"""

import json
import multiprocessing as mp
import os
import statistics
import time

WARMUP = 50
MEASURE = 600
GATE = 1.15
# Screen the grid cheaply at REPEATS=1, then re-run the survivor at a high
# repeat count. A single measurement cannot resolve the 1.15 gate, so a screen
# pass is a shortlist, never a verdict.
REPEATS = int(os.environ.get("TEXT_REPEATS", "5"))
LEVELS = (1, 2, 3, 4, 5, 6)
MAX_SOLO_CV = 0.10

# Tesla V100-SXM2 fp32 peak. Volta has no TF32 path, so this is the ceiling
# these dense GEMMs are measured against.
V100_FP32_PEAK_TFLOPS = 15.7
# A job must actually be doing arithmetic to count. Step time is NOT a valid
# proxy for this: f=5000 (overhead-bound, GPU mostly idle) and f=20000
# (compute-bound at peak, GPU completely full) both measured 1.9 ms/step, so a
# step-time threshold misclassified both. Achieved FLOP rate separates them.
MIN_SOLO_DEMAND = 0.05
# Above this a single job already fills the device and packing is pointless.
MAX_SOLO_DEMAND = 0.35


def step_gflop(f, w, d, b, c):
    """Analytic FLOPs per training step for the MLP under test, in GFLOP.

    Args:
        f : int, input feature count
        w : int, hidden width
        d : int, hidden layer count
        b : int, batch size
        c : int, class count

    Returns:
        float, billions of floating point operations per optimizer step.

    A Linear of shape (in, out) at batch b costs 2*b*in*out in the forward
    pass, and the backward pass adds an input-gradient and a weight-gradient
    GEMM of the same size, so one training step costs 6*b*in*out. BatchNorm and
    ReLU are elementwise and negligible beside the GEMMs.
    """
    flop = 6.0 * b * f * w                      # first Linear, the large GEMM
    flop += 6.0 * b * w * w * max(0, d - 1)     # remaining hidden Linears
    flop += 6.0 * b * w * c                     # classifier head
    return flop / 1e9

# (label, f, w, d, b, c). f = 5000 is a typical TF-IDF truncation; c = 3 matches
# spooky-author-identification, c = 2 matches the toxic-comment binary head.
# Measured anchors, all at 1.9 ms/step yet spanning the whole range of device
# demand, which is why step time had to be abandoned as the work test:
#
#   f=5000  w=256 b=512    4.1 GFLOP/step   2.15 TFLOP/s    14% of peak
#   f=5000  w=384 b=512    6.4 GFLOP/step   3.40 TFLOP/s    22% of peak
#   f=20000 w=512 b=512   31.5 GFLOP/step  16.10 TFLOP/s   100% of peak
#
# Four to five concurrent jobs fill the device when each demands roughly 20-25%
# of it, so this grid brackets that band. These are ordinary TF-IDF shapes for
# the MLEBench-Lite text tasks (jigsaw-toxic-comment-classification-challenge,
# spooky-author-identification), not contrived sizes.
CONFIGS = [
    ("text_f5000_w256_d2_b512",  5000,  256, 2,  512, 3),
    ("text_f5000_w384_d2_b512",  5000,  384, 2,  512, 3),
    ("text_f5000_w512_d2_b512",  5000,  512, 2,  512, 3),
    ("text_f8000_w384_d2_b512",  8000,  384, 2,  512, 3),
    ("text_f5000_w384_d2_b1024", 5000,  384, 2, 1024, 3),
    ("text_f10000_w256_d2_b512", 10000, 256, 2,  512, 2),
]


def build(f, w, d, c, dev):
    """Build the MLP under test.

    Args:
        f   : int, input feature count (TF-IDF vocabulary size)
        w   : int, hidden width
        d   : int, hidden layer count
        c   : int, class count
        dev : torch.device, target device

    Returns:
        nn.Sequential mapping x(b, f) -> logits(b, c), via
        [Linear(., w) -> BatchNorm1d(w) -> ReLU] x d -> Linear(w, c).
        All layers use PyTorch default init (Kaiming-uniform weights and
        uniform bias for Linear; weight=1, bias=0 for BatchNorm1d).
        The first Linear carries the (b, f) x (f, w) GEMM this test is about.
    """
    import torch.nn as nn
    layers = []
    in_dim = f
    for _ in range(d):
        layers += [nn.Linear(in_dim, w), nn.BatchNorm1d(w), nn.ReLU()]
        in_dim = w
    layers += [nn.Linear(in_dim, c)]
    return nn.Sequential(*layers).to(dev)


def worker(f, w, d, b, c, rq):
    """Train one instance to steady state and report its per-job rate.

    Args:
        f  : int, input feature count
        w  : int, hidden width
        d  : int, hidden layer count
        b  : int, batch size
        c  : int, class count
        rq : mp.Queue, receives
             (steps_per_sec: float, step_ms: float, peak_vram_mib: float)

    Variables:
        x      : shape (b, f), dim 0 = sample, dim 1 = TF-IDF feature.
                 source = torch.randn, init = N(0, 1)
        target : shape (b,), dim 0 = sample, value = class index in [0, c).
                 source = torch.randint, init = uniform over [0, c)
        logits : shape (b, c), dim 0 = sample, dim 1 = class score.
                 source = model(x)
        loss   : shape (), scalar. source = CrossEntropyLoss(logits, target)
    """
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    try:
        torch.cuda.reset_peak_memory_stats(dev)
    except RuntimeError:
        # Raised under MPS; peak stats still report correctly afterwards.
        pass
    model = build(f, w, d, c, dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(b, f, device=dev)                      # x(b, f)
    target = torch.randint(0, c, (b,), device=dev)         # target(b,)

    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)                   # logits(b, c) -> loss()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rq.put((
        MEASURE / elapsed,
        1000.0 * elapsed / MEASURE,
        torch.cuda.max_memory_allocated(dev) / (1024 ** 2),
    ))


def run_n(f, w, d, b, c, n):
    """Run n concurrent copies; return (mean steps/sec, mean step_ms, peak VRAM)."""
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(f, w, d, b, c, q)) for _ in range(n)]
    for p in ps:
        p.start()
    res = [q.get(timeout=1800) for _ in range(n)]
    for p in ps:
        p.join(timeout=120)
    return (
        sum(v[0] for v in res) / len(res),
        sum(v[1] for v in res) / len(res),
        max(v[2] for v in res),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    results = []
    for label, f, w, d, b, c in CONFIGS:
        print(f"\n=== {label} (f={f}, w={w}, d={d}, b={b}, c={c}) ===", flush=True)
        samples = {}
        for n in LEVELS:
            rates, step_ms, vram = [], [], 0.0
            for _ in range(REPEATS):
                rate, ms, mib = run_n(f, w, d, b, c, n)
                rates.append(rate)
                step_ms.append(ms)
                vram = max(vram, mib)
            samples[n] = {
                "median_sps": statistics.median(rates),
                "mean_sps": statistics.fmean(rates),
                "stdev_sps": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "median_step_ms": statistics.median(step_ms),
                "vram_mib": vram,
            }
            cv = samples[n]["stdev_sps"] / samples[n]["mean_sps"]
            print(f"  N={n}: median {samples[n]['median_sps']:8.2f} steps/s  "
                  f"CV {cv:5.1%}  step {samples[n]['median_step_ms']:6.2f} ms  "
                  f"vram {vram:8.1f} MiB", flush=True)

        base = samples[1]["median_sps"]
        aggs, sds = {}, {}
        for n in LEVELS:
            sds[n] = base / samples[n]["median_sps"]
            aggs[n] = n / sds[n]
        peak_n = max(aggs, key=lambda k: aggs[k])
        solo_cv = samples[1]["stdev_sps"] / samples[1]["mean_sps"]

        gflop = step_gflop(f, w, d, b, c)
        solo_tflops = gflop * base / 1000.0
        solo_demand = solo_tflops / V100_FP32_PEAK_TFLOPS
        # Fraction of the device the whole pack is driving at its best point.
        peak_demand = solo_demand * aggs[peak_n]

        # The operative number is the largest pack that stays inside the gate,
        # not the raw aggregate peak. Aggregate can keep creeping up past the
        # gate by making every job slower, which is not a usable placement:
        # w=384 rose from 4.49 at N=5 to 4.96 at N=6 only by pushing per-job
        # slowdown from 1.11 to 1.21.
        gated = [n for n in LEVELS if sds[n] <= GATE]
        n_gate = max(gated) if gated else 1
        demand_at_gate = solo_demand * aggs[n_gate]

        checks = {
            # real arithmetic, not an idle GPU spinning on kernel launches
            "does_real_work": solo_demand >= MIN_SOLO_DEMAND,
            # one job must not already fill the device, or packing is moot
            "leaves_headroom": solo_demand <= MAX_SOLO_DEMAND,
            # 4-5 jobs must be schedulable together within the gate
            "gate_holds_at_4_or_5": n_gate in (4, 5),
            # and that pack must actually fill the device
            "fills_device": demand_at_gate >= 0.85,
            "solo_stable": solo_cv <= MAX_SOLO_CV,
        }
        fills = all(checks.values())
        print(f"  {gflop:.2f} GFLOP/step  solo {solo_tflops:.2f} TFLOP/s  "
              f"solo demand {solo_demand:.1%}  demand at peak {peak_demand:.1%}",
              flush=True)
        print("  " + "  ".join(f"N{n} sd {sds[n]:.2f} agg {aggs[n]:.2f}"
                               for n in LEVELS), flush=True)
        print(f"  peak N={peak_n} agg {aggs[peak_n]:.2f} sd {sds[peak_n]:.2f} "
              f"solo_cv {solo_cv:.1%}", flush=True)
        print(f"  largest pack within gate: N={n_gate} agg {aggs[n_gate]:.2f} "
              f"sd {sds[n_gate]:.2f} driving {demand_at_gate:.1%} of peak",
              flush=True)
        print(f"  --> fills GPU at N=4-5: {fills}  {checks}", flush=True)
        results.append({"label": label, "f": f, "w": w, "d": d, "b": b,
                        "step_ms": round(samples[1]["median_step_ms"], 2),
                        "gflop_per_step": round(gflop, 3),
                        "solo_tflops": round(solo_tflops, 3),
                        "solo_demand": round(solo_demand, 4),
                        "peak_demand": round(peak_demand, 4),
                        "slowdowns": {n: round(sds[n], 3) for n in LEVELS},
                        "aggregates": {n: round(aggs[n], 3) for n in LEVELS},
                        "peak_n": peak_n, "n_gate": n_gate,
                        "demand_at_gate": round(demand_at_gate, 4),
                        "solo_cv": round(solo_cv, 4),
                        "checks": checks, "fills": fills})

    print("\n=== SUMMARY (want solo demand near 20-25% and peak N of 4-5) ===")
    print(f"{'config':<26} {'GFLOP':>7} {'solo%':>7} {'peak%':>7} {'peak_N':>7} "
          f"{'agg@peak':>9} {'sd@4':>6} {'sd@5':>6} {'fills':>6}")
    for r in results:
        print(f"{r['label']:<26} {r['gflop_per_step']:>7.2f} "
              f"{r['solo_demand']:>6.1%} {r['peak_demand']:>6.1%} "
              f"{r['peak_n']:>7} {r['aggregates'][r['peak_n']]:>9.2f} "
              f"{r['slowdowns'][4]:>6.2f} {r['slowdowns'][5]:>6.2f} "
              f"{str(r['fills']):>6}")
    print("\nJSON " + json.dumps(results))
