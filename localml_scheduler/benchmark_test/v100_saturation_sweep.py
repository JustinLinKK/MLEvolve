"""Find MLEBench-Lite tabular configs that TRULY saturate a V100 at N=4-5.

The earlier sweep (v100_tabular_sweep.py) scored configs by `nvidia-smi dmon`
SM%, which is a busy-time indicator, not a capacity measure. Four tiny MLPs
report 99% SM while each kernel occupies a handful of the 80 SMs, so 99% there
did not mean the device was full. The proof is that per-job slowdown stayed at
1.00x: a compute-saturated device cannot absorb a 4th job for free.

This sweep uses an unambiguous, metric-free criterion instead. For N
concurrent copies of the same job, define aggregate throughput relative to
solo:

    aggregate(N) = N / slowdown(N)

If the device still has headroom, aggregate(N) keeps climbing with N. Once the
device is full, adding another job buys nothing and aggregate(N) flattens. The
saturation point N* is the smallest N whose next step gains less than
PLATEAU_GAIN. A config "fills the GPU with 4-5 jobs" iff N* is 4 or 5.

Shape symbols:
    b = batch size, f = input features, w = hidden width
    d = hidden layer count, c = class count

Loss and optimizer are the plain ones, not special: CrossEntropyLoss and SGD
at lr = 1e-3, matching the earlier sweep so slowdowns stay comparable.
"""

import json
import multiprocessing as mp
import os
import time

WARMUP = 50
MEASURE = 300
GATE = 1.15
# Aggregate throughput must gain at least this factor to count as "still scaling".
PLATEAU_GAIN = 1.05
MAX_N = 8

# (label, f, w, d, b, c). f/c drawn from the two MLEBench-Lite tabular tasks:
#   tabular-playground-series-dec-2021 : 54 features, 7 classes
#   tabular-playground-series-may-2022 : 31 features, 2 classes
# Widths fill the gap between w256 (never saturates) and w512 (saturates at
# N=2-3), which is where an N*=4-5 config must live if one exists.
CONFIGS = [
    ("tps_w256_d2_b1024", 54, 256, 2, 1024, 7),
    ("tps_w320_d2_b1024", 54, 320, 2, 1024, 7),
    ("tps_w384_d2_b1024", 54, 384, 2, 1024, 7),
    ("tps_w448_d2_b1024", 54, 448, 2, 1024, 7),
    ("tps_w512_d2_b1024", 54, 512, 2, 1024, 7),
    ("tps_w256_d3_b2048", 54, 256, 3, 2048, 7),
    ("tps_w320_d3_b2048", 54, 320, 3, 2048, 7),
    ("tps_w384_d3_b2048", 54, 384, 3, 2048, 7),
    ("may_w384_d2_b1024", 31, 384, 2, 1024, 2),
    ("may_w512_d3_b2048", 31, 512, 3, 2048, 2),
]


def build(f, w, d, c, dev):
    """Build the MLP under test.

    Args:
        f   : int, input feature count
        w   : int, hidden width
        d   : int, hidden layer count
        c   : int, class count
        dev : torch.device, target device

    Returns:
        nn.Sequential mapping x(b, f) -> logits(b, c), via
        [Linear(., w) -> BatchNorm1d(w) -> ReLU] x d -> Linear(w, c).
        All layers use PyTorch default init (Kaiming-uniform weights,
        uniform bias for Linear; weight=1 bias=0 for BatchNorm1d).
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
    """Train one instance to steady state and report its solo-comparable rate.

    Args:
        f  : int, input feature count
        w  : int, hidden width
        d  : int, hidden layer count
        b  : int, batch size
        c  : int, class count
        rq : mp.Queue, receives (steps_per_sec: float, peak_vram_mib: float)

    Variables:
        x      : shape (b, f), dim 0 = sample, dim 1 = feature.
                 source = torch.randn, init = N(0, 1)
        target : shape (b,),   dim 0 = sample, value = class index in [0, c).
                 source = torch.randint, init = uniform over [0, c)
        logits : shape (b, c), dim 0 = sample, dim 1 = class score.
                 source = model(x)
        loss   : shape (), scalar. source = CrossEntropyLoss(logits, target)
    """
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
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
    rq.put((MEASURE / elapsed, torch.cuda.max_memory_allocated(dev) / (1024 ** 2)))


def run_n(f, w, d, b, c, n):
    """Run n concurrent copies; return (mean steps/sec per job, peak VRAM MiB).

    All n workers are started before any is measured, so the timed window of
    each overlaps the others. The returned rate is the per-job rate, which
    divided into the solo rate gives that config's slowdown at concurrency n.
    """
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(f, w, d, b, c, q)) for _ in range(n)]
    for p in ps:
        p.start()
    res = [q.get(timeout=900) for _ in range(n)]
    for p in ps:
        p.join(timeout=60)
    rates = [r[0] for r in res]
    return sum(rates) / len(rates), max(r[1] for r in res)


def saturation_point(aggregate):
    """Smallest N whose next concurrency step gains less than PLATEAU_GAIN.

    Args:
        aggregate : dict[int, float], N -> aggregate throughput relative to solo

    Returns:
        int or None. None means throughput was still scaling at MAX_N, i.e.
        the device never filled within the tested range.
    """
    for n in sorted(aggregate):
        nxt = aggregate.get(n + 1)
        if nxt is None:
            return None
        if nxt / aggregate[n] < PLATEAU_GAIN:
            return n
    return None


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    rows = []
    for label, f, w, d, b, c in CONFIGS:
        print(f"\n=== {label} (f={f}, w={w}, d={d}, b={b}, c={c}) ===", flush=True)
        base, vram = run_n(f, w, d, b, c, 1)
        print(f"  N=1: {base:9.1f} steps/s  vram={vram:7.1f} MiB", flush=True)

        aggregate = {1: 1.0}
        row = {"label": label, "f": f, "w": w, "depth": d, "b": b,
               "base_sps": round(base, 1), "vram_mib": round(vram, 1)}
        for n in range(2, MAX_N + 1):
            avg, _ = run_n(f, w, d, b, c, n)
            sd = base / avg
            agg = n / sd
            aggregate[n] = agg
            row[f"sd_n{n}"] = round(sd, 3)
            row[f"agg_n{n}"] = round(agg, 3)
            print(f"  N={n}: {avg:9.1f} steps/s  slowdown={sd:5.2f}x  "
                  f"aggregate={agg:5.2f}x  {'PASS' if sd <= GATE else 'FAIL'}",
                  flush=True)

        n_star = saturation_point(aggregate)
        row["n_star"] = n_star
        row["hits_target"] = n_star in (4, 5)
        print(f"  --> saturation N* = {n_star}  target_4_or_5={row['hits_target']}",
              flush=True)
        rows.append(row)

    print("\n=== SUMMARY (want N* in {4, 5}) ===")
    print(f"{'config':<20} {'vram':>8} {'N*':>4} {'agg@4':>7} {'agg@5':>7} "
          f"{'sd@4':>6} {'sd@5':>6} {'target':>7}")
    for r in rows:
        print(f"{r['label']:<20} {r['vram_mib']:>8.1f} {str(r['n_star']):>4} "
              f"{r.get('agg_n4', 0):>7.2f} {r.get('agg_n5', 0):>7.2f} "
              f"{r.get('sd_n4', 0):>6.2f} {r.get('sd_n5', 0):>6.2f} "
              f"{str(r['hits_target']):>7}")
    print("\nJSON " + json.dumps(rows))
