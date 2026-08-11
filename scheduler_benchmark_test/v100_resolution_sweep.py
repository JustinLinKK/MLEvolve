"""Find image configs whose 4-5 concurrent copies truly saturate a V100.

Sweep 1 (v100_saturation_sweep.py) showed small tabular MLPs are CUDA-launch
bound, not compute bound: per-job throughput RISES with co-runners, which
cannot happen under genuine contention. Those configs can never fill the
device, so their earlier 99% `nvidia-smi dmon` reading was busy-time only.

This sweep attacks the problem from the other side. Convolution cost scales
with spatial resolution squared, and the earlier CNN sweep measured every
model at 224x224 sitting at 100% SM solo. Dropping to resolution r should
scale solo compute by (r / 224)^2, so r = 112 predicts roughly 25% solo, which
is what 4 concurrent jobs need to fill the device.

This is also faithful to MLEBench-Lite: aerial-cactus-identification images
are natively 32x32, so a reduced-resolution CNN is a solution an agent would
plausibly write, not a contrivance.

Criterion is metric-free, as in sweep 1. For N concurrent copies:

    aggregate(N) = N / slowdown(N)

Aggregate climbs while headroom remains and flattens once the device is full.
N* is the smallest N whose next step gains less than PLATEAU_GAIN. A config
meets the requirement iff N* is 4 or 5 AND per-step GPU time is well above
launch overhead, so the measurement reflects compute rather than Python.

Shape symbols:
    b = batch size, r = input resolution (height = width)
    ch = input channels, c = class count

Loss and optimizer are the plain ones, not special: CrossEntropyLoss and SGD
at lr = 1e-3, matching the earlier sweeps so slowdowns stay comparable.
"""

import json
import multiprocessing as mp
import os
import time

WARMUP = 20
MEASURE = 100
GATE = 1.15
PLATEAU_GAIN = 1.05
MAX_N = 6
# Below this per-step wall-clock the measurement is dominated by Python and
# CUDA launch overhead rather than by GPU compute, so N* would be meaningless.
MIN_STEP_MS = 5.0

# (label, timm_model, r, b). ch = 3 and c = 2 throughout: aerial-cactus-
# identification is binary classification on 3-channel images.
CONFIGS = [
    ("resnet18_r64_b64",       "resnet18",              64,  64),
    ("resnet18_r96_b64",       "resnet18",              96,  64),
    ("resnet18_r112_b64",      "resnet18",             112,  64),
    ("resnet18_r128_b64",      "resnet18",             128,  64),
    ("resnet18_r112_b128",     "resnet18",             112, 128),
    ("mobilenetv3_r112_b64",   "mobilenetv3_large_100", 112,  64),
    ("mobilenetv3_r128_b128",  "mobilenetv3_large_100", 128, 128),
    ("efficientnet_r96_b64",   "efficientnet_b0",       96,  64),
    ("efficientnet_r112_b64",  "efficientnet_b0",      112,  64),
    ("resnet50_r96_b64",       "resnet50",              96,  64),
]

NUM_CLASSES = 2
IN_CHANNELS = 3


def worker(model_name, r, b, rq):
    """Train one instance to steady state and report its per-job rate.

    Args:
        model_name : str, timm architecture name, constructed pretrained=False
        r          : int, input resolution (height and width)
        b          : int, batch size
        rq         : mp.Queue, receives
                     (steps_per_sec: float, step_ms: float, peak_vram_mib: float)

    Variables:
        x      : shape (b, ch, r, r), dim 0 = sample, dim 1 = RGB channel,
                 dims 2-3 = spatial height and width.
                 source = torch.randn, init = N(0, 1)
        target : shape (b,), dim 0 = sample, value = class index in [0, c).
                 source = torch.randint, init = uniform over [0, c)
        logits : shape (b, c), dim 0 = sample, dim 1 = class score.
                 source = model(x)
        loss   : shape (), scalar. source = CrossEntropyLoss(logits, target)
    """
    # CUDA_VISIBLE_DEVICES is inherited from the launcher on purpose, so this
    # sweep can run on a different physical GPU than a concurrent sweep.
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import timm
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    model = timm.create_model(
        model_name, pretrained=False, num_classes=NUM_CLASSES
    ).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(b, IN_CHANNELS, r, r, device=dev)          # x(b, ch, r, r)
    target = torch.randint(0, NUM_CLASSES, (b,), device=dev)   # target(b,)

    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)                       # logits(b, c) -> loss()
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


def run_n(model_name, r, b, n):
    """Run n concurrent copies; return (mean steps/sec, mean step_ms, peak VRAM MiB).

    All n workers start before any is timed, so each timed window overlaps the
    others. The returned rate divided into the solo rate gives the slowdown at
    concurrency n.
    """
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(model_name, r, b, q)) for _ in range(n)]
    for p in ps:
        p.start()
    res = [q.get(timeout=1200) for _ in range(n)]
    for p in ps:
        p.join(timeout=120)
    return (
        sum(x[0] for x in res) / len(res),
        sum(x[1] for x in res) / len(res),
        max(x[2] for x in res),
    )


def saturation_point(aggregate):
    """Smallest N whose next concurrency step gains less than PLATEAU_GAIN.

    Args:
        aggregate : dict[int, float], N -> aggregate throughput relative to solo

    Returns:
        int or None. None means throughput was still scaling at MAX_N, so the
        device never filled within the tested range.
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
    for label, model_name, r, b in CONFIGS:
        print(f"\n=== {label} ({model_name}, r={r}, b={b}) ===", flush=True)
        base, step_ms, vram = run_n(model_name, r, b, 1)
        launch_bound = step_ms < MIN_STEP_MS
        print(f"  N=1: {base:9.1f} steps/s  step={step_ms:6.2f} ms  "
              f"vram={vram:8.1f} MiB  launch_bound={launch_bound}", flush=True)

        aggregate = {1: 1.0}
        row = {"label": label, "model": model_name, "r": r, "b": b,
               "base_sps": round(base, 1), "step_ms": round(step_ms, 2),
               "vram_mib": round(vram, 1), "launch_bound": launch_bound}
        for n in range(2, MAX_N + 1):
            avg, _, _ = run_n(model_name, r, b, n)
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
        row["hits_target"] = (n_star in (4, 5)) and not launch_bound
        print(f"  --> saturation N* = {n_star}  target_4_or_5={row['hits_target']}",
              flush=True)
        rows.append(row)

    print("\n=== SUMMARY (want N* in {4, 5} and launch_bound False) ===")
    print(f"{'config':<24} {'step_ms':>8} {'vram':>9} {'N*':>4} "
          f"{'agg@4':>7} {'agg@5':>7} {'sd@4':>6} {'target':>7}")
    for r_ in rows:
        print(f"{r_['label']:<24} {r_['step_ms']:>8.2f} {r_['vram_mib']:>9.1f} "
              f"{str(r_['n_star']):>4} {r_.get('agg_n4', 0):>7.2f} "
              f"{r_.get('agg_n5', 0):>7.2f} {r_.get('sd_n4', 0):>6.2f} "
              f"{str(r_['hits_target']):>7}")
    print("\nJSON " + json.dumps(rows))
