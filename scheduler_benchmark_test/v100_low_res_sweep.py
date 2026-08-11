"""Bracket the image config whose 4-5 concurrent copies fill a V100.

Two earlier sweeps bracketed the answer from both sides:

  v100_saturation_sweep.py  tabular MLPs at 54 and 31 input features run at
                            ~490 steps/s regardless of hidden width or
                            concurrency. Launch-bound, never saturate.
  v100_resolution_sweep.py  resnet18 at r=64, b=64 costs 14.55 ms/step (real
                            GPU work) but collapses at N=3, so its solo demand
                            is already about half the device.

The target therefore sits BELOW r=64 b=64, not above it. Without MPS the V100
time-slices, so N jobs each demanding 1/N of the device is the best case; to
get N* of 4-5 a job must demand roughly 20-25% of the device, i.e. about half
the compute of resnet18 at r=64 b=64.

Convolution cost scales with resolution squared and linearly with batch, so
this sweep walks both knobs down from that reference point. r=32 is also the
native resolution of aerial-cactus-identification, so the small end of this
grid is what an agent would actually write for that MLEBench-Lite task.

Criterion is the same metric-free one used in both earlier sweeps:

    aggregate(N) = N / slowdown(N)

Aggregate climbs while headroom remains, flattens once the device is full.
N* is the smallest N whose next step gains less than PLATEAU_GAIN. To reject
the noise-driven false positive seen in sweep 1, a plateau must also HOLD:
every later N must stay within PLATEAU_TOLERANCE of the plateau value, so a
single dip followed by a recovery no longer counts as saturation.

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
# Each (config, N) point is measured this many times and the median is kept.
# Single measurements produced non-monotonic curves that made scoring unusable.
REPEATS = 3
MAX_N = 6
MIN_STEP_MS = 5.0

# (label, timm_model, r, b), walking compute DOWN from the resnet18 r64 b64
# reference that saturated at N=2. ch = 3 and c = 2 throughout:
# aerial-cactus-identification is binary classification on 3-channel images.
CONFIGS = [
    ("resnet18_r64_b32",       "resnet18",              64,  32),
    ("resnet18_r48_b64",       "resnet18",              48,  64),
    ("resnet18_r48_b32",       "resnet18",              48,  32),
    ("resnet18_r32_b64",       "resnet18",              32,  64),
    ("resnet18_r32_b128",      "resnet18",              32, 128),
    ("resnet34_r48_b32",       "resnet34",              48,  32),
    ("mobilenetv3_r64_b64",    "mobilenetv3_large_100", 64,  64),
    ("mobilenetv3_r48_b64",    "mobilenetv3_large_100", 48,  64),
    ("efficientnet_r48_b64",   "efficientnet_b0",       48,  64),
    ("efficientnet_r64_b32",   "efficientnet_b0",       64,  32),
]

NUM_CLASSES = 2
IN_CHANNELS = 3


def worker(model_name, r, b, rq):
    """Train one instance to steady state and report its per-job rate.

    Args:
        model_name : str, timm architecture name, constructed pretrained=False
                     so weights use each layer's default init
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
    try:
        torch.cuda.reset_peak_memory_stats(dev)
    except RuntimeError:
        # Under MPS this raises "invalid argument"; peak stats still work, the
        # reported peak is just not reset from a prior context in-process.
        pass
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


def meets_requirement(step_ms, slowdowns):
    """Decide whether 4-5 concurrent copies genuinely fill the device.

    Args:
        step_ms   : float, solo wall-clock per training step in milliseconds
        slowdowns : dict[int, float], N -> per-job slowdown relative to solo

    Returns:
        (ok: bool, reason: str)

    Two conditions must hold together, and the earlier sweeps failed one each:

      real work    step_ms >= MIN_STEP_MS. The tabular MLPs ran at ~2 ms/step
                   with throughput independent of both hidden width and
                   concurrency, so their 1.00x slowdown meant the GPU was idle,
                   not that packing was efficient.
      near-linear  slowdown at N=4 and N=5 within GATE. Aggregate throughput is
                   N / slowdown(N), so slowdown <= 1.15 at N=4 means the device
                   really is delivering about four jobs' worth of work. A
                   config whose aggregate ceilings at 2 has "plateaued" but
                   packs nothing, which is why a plateau test alone was wrong.
    """
    if step_ms < MIN_STEP_MS:
        return False, f"launch-bound ({step_ms:.2f} ms/step)"
    sd4 = slowdowns.get(4)
    sd5 = slowdowns.get(5)
    if sd4 is None or sd4 > GATE:
        return False, f"slowdown@4 = {sd4:.2f}x exceeds gate {GATE}"
    if sd5 is not None and sd5 > GATE:
        return False, f"slowdown@5 = {sd5:.2f}x exceeds gate {GATE}"
    return True, f"slowdown@4 = {sd4:.2f}x with {step_ms:.2f} ms/step"


def median_run(model_name, r, b, n):
    """Median of REPEATS measurements of run_n, to suppress run-to-run noise.

    Sweeps 1 and 2 produced non-monotonic aggregate curves (2.03 at N=4, 2.97
    at N=5, 2.03 at N=6) from single measurements, which is what made a plateau
    test unusable. Returns (steps_per_sec, step_ms, peak_vram_mib).
    """
    trials = [run_n(model_name, r, b, n) for _ in range(REPEATS)]
    trials.sort(key=lambda t: t[0])
    return trials[len(trials) // 2]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    rows = []
    for label, model_name, r, b in CONFIGS:
        print(f"\n=== {label} ({model_name}, r={r}, b={b}) ===", flush=True)
        base, step_ms, vram = median_run(model_name, r, b, 1)
        print(f"  N=1: {base:9.1f} steps/s  step={step_ms:6.2f} ms  "
              f"vram={vram:8.1f} MiB", flush=True)

        slowdowns = {1: 1.0}
        row = {"label": label, "model": model_name, "r": r, "b": b,
               "base_sps": round(base, 1), "step_ms": round(step_ms, 2),
               "vram_mib": round(vram, 1)}
        for n in range(2, MAX_N + 1):
            avg, _, _ = median_run(model_name, r, b, n)
            sd = base / avg
            slowdowns[n] = sd
            row[f"sd_n{n}"] = round(sd, 3)
            row[f"agg_n{n}"] = round(n / sd, 3)
            print(f"  N={n}: {avg:9.1f} steps/s  slowdown={sd:5.2f}x  "
                  f"aggregate={n / sd:5.2f}x  {'PASS' if sd <= GATE else 'FAIL'}",
                  flush=True)

        ok, reason = meets_requirement(step_ms, slowdowns)
        row["hits_target"] = ok
        row["reason"] = reason
        print(f"  --> fills GPU at N=4-5: {ok}  ({reason})", flush=True)
        rows.append(row)

    print("\n=== SUMMARY (want real GPU work AND slowdown <= gate at N=4-5) ===")
    print(f"{'config':<24} {'step_ms':>8} {'vram':>9} {'sd@4':>6} {'sd@5':>6} "
          f"{'agg@4':>7} {'agg@5':>7} {'target':>7}")
    for r_ in rows:
        print(f"{r_['label']:<24} {r_['step_ms']:>8.2f} {r_['vram_mib']:>9.1f} "
              f"{r_.get('sd_n4', 0):>6.2f} {r_.get('sd_n5', 0):>6.2f} "
              f"{r_.get('agg_n4', 0):>7.2f} {r_.get('agg_n5', 0):>7.2f} "
              f"{str(r_['hits_target']):>7}")
    print("\nJSON " + json.dumps(rows))
