"""Confirm the packing candidate with high-repeat measurements.

The exploratory sweep found resnet18 at r=64, b=32 under MPS scaling to an
aggregate of 4.63x at N=5 with 1.08x per-job slowdown, then falling to 3.55x at
N=6. That shape is real saturation at N=5. One point disagreed with its
neighbours, N=4 reading 1.21x between 0.99x at N=3 and 1.08x at N=5, so the
whole curve is re-measured here with many more repeats before it is used.

The same config is measured with and without MPS in one process run per mode,
because the entire claim depends on MPS: without it the V100 time-slices whole
contexts and aggregate throughput ceilings near 2.7 regardless of job size.

Run twice, once per mode:
    CUDA_VISIBLE_DEVICES=0 python3 v100_confirm_winner.py nomps
    CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_g3 \
        CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_g3 python3 v100_confirm_winner.py mps

Shape symbols:
    b = batch size, r = input resolution (height = width)
    ch = input channels, c = class count

Loss and optimizer are the plain ones, not special: CrossEntropyLoss and SGD
at lr = 1e-3, matching the sweeps so slowdowns stay comparable.
"""

import json
import multiprocessing as mp
import os
import statistics
import sys
import time

WARMUP = 50
# A 100-step window ran ~1.5 s and gave a solo coefficient of variation of 13%,
# which cannot resolve a 1.15 gate. 600 steps puts each sample near 9 s so
# scheduler jitter and clock ramp average out.
MEASURE = 600
GATE = 1.15
REPEATS = 7
LEVELS = (1, 2, 3, 4, 5, 6)
# Reject the whole measurement if solo repeatability is worse than this, the
# stability bar the experiment design already requires of solo timings.
MAX_SOLO_CV = 0.10

MODEL = os.environ.get("CONFIRM_MODEL", "resnet18")
RESOLUTION = int(os.environ.get("CONFIRM_R", "64"))
BATCH = int(os.environ.get("CONFIRM_B", "32"))
NUM_CLASSES = 2
IN_CHANNELS = 3


def worker(rq):
    """Train one resnet18 instance to steady state and report its rate.

    Args:
        rq : mp.Queue, receives
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
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import timm
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    try:
        torch.cuda.reset_peak_memory_stats(dev)
    except RuntimeError:
        # Raised under MPS; peak stats still report correctly afterwards.
        pass
    model = timm.create_model(
        MODEL, pretrained=False, num_classes=NUM_CLASSES
    ).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(BATCH, IN_CHANNELS, RESOLUTION, RESOLUTION, device=dev)
    target = torch.randint(0, NUM_CLASSES, (BATCH,), device=dev)

    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)
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


def run_n(n):
    """Run n concurrent copies; return (mean steps/sec, mean step_ms, peak VRAM)."""
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(q,)) for _ in range(n)]
    for p in ps:
        p.start()
    res = [q.get(timeout=1200) for _ in range(n)]
    for p in ps:
        p.join(timeout=120)
    return (
        sum(v[0] for v in res) / len(res),
        sum(v[1] for v in res) / len(res),
        max(v[2] for v in res),
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nomps"
    mp.set_start_method("spawn", force=True)

    print(f"mode={mode}  model={MODEL}  r={RESOLUTION}  b={BATCH}  "
          f"repeats={REPEATS}", flush=True)

    samples = {}
    for n in LEVELS:
        rates = []
        step_ms = []
        vram = 0.0
        for _ in range(REPEATS):
            rate, ms, mib = run_n(n)
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
              f"mean {samples[n]['mean_sps']:8.2f}  CV {cv:5.1%}  "
              f"step {samples[n]['median_step_ms']:6.2f} ms  "
              f"vram {vram:8.1f} MiB", flush=True)

    base = samples[1]["median_sps"]
    print(f"\n{'N':>3} {'slowdown':>9} {'aggregate':>10} {'gate':>6}")
    rows = []
    for n in LEVELS:
        sd = base / samples[n]["median_sps"]
        agg = n / sd
        rows.append({"n": n, "slowdown": round(sd, 3), "aggregate": round(agg, 3),
                     "cv": round(samples[n]["stdev_sps"] / samples[n]["mean_sps"], 4)})
        print(f"{n:>3} {sd:>9.3f} {agg:>10.3f} "
              f"{'PASS' if sd <= GATE else 'FAIL':>6}")

    aggs = {r["n"]: r["aggregate"] for r in rows}
    peak_n = max(aggs, key=lambda k: aggs[k])
    sd_peak = base / samples[peak_n]["median_sps"]
    solo_cv = samples[1]["stdev_sps"] / samples[1]["mean_sps"]

    # Every condition must hold. Each one corresponds to a way an earlier
    # measurement in this search produced a false positive.
    checks = {
        # tabular MLPs read 1.00x slowdown only because the GPU sat idle
        "real_gpu_work": samples[1]["median_step_ms"] >= 5.0,
        # a config whose aggregate ceilings at 2 has plateaued but packs nothing
        "peak_at_4_or_5": peak_n in (4, 5),
        "within_gate_at_peak": sd_peak <= GATE,
        # a 13% solo CV cannot resolve a 1.15 gate at all
        "solo_stable": solo_cv <= MAX_SOLO_CV,
    }
    fills = all(checks.values())

    print(f"\npeak aggregate at N={peak_n} ({aggs[peak_n]:.3f}x), "
          f"slowdown there {sd_peak:.3f}x, solo CV {solo_cv:.1%}")
    for name, ok in checks.items():
        print(f"  {name:<22} {'PASS' if ok else 'FAIL'}")
    print(f"fills GPU at N=4-5: {fills}")
    print("\nJSON " + json.dumps({"mode": mode, "model": MODEL, "r": RESOLUTION,
                                  "b": BATCH, "rows": rows, "peak_n": peak_n,
                                  "solo_cv": round(solo_cv, 4),
                                  "checks": checks, "fills": fills}))
