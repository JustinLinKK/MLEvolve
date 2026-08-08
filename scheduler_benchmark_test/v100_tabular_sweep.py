"""Sweep tabular/text-scale models on V100 for the 4-5 job packing target.

MLEBench-Lite non-image tasks (tabular-playground-series-dec-2021,
spooky-author-identification, random-acts-of-pizza, detecting-insults)
produce small MLPs rather than CNNs. Those are the candidates that can
leave SM headroom at N=1 and still fill the device at N=4-5.

For each config: measure solo SM% (nvidia-smi dmon), packed SM% at N=4,
and per-job slowdown at N=1..5.

Shape symbols:
    b = batch size, f = input features, w = hidden width, c = classes

Variables:
    x      : shape (b, f), source = torch.randn, init = N(0,1)
    target : shape (b,),   source = torch.randint, init = uniform [0, c)
    logits : shape (b, c), model output
    loss   : scalar, CrossEntropy(logits, target)
"""

import json
import multiprocessing as mp
import os
import re
import subprocess
import time

WARMUP = 50
MEASURE = 300
GATE = 1.15

# (label, f, w, depth, b, c) drawn from MLEBench-Lite non-image task shapes.
# tps-dec-2021: 54 features / 7 classes. text tasks: TF-IDF vectors -> binary or 3-class.
CONFIGS = [
    ("tps_w128_d1_b512",   54,   128, 1,  512, 7),
    ("tps_w256_d2_b1024",  54,   256, 2, 1024, 7),
    ("tps_w512_d3_b2048",  54,   512, 3, 2048, 7),
    ("tps_w1024_d4_b4096", 54,  1024, 4, 4096, 7),
    ("tps_w2048_d4_b4096", 54,  2048, 4, 4096, 7),
    ("tps_w2048_d6_b8192", 54,  2048, 6, 8192, 7),
    ("text_w512_d3_b1024", 5000, 512, 3, 1024, 3),
    ("text_w1024_d3_b2048",5000,1024, 3, 2048, 3),
]


def build(f, w, depth, c, dev):
    """Build MLP: x(b, f) -> [Linear(.,w)+BN(w)+ReLU] x depth -> Linear(w, c) -> logits(b, c)."""
    import torch.nn as nn
    layers = []
    in_dim = f
    for _ in range(depth):
        layers += [nn.Linear(in_dim, w), nn.BatchNorm1d(w), nn.ReLU()]
        in_dim = w
    layers += [nn.Linear(in_dim, c)]
    return nn.Sequential(*layers).to(dev)


def worker(rank, f, w, depth, b, c, sustain_s, rq):
    """Train one instance; report steps/sec and peak VRAM.

    Args:
        rank      : int, worker index
        f         : int, input feature count
        w         : int, hidden width
        depth     : int, hidden layer count
        b         : int, batch size
        c         : int, class count
        sustain_s : float, extra seconds to keep training after timing (for SM sampling)
        rq        : mp.Queue, receives (steps_per_sec, peak_vram_mib)

    Variables:
        x      : shape (b, f), source = torch.randn, init = N(0,1)
        target : shape (b,),   source = torch.randint, init = uniform [0, c)
        logits : shape (b, c), model output
        loss   : scalar, CrossEntropy(logits, target)
    """
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    import torch.nn as nn

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    model = build(f, w, depth, c, dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(b, f, device=dev)                      # x(b, f)
    target = torch.randint(0, c, (b,), device=dev)         # target(b,)

    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)                   # logits(b, c) -> scalar
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
    peak_mib = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    rq.put((MEASURE / elapsed, peak_mib))

    end = time.time() + sustain_s
    while time.time() < end:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), target)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()


def sample_sm(seconds):
    """Average SM% over `seconds` using nvidia-smi dmon (correct on Volta)."""
    proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-i", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    time.sleep(seconds)
    proc.terminate()
    out, _ = proc.communicate(timeout=10)
    vals = []
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"\d+", parts[1]):
            vals.append(int(parts[1]))
    return sum(vals) / len(vals) if vals else None


def run_n(f, w, depth, b, c, n, *, sample=False):
    """Run n workers; return (avg steps/sec, peak VRAM MiB, sm_pct or None)."""
    sustain = 12.0 if sample else 0.0
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(i, f, w, depth, b, c, sustain, q)) for i in range(n)]
    for p in ps:
        p.start()
    res = [q.get(timeout=600) for _ in range(n)]
    sm = sample_sm(8) if sample else None
    for p in ps:
        p.join(timeout=60)
    rates = [r[0] for r in res]
    return sum(rates) / len(rates), max(r[1] for r in res), sm


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    rows = []
    for label, f, w, depth, b, c in CONFIGS:
        print(f"\n=== {label} (f={f}, w={w}, d={depth}, b={b}, c={c}) ===", flush=True)
        base, vram, sm1 = run_n(f, w, depth, b, c, 1, sample=True)
        print(f"  N=1: {base:8.1f} steps/s  vram={vram:7.1f} MiB  SM={sm1}%", flush=True)
        row = {"label": label, "f": f, "w": w, "depth": depth, "b": b,
               "base_sps": round(base, 1), "vram_mib": round(vram, 1), "sm_n1": sm1}
        for n in (2, 3, 4, 5):
            avg, vram_n, sm_n = run_n(f, w, depth, b, c, n, sample=(n in (4, 5)))
            sd = base / avg
            row[f"sd_n{n}"] = round(sd, 3)
            if sm_n is not None:
                row[f"sm_n{n}"] = sm_n
            tag = "PASS" if sd <= GATE else "FAIL"
            smtxt = f"  SM={sm_n}%" if sm_n is not None else ""
            print(f"  N={n}: {avg:8.1f} steps/s  slowdown={sd:.2f}x {tag}{smtxt}", flush=True)
        rows.append(row)

    print("\n=== SUMMARY (target: SM_n4 near 100 and sd_n4 <= 1.15) ===")
    print(f"{'config':<22} {'vram':>8} {'SM@1':>5} {'SM@4':>5} {'SM@5':>5} {'sd@4':>6} {'sd@5':>6}")
    for r in rows:
        print(f"{r['label']:<22} {r['vram_mib']:>8.1f} {str(r.get('sm_n1')):>5} "
              f"{str(r.get('sm_n4')):>5} {str(r.get('sm_n5')):>5} "
              f"{r['sd_n4']:>6.2f} {r['sd_n5']:>6.2f}")
    print("\nJSON " + json.dumps(rows))
