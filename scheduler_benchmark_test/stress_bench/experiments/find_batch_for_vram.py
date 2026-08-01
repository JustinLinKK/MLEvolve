"""Per-model: pick the largest 2^n batch whose peak VRAM is <= target (so ~3 jobs fill the
23 GiB A10), and measure its step time for 30-min sizing."""
from __future__ import annotations
import json, time, sys, argparse, importlib.util, statistics as s
from pathlib import Path
import torch, torch.nn.functional as F

REPO = Path("/root/downeyflyfan/perfseer_test/exp_run")
sys.path.insert(0, str(REPO))
FIX = REPO / "scheduler_benchmark_test/fixtures/stress_test_data_v1.0"
MODEL_SOURCE = FIX / "model_source.py"

def load_build():
    spec = importlib.util.spec_from_file_location("ms", MODEL_SOURCE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.build_model

def autocast(prec, dev):
    if prec == "bf16_amp": return torch.autocast(dev.type, dtype=torch.bfloat16)
    if prec == "fp16_amp": return torch.autocast(dev.type, dtype=torch.float16)
    return torch.autocast(dev.type, enabled=False)

def measure(build, kwargs, base_shape, bs, prec, dev, warmup, iters):
    torch.backends.cuda.matmul.allow_tf32 = (prec == "tf32")
    shape = [bs if v == "$batch" else v for v in base_shape]
    model = build(**kwargs).to(dev); model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler(dev.type, enabled=(prec == "fp16_amp"))
    x = torch.randn(shape, device=dev)
    def step():
        opt.zero_grad(set_to_none=True)
        with autocast(prec, dev):
            out = model(x); loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
        if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
    torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(warmup): step()
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    for _ in range(iters): step()
    torch.cuda.synchronize(dev); dt = (time.perf_counter()-t0)/iters*1000.0
    peak = torch.cuda.max_memory_reserved(dev)/(1024*1024)
    del model, x, opt, scaler; torch.cuda.empty_cache()
    return dt, peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-peak-mib", type=float, default=6500)
    ap.add_argument("--candidates", default="128,256,512,1024,2048,4096,8192,16384,32768,65536,131072")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda:0"); build = load_build()
    jobs = json.loads((FIX/"joblist.json").read_text())["jobs"]
    cands = [int(c) for c in a.candidates.split(",")]
    rows = []; t0 = time.time()
    for i, item in enumerate(jobs, 1):
        m = item["perfseer_model"]; base = m["input_shapes"][0]; prec = m["precision"]
        chosen = None
        for bs in cands:
            try:
                dt, peak = measure(build, m["constructor_kwargs"], base, bs, prec, dev, a.warmup, a.iters)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                if "out of memory" in str(e).lower(): break
                raise
            if peak <= a.target_peak_mib:
                chosen = {"batch_size": bs, "ms_per_step": dt, "peak_mib": peak}
            else:
                if chosen is None: chosen = {"batch_size": bs, "ms_per_step": dt, "peak_mib": peak}
                break
        rec = {"id": item["id"], "architecture": item["architecture"], "family": item["family"],
               "variant": item["variant"], "precision": prec, **chosen}
        rows.append(rec)
        print(f"[{i:3d}/100] {item['id']} {item['architecture']:<20} -> bs={rec['batch_size']:<6} "
              f"peak={rec['peak_mib']:.0f}MiB step={rec['ms_per_step']:.2f}ms  ({time.time()-t0:.0f}s)", flush=True)
    peaks = [r["peak_mib"] for r in rows]
    summary = {"target_peak_mib": a.target_peak_mib, "device": torch.cuda.get_device_name(0),
        "avg_peak_mib": round(s.mean(peaks),1), "median_peak_mib": round(s.median(peaks),1),
        "avg_job_total_gib": round((s.mean(peaks)+760)/1024,2),
        "jobs_to_fill_23gib_avg": round(23552/(s.mean(peaks)+760),2), "rows": rows}
    Path(a.output).write_text(json.dumps(summary, indent=2))
    print(f"\navg peak={summary['avg_peak_mib']} MiB (+760 ctx = {summary['avg_job_total_gib']} GiB/job); "
          f"~{summary['jobs_to_fill_23gib_avg']} jobs fill 23 GiB. wrote {a.output}")

if __name__ == "__main__":
    main()
