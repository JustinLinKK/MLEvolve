"""Builder: per-job batch (from VRAM finder), sized so ~3 jobs fill the GPU; each job
~target_min via per-job batches_per_epoch; 25 packs of 4; Poisson lambda=4/min."""
from __future__ import annotations
import json, argparse, random, statistics as s
from pathlib import Path

REPO = Path("/root/downeyflyfan/perfseer_test/exp_run")
FIX = REPO / "scheduler_benchmark_test/fixtures/stress_test_data_v1.0"
MODEL_SOURCE = FIX / "model_source.py"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-profile", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--target-min", type=float, default=30.0)
    ap.add_argument("--min-bpe", type=int, default=10)
    ap.add_argument("--context-mib", type=int, default=760)
    ap.add_argument("--lambda-jobs-per-min", type=float, default=4.0)
    ap.add_argument("--pack-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    prof = json.loads(Path(a.batch_profile).read_text())
    fixture = {j["id"]: j for j in json.loads((FIX/"joblist.json").read_text())["jobs"]}
    target_ms = a.target_min*60*1000.0
    jobs = []
    for r in prof["rows"]:
        step_ms = r["ms_per_step"]; bs = r["batch_size"]
        bpe = max(a.min_bpe, int(round((target_ms/step_ms)/a.epochs)))
        jobs.append({"id": r["id"], "architecture": r["architecture"], "family": r["family"],
            "variant": r["variant"], "precision": r["precision"], "batch_size": bs,
            "step_ms": round(step_ms,3), "peak_mib": round(r["peak_mib"],1),
            "estimated_vram_mb": int(round(r["peak_mib"]))+a.context_mib,
            "batches_per_epoch": bpe, "est_job_min": round(a.epochs*bpe*step_ms/60000.0,2)})
    jobs.sort(key=lambda j: j["id"]); jobmap = {j["id"]: j for j in jobs}
    n_groups = len(jobs)//a.pack_size
    groups = [[] for _ in range(n_groups)]; load=[0.0]*n_groups
    for j in sorted(jobs, key=lambda x: -x["est_job_min"]):
        g = min(range(n_groups), key=lambda i: (len(groups[i])>=a.pack_size, load[i]))
        groups[g].append(j); load[g]+=j["est_job_min"]
    group_list=[{"group_id":f"g{gi:02d}","job_ids":[j["id"] for j in g]} for gi,g in enumerate(groups)]
    rng = random.Random(a.seed); lam = (a.lambda_jobs_per_min/a.pack_size)/60.0
    order=list(range(n_groups)); rng.shuffle(order)
    arrival=0.0; trace=[]; step=0
    for gi in order:
        arrival += rng.expovariate(lam)
        for jid in group_list[gi]["job_ids"]:
            j=jobmap[jid]; item=fixture[jid]; m=item["perfseer_model"]
            shape=[j["batch_size"] if v=="$batch" else int(v) for v in m["input_shapes"][0]]
            trace.append({"step_idx":step,"job_id":jid,"group_id":group_list[gi]["group_id"],
                "arrival_offset_s":round(arrival,3),"family":j["family"],"architecture":j["architecture"],
                "variant":j["variant"],"precision":j["precision"],"batch_size":j["batch_size"],
                "epochs":a.epochs,"batches_per_epoch":j["batches_per_epoch"],"stream_data":True,
                "source_path":str(MODEL_SOURCE),"entry":m["entry"],"constructor_kwargs":m["constructor_kwargs"],
                "input_shape":shape,"input_dtypes":m["input_dtypes"],
                "packing_signature":f"{j['family']}:{j['architecture']}:{j['variant']}:bs{j['batch_size']}",
                "estimated_vram_mb":j["estimated_vram_mb"],"est_job_min":j["est_job_min"]}); step+=1
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    (out/"poisson_trace.jsonl").write_text("\n".join(json.dumps(t,sort_keys=True) for t in trace)+"\n")
    (out/"group_list.json").write_text(json.dumps({"pack_size":a.pack_size,"n_groups":n_groups,"groups":group_list},indent=2))
    peaks=[j["peak_mib"] for j in jobs]; ests=[j["est_job_min"] for j in jobs]; vram=[j["estimated_vram_mb"] for j in jobs]
    man={"epochs":a.epochs,"target_min":a.target_min,"lambda_jobs_per_min":a.lambda_jobs_per_min,
        "pack_size":a.pack_size,"n_jobs":len(jobs),"n_groups":n_groups,"arrival_span_s":round(arrival,1),
        "model_peak_mib":{"median":round(s.median(peaks),0),"mean":round(s.mean(peaks),0),"max":round(max(peaks),0)},
        "avg_job_total_gib":round(s.mean(vram)/1024,2),"jobs_to_fill_23gib_avg":round(23552/s.mean(vram),2),
        "est_job_min":{"min":round(min(ests),1),"median":round(s.median(ests),1),"max":round(max(ests),1)},
        "total_job_minutes":round(sum(ests),0),"seed":a.seed}
    (out/"manifest.json").write_text(json.dumps(man,indent=2))
    print(json.dumps(man,indent=2))
    print(f"bpe range {min(j['batches_per_epoch'] for j in jobs)}..{max(j['batches_per_epoch'] for j in jobs)}; wrote {out}")

if __name__ == "__main__":
    main()
