"""Strict-source check: for every Schema B chunk, each factual claim listed in
CLAIMS[chunk_id] must be directly traceable (case-insensitive substring) to
the chunk's declared source_url. If any chunk has unsupported claims, the
chunk text is overreaching beyond the source and must be edited (or its
source_url replaced with one that does support the claim).

This supersedes verify_fp4_chunk_text.py (fp4 is now one entry in CLAIMS).
"""
from __future__ import annotations
import html as html_mod
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import load_records  # noqa: E402

USER_AGENT = "Mozilla/5.0 (verify-chunk-texts)"

# CLAIMS: chunk_id -> list of (label, substring). The substring must be present
# (case-insensitive, after HTML-strip and whitespace-collapse) in the fetched
# source_url page. Substrings are tight, 4-15 words, chosen to be specific
# enough that no accidental match is plausible.
CLAIMS: dict[str, list[tuple[str, str]]] = {
    "cuda.graphs.training_loop.001": [
        ("CUDA Graphs capture kernel launches",
         "kernel launches"),
        ("overhead amortized over many executions",
         "overhead"),
        ("graph = snapshot of workflow incl kernels + dependencies",
         "snapshot"),
        ("fixed launch parameters",
         "fixed"),
    ],
    "flash_attn.v3.usage.001": [
        ("FlashAttention-3 beta",
         "flashattention-3"),
        ("optimized for Hopper",
         "hopper"),
        ("Blackwell support",
         "blackwell"),
        ("supports fp16",
         "fp16"),
        ("supports bf16",
         "bf16"),
        ("flash_attn_func API",
         "flash_attn_func"),
    ],
    "nvidia.ada_lovelace.architecture.001": [
        ("Ada Lovelace branding (RTX 40 Series)",
         "rtx 40 series"),
        ("4th-gen Tensor Cores",
         "fourth-generation tensor cores"),
        ("FP8 Transformer Engine from Hopper H100",
         "fp8 transformer engine"),
        ("3rd-gen RT cores 2x ray-triangle",
         "ray-triangle"),
        ("AV1 encoders (NVENC)",
         "av1"),
    ],
    "nvidia.fp4.blackwell.001": [
        ("FP4 = E2M1 = 1 sign, 2 exp, 1 mantissa",
         "1 sign, 2 exponent, 1 mantissa"),
        ("Fifth-gen Blackwell Tensor Cores accelerate FP4",
         "fifth-generation nvidia blackwell tensor cores"),
        ("MXFP4 block size 32",
         "mxfp4, which used 32 values"),
        ("NVFP4 block size 16",
         "smaller block size of 16 values"),
        ("MXFP4 scale dtype e8m0",
         "e8m0"),
        ("NVFP4 scale e4m3 + fp32 second-level",
         "e4m3 scaling factor and a second-level fp32 scalar"),
        ("NVFP4 used for PTQ inference",
         "post-training quantization (ptq)"),
    ],
    "nvidia.fp8.e4m3_vs_e5m2.001": [
        ("E4M3 4-bit exponent 3-bit mantissa",
         "e4m3"),
        ("E5M2 5-bit exponent 2-bit mantissa",
         "e5m2"),
        ("range +/-448 for E4M3",
         "448"),
        ("range +/-57344 for E5M2",
         "57344"),
        ("forward pass uses E4M3",
         "forward"),
        ("backward pass uses E5M2",
         "backward"),
        ("Delayed Scaling recipe",
         "delayed scaling"),
        ("amax history",
         "amax"),
        ("H100 (Hopper) FP8 support",
         "h100"),
        ("Blackwell FP8 support",
         "blackwell"),
    ],
    "nvidia.int8.quantization.training.001": [
        ("torch.ao.quantization namespace",
         "torch.ao.quantization"),
        ("quantize_dynamic",
         "quantize_dynamic"),
        ("per-channel quantization",
         "quantize_per_channel"),
        ("per-tensor quantization",
         "quantize_per_tensor"),
        ("QAT (fake quant)",
         "qat"),
        ("fake_quantize API",
         "fake_quantize"),
    ],
    "nvidia.tensor_cores.generations.001": [
        ("fifth-gen Tensor Cores on Blackwell",
         "fifth generation"),
        ("FP4 precision on Blackwell, doubling performance",
         "fp4"),
        ("supported precisions list (NVFP4, FP64, TF32, BF16, FP16, FP8, INT8)",
         "nvfp4"),
        ("Blackwell brand",
         "blackwell"),
    ],
    "nvidia.tf32.matmul.policy.001": [
        ("TF32 third generation of Tensor Cores",
         "third generation"),
        ("introduced with NVIDIA Ampere",
         "ampere"),
        ("10 bits of mantissa + sign bit",
         "10 bits of mantissa"),
        ("same range as FP32",
         "same range"),
        ("TF32 Tensor Cores in A100",
         "a100"),
    ],
    "pytorch.amp.autocast.bf16.training.001": [
        ("torch.autocast context manager",
         "torch.autocast"),
        ("supports torch.bfloat16",
         "torch.bfloat16"),
        ("supports torch.float16",
         "torch.float16"),
        ("GradScaler for fp16",
         "gradscaler"),
        ("gradient scaling",
         "gradient scaling"),
    ],
    "pytorch.compile.max_autotune.001": [
        ("max-autotune mode string",
         "max-autotune"),
        ("Inductor backend default",
         "inductor"),
        ("Triton kernels",
         "triton"),
        ("epilogue_fusion option",
         "epilogue_fusion"),
        ("recompile_limit knob",
         "recompile_limit"),
    ],
    "pytorch.cudnn.benchmark.policy.001": [
        ("cuDNN benchmark fastest algorithm",
         "fastest"),
        ("cudnn.deterministic flag",
         "deterministic"),
        ("benchmarking different algorithms",
         "benchmarking"),
        ("seeded RNGs for reproducibility",
         "reproducibility"),
    ],
    "pytorch.dataloader.pin_memory_workers.001": [
        ("pin_memory=True",
         "pin_memory"),
        ("pinned (page-locked) memory",
         "pinned"),
        ("host-to-GPU copies faster from pinned",
         "host to gpu"),
        ("num_workers > 0 multi-process",
         "num_workers"),
        ("prefetch_factor",
         "prefetch_factor"),
        ("persistent_workers",
         "persistent_workers"),
    ],
    "pytorch.vram_probe.budget.001": [
        ("max_memory_allocated to monitor VRAM",
         "max_memory_allocated"),
        ("memory_allocated function",
         "memory_allocated"),
        ("running out of memory keyword",
         "running out of memory"),
        ("fragmentation can prevent borderline workloads",
         "fragmentation"),
    ],
    "transformer_engine.fp8.inference.001": [
        ("FP8 training and inference",
         "fp8"),
        ("Hopper GPUs",
         "hopper"),
        ("Ada GPUs",
         "ada"),
        ("Blackwell GPUs",
         "blackwell"),
        ("DelayedScaling recipe",
         "delayedscaling"),
        ("CUDA 12.1 requirement",
         "12.1"),
        ("cuDNN requirement",
         "cudnn"),
        ("FlashAttention bundled",
         "flashattention"),
    ],
}


def _fetch_raw(url: str) -> tuple[str, int]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(
            req, timeout=20, context=ssl.create_default_context()
        ) as r:
            if r.status != 200:
                return "", r.status
            return r.read().decode("utf-8", errors="ignore"), 200
    except urllib.error.HTTPError as exc:
        return "", exc.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return "", 0


def fetch_text(url: str, _cache: dict[str, str] = {}, _depth: int = 0) -> tuple[str, int]:
    """Fetch URL, return (lowercased plain text, status). Follows JS
    location.replace redirects up to depth 2 (PyTorch docs serve a tiny
    JS-redirect page for /docs/stable/ -> /docs/<latest>/)."""
    if url in _cache:
        return _cache[url], 200
    raw, status = _fetch_raw(url)
    if status != 200:
        return "", status
    # Detect JS redirect: location.replace("../2.12/amp.html" + ...)
    if _depth < 2 and len(raw) < 5000 and "location.replace" in raw:
        m = re.search(r'location\.replace\(\s*"([^"]+)"', raw)
        if m:
            target = m.group(1)
            if target.startswith("http"):
                next_url = target
            else:
                next_url = urllib.parse.urljoin(url, target.split("'")[0].split("+")[0].strip())
            text, st = fetch_text(next_url, _cache, _depth + 1)
            if st == 200:
                _cache[url] = text
            return text, st
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text).lower()
    _cache[url] = text
    return text, 200


def main() -> int:
    chunks = {r["chunk_id"]: r for r in load_records("code_doc_chunks")}
    missing_in_claims = set(chunks) - set(CLAIMS)
    extra_in_claims = set(CLAIMS) - set(chunks)
    if missing_in_claims:
        print(f"WARN chunks without CLAIMS entry: {sorted(missing_in_claims)}")
    if extra_in_claims:
        print(f"WARN CLAIMS entries with no chunk: {sorted(extra_in_claims)}")

    fails: list[str] = []
    per_chunk: list[tuple[str, int, int]] = []
    for cid in sorted(CLAIMS):
        rec = chunks.get(cid)
        if rec is None:
            continue
        url = rec.get("source_url", "")
        if not url:
            fails.append(f"[{cid}] missing source_url")
            continue
        print(f"\n--- {cid} ---")
        print(f"  source_url: {url}")
        text, status = fetch_text(url)
        if status != 200:
            print(f"  FAIL HTTP {status}")
            fails.append(f"[{cid}] HTTP {status}: {url}")
            continue
        n_ok = 0
        n_fail = 0
        for label, needle in CLAIMS[cid]:
            ok = needle.lower() in text
            if ok:
                n_ok += 1
                print(f"  OK   {label}")
            else:
                n_fail += 1
                print(f"  FAIL {label}  (needle: '{needle}')")
                fails.append(f"[{cid}] unsupported: {label} (needle '{needle}')")
        per_chunk.append((cid, n_ok, n_fail))

    print("\n=== summary ===")
    for cid, ok, bad in per_chunk:
        mark = "OK  " if bad == 0 else "FAIL"
        print(f"  {mark} {cid}  {ok} OK / {bad} FAIL")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"=== all {len(per_chunk)} chunks: text 100% sourced ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
