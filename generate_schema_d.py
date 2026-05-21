"""Generate schema/api_symbol_chunks.yaml (Schema D) from pytorch.org docs.

For each target API:
  1. probe docs.pytorch.org/docs/<v>/<page>.html for v in 2.1 .. 2.12
  2. extract the signature parameter list from the <dt id="<sym>"> block
  3. normalize to (name, default) tuples ignoring annotation noise
  4. group consecutive versions with identical normalized signatures
  5. emit one Schema D record per (api_symbol, version-group) with
     framework_version = "<from>-<to>" or single version if 1-version group

This means an API with stable signature 2.1 -> 2.12 produces 1 record.
An API whose signature changed at 2.6 produces 2 records: one "2.1-2.5"
and one "2.6-2.12".
"""
from __future__ import annotations
import html as html_mod
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from schema_io import save_record  # noqa: E402

DOCS_BASE = "https://docs.pytorch.org/docs"
VERSIONS = ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
            "2.8", "2.9", "2.10", "2.11", "2.12"]
USER_AGENT = "Mozilla/5.0 (schema-d-probe)"


def memory_url(version: str, fn: str) -> str:
    """torch.cuda.memory.*.html for >=2.8, torch.cuda.*.html for <=2.7."""
    minor = int(version.split(".")[1])
    seg = "torch.cuda.memory." if minor >= 8 else "torch.cuda."
    return f"{DOCS_BASE}/{version}/generated/{seg}{fn}.html"


def memory_dt_id(version: str, fn: str) -> str:
    """The dt id changed namespace at 2.8 alongside the URL."""
    minor = int(version.split(".")[1])
    seg = "torch.cuda.memory." if minor >= 8 else "torch.cuda."
    return f"{seg}{fn}"


APIS = [
    {
        "api_symbol": "torch.autocast",
        "dt_id": "torch.autocast",
        "url_fn": lambda v: f"{DOCS_BASE}/{v}/amp.html",
        "text": "Context manager that runs ops in lower precision (bf16/fp16) where safe; activates tensor cores",
        "example_code": (
            "import torch\n"
            "with torch.autocast(device_type='cuda', dtype=torch.bfloat16):\n"
            "    out = model(x)\n"
            "    loss = criterion(out, y)"
        ),
    },
    {
        "api_symbol": "torch.cuda.amp.GradScaler",
        "dt_id": "torch.cuda.amp.GradScaler",
        "url_fn": lambda v: f"{DOCS_BASE}/{v}/amp.html",
        "text": "Loss-scaling helper required for fp16 autocast (not needed for bf16) to avoid gradient underflow",
        "example_code": (
            "import torch\n"
            "scaler = torch.cuda.amp.GradScaler()\n"
            "with torch.autocast('cuda', dtype=torch.float16):\n"
            "    out = model(x); loss = criterion(out, y)\n"
            "scaler.scale(loss).backward()\n"
            "scaler.step(optimizer); scaler.update()"
        ),
    },
    {
        "api_symbol": "torch.compile",
        "dt_id": "torch.compile",
        "url_fn": lambda v: f"{DOCS_BASE}/{v}/generated/torch.compile.html",
        "text": "JIT-compiles a model via TorchDynamo + Inductor; mode='max-autotune' searches kernel schedules",
        "example_code": (
            "import torch\n"
            "compiled = torch.compile(model, mode='max-autotune')\n"
            "out = compiled(x)"
        ),
    },
    {
        "api_symbol": "torch.cuda.graph",
        "dt_id": "torch.cuda.graph",
        "url_fn": lambda v: f"{DOCS_BASE}/{v}/generated/torch.cuda.graph.html",
        "text": "Context manager capturing kernel launches into a CUDA Graph for low-overhead replay",
        "example_code": (
            "import torch\n"
            "g = torch.cuda.CUDAGraph()\n"
            "for _ in range(3):\n"
            "    out = model(x)\n"
            "with torch.cuda.graph(g):\n"
            "    out = model(x)\n"
            "g.replay()"
        ),
    },
    {
        "api_symbol": "torch.cuda.CUDAGraph",
        "dt_id": "torch.cuda.CUDAGraph",
        "url_fn": lambda v: f"{DOCS_BASE}/{v}/generated/torch.cuda.CUDAGraph.html",
        "text": "CUDA Graph object capturable via torch.cuda.graph context, replayable via .replay()",
        "example_code": (
            "import torch\n"
            "g = torch.cuda.CUDAGraph()\n"
            "with torch.cuda.graph(g):\n"
            "    out = model(x)\n"
            "g.replay()"
        ),
    },
    {
        "api_symbol": "torch.utils.data.DataLoader",
        "dt_id": "torch.utils.data.DataLoader",
        "url_fn": lambda v: f"{DOCS_BASE}/{v}/data.html",
        "text": "Iterable wrapper around a Dataset with batching, shuffling, parallel loading, and pinning",
        "example_code": (
            "from torch.utils.data import DataLoader\n"
            "loader = DataLoader(\n"
            "    dataset, batch_size=32, shuffle=True,\n"
            "    num_workers=10, pin_memory=True,\n"
            "    persistent_workers=True, prefetch_factor=4,\n"
            ")"
        ),
    },
    {
        "api_symbol": "torch.cuda.max_memory_allocated",
        "dt_id_fn": lambda v: memory_dt_id(v, "max_memory_allocated"),
        "url_fn": lambda v: memory_url(v, "max_memory_allocated"),
        "text": "Returns peak VRAM (bytes) allocated by tensors on a given device since last reset",
        "example_code": (
            "import torch\n"
            "torch.cuda.reset_peak_memory_stats()\n"
            "out = model(x); out.sum().backward()\n"
            "peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024"
        ),
    },
    {
        "api_symbol": "torch.cuda.reset_peak_memory_stats",
        "dt_id_fn": lambda v: memory_dt_id(v, "reset_peak_memory_stats"),
        "url_fn": lambda v: memory_url(v, "reset_peak_memory_stats"),
        "text": "Resets the peak memory counter to the current allocated value (call before probing)",
        "example_code": (
            "import torch\n"
            "torch.cuda.reset_peak_memory_stats()\n"
            "peak = torch.cuda.max_memory_allocated()"
        ),
    },
]


def fetch(url: str, _cache: dict[str, str] = {}) -> str | None:
    if url in _cache:
        return _cache[url]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        ctx = ssl.create_default_context()
        body = urllib.request.urlopen(req, timeout=20, context=ctx).read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        body = None
    _cache[url] = body
    return body


def extract_param_entries(html: str, dt_id: str) -> list[str] | None:
    m = re.search(r'<dt[^>]*id="' + re.escape(dt_id) + r'"[^>]*>(.*?)</dt>', html, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    params = re.findall(r'<em class="sig-param"[^>]*>(.*?)</em>', block, re.DOTALL)
    out = []
    for p in params:
        text = re.sub(r"<[^>]+>", " ", p)
        text = html_mod.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        out.append(text)
    return out


def parse_entry(entry: str) -> tuple[str, str]:
    """Normalize 'name : annotation = default' -> (name, default).

    Annotations are stripped because they evolve as cosmetic changes
    (Optional[X] -> X | None) without altering the API contract.
    Returns ('*', '') for the keyword-only marker.
    """
    entry = entry.strip()
    if entry == "*":
        return ("*", "")
    if "=" in entry:
        left, default = entry.split("=", 1)
        default = default.strip()
    else:
        left = entry
        default = ""
    name = left.split(":", 1)[0].strip()
    return (name, default)


def normalize(entries: list[str]) -> list[tuple[str, str]]:
    return [parse_entry(e) for e in entries]


def signature_string(norm: list[tuple[str, str]]) -> str:
    parts = []
    for name, default in norm:
        if name == "*":
            parts.append("*")
        elif default:
            parts.append(f"{name}={default}")
        else:
            parts.append(name)
    return "(" + ", ".join(parts) + ")"


def params_json(norm: list[tuple[str, str]]) -> str:
    arr = [{"name": n, "default": d} for n, d in norm if n != "*"]
    return json.dumps(arr, ensure_ascii=False)


def group_contiguous(per_version: dict[str, list[tuple[str, str]]]) -> list[tuple[list[str], list[tuple[str, str]]]]:
    """Returns [(version_list, normalized_sig), ...]."""
    groups = []
    cur_vers: list[str] = []
    cur_sig: list[tuple[str, str]] | None = None
    for v in VERSIONS:
        s = per_version.get(v)
        if s is None:
            if cur_vers:
                groups.append((cur_vers, cur_sig))
                cur_vers, cur_sig = [], None
            continue
        if cur_sig is None:
            cur_sig = s
            cur_vers = [v]
        elif s == cur_sig:
            cur_vers.append(v)
        else:
            groups.append((cur_vers, cur_sig))
            cur_sig = s
            cur_vers = [v]
    if cur_vers:
        groups.append((cur_vers, cur_sig))
    return groups


def version_range_label(versions: list[str]) -> str:
    if len(versions) == 1:
        return versions[0]
    return f"{versions[0]}-{versions[-1]}"


def main() -> int:
    print(f"Probing {len(APIS)} APIs across {len(VERSIONS)} PyTorch versions ({VERSIONS[0]} to {VERSIONS[-1]})...")
    out_records = []
    for api in APIS:
        sym = api["api_symbol"]
        dt_id_fn = api.get("dt_id_fn") or (lambda v, _id=api.get("dt_id"): _id)
        per_version: dict[str, list[tuple[str, str]]] = {}
        for v in VERSIONS:
            url = api["url_fn"](v)
            html = fetch(url)
            if html is None:
                continue
            entries = extract_param_entries(html, dt_id_fn(v))
            if entries is None:
                continue
            per_version[v] = normalize(entries)
        if not per_version:
            print(f"  [{sym}] NO_DOCS_FOUND across any version - skipping")
            continue

        groups = group_contiguous(per_version)
        print(f"  [{sym}] {len(groups)} signature group(s):")
        for vers, norm in groups:
            label = version_range_label(vers)
            sig_str = signature_string(norm)
            print(f"      {label}: {sig_str}")
            rec = {
                "schema_version": "api_symbol_chunk_v1",
                "api_symbol_id": f"{sym}.v{label}",
                "title": f"{sym} (PyTorch {label})",
                "api_symbol": sym,
                "signature": sig_str,
                "parameters_json": params_json(norm),
                "usage_summary": api["text"].rstrip("."),
                "example_code": api["example_code"].strip(),
                "text": api["text"].strip(),
                "framework": "pytorch",
                "framework_version": label,
                "docs_url": api["url_fn"](vers[-1]),
                "covers_versions": list(vers),
                "deprecated": False,
            }
            out_records.append(rec)

    for rec in out_records:
        save_record("api_symbol_chunks", rec["api_symbol_id"], rec)
    print(f"\nwrote {len(out_records)} records to schema/D/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
