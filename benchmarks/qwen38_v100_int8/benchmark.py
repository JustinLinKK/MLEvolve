"""Measure TTFT and generation tokens per second from the local API."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from transformers import AutoTokenizer

MODEL_PATH = "models/Qwen3.8-27B"
URL = "http://127.0.0.1:8000/v1/chat/completions"
OUT = Path("results/qwen38_v100_int8_benchmark.json")
PLOT = Path("results/qwen38_v100_int8_benchmark.png")
PROMPT = "Explain in one sentence why profile-based GPU scheduling needs runtime estimates."


def request_once(tokenizer) -> dict[str, float]:
    started = time.perf_counter()
    first_token_at: float | None = None
    text = ""
    response = requests.post(
        URL,
        json={
            "model": "Qwen3.8-27B-INT8-V100",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 128,
            "temperature": 0.0,
            "stream": True,
        },
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        fragment = json.loads(payload)["choices"][0]["delta"].get("content", "")
        if fragment and first_token_at is None:
            first_token_at = time.perf_counter()
        text += fragment
    finished = time.perf_counter()
    completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    ttft = (first_token_at or finished) - started
    decode_seconds = max(1e-9, finished - (first_token_at or finished))
    return {
        "ttft_seconds": ttft,
        "completion_tokens": completion_tokens,
        "tokens_per_second": completion_tokens / decode_seconds,
        "total_seconds": finished - started,
    }


def draw(records: list[dict[str, float]]) -> None:
    fig, (gantt, metrics) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=(1, 1.4))
    cursor = 0.0
    for index, record in enumerate(records):
        gantt.broken_barh([(cursor, record["total_seconds"])], (index - 0.35, 0.7), facecolors="#2b7bba")
        gantt.axvline(cursor + record["ttft_seconds"], color="#d1622b", linewidth=1.4)
        cursor += record["total_seconds"]
    gantt.set_yticks(range(len(records)), [f"request {index + 1}" for index in range(len(records))])
    gantt.set_xlabel("sequential serving time (seconds)")
    gantt.set_title("Gantt: two-V100 Qwen3.8-27B INT8 benchmark; orange = first token")
    gantt.grid(axis="x", alpha=0.25)

    xs = list(range(1, len(records) + 1))
    metrics.plot(xs, [item["ttft_seconds"] for item in records], "o-", color="#d1622b", label="TTFT (seconds)")
    right = metrics.twinx()
    right.plot(xs, [item["tokens_per_second"] for item in records], "s-", color="#2b7bba", label="generation TPS")
    metrics.set_xlabel("measured request")
    metrics.set_ylabel("TTFT (seconds)", color="#d1622b")
    right.set_ylabel("tokens per second", color="#2b7bba")
    metrics.set_title("Metric-node graph: post-warmup response latency and throughput")
    metrics.grid(alpha=0.25)
    fig.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=160)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    request_once(tokenizer)  # warmup
    records = [request_once(tokenizer) for _ in range(3)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"records": records, "median": {key: statistics.median(item[key] for item in records) for key in records[0]}}, indent=2))
    draw(records)
    print(OUT)


if __name__ == "__main__":
    main()
