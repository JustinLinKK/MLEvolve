#!/usr/bin/env bash
# Short sensitivity check for packed co-location by model composition.
#
# Compares total time for:
# - same exact model packed together
# - same architecture family but different models packed together
# - different architecture families packed together
#
# Each composition is run twice:
# - serial baseline
# - packed stream baseline
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO/scheduler_benchmark_test"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

RESULTS_BASE="${RESULTS_BASE:-$REPO/results/scheduler_benchmark_test/arch_sensitivity}"
TRACE_DIR="$BENCH_DIR/arch_sensitivity_inputs"
CODE_CACHE="$TRACE_DIR/replay_codes"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-900}"
COMMON_VRAM_BUDGET_GIB="${COMMON_VRAM_BUDGET_GIB:-30.0}"
COMMON_CACHE_WARM_POLICY="${COMMON_CACHE_WARM_POLICY:-top_k}"
COMMON_CACHE_WARM_TOP_K="${COMMON_CACHE_WARM_TOP_K:-2}"
COMMON_CACHE_ENTRY_CAPACITY="${COMMON_CACHE_ENTRY_CAPACITY:-4}"
COMMON_CACHE_MAX_RAM_PERCENT="${COMMON_CACHE_MAX_RAM_PERCENT:-0.10}"
COMMON_CACHE_MEMORY_BUDGET_GIB="${COMMON_CACHE_MEMORY_BUDGET_GIB:-8.0}"

mkdir -p "$RESULTS_BASE" "$CODE_CACHE"

cleanup_gpu() {
    pkill -9 -f "replay_scheduler.py|step_[0-9]+\\.py|nvidia-cuda-mps" 2>/dev/null || true
    sleep 2
}

run_sched() {
    local id="$1" trace_name="$2" mode="$3" backend="$4" probe="$5"
    shift 5
    local trace="$TRACE_DIR/${trace_name}.jsonl"
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/$id" "/tmp/scheduler_benchmark_runtime_$id"

    echo "[$(date -Iseconds)] === $id trace=$trace_name mode=$mode backend=$backend probe=$probe ==="
    local t0
    t0=$(date +%s.%N)
    timeout "$CONFIG_TIMEOUT" "$PY" "$BENCH_DIR/replay_scheduler.py" \
        --config-id "$id" \
        --mode "$mode" \
        --backend "$backend" \
        --batch-search "$probe" \
        --trace "$trace" \
        --runtime-root "/tmp/scheduler_benchmark_runtime_$id" \
        --results-dir "$cfg_dir/results" \
        --summary "$cfg_dir/summary.json" \
        --code-cache-dir "$CODE_CACHE" \
        --duration-s $(( CONFIG_TIMEOUT - 60 )) \
        --vram-budget-gib "$COMMON_VRAM_BUDGET_GIB" \
        --cache-warm-policy "$COMMON_CACHE_WARM_POLICY" \
        --cache-warm-top-k "$COMMON_CACHE_WARM_TOP_K" \
        --cache-entry-capacity "$COMMON_CACHE_ENTRY_CAPACITY" \
        --cache-max-ram-percent "$COMMON_CACHE_MAX_RAM_PERCENT" \
        --cache-memory-budget-gib "$COMMON_CACHE_MEMORY_BUDGET_GIB" \
        "$@" \
        > "$cfg_dir/replay.log" 2>&1
    local rc=$?
    local t1
    t1=$(date +%s.%N)
    local elapsed
    elapsed=$("$PY" - <<PY
t0 = float("$t0")
t1 = float("$t1")
print(t1 - t0)
PY
)
    echo "$elapsed" > "$cfg_dir/wall_clock.txt"
    echo "$rc" > "$cfg_dir/rc.txt"
    echo "[$(date -Iseconds)] $id rc=$rc elapsed=${elapsed}s"
}

echo "[setup] checking benchmark Python environment..."
"$PY" "$BENCH_DIR/check_benchmark_env.py"

echo "[setup] generating short sensitivity traces..."
OUT_DIR="$BENCH_DIR" "$PY" "$BENCH_DIR/gen_arch_sensitivity_traces.py"

run_sched SM_S same_model_pair serial_basic exclusive off
run_sched SA_S same_arch_pair serial_basic exclusive off
run_sched DA_S cross_arch_pair serial_basic exclusive off

run_sched SM_P same_model_pair parallel_default stream off
run_sched SA_P same_arch_pair parallel_default stream off
run_sched DA_P cross_arch_pair parallel_default stream off

echo "[summary] building markdown report..."
"$PY" - <<PY
import json
from pathlib import Path

base = Path("$RESULTS_BASE")
cases = [
    ("SM_S", "same exact model", "serial"),
    ("SA_S", "same arch, different models", "serial"),
    ("DA_S", "different arch", "serial"),
    ("SM_P", "same exact model", "packed stream"),
    ("SA_P", "same arch, different models", "packed stream"),
    ("DA_P", "different arch", "packed stream"),
]
serial_map = {}
rows = []
for case_id, composition, mode in cases:
    cfg_dir = base / case_id
    summary = json.loads((cfg_dir / "summary.json").read_text())
    wall = float((cfg_dir / "wall_clock.txt").read_text().strip())
    completed = int((summary.get("by_status") or {}).get("COMPLETED", 0))
    treat_elapsed = float(summary.get("treat_elapsed_s") or wall)
    packed = int(summary.get("n_pack_dispatches") or 0)
    if mode == "serial":
        serial_map[composition] = wall
    rows.append(
        {
            "id": case_id,
            "composition": composition,
            "mode": mode,
            "wall": wall,
            "treat": treat_elapsed,
            "completed": completed,
            "packed": packed,
        }
    )

for row in rows:
    base_wall = serial_map.get(row["composition"])
    if row["mode"] == "packed stream" and base_wall:
        row["speedup"] = base_wall / max(row["wall"], 1e-9)
    else:
        row["speedup"] = None

lines = [
    "# Arch Sensitivity Summary",
    "",
    "Short sensitivity check for co-locating two training jobs.",
    "",
    "| Config ID | Composition | Mode | Total wall time (s) | Replay elapsed (s) | Completed jobs | Packed dispatches | Speedup vs same-composition serial |",
    "|---|---|---|---|---|---|---|---|",
]
for row in rows:
    speedup = f'{row["speedup"]:.2f}x' if row["speedup"] is not None else "-"
    lines.append(
        f'| `{row["id"]}` | {row["composition"]} | {row["mode"]} | {row["wall"]:.3f} | {row["treat"]:.3f} | {row["completed"]} | {row["packed"]} | {speedup} |'
    )

lines.extend(
    [
        "",
        "Interpretation:",
        "- `same exact model`: two copies of `convnext_base`",
        "- `same arch, different models`: `convnext_base` + `efficientnet_b3`",
        "- `different arch`: `convnext_base` + `vit_base_patch16_224`",
        "",
        "Packed cases use `parallel_default` + `stream` with batch search disabled so the comparison stays focused on co-location sensitivity rather than batch-size tuning.",
    ]
)
(base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((base / "summary.md").read_text())
PY

echo
echo "Results written to: $RESULTS_BASE"
echo "Markdown summary: $RESULTS_BASE/summary.md"
