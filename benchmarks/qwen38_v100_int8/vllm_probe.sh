#!/usr/bin/env bash
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.17.0}"
VLLM_VENV="${VLLM_VENV:-/tmp/qwen38-vllm-v100-probe-venv}"
RESULT_PATH="${RESULT_PATH:-results/vllm_v100_probe.json}"

/root/downeyflyfan/.local/bin/uv venv "$VLLM_VENV" --python 3.12
/root/downeyflyfan/.local/bin/uv pip install --python "$VLLM_VENV/bin/python" "vllm==${VLLM_VERSION}"

RESULT_PATH="$RESULT_PATH" "$VLLM_VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import torch
import vllm
from vllm_config import visible_device_index

device = visible_device_index(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
tensor = torch.ones(8, device=f"cuda:{device}", dtype=torch.float16)
payload = {
    "vllm_version": vllm.__version__,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "device": device,
    "compute_capability": list(torch.cuda.get_device_capability(device)),
    "cuda_tensor_sum": float(tensor.sum()),
}
path = Path(os.environ["RESULT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2))
print(path)
PY
