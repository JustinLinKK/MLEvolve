#!/usr/bin/env bash
set -euo pipefail

uv venv .venv --python 3.12
# PyTorch 2.7+ CUDA wheels omit Volta (SM70) kernels; V100 requires this
# final compatible CUDA 11.8 release.
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0+cu118 torchvision==0.21.0+cu118
uv pip install --python .venv/bin/python \
  'transformers>=5.10.0' 'accelerate>=1.12.0' 'bitsandbytes>=0.49.0' \
  'fastapi>=0.128.0' 'uvicorn[standard]>=0.40.0' 'requests>=2.32.0' \
  'matplotlib>=3.10.0' 'huggingface_hub>=1.0.0'

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.8-27B",
    local_dir="models/Qwen3.8-27B",
)
PY
