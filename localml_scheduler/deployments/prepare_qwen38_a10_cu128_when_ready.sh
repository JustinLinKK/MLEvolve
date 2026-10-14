#!/usr/bin/env bash
set -euo pipefail

namespace="${NAUTILUS_NAMESPACE:-ecepxie}"
selector="${A10_SELECTOR:-app=gpu-dev-a10}"

while true; do
  pod="$({
    kubectl get pods -n "$namespace" -l "$selector" -o json
  } | jq -r '
    [.items[]
      | select(.status.phase == "Running")
      | select(any(.status.conditions[]?; .type == "Ready" and .status == "True"))
      | .metadata.name][0] // empty
  ')"
  if [[ -n "$pod" ]]; then
    break
  fi
  sleep 15
done

echo "Preparing CUDA 12.8 runtime in $pod"
kubectl exec -n "$namespace" "$pod" -- bash -lc '
  set -euo pipefail
  env_root=/tmp/vllm-026-cu128
  cache_root=/tmp/uv-cache
  wheel_url=https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl

  UV_CACHE_DIR="$cache_root" uv venv --python /usr/bin/python3.12 "$env_root"
  UV_CACHE_DIR="$cache_root" uv pip install \
    --python "$env_root/bin/python" vllm==0.26.0
  UV_CACHE_DIR="$cache_root" uv pip install \
    --python "$env_root/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
  UV_CACHE_DIR="$cache_root" uv pip install \
    --python "$env_root/bin/python" --reinstall --no-deps "$wheel_url"

  CUDA_VISIBLE_DEVICES=0 "$env_root/bin/python" - <<"PY"
import torch
import vllm

print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
values = torch.randn(4, 248_320, device="cuda", dtype=torch.bfloat16)
ordered, _ = values.sort(dim=-1)
torch.cuda.synchronize()
print("A10_CU128_SORT_OK", tuple(ordered.shape), float(ordered[0, 0]))
PY
  touch /tmp/a10_cu128_compat.pass
'

echo "A10 CUDA 12.8 compatibility check passed in $pod"
