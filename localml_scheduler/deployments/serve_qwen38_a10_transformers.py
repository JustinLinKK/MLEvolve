"""OpenAI-compatible, text-only Qwen3.8 server for two A10 GPUs.

The model is split by layer with Transformers ``device_map``.  This avoids
the CUDA/NCCL collectives used by vLLM tensor parallelism on hosts where
peer-to-peer GPU communication is unavailable or unstable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Convert Qwen tool-call markup to the OpenAI message representation."""

    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            raw = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, calls


def encode_sse(payload: dict[str, Any]) -> str:
    """Encode one OpenAI-compatible Server-Sent Events data frame."""

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


def build_max_memory(num_gpus: int) -> dict[int | str, str]:
    """Reserve one GiB of headroom on each 22 GiB A10."""

    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    return {**{index: "21GiB" for index in range(num_gpus)}, "cpu": "6GiB"}


def unwrap_compressed_tensors_forward(module: Any) -> None:
    """Remove a compressed-tensors forward saved behind an Accelerate hook."""

    raw_forward = module.__class__.forward.__get__(module)
    accelerate_hook = getattr(module, "_hf_hook", None)
    if accelerate_hook is None:
        module.forward = raw_forward
        if hasattr(module, "_old_forward"):
            module._old_forward = raw_forward
        return

    from accelerate.hooks import add_hook_to_module, remove_hook_from_module

    remove_hook_from_module(module)
    module.forward = raw_forward
    add_hook_to_module(module, accelerate_hook)


def enable_cpu_staged_cross_gpu_transfers(torch: Any) -> None:
    """Avoid broken CUDA peer copies while preserving Accelerate device hooks."""

    import accelerate.hooks

    original_send_to_device = accelerate.hooks.send_to_device

    def staged_send_to_device(
        value: Any, device: Any, non_blocking: bool = False, skip_keys: Any = None
    ) -> Any:
        if torch.is_tensor(value) and value.device.type == "cuda":
            target = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
            if target.type == "cuda" and target.index != value.device.index:
                value = value.to("cpu")
                non_blocking = False
        return original_send_to_device(
            value, device, non_blocking=non_blocking, skip_keys=skip_keys
        )

    accelerate.hooks.send_to_device = staged_send_to_device


def convert_compressed_weights_to_torchao_int8(model: Any) -> int:
    """Replace packed checkpoint weights with executable TorchAO INT8 weights.

    ``compressed-tensors`` otherwise installs a pre-forward hook that expands
    every weight to FP16 at once.  Converting one linear layer at a time keeps
    peak memory bounded and leaves the resident model genuinely INT8.
    """

    import torch
    from compressed_tensors.compressors import decompress_module
    from compressed_tensors.quantization.quant_metadata import QuantizationMetadata
    from compressed_tensors.quantization.utils.helpers import is_module_quantized
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    from tqdm.auto import tqdm

    decompression_hook = getattr(model, "ct_decompress_hook", None)
    if decompression_hook is not None:
        decompression_hook.remove()
        delattr(model, "ct_decompress_hook")

    modules = [module for module in model.modules() if is_module_quantized(module)]
    config = Int8WeightOnlyConfig(group_size=128, set_inductor_config=True)
    for module in tqdm(modules, desc="Converting packed weights to TorchAO INT8"):
        packed_weight = getattr(module, "weight_packed", None)
        device = packed_weight.device if packed_weight is not None else None
        device_context = (
            torch.cuda.device(device) if device is not None and device.type == "cuda" else None
        )
        if device_context is None:
            decompress_module(module)
            QuantizationMetadata.clear_quantization(module)
            unwrap_compressed_tensors_forward(module)
            quantize_(
                module,
                config,
                filter_fn=lambda candidate, _name, target=module: candidate is target,
            )
        else:
            with device_context:
                decompress_module(module)
                QuantizationMetadata.clear_quantization(module)
                unwrap_compressed_tensors_forward(module)
                quantize_(
                    module,
                    config,
                    filter_fn=lambda candidate, _name, target=module: candidate is target,
                )
                torch.cuda.empty_cache()
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            unwrap_compressed_tensors_forward(module)
    stale_wrappers = []
    for name, module in model.named_modules():
        candidates = (
            [module._old_forward]
            if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward")
            else [module.forward]
        )
        for candidate in candidates:
            forward_function = getattr(candidate, "__func__", candidate)
            if str(getattr(forward_function, "__module__", "")).startswith(
                "compressed_tensors"
            ):
                stale_wrappers.append(name)
                break
    if stale_wrappers:
        raise RuntimeError(
            f"compressed-tensors forward wrappers remain: {stale_wrappers[:5]}"
        )
    return len(modules)


class A10LayerSplitModel:
    def __init__(
        self, model_path: str, model_name: str, max_context: int, num_gpus: int
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_context = max_context
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="balanced",
            max_memory=build_max_memory(num_gpus),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        converted = convert_compressed_weights_to_torchao_int8(self.model)
        if converted == 0:
            raise RuntimeError("No compressed linear weights were converted to TorchAO INT8")
        enable_cpu_staged_cross_gpu_transfers(torch)
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    def _render_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, body: dict[str, Any]) -> GenerationResult:
        max_new_tokens = max(1, min(int(body.get("max_tokens") or 1024), 8192))
        prompt = self._render_prompt(body.get("messages") or [], body.get("tools"))
        prompt_limit = max(1, self.max_context - max_new_tokens)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_limit,
        ).to(self.input_device)
        temperature = float(body.get("temperature", 0.7))
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(body.get("top_p", 0.95))
        with self.torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        prompt_tokens = int(inputs.input_ids.shape[1])
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated.shape[0]),
        )


def _stream_frames(
    completion_id: str,
    model_name: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    usage: dict[str, int],
    include_usage: bool,
) -> Iterable[str]:
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
    }
    yield encode_sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    for offset in range(0, len(content), 128):
        yield encode_sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content[offset : offset + 128]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    if tool_calls:
        yield encode_sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": None,
                    }
                ],
            }
        )
    yield encode_sse(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        }
    )
    if include_usage:
        yield encode_sse({**base, "choices": [], "usage": usage})
    yield "data: [DONE]\n\n"


def create_app(model: A10LayerSplitModel):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model.model_name}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model.model_name, "object": "model", "owned_by": "downeyflyfan"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any]):
        async with lock:
            result = await asyncio.to_thread(model.generate, body)
        content, tool_calls = extract_tool_calls(result.text)
        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return StreamingResponse(
                _stream_frames(
                    completion_id,
                    model.model_name,
                    content,
                    tool_calls,
                    usage,
                    include_usage,
                ),
                media_type="text/event-stream",
            )
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": usage,
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="qwen3.8-27b-int8-a10-transformers-3gpu-text")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--num-gpus", type=int, default=3)
    args = parser.parse_args()

    import uvicorn

    model = A10LayerSplitModel(
        args.model_path, args.model_name, args.max_context, args.num_gpus
    )
    uvicorn.run(create_app(model), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
