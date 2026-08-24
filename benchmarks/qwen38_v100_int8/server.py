"""OpenAI-compatible local Qwen3.8-27B INT8 service for three V100 GPUs."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "models/Qwen3.8-27B")
QUANTIZATION = os.environ.get("QWEN_QUANTIZATION", "int8").lower()
SERVED_MODEL_NAME = "Qwen3.8-27B-INT8-V100" if QUANTIZATION == "int8" else "Qwen3.8-27B-FP16-V100"
GPU_COUNT = int(os.environ.get("QWEN_GPU_COUNT", "3"))
MAX_GPU_MEMORY_GIB = os.environ.get("QWEN_MAX_GPU_MEMORY_GIB", "30")


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


app = FastAPI()
processor: Any | None = None
model: Any | None = None


@app.on_event("startup")
def load_model() -> None:
    global model, processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": "balanced",
        "max_memory": {index: f"{MAX_GPU_MEMORY_GIB}GiB" for index in range(GPU_COUNT)},
    }
    if QUANTIZATION == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif QUANTIZATION != "fp16":
        raise ValueError(f"unsupported QWEN_QUANTIZATION: {QUANTIZATION}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        **model_kwargs,
    ).eval()


def prompt_inputs(messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if processor is None or model is None:
        raise RuntimeError("model is not loaded")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[text], return_tensors="pt")
    first_device = next(model.parameters()).device
    return {name: value.to(first_device) for name, value in inputs.items()}


def generation_kwargs(request: ChatRequest, streamer: TextIteratorStreamer | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, min(int(request.max_tokens), 512)),
        "do_sample": float(request.temperature) > 0.0,
        "use_cache": True,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(request.temperature)
    if streamer is not None:
        kwargs["streamer"] = streamer
    return kwargs


def sse_response(request: ChatRequest) -> StreamingResponse:
    assert processor is not None and model is not None
    inputs = prompt_inputs(request.messages)
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    worker = threading.Thread(
        target=model.generate,
        kwargs={**inputs, **generation_kwargs(request, streamer)},
        daemon=True,
    )
    worker.start()

    def events():
        for fragment in streamer:
            payload = {"choices": [{"delta": {"content": fragment}, "index": 0}]}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ready": model is not None,
        "model": SERVED_MODEL_NAME,
        "quantization": QUANTIZATION,
        "visible_gpus": torch.cuda.device_count(),
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    assert processor is not None and model is not None
    if request.stream:
        return sse_response(request)
    inputs = prompt_inputs(request.messages)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs(request))
    continuation = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(continuation, skip_special_tokens=True)[0]
    return JSONResponse(
        {
            "model": SERVED_MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": int(continuation.shape[1])},
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
