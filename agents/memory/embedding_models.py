"""
Embedding Models Module

Provides unified interface for different embedding model types:
- Local: sentence-transformers models
- OpenAI: OpenAI embedding API
- Azure: Azure OpenAI embedding API
- OpenRouter: OpenRouter embedding API
- Custom: User-defined embedding services
"""

# Disable TensorFlow backend before any imports
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import json
import time
import urllib.error
import urllib.request
import numpy as np
from typing import Any, List, Literal, Optional


_RETRYABLE_OPENROUTER_STATUSES = {429, 500, 502, 503, 504, 529}


class EmbeddingModel:
    """
    Unified embedding model supporting both local and remote APIs

    Supported types:
    - local: sentence-transformers models
    - openai: OpenAI embedding API
    - azure: Azure OpenAI embedding API
    - openrouter: OpenRouter embedding API
    - custom: Custom API endpoint
    """

    def __init__(
        self,
        model_type: Literal["local", "openai", "azure", "openrouter", "custom"] = "local",
        model_name: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize embedding model

        Args:
            model_type: Type of model (local/openai/azure/custom)
            model_name: Model name or identifier
            api_key: API key for remote models
            base_url: Base URL for API endpoints
            dimension: Embedding dimension (auto-detected for local models)
            device: Device to use for local models ("cpu" or "cuda")
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.device = device or "cpu"
        self.kwargs = kwargs

        if model_type == "local":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
        elif model_type == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            # OpenAI text-embedding-3-small: 1536, text-embedding-ada-002: 1536
            self.dimension = dimension or 1536
        elif model_type == "azure":
            import openai
            self.client = openai.AzureOpenAI(
                api_key=api_key,
                api_version=kwargs.get("api_version", "2023-05-15"),
                azure_endpoint=base_url
            )
            self.dimension = dimension or 1536
        elif model_type == "openrouter":
            self.model_name = str(model_name or "").strip()
            if not self.model_name:
                raise ValueError("embedding_model_name is required for OpenRouter embeddings")
            if not dimension:
                raise ValueError("embedding_dimension is required for OpenRouter embeddings")
            self.dimension = int(dimension)
            api_key_env = str(kwargs.get("api_key_env") or "OPENROUTER_API_KEY").strip()
            self.api_key = api_key or (os.getenv(api_key_env) if api_key_env else None)
            if not self.api_key:
                raise ValueError(f"OpenRouter embedding API key is missing; set {api_key_env}")
            self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
            self.batch_size = max(1, int(32 if kwargs.get("batch_size") is None else kwargs.get("batch_size")))
            self.max_retries = max(0, int(4 if kwargs.get("max_retries") is None else kwargs.get("max_retries")))
            self.retry_delay_seconds = max(
                0.0,
                float(1.0 if kwargs.get("retry_delay_seconds") is None else kwargs.get("retry_delay_seconds")),
            )
            self.timeout_seconds = max(1.0, float(60.0 if kwargs.get("timeout_seconds") is None else kwargs.get("timeout_seconds")))
        elif model_type == "custom":
            # Custom API - user needs to implement their own logic
            self.dimension = dimension
            if not dimension:
                raise ValueError("dimension must be specified for custom model type")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings
            show_progress_bar: Show progress bar (only for local models)

        Returns:
            embeddings: numpy array of shape (len(texts), dimension)
        """
        if self.model_type == "local":
            embeddings = self.model.encode(texts, show_progress_bar=show_progress_bar)
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "openai":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "azure":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "openrouter":
            embeddings: list[list[float]] = []
            for start in range(0, len(texts), self.batch_size):
                embeddings.extend(self._openrouter_embeddings_batch(texts[start:start + self.batch_size]))
            array = np.array(embeddings, dtype=np.float32)
            if array.size and array.shape[1] != self.dimension:
                raise ValueError(
                    f"OpenRouter embedding dimension mismatch for {self.model_name}: "
                    f"configured {self.dimension}, received {array.shape[1]}"
                )
            return array

        elif self.model_type == "custom":
            # Placeholder for custom implementation
            raise NotImplementedError(
                "Custom embedding model requires user implementation. "
                "Please subclass EmbeddingModel and override the encode method."
            )

        return np.array([])

    def _openrouter_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._openrouter_post_json(url, payload, headers)
        data = response.get("data")
        if not isinstance(data, list):
            raise ValueError("OpenRouter embeddings response missing data list")
        embeddings: list[list[float]] = []
        for index, item in enumerate(data):
            embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            if not isinstance(embedding, list):
                raise ValueError(f"OpenRouter embeddings response item {index} missing embedding")
            embeddings.append([float(value) for value in embedding])
        if len(embeddings) != len(texts):
            raise ValueError(f"OpenRouter returned {len(embeddings)} embeddings for {len(texts)} inputs")
        return embeddings

    def _openrouter_post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"OpenRouter embeddings HTTP {exc.code}: {detail}")
                if exc.code not in _RETRYABLE_OPENROUTER_STATUSES or attempt >= attempts - 1:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"OpenRouter embeddings request failed: {exc.reason}")
                if attempt >= attempts - 1:
                    raise last_error from exc
            if self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds * (2 ** attempt))
        assert last_error is not None
        raise last_error
