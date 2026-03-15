from __future__ import annotations

"""Embedding abstraction layer.

This module defines a small, pluggable interface that mirrors the
`memory-lancedb-pro` embedder design, but implemented in Python.

The default implementation is an OpenAI-compatible HTTP client that can
also run in **dry-run** mode (generating random vectors) so that
pipelines and tests can run without any external API key.
"""

import logging
import os
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


class EmbedderAdapter(Protocol):
    """Abstract embedding interface.

    Implementations should be stateless or lightly stateful and safe to
    reuse across batches.
    """

    @property
    def dimensions(self) -> int:  # pragma: no cover - simple property
        """Dimension of the embedding vectors produced by this embedder."""

    def embed_query(self, text: str) -> List[float]:
        """Embed a query string (short, retrieval-style)."""

    def embed_passage(self, text: str) -> List[float]:
        """Embed a passage/document string (may be longer)."""

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        """Batch passage embedding.

        Implementations should keep relative order of `texts` in the
        returned list.
        """


@dataclass
class OpenAICompatibleConfig:
    """Configuration for an OpenAI-compatible embeddings endpoint.

    This intentionally mirrors the configuration surface used by
    `memory-lancedb-pro`:

    - `base_url`  : HTTP base URL of the embedding provider
    - `model`     : embedding model name
    - `api_key`   : API key (can come from env var as well)
    - `dimensions`: expected output vector dimension (required)
    - `task_query`/`task_passage`: optional task hints for providers
    - `dry_run`   : if True, skip HTTP calls and emit random vectors
    """

    model: str
    dimensions: int
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    task_query: Optional[str] = None
    task_passage: Optional[str] = None
    dry_run: bool = False


class OpenAICompatibleEmbedder(EmbedderAdapter):
    """OpenAI-compatible embedding adapter with optional dry-run.

    When ``dry_run=True``, the adapter does **not** make any network
    calls and instead returns deterministic pseudo-random vectors. This
    is ideal for local smoke tests and CI environments where secrets are
    not available.
    """

    def __init__(self, config: OpenAICompatibleConfig, *, rng_seed: int = 42) -> None:
        if config.dimensions <= 0:
            raise ValueError("embedding dimensions must be a positive integer")

        self._config = config
        self._dimensions = int(config.dimensions)
        self._rng = random.Random(rng_seed)

        # Try to detect if the base_url is a full endpoint URL (e.g. Doubao specific)
        self._is_custom_endpoint = False
        if config.base_url:
            # Simple heuristic: if it ends with "embeddings" or "embeddings/multimodal", it's likely a full endpoint
            # Standard OpenAI base_url usually ends with /v1
            lower_url = config.base_url.lower()
            if lower_url.endswith("/embeddings") or "embeddings/" in lower_url:
                self._is_custom_endpoint = True
                logger.info("Detected custom embedding endpoint: %s", config.base_url)

        self._client = None
        if not config.dry_run:
            # Lazily import to keep the package importable even if
            # python-openai is not installed (e.g. when only using dry-run).
            from openai import OpenAI  # type: ignore

            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Embedding API key is required unless dry_run=True or OPENAI_API_KEY is set",
                )

            client_kwargs = {"api_key": api_key}
            if config.base_url:
                if self._is_custom_endpoint:
                    # For OpenAI client, base_url is usually the root.
                    # If we have a full path, we might need to handle it manually or trick the client.
                    # However, standard OpenAI client appends /embeddings to base_url.
                    # If the user provided ".../embeddings/multimodal", appending /embeddings -> ".../embeddings/multimodal/embeddings" (Wrong)
                    # So if it's a custom endpoint, we might prefer using httpx directly in _embed_many
                    # But to initialize the client, we can just pass the base_url as is for now,
                    # or pass a dummy one if we are going to bypass it.
                    # Let's keep the client for potential future use or non-embedding calls,
                    # but _embed_many will handle the custom logic.
                    client_kwargs["base_url"] = config.base_url
                else:
                    client_kwargs["base_url"] = config.base_url
            
            self._client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:  # pragma: no cover - trivial
        return self._dimensions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text, task=self._config.task_query)

    def embed_passage(self, text: str) -> List[float]:
        return self._embed_single(text, task=self._config.task_passage)

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_many(texts, task=self._config.task_passage)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_single(self, text: str, task: Optional[str]) -> List[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        vectors = self._embed_many([text], task=task)
        return vectors[0]

    def _embed_many(self, texts: Sequence[str], task: Optional[str]) -> List[List[float]]:
        texts_list = [t.strip() for t in texts if (t or "").strip()]
        if not texts_list:
            return []

        if self._config.dry_run:
            # Deterministic pseudo-random vectors for reproducible tests.
            return [self._random_vector() for _ in texts_list]

        # Special handling for Doubao Vision models on custom endpoints (no batch support)
        if self._is_custom_endpoint and self._config.base_url and "vision" in self._config.model:
             import httpx
             headers = {
                 "Content-Type": "application/json",
                 "Authorization": f"Bearer {self._config.api_key or os.getenv('OPENAI_API_KEY')}",
             }
             vectors = []
             with httpx.Client(timeout=60.0) as http_client:
                 for text in texts_list:
                     payload = {
                         "model": self._config.model,
                         "input": [{"type": "text", "text": text}],
                     }
                     if task:
                         payload["task"] = task
                     
                     try:
                         # print(f"[DEBUG] Sending payload to {self._config.base_url}: {json.dumps(payload)}")
                         resp = http_client.post(self._config.base_url, json=payload, headers=headers)
                         resp.raise_for_status()
                         data = resp.json()
                         
                         # data['data'] is expected to be a dict with 'embedding' key for single item
                         emb = None
                         d_data = data.get("data")
                         if isinstance(d_data, dict):
                             emb = d_data.get("embedding")
                         elif isinstance(d_data, list) and len(d_data) > 0:
                             emb = d_data[0].get("embedding")
                             
                         if not emb:
                             print(f"[ERROR] Unexpected response format: {data}")
                             raise RuntimeError("No embedding found in response")
                                 
                         vectors.append(emb)
                     except Exception as exc:
                         print(f"[ERROR] Failed to embed text: {exc}")
                         if isinstance(exc, httpx.HTTPStatusError):
                             print(f"[ERROR] Response text: {exc.response.text}")
                         raise
             return vectors

        payload: dict = {
            "model": self._config.model,
            "input": texts_list,
        }
        
        # Only add encoding_format if not custom endpoint or known to support it
        if not self._is_custom_endpoint:
            payload["encoding_format"] = "float"
        elif "vision" in self._config.model:
             # Hypothesis: Doubao vision model expects structured input
             payload["input"] = [{"type": "text", "text": t} for t in texts_list]

        if task:
            payload["task"] = task
            
        # Some providers (like Doubao/Volcengine) do not support 'dimensions' parameter
        # Only add it if we are sure, or if it's OpenAI
        if self._dimensions and not self._is_custom_endpoint:
            payload["dimensions"] = self._dimensions

        try:
            if self._is_custom_endpoint and self._config.base_url:
                # Direct HTTP call for custom endpoints (e.g. Doubao) to avoid OpenAI client path appending logic
                import httpx
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._config.api_key or os.getenv('OPENAI_API_KEY')}",
                }
                
                # Use a new client or the one from openai if exposed (it's not easily exposed)
                # Just use httpx.post
                with httpx.Client(timeout=60.0) as http_client:
                    try:
                        print(f"[DEBUG] Sending payload to {self._config.base_url}: {json.dumps(payload)}")
                        resp = http_client.post(self._config.base_url, json=payload, headers=headers)
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        print(f"[ERROR] HTTPStatusError: {exc}")
                        print(f"[ERROR] Response text: {exc.response.text}")
                        raise

                    # Mocking an object that looks like OpenAI response for downstream compatibility
                    data = resp.json()
                    
                    # Wrap in a simple structure to match the loop below
                    class MockItem:
                        def __init__(self, d):
                            self.embedding = d.get("embedding")
                            self.index = d.get("index")
                            
                    class MockResponse:
                        def __init__(self, d):
                            self.data = [MockItem(i) for i in d.get("data", [])]
                            
                    response = MockResponse(data)
            else:
                if self._client is None:
                     # Deterministic pseudo-random vectors for reproducible tests.
                    return [self._random_vector() for _ in texts_list]
                
                # Chunk inputs to respect provider limits (Doubao: 256, OpenAI: 2048)
                chunk_size = 200
                all_data = []
                
                for i in range(0, len(texts_list), chunk_size):
                    chunk = texts_list[i : i + chunk_size]
                    chunk_payload = payload.copy()
                    chunk_payload["input"] = chunk
                    
                    response = self._client.embeddings.create(**chunk_payload)  # type: ignore[operator]
                    all_data.extend(getattr(response, "data", []))
                
                # Wrap in a simple structure to match the loop below
                class UnifiedResponse:
                    def __init__(self, data):
                        self.data = data
                response = UnifiedResponse(all_data)
        except Exception as exc:  # pragma: no cover - network path
            logger.error("Failed to call embedding provider: %s", exc)
            raise

        vectors: List[List[float]] = []
        for item in getattr(response, "data", []):  # type: ignore[assignment]
            emb = getattr(item, "embedding", None)
            if emb is None:
                raise RuntimeError("Embedding provider returned item without 'embedding' field")
            vec = list(emb)
            if len(vec) != self._dimensions:
                logger.warning(
                    "Embedding dimension mismatch (expected %d, got %d)",
                    self._dimensions,
                    len(vec),
                )
            vectors.append(vec)

        if len(vectors) != len(texts_list):
            logger.warning(
                "Embedding provider returned %d vectors for %d inputs",
                len(vectors),
                len(texts_list),
            )

        return vectors

    def _random_vector(self) -> List[float]:
        # Simple uniform distribution in [-1, 1]. For most retrieval
        # algorithms only direction matters; magnitude does not.
        return [self._rng.uniform(-1.0, 1.0) for _ in range(self._dimensions)]


class RandomEmbedder(EmbedderAdapter):
    """Pure random embedder.

    This is mostly useful as a stand-alone adapter when you want to run
    the full pipeline without configuring any remote embedding provider.
    """

    def __init__(self, dimensions: int, *, seed: int = 42) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self._dimensions = int(dimensions)
        self._rng = random.Random(seed)

    @property
    def dimensions(self) -> int:  # pragma: no cover - trivial
        return self._dimensions

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover - trivial wrapper
        return self._random_vector()

    def embed_passage(self, text: str) -> List[float]:  # pragma: no cover - trivial wrapper
        return self._random_vector()

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._random_vector() for _ in texts]

    def _random_vector(self) -> List[float]:
        return [self._rng.uniform(-1.0, 1.0) for _ in range(self._dimensions)]


@dataclass
class LocalHuggingFaceConfig:
    """Configuration for local HuggingFace/SentenceTransformers models."""
    model_name_or_path: str
    dimensions: int = 1024
    device: str = "cpu"
    normalize_embeddings: bool = True
    query_instruction: str = "" # e.g. "Represent this sentence for searching relevant passages: "
    batch_size: int = 32

class LocalHuggingFaceEmbedder(EmbedderAdapter):
    """Adapter for local SentenceTransformers models (e.g. BGE)."""

    def __init__(self, config: LocalHuggingFaceConfig) -> None:
        self._config = config
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Please install sentence-transformers to use LocalHuggingFaceEmbedder: "
                "pip install sentence-transformers"
            ) from exc

        # Initialize model
        # We assume this is called once per process or handled by Daft
        import torch
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self._is_gguf = str(config.model_name_or_path).lower().endswith(".gguf")
        
        if self._is_gguf:
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "You are trying to use a GGUF model but `llama-cpp-python` is not installed.\n"
                    "Please install it: `pip install llama-cpp-python`"
                ) from e
            
            n_gpu_layers = -1 if device == "cuda" else 0
            logger.info(f"Loading GGUF model: {config.model_name_or_path} (n_gpu_layers={n_gpu_layers})")
            
            self._model = Llama(
                model_path=config.model_name_or_path,
                embedding=True,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            self._dimensions = config.dimensions
        else:
            try:
                self._model = SentenceTransformer(
                    config.model_name_or_path, 
                    device=device
                )
            except Exception as exc:
                msg = str(exc)
                # Handle Network errors by retrying in offline mode
                if "Network is unreachable" in msg or "Connection refused" in msg or "offline" in msg or "Errno 101" in msg:
                    logger.warning(f"Network error during model load: {exc}. Retrying with HF_HUB_OFFLINE=1 (Offline Mode).")
                    import os
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    try:
                        self._model = SentenceTransformer(config.model_name_or_path, device=device)
                    except Exception as e2:
                        # If still fails, re-raise original or new
                        raise e2
                else:
                    raise exc
            
            # Update dimensions from model if possible
            if self._model.get_sentence_embedding_dimension():
                self._dimensions = self._model.get_sentence_embedding_dimension()
            else:
                self._dimensions = config.dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dimensions
            
        if self._is_gguf:
            # GGUF model
            query_text = text
            if self._config.query_instruction:
                query_text = f"{self._config.query_instruction}{text}"
            
            # Llama.create_embedding returns dict with 'data' list
            resp = self._model.create_embedding(query_text)
            embedding = resp['data'][0]['embedding']
            return embedding
        
        # Standard SentenceTransformer
        query_text = text
        if self._config.query_instruction:
            query_text = f"{self._config.query_instruction}{text}"
            
        embedding = self._model.encode(
            query_text,
            normalize_embeddings=self._config.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_passage(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dimensions
        
        if self._is_gguf:
            resp = self._model.create_embedding(text)
            return resp['data'][0]['embedding']
        
        embedding = self._model.encode(
            text,
            normalize_embeddings=self._config.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_passages(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if self._is_gguf:
            # LlamaCpp doesn't support batch embedding natively in one call like ST
            # We loop manually
            results = []
            for t in texts:
                resp = self._model.create_embedding(t)
                results.append(resp['data'][0]['embedding'])
            return results
        
        embeddings = self._model.encode(
            list(texts), 
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
