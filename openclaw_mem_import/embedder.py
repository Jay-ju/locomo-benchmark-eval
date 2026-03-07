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

        if self._config.dry_run or self._client is None:
            # Deterministic pseudo-random vectors for reproducible tests.
            return [self._random_vector() for _ in texts_list]

        payload: dict = {
            "model": self._config.model,
            "input": texts_list,
            "encoding_format": "float",
        }
        if task:
            payload["task"] = task
        if self._dimensions:
            payload["dimensions"] = self._dimensions

        try:
            response = self._client.embeddings.create(**payload)  # type: ignore[operator]
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
