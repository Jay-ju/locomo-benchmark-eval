"""locomo_eval

Locomo Benchmark Eval (locomo_eval) - Universal Memory Import & Evaluation CLI for the OpenClaw ecosystem.

This package provides:
- EmbedderAdapter: pluggable embedding backends (OpenAI-compatible + dry-run).
- VectorStoreAdapter: pluggable vector stores (LanceDB by default).
- Batch import pipelines based on Daft (with graceful fallback to local processing).
- A Typer-based CLI exposed as ``locomo_eval``.
- A Daft-based prompt runner for session/text distillation.
"""

from .core import models, pipeline, daft_runner, search
from .core.prompts import extraction
from .adapters import embedder, vector_store

__all__ = [
    "models",
    "embedder",
    "vector_store",
    "pipeline",
    "extraction",
    "daft_runner",
    "search",
]

__version__ = "0.1.0"
