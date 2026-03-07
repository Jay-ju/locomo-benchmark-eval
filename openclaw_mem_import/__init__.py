"""openclaw_mem_import

Generic LanceDB-compatible memory import toolkit for OpenClaw-style long-term memory.

This package provides:
- EmbedderAdapter: pluggable embedding backends (OpenAI-compatible + dry-run).
- VectorStoreAdapter: pluggable vector stores (LanceDB by default).
- Batch import pipelines based on Daft (with graceful fallback to local processing).
- A Typer-based CLI exposed as ``add_lance_memory``.
- A Daft-based prompt runner for session/text distillation.
"""

from . import models, embedder, vector_store, pipeline, distill_prompts, daft_prompt  # noqa: F401

__all__ = [
    "models",
    "embedder",
    "vector_store",
    "pipeline",
    "distill_prompts",
    "daft_prompt",
]

__version__ = "0.1.0"
