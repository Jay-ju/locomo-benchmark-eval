from __future__ import annotations

"""Search utilities built on top of the existing adapters.

This module provides :class:`SearchRunner`, which wires together an
``EmbedderAdapter`` and a ``VectorStoreAdapter`` to perform vector-based
search over the ``memories`` table.

Key goals:

- Keep the interface *system-neutral* – no direct dependency on OpenClaw.
- Make the result shape convenient for CLI / JSON output
  (id/text/category/scope/importance/timestamp/metadata/score).
- Use Daft for **batch** search concurrency when available, while keeping
  single-query search usable without Daft.

Only vector search is implemented at the moment (``mode="vector"``). A
future ``"hybrid"`` mode could combine BM25 + vector similar to
``memory-lancedb-pro``.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from .udfs import SearchUDF
from ..adapters.embedder import EmbedderAdapter, OpenAICompatibleEmbedder
from .models import StoredMemoryEntry
from ..adapters.vector_store import VectorStoreAdapter

logger = logging.getLogger(__name__)


SearchMode = Literal["vector"]  # placeholder for future: "hybrid"


@dataclass
class SearchRunner:
    """High-level search orchestrator.

    Parameters
    ----------
    store:
        Vector store adapter that provides ``vector_search(...)``.
    embedder:
        Embedding adapter used to turn queries into vectors.
    parallelism:
        Logical parallelism hint used by Daft in ``search_batch``.
    mode:
        Search mode. Currently only ``"vector"`` is implemented.
    """

    store: VectorStoreAdapter
    embedder: EmbedderAdapter
    parallelism: int = 4
    mode: SearchMode = "vector"

    # ------------------------------------------------------------------
    # Single-query API
    # ------------------------------------------------------------------

    def search_one(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search a single query and return ranked memory dicts.

        The return value is a list of dicts with the following keys:

        - ``id``
        - ``text``
        - ``category``
        - ``scope``
        - ``importance``
        - ``timestamp``
        - ``metadata`` (parsed JSON when possible)
        - ``score`` (float)
        """

        if not query or not str(query).strip():
            return []

        if self.mode != "vector":  # pragma: no cover - future extension guard
            raise ValueError(f"Unsupported search mode: {self.mode}")

        vector = self.embedder.embed_query(str(query))
        raw_results = self.store.vector_search(vector, top_k=top_k, min_score=min_score)
        return [self._normalize_result(r) for r in raw_results]

    # ------------------------------------------------------------------
    # Batch API (Daft-based)
    # ------------------------------------------------------------------

    def search_batch(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[List[Dict[str, Any]]]:
        """Batch search using Daft for per-query parallelism.

        Parameters
        ----------
        queries:
            A sequence of user queries (one per question).

        Returns
        -------
        List[List[dict]]
            For each input query (same index), a list of result dicts with
            the same shape as :meth:`search_one`.

        Notes
        -----
        - This method *requires* ``daft`` to be installed. If daft is not
          available, a :class:`RuntimeError` is raised with a clear message.
        - The embedder and vector store are re-used inside the UDF, relying
          on Daft's ability to execute Python UDFs in the same process for
          single-node workloads.
        """

        # Normalise input and preserve original indices so we can
        # reconstruct output alignment even if some queries are empty.
        indexed: List[Dict[str, Any]] = []
        for idx, q in enumerate(queries):
            text = (q or "").strip()
            if not text:
                continue
            indexed.append({"index": idx, "query": text})

        if not indexed:
            return [[] for _ in queries]

        try:  # pragma: no cover - optional dependency
            import daft  # type: ignore
            from daft import DataType, col  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SearchRunner.search_batch requires 'daft' library. Please install it and try again.",
            ) from exc

        df = daft.from_pydict({"payload": indexed})

        # Prepare config for UDF
        embed_config_json = "{}"
        if hasattr(self.embedder, "_config"):
             # Assuming OpenAICompatibleEmbedder
             import dataclasses, json
             # We need to serialize the dataclass. asdict might work if fields are simple.
             try:
                 cfg = dataclasses.asdict(self.embedder._config)
                 embed_config_json = json.dumps(cfg)
             except Exception as e:
                 logger.warning("Failed to serialize embedder config: %s", e)
        
        store_config_json = "{}"
        if hasattr(self.store, "config") and self.store.config:
             import dataclasses, json
             try:
                 cfg = dataclasses.asdict(self.store.config)
                 # Add path if not in config
                 if hasattr(self.store, "_db_path"):
                     cfg["db_path"] = str(self.store._db_path)
                 if hasattr(self.store, "_table_name"):
                     cfg["table_name"] = self.store._table_name
                 store_config_json = json.dumps(cfg)
             except Exception as e:
                 logger.warning("Failed to serialize store config: %s", e)
        
        # Apply SearchUDF (Class UDF)
        df_with = df.with_column(
            "results",
            SearchUDF.with_init_args(
                embed_config_json=embed_config_json,
                store_config_json=store_config_json,
                top_k=top_k,
                min_score=min_score,
                mode=self.mode
            )(col("payload"))
        )
        
        collected = df_with.collect().to_pydict()

        payload_col = collected.get("payload", [])
        results_col = collected.get("results", [])

        # Build a mapping from original index -> result list
        by_index: Dict[int, List[Dict[str, Any]]] = {}
        for payload, res in zip(payload_col, results_col):
            if not isinstance(payload, dict):
                continue
            idx_val = payload.get("index")
            if not isinstance(idx_val, int):
                continue
            if isinstance(res, list):
                by_index[idx_val] = res
            else:
                by_index[idx_val] = []

        # Reconstruct output aligned to the original queries order
        output: List[List[Dict[str, Any]]] = []
        for i in range(len(queries)):
            output.append(by_index.get(i, []))

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_result(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a raw result row from the vector store.

        The LanceDB adapter already returns a dict with scalar fields and
        a ``score`` key. This helper adds minimal type coercions and JSON
        parsing for ``metadata``.
        """

        # Shallow copy to avoid mutating the original dict from adapter.
        r = dict(row)

        # Metadata is stored as JSON string in the table; parse to dict
        # where possible to make downstream consumption easier.
        metadata = r.get("metadata")
        if isinstance(metadata, str):
            try:
                r["metadata"] = json.loads(metadata)
            except Exception:
                # Keep raw string if parsing fails.
                pass

        # Scope defaulting for backward compatibility.
        scope = r.get("scope")
        if scope is None:
            r["scope"] = "global"

        # Score as float.
        if "score" in r:
            try:
                r["score"] = float(r["score"])
            except Exception:
                r["score"] = 0.0

        return r
