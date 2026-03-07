from __future__ import annotations

"""Vector store abstraction layer.

Currently only LanceDB is implemented, but the interface is designed so
that FAISS / Milvus / internal systems (e.g. Viking) can plug in later
without changing the import or retrieval pipeline.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

from .models import StoredMemoryEntry

logger = logging.getLogger(__name__)


class VectorStoreAdapter(Protocol):
    """Abstract vector store interface used by the pipelines.

    The interface is intentionally small: batch insert + optional flush
    + vector search. Higher level logic (hybrid retrieval / rerank) is
    implemented in separate components.
    """

    def insert_memories(self, entries: Sequence[StoredMemoryEntry]) -> int:
        """Insert a batch of memories.

        Implementations should be idempotent with respect to the
        underlying table schema (i.e. creating the table if it does not
        exist) but **not** necessarily de-duplicate by content.
        """

    def flush(self) -> None:
        """Flush any buffered writes (optional)."""

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Vector search over the stored memories.

        Parameters
        ----------
        query_vector:
            Embedding vector of the query.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum similarity score (0–1) to keep a result.

        Returns
        -------
        List[dict]
            Each dict should contain at least:

            - ``id``
            - ``text``
            - ``category``
            - ``scope``
            - ``importance``
            - ``timestamp``
            - ``metadata`` (usually JSON string)
            - ``score`` (float)
        """


@dataclass
class LanceDBConfig:
    """Configuration for LanceDB-backed vector store."""

    db_path: str
    table_name: str = "memories"
    vector_dim: int = 1024


class LanceDBVectorStore(VectorStoreAdapter):
    """Simple LanceDB-backed implementation of :class:`VectorStoreAdapter`.

    The table schema is aligned with ``memory-lancedb-pro``:

    - id: string (UUID)
    - text: string
    - vector: float[]
    - category: string
    - scope: string
    - importance: float
    - timestamp: int64 (ms)
    - metadata: string (JSON)
    """

    def __init__(self, config: LanceDBConfig) -> None:
        self.config = config
        self._db = None
        self._table = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        if self._table is not None:
            return

        import lancedb  # type: ignore

        db_path = os.path.expanduser(self.config.db_path)
        Path(db_path).mkdir(parents=True, exist_ok=True)

        db = lancedb.connect(db_path)
        table_name = self.config.table_name

        try:
            table = db.open_table(table_name)
            logger.info("Opened existing LanceDB table '%s' at %s", table_name, db_path)
        except Exception:  # pragma: no cover - table creation path
            # Create a schema entry row similar to memory-lancedb-pro so
            # that the Lance schema is fully specified from the start.
            schema_entry = {
                "id": "__schema__",
                "text": "",
                "vector": [0.0] * self.config.vector_dim,
                "category": "other",
                "scope": "global",
                "importance": 0.0,
                "timestamp": 0,
                "metadata": "{}",
            }
            table = db.create_table(table_name, [schema_entry])
            try:
                table.delete(where="id = '__schema__'")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to delete schema marker row: %s", exc)

            logger.info("Created new LanceDB table '%s' at %s", table_name, db_path)

        self._db = db
        self._table = table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_memories(self, entries: Sequence[StoredMemoryEntry]) -> int:
        if not entries:
            return 0

        self._ensure_table()

        # Vector dimension check at the adapter boundary. This protects us
        # from accidentally mixing tables created with different embedding
        # models.
        for e in entries:
            if len(e.vector) != self.config.vector_dim:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.config.vector_dim}, got {len(e.vector)}",
                )

        records: List[dict] = [
            {
                "id": e.id,
                "text": e.text,
                "vector": e.vector,
                "category": e.category,
                "scope": e.scope,
                "importance": float(e.importance),
                "timestamp": int(e.timestamp),
                "metadata": e.metadata,
            }
            for e in entries
        ]

        assert self._table is not None  # for type checkers
        self._table.add(records)
        return len(records)

    def flush(self) -> None:  # pragma: no cover - no-op for LanceDB
        # LanceDB writes are synchronous for the Python client, so there
        # is nothing to flush here. The method exists to satisfy the
        # interface and to keep a consistent API for future adapters.
        return None

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Vector search using LanceDB's search API.

        The scoring semantics follow ``memory-lancedb-pro``: LanceDB's
        internal distance (``_distance``) is converted to a similarity
        score via ``score = 1 / (1 + distance)`` and filtered by
        ``min_score``. Results are truncated to ``top_k``.
        """

        if top_k <= 0:
            return []

        if len(query_vector) != self.config.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.config.vector_dim}, got {len(query_vector)}",
            )

        self._ensure_table()
        assert self._table is not None

        # Over-fetch to give min_score some headroom, mirroring the TS
        # implementation in memory-lancedb-pro.
        safe_k = max(1, int(top_k))
        fetch_limit = min(safe_k * 10, 200)

        table = self._table
        # Best-effort support for different LanceDB Python APIs.
        search_method = getattr(table, "search", None)
        if callable(search_method):
            query = search_method(list(query_vector))
        else:  # pragma: no cover - fallback path
            alt = getattr(table, "vector_search", None) or getattr(table, "vectorSearch", None)
            if not callable(alt):
                raise RuntimeError(
                    "LanceDB table does not expose a compatible vector search API ('search' or 'vector_search').",
                )
            query = alt(list(query_vector))

        try:  # pragma: no cover - depends on lancedb runtime
            rows = query.limit(fetch_limit).to_list()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("LanceDB vector_search failed: %s", exc)
            raise

        results: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            try:
                dist = float(row.get("_distance", 0.0))
            except Exception:
                dist = 0.0
            score = 1.0 / (1.0 + dist)
            if score < min_score:
                continue

            scope_val = row.get("scope")
            scope_str = scope_val if isinstance(scope_val, str) and scope_val else "global"

            entry: Dict[str, Any] = {
                "id": row.get("id"),
                "text": row.get("text", ""),
                "vector": list(row.get("vector") or []),
                "category": row.get("category", "other"),
                "scope": scope_str,
                "importance": float(row.get("importance", 0.0)),
                "timestamp": int(row.get("timestamp", 0)),
                "metadata": row.get("metadata", "{}"),
                "score": score,
            }
            results.append(entry)

            if len(results) >= safe_k:
                break

        return results


# ----------------------------------------------------------------------
# Adapter stubs for other vector stores
# ----------------------------------------------------------------------

# The following classes are intentionally left as thin stubs / TODOs so
# that downstream users understand how to plug in their own vector
# stores. Implementations are **not** provided in this repo to keep
# dependencies minimal.


class FaissVectorStore(VectorStoreAdapter):  # pragma: no cover - placeholder
    """Placeholder for a FAISS-backed implementation.

    To implement this adapter, you would:
    1. Maintain a FAISS index for the `vector` column.
    2. Persist metadata (id/text/category/scope/importance/timestamp/
       metadata) in a sidecar store (e.g. SQLite, DuckDB, or Parquet).
    3. Ensure that `insert_memories` appends both the vectors and the
       metadata transactionally.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError("FAISS adapter is not implemented in this repo.")

    def insert_memories(self, entries: Sequence[StoredMemoryEntry]) -> int:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class MilvusVectorStore(VectorStoreAdapter):  # pragma: no cover - placeholder
    """Placeholder for a Milvus-backed implementation.

    A Milvus adapter would map the LanceDB schema into a Milvus
    collection and insert vectors + payload in batches.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError("Milvus adapter is not implemented in this repo.")

    def insert_memories(self, entries: Sequence[StoredMemoryEntry]) -> int:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class VikingVectorStore(VectorStoreAdapter):  # pragma: no cover - placeholder
    """Placeholder for an internal Viking-backed implementation.

    A Viking adapter is expected to wrap the corresponding HTTP/gRPC
    API while preserving the same logical schema (id/text/vector/
    category/scope/importance/timestamp/metadata).
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError("Viking adapter is not implemented in this repo.")

    def insert_memories(self, entries: Sequence[StoredMemoryEntry]) -> int:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def create_vector_store(store_type: str, **kwargs: object) -> VectorStoreAdapter:
    """Factory to create a :class:`VectorStoreAdapter`.

    Parameters
    ----------
    store_type:
        Backend type. Current options: ``lancedb`` (implemented),
        ``faiss`` / ``milvus`` / ``viking`` (placeholders raising
        :class:`NotImplementedError`).

    Common keyword arguments
    ------------------------
    For ``store_type="lancedb"`` the following kwargs are recognised:

    - ``db_path`` (str, required): LanceDB directory path.
    - ``table_name`` (str, optional): table name, default ``"memories"``.
    - ``vector_dim`` (int, optional): expected vector dimension, default 1024.

    Other store types may accept different kwargs; see their adapter
    implementations when they are added.
    """

    normalized = store_type.lower()

    if normalized == "lancedb":
        db_path_obj = kwargs.get("db_path")
        if not isinstance(db_path_obj, str):
            raise ValueError("db_path (str) is required for LanceDB vector store")

        table_name_obj = kwargs.get("table_name", "memories")
        if not isinstance(table_name_obj, str):
            raise ValueError("table_name must be a string for LanceDB vector store")

        vector_dim_obj = kwargs.get("vector_dim", 1024)
        if not isinstance(vector_dim_obj, int):
            raise ValueError("vector_dim must be an int for LanceDB vector store")

        cfg = LanceDBConfig(
            db_path=db_path_obj,
            table_name=table_name_obj,
            vector_dim=vector_dim_obj,
        )
        return LanceDBVectorStore(cfg)

    if normalized == "faiss":
        return FaissVectorStore(**kwargs)

    if normalized == "milvus":
        return MilvusVectorStore(**kwargs)

    if normalized == "viking":
        return VikingVectorStore(**kwargs)

    raise ValueError(
        f"Unsupported store_type: {store_type}. Expected one of: lancedb | faiss | milvus | viking.",
    )
