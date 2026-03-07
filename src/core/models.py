from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


MemoryCategory = Literal[
    "preference",
    "fact",
    "decision",
    "entity",
    "other",
    "reflection",
]


@dataclass
class MemoryEntryPayload:
    """Input memory payload before embedding.

    This is the structure that parsers (markdown / jsonl / custom loaders)
    should produce. It intentionally does *not* require id / vector / timestamp;
    those will be filled by the pipeline + vector store.
    """

    text: str
    category: MemoryCategory = "other"
    scope: str = "global"
    importance: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional pre-specified id / timestamp / vector for advanced migrations.
    id: Optional[str] = None
    timestamp: Optional[int] = None
    vector: Optional[List[float]] = None


@dataclass
class StoredMemoryEntry:
    """Fully materialized memory ready to be written into LanceDB.

    This aligns with the LanceDB schema used by memory-lancedb-pro:

    - id: string (UUID)
    - text: string
    - vector: float[]
    - category: string enum
    - scope: string
    - importance: float
    - timestamp: int64 (ms)
    - metadata: JSON string
    """

    id: str
    text: str
    vector: List[float]
    category: MemoryCategory
    scope: str
    importance: float
    timestamp: int
    metadata: str

    @classmethod
    def from_payload(
        cls,
        payload: MemoryEntryPayload,
        vector: List[float],
        *,
        now_ms: Optional[int] = None,
    ) -> "StoredMemoryEntry":
        if not payload.text or not payload.text.strip():
            raise ValueError("Memory text must be non-empty")

        ts = payload.timestamp if payload.timestamp and payload.timestamp > 0 else int(
            now_ms or time.time() * 1000
        )
        importance = payload.importance
        if not (0.0 <= importance <= 1.0):
            # Clamp to [0, 1]
            importance = max(0.0, min(1.0, importance))

        import json

        metadata_str: str
        if isinstance(payload.metadata, str):
            metadata_str = payload.metadata
        else:
            metadata_str = json.dumps(payload.metadata or {}, ensure_ascii=False)

        return cls(
            id=payload.id or str(uuid.uuid4()),
            text=payload.text.strip(),
            vector=vector,
            category=payload.category,
            scope=payload.scope or "global",
            importance=float(importance),
            timestamp=int(ts),
            metadata=metadata_str,
        )
