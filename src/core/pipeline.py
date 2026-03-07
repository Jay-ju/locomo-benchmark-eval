from __future__ import annotations

"""Import pipeline built on top of Daft + pluggable adapters.

The pipeline is intentionally small and focused:

- It knows how to expand directories and discover input files.
- It parses Markdown / JSONL / LoCoMo result JSON into ``MemoryEntryPayload``.
- It batches texts, calls an :class:`EmbedderAdapter`, and writes
  :class:`StoredMemoryEntry` objects into a :class:`VectorStoreAdapter`.

Daft is used opportunistically when available (for larger datasets) but
falls back to a simple in-process loop so that small demos and tests do
not depend on a distributed runtime.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..adapters.embedder import EmbedderAdapter
from .models import MemoryEntryPayload, StoredMemoryEntry
from ..adapters.vector_store import VectorStoreAdapter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration knobs for the import pipeline."""

    batch_size: int = 32
    use_daft: bool = True
    parallelism: int = 4
    max_locomo_items_per_file: int = 32


class ImportPipeline:
    """High-level batch import pipeline.

    This class is deliberately minimal and keeps most logic in pure
    Python so that it can run inside agent sandboxes, notebooks, or
    simple scripts without complex orchestration.
    """

    def __init__(
        self,
        embedder: EmbedderAdapter,
        store: VectorStoreAdapter,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or PipelineConfig()

        self._daft = None
        if self.config.use_daft:
            try:  # pragma: no cover - optional dependency
                import daft  # type: ignore

                self._daft = daft
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.info("Daft is not available: %s. Falling back to local loop.", exc)
                self._daft = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_direct_import(
        self,
        input_paths: Sequence[Path],
        *,
        input_format: str = "auto",
    ) -> int:
        """Directly import structured memories from files.

        ``input_format`` can be one of:

        - ``auto``          : infer from file extension / filename
        - ``markdown``      : each ``.md`` file becomes one memory
        - ``jsonl``         : each line is a JSON object with memory fields
        - ``locomo-result`` : MemoryLake LoCoMo result JSON files
        - ``text``          : plain text files, one memory per file
        """

        files = self._expand_input_paths(input_paths)
        if not files:
            logger.warning("No input files found for direct-import")
            return 0

        payloads: List[MemoryEntryPayload] = []
        for path in files:
            parsed = self._parse_file(path, input_format=input_format)
            if not parsed:
                continue
            payloads.extend(parsed)

        if not payloads:
            logger.warning("No MemoryEntry payloads generated from input")
            return 0

        return self._embed_and_store(payloads)

    # ------------------------------------------------------------------
    # Embedding + storage
    # ------------------------------------------------------------------

    def _embed_and_store(self, payloads: Sequence[MemoryEntryPayload]) -> int:
        batch_size = max(1, int(self.config.batch_size))
        total_inserted = 0

        for i in range(0, len(payloads), batch_size):
            batch = list(payloads[i : i + batch_size])
            texts = [p.text for p in batch]

            vectors = self.embedder.embed_passages(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding adapter returned {len(vectors)} vectors for {len(batch)} payloads",
                )

            entries: List[StoredMemoryEntry] = [
                StoredMemoryEntry.from_payload(payload, vector)
                for payload, vector in zip(batch, vectors)
            ]

            inserted = self.store.insert_memories(entries)
            total_inserted += inserted

            logger.info(
                "Inserted %d memories (batch %d-%d)",
                inserted,
                i,
                i + len(batch),
            )

        # Most vector stores used here are synchronous, but we keep a
        # flush hook for completeness / future adapters.
        self.store.flush()
        return total_inserted

    # ------------------------------------------------------------------
    # File discovery & parsing
    # ------------------------------------------------------------------

    def _expand_input_paths(self, input_paths: Sequence[Path]) -> List[Path]:
        files: List[Path] = []
        for raw in input_paths:
            path = Path(raw)
            if path.is_dir():
                for sub in path.rglob("*"):
                    if sub.is_file():
                        files.append(sub)
            elif path.is_file():
                files.append(path)
            else:
                logger.warning("Input path does not exist: %s", path)

        # Deterministic order for easier debugging
        return sorted(files)

    def _infer_category_from_path(self, path: Path) -> str:
        normalized = str(path).replace(os.sep, "/")
        name = path.name.lower()

        if "/preferences/" in normalized or name.startswith("profile"):
            return "preference"
        if "/events/" in normalized:
            return "fact"
        if "/entities/" in normalized:
            return "entity"
        return "other"

    def _parse_file(self, path: Path, *, input_format: str) -> List[MemoryEntryPayload]:
        fmt = input_format
        if fmt == "auto":
            suffix = path.suffix.lower()
            if suffix in {".md", ".markdown"}:
                fmt = "markdown"
            elif suffix == ".jsonl":
                fmt = "jsonl"
            elif suffix == ".json" and path.name.startswith("locomo_result_"):
                fmt = "locomo-result"
            else:
                fmt = "text"

        if fmt == "markdown":
            return self._parse_markdown_file(path)
        if fmt == "jsonl":
            return self._parse_jsonl_file(path)
        if fmt == "locomo-result":
            return self._parse_locomo_result_file(path)
        if fmt == "text":
            return self._parse_plain_text_file(path)

        raise ValueError(f"Unsupported input_format: {fmt}")

    # Individual parsers ------------------------------------------------

    def _parse_markdown_file(self, path: Path) -> List[MemoryEntryPayload]:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pragma: no cover - IO errors
            logger.warning("Failed to read markdown file %s: %s", path, exc)
            return []

        if not text:
            return []

        category = self._infer_category_from_path(path)
        scope = "global"

        metadata = {
            "source": "markdown",
            "source_path": str(path),
        }

        return [
            MemoryEntryPayload(
                text=text,
                category=category,  # type: ignore[arg-type]
                scope=scope,
                importance=0.8 if category in {"preference", "fact", "entity"} else 0.7,
                metadata=metadata,
            )
        ]

    def _parse_plain_text_file(self, path: Path) -> List[MemoryEntryPayload]:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pragma: no cover - IO errors
            logger.warning("Failed to read text file %s: %s", path, exc)
            return []

        if not text:
            return []

        metadata = {
            "source": "text",
            "source_path": str(path),
        }

        return [
            MemoryEntryPayload(
                text=text,
                category="other",  # type: ignore[arg-type]
                scope="global",
                importance=0.6,
                metadata=metadata,
            )
        ]

    def _parse_jsonl_file(self, path: Path) -> List[MemoryEntryPayload]:
        payloads: List[MemoryEntryPayload] = []

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skip invalid JSON line in %s: %s", path, exc)
                        continue

                    text = (obj.get("text") or obj.get("content") or "").strip()
                    if not text:
                        continue

                    category = obj.get("category", "other")
                    scope = obj.get("scope", "global")
                    importance = float(obj.get("importance", 0.7))
                    metadata = obj.get("metadata") or {}

                    if not isinstance(metadata, dict):
                        metadata = {"metadata_raw": metadata}

                    metadata.setdefault("source", "jsonl")
                    metadata.setdefault("source_path", str(path))

                    payloads.append(
                        MemoryEntryPayload(
                            text=text,
                            category=category,  # type: ignore[arg-type]
                            scope=scope,
                            importance=importance,
                            metadata=metadata,
                            id=obj.get("id"),
                            timestamp=obj.get("timestamp"),
                        ),
                    )
        except Exception as exc:  # pragma: no cover - IO errors
            logger.warning("Failed to read jsonl file %s: %s", path, exc)

        return payloads

    def _parse_locomo_result_file(self, path: Path) -> List[MemoryEntryPayload]:
        """Parse MemoryLake LoCoMo benchmark result JSON.

        Each file (``locomo_result_*.json``) contains an array of
        question-level evaluation results with fields like:

        - ``category``
        - ``ground_truth``
        - ``question``
        - ``question_index``
        - ``score``
        """

        max_items = max(1, int(self.config.max_locomo_items_per_file))

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:  # pragma: no cover - IO errors
            logger.warning("Failed to read locomo result file %s: %s", path, exc)
            return []

        if not isinstance(data, list):
            logger.warning("Unexpected JSON structure in %s: expected list", path)
            return []

        record_suffix = path.stem.split("_")[-1]
        record_id = f"R{record_suffix}" if record_suffix.isdigit() else path.stem

        payloads: List[MemoryEntryPayload] = []
        for obj in data[:max_items]:
            if not isinstance(obj, dict):
                continue

            category_raw = obj.get("category", "?")
            q_index = obj.get("question_index")

            gt_val = obj.get("ground_truth")
            ground_truth = str(gt_val).strip() if gt_val is not None else ""

            score = obj.get("score")

            q_val = obj.get("question")
            question = str(q_val).strip() if q_val is not None else ""

            if not ground_truth and not question:
                continue

            text_parts = [
                f"LoCoMo record {record_id}, category {category_raw}, question {q_index} evaluation result.",
            ]
            if question:
                text_parts.append(f"Question: {question}")
            if ground_truth:
                text_parts.append(f"Ground Truth: {ground_truth}")
            if score is not None:
                text_parts.append(f"MemoryLake F1 score for this question is approximately {score}.")

            text = " ".join(text_parts)

            keywords_line = f"Keywords: LoCoMo; {record_id}; {category_raw}; evaluation; F1; MemoryLake"

            metadata = {
                "source": "locomo_result",
                "source_path": str(path),
                "record_id": record_id,
                "question_index": q_index,
                "category_raw": category_raw,
                "score": score,
                "keywords_line": keywords_line,
            }

            payloads.append(
                MemoryEntryPayload(
                    text=text,
                    category="fact",  # type: ignore[arg-type]
                    scope="global",
                    importance=0.6,
                    metadata=metadata,
                ),
            )

        return payloads
