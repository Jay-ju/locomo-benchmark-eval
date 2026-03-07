import os
import logging
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass
from pathlib import Path

from .prompts.extraction import SESSION_EXTRACTION_PROMPT
from .prompts.judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from .prompts.qa import QA_SYSTEM_PROMPT
from .udfs import read_file_udf, ExtractionUDF, QAUDF, JudgeUDF
from ..adapters.embedder import EmbedderAdapter, OpenAICompatibleEmbedder, OpenAICompatibleConfig
from .models import MemoryEntryPayload, StoredMemoryEntry
from ..adapters.vector_store import VectorStoreAdapter

logger = logging.getLogger(__name__)


@dataclass
class DaftPromptRunner:
    """Run LLM-based distillation over files using Daft for parallelism.

    Parameters
    ----------
    prompt:
        System prompt to use for distillation. Defaults to
        :data:`SESSION_EXTRACTION_PROMPT`.
    openai_base_url:
        Base URL of the OpenAI-compatible HTTP endpoint (e.g.
        ``https://api.openai.com/v1`` or OpenClaw Gateway's ``http://127.0.0.1:18789/v1``).
    openai_api_key:
        API key for the endpoint. When omitted, falls back to
        ``OPENAI_API_KEY`` environment variable (required only when ``dry_run=False``).
    model:
        Chat / Responses model name used for executing the distillation prompt (e.g. ``gpt-4o-mini`` or
        OpenClaw's ``openclaw`` etc.).
    max_tokens:
        Maximum number of tokens generated per distillation call.
    temperature:
        Sampling temperature.
    parallelism:
        Logical parallelism hint, used to control Daft execution scale (currently mainly controls batch size).
    dry_run:
        If True, will not call any remote LLM, but generate one example memory per file,
        useful for verifying Daft flow and vector store writing in no-network/no-key environments.
    """

    prompt: str = SESSION_EXTRACTION_PROMPT
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.2
    parallelism: int = 4
    dry_run: bool = False

    def __post_init__(self) -> None:
        # Lazy import daft so that the rest of the package remains usable
        # without it. If Daft is not available, we fail fast with a clear
        # message.
        try:  # pragma: no cover - optional dependency
            import daft  # type: ignore
            from daft import DataType, col  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DaftPromptRunner requires 'daft' library. Please install it and try again.") from exc

        self._daft = daft
        self._daft_col = col
        self._daft_DataType = DataType

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_paths: Sequence[str]) -> List[Dict[str, Any]]:
        """Run distillation over the given input paths."""

        files = self._expand_input_paths(input_paths)
        if not files:
            logger.warning("DaftPromptRunner: No input files found, skipping distillation.")
            return []

        daft = self._daft
        col = self._daft_col
        DataType = self._daft_DataType

        # Create DataFrame from file paths
        df = daft.from_pydict({"path": files})

        # Apply ReadFileUDF
        df = df.with_column("text", read_file_udf(col("path")))

        # Apply daft.functions.prompt
        if self.dry_run:
            df = df.with_column(
                "entries",
                ExtractionUDF.with_init_args(
                    api_key="",
                    base_url="",
                    model=self.model,
                    prompt=self.prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    dry_run=True
                )(col("text"), col("path"))
            )
        else:
            # Configure Daft provider
            daft.set_provider(
                "openai",
                base_url=self.openai_base_url or os.getenv("OPENAI_BASE_URL"),
                api_key=self.openai_api_key or os.getenv("OPENAI_API_KEY"),
            )
            
            # Use daft.functions.prompt
            # Note: daft.functions.prompt returns a string expression
            # We need to parse it later or use a UDF to parse the result string
            
            # The prompt function usage:
            # daft.functions.prompt(col("text"), system_message=self.prompt, model=self.model)
            
            df = df.with_column(
                "llm_response",
                daft.functions.prompt(
                    col("text"),
                    system_message=self.prompt,
                    model=self.model,
                    temperature=self.temperature,
                    # max_tokens=self.max_tokens  # Removed max_tokens as it causes TypeError
                )
            )
            
            # Parse the JSON string result using a simple UDF
            @daft.udf(return_dtype=DataType.python())
            def parse_response_udf(response_col, path_col):
                import json
                results = []
                for content, path in zip(response_col, path_col):
                    if not content:
                        results.append([])
                        continue
                    
                    content = content.strip()
                    if content.startswith("```"):
                        lines = content.splitlines()
                        if len(lines) >= 3:
                            content = "\n".join(lines[1:-1]).strip()
                    
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            normalized = []
                            for item in parsed:
                                if isinstance(item, dict):
                                    if "metadata" not in item:
                                        item["metadata"] = {}
                                    item["metadata"]["source_path"] = path
                                    normalized.append(item)
                            results.append(normalized)
                        else:
                            results.append([])
                    except Exception as e:
                        logger.error(f"Failed to parse JSON for {path}: {e}")
                        results.append([])
                return results

            df = df.with_column("entries", parse_response_udf(col("llm_response"), col("path")))

        collected = df.collect()
        result = collected.to_pydict()

        all_entries: List[Dict[str, Any]] = []
        entries_col = result.get("entries", [])
        
        for entries in entries_col:
            if not entries:
                continue
            if isinstance(entries, list):
                all_entries.extend(entries)

        return all_entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_input_paths(self, paths: Sequence[str]) -> List[str]:
        files: List[str] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in {".txt", ".md", ".markdown", ".jsonl"}:
                        files.append(str(sub))
            elif p.is_file():
                if p.suffix.lower() in {".txt", ".md", ".markdown", ".jsonl"}:
                    files.append(str(p))
            else:
                logger.warning("DaftPromptRunner: Input path does not exist: %s", p)

        return sorted(files)

    # ------------------------------------------------------------------
    # Storage helper
    # ------------------------------------------------------------------

    def store_to_vector(
        self,
        db_adapter: VectorStoreAdapter,
        embedder: EmbedderAdapter,
        entries_json: List[Dict[str, Any]],
    ) -> int:
        """Embed distilled entries and write them into the vector store."""

        if not entries_json:
            return 0

        payloads: List[MemoryEntryPayload] = []
        for idx, item in enumerate(entries_json):
            try:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                category = item.get("category", "other")
                scope = item.get("scope", "global")

                importance_val = item.get("importance", 0.7)
                try:
                    importance = float(importance_val)
                except Exception:
                    importance = 0.7

                metadata = item.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {"metadata_raw": metadata}

                payloads.append(
                    MemoryEntryPayload(
                        text=text,
                        category=category,  # type: ignore[arg-type]
                        scope=scope,
                        importance=importance,
                        metadata=metadata,
                    ),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("DaftPromptRunner: Skipping invalid entry index=%d: %s", idx, exc)

        if not payloads:
            return 0
        
        # Optimization: Use Daft for parallel embedding if possible
        # We need the embedder configuration to replicate it in workers
        embed_config = getattr(embedder, "_config", None)
        if self._daft and isinstance(embedder, OpenAICompatibleEmbedder) and embed_config:
            try:
                return self._store_with_daft_embedding(db_adapter, embed_config, payloads)
            except Exception as exc:
                logger.warning(
                    "Daft parallel embedding failed: %s. Fallback to serial loop.",
                    exc,
                )

        batch_size = max(1, int(self.parallelism) * 4)
        total_inserted = 0

        for i in range(0, len(payloads), batch_size):
            batch = payloads[i : i + batch_size]
            texts = [p.text for p in batch]

            vectors = embedder.embed_passages(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedding returned {len(vectors)} vectors but expected {len(batch)}",
                )

            entries: List[StoredMemoryEntry] = [
                StoredMemoryEntry.from_payload(payload, vector)
                for payload, vector in zip(batch, vectors)
            ]

            inserted = db_adapter.insert_memories(entries)
            total_inserted += inserted

        return total_inserted

    def _store_with_daft_embedding(
        self,
        db_adapter: VectorStoreAdapter,
        embed_config: OpenAICompatibleConfig,
        payloads: List[MemoryEntryPayload],
    ) -> int:
        """Embed payloads in parallel using Daft UDF, then insert."""
        
        batch_size = max(1, int(self.parallelism) * 4)
        
        # 1. Group payloads into batches to process in UDF
        batches = []
        for i in range(0, len(payloads), batch_size):
            chunk = payloads[i : i + batch_size]
            batches.append({
                "batch_id": i,
                "texts": [p.text for p in chunk],
            })

        daft = self._daft
        col = self._daft_col
        DataType = self._daft_DataType
        
        df = daft.from_pydict({"data": batches})
        
        # 2. Define UDF
        def _embed_batch(data: Dict[str, Any]) -> List[List[float]]:
            texts = data.get("texts", [])
            if not texts:
                return []
            
            # Re-init embedder
            embedder = OpenAICompatibleEmbedder(embed_config)
            return embedder.embed_passages(texts)

        # 3. Apply UDF
        df_with = df.with_column(
            "vectors",
            col("data").apply(_embed_batch, return_dtype=DataType.python()),
        )
        
        collected = df_with.collect().to_pydict()
        
        # 4. Reconstruct and insert
        total_inserted = 0
        data_col = collected.get("data", [])
        vectors_col = collected.get("vectors", [])
        
        batch_map = {}
        for d, v in zip(data_col, vectors_col):
            if isinstance(d, dict) and isinstance(v, list):
                bid = d.get("batch_id")
                batch_map[bid] = v
                
        for i in range(0, len(payloads), batch_size):
            chunk = payloads[i : i + batch_size]
            vectors = batch_map.get(i)
            
            if not vectors or len(vectors) != len(chunk):
                logger.warning("Batch embedding missing or size mismatch for batch %d", i)
                continue
                
            entries = [
                StoredMemoryEntry.from_payload(p, v)
                for p, v in zip(chunk, vectors)
            ]
            
            total_inserted += db_adapter.insert_memories(entries)
            
        return total_inserted

    # ------------------------------------------------------------------
    # QA helper
    # ------------------------------------------------------------------

    def answer_batch(self, qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch QA answering using Daft."""
        if not qa_items:
            return []

        daft = self._daft
        col = self._daft_col
        DataType = self._daft_DataType

        questions = [item.get("question", "") for item in qa_items]
        contexts = [item.get("context", "") for item in qa_items]
        
        df = daft.from_pydict({
            "question": questions,
            "context": contexts,
            "original_item": qa_items # Daft handles list of dicts
        })

        # Use QAUDF (Class UDF)
        # We need to call .with_init_args() on the UDF wrapper first
        df = df.with_column(
            "answer",
            QAUDF.with_init_args(
                api_key=self.openai_api_key or os.getenv("OPENAI_API_KEY") or "",
                base_url=self.openai_base_url or "",
                model=self.model,
                prompt=self.prompt, # This will be the QA system prompt
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                dry_run=self.dry_run
            )(col("question"), col("context"))
        )

        collected = df.collect().to_pydict()
        
        results = []
        answers = collected.get("answer", [])
        originals = collected.get("original_item", [])
        
        for ans, item in zip(answers, originals):
            if isinstance(item, dict):
                item["answer"] = ans
                results.append(item)
        
        return results

    # ------------------------------------------------------------------
    # Judge helper
    # ------------------------------------------------------------------

    def judge_batch(self, qa_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch LLM-as-a-Judge evaluation using Daft."""
        if not qa_results:
            return []

        daft = self._daft
        col = self._daft_col
        DataType = self._daft_DataType

        questions = [item.get("question", "") for item in qa_results]
        answers = [item.get("answer", "") for item in qa_results]
        ground_truths = [item.get("ground_truth", "") for item in qa_results]
        
        df = daft.from_pydict({
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "original_item": qa_results
        })

        # Use JudgeUDF (Class UDF)
        df = df.with_column(
            "judge_result",
            JudgeUDF.with_init_args(
                api_key=self.openai_api_key or os.getenv("OPENAI_API_KEY") or "",
                base_url=self.openai_base_url or "",
                model=self.model,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_template=JUDGE_USER_TEMPLATE,
                dry_run=self.dry_run
            )(col("question"), col("answer"), col("ground_truth"))
        )

        collected = df.collect().to_pydict()
        
        results = []
        judge_results = collected.get("judge_result", [])
        originals = collected.get("original_item", [])
        
        for jr, item in zip(judge_results, originals):
            if isinstance(item, dict):
                item["judge_result"] = jr
                results.append(item)
        
        return results