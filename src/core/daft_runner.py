import os
import logging
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass
from pathlib import Path

from .prompts.extraction import SESSION_EXTRACTION_PROMPT
from .prompts.judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from .prompts.qa import QA_SYSTEM_PROMPT
from .udfs import make_judge_udf, make_qa_udf, SearchUDF, read_file_udf
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
    mode: str = "agent"
    memory_type: str = "lancedb"

    def __post_init__(self) -> None:
        # Lazy import daft
        try:  # pragma: no cover - optional dependency
            import daft  # type: ignore
            from daft import DataType, col  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DaftPromptRunner requires 'daft' library. Please install it and try again.") from exc

        self._daft = daft
        self._daft_col = col
        self._daft_DataType = DataType
        
        # Load Memory Plugin
        from .memory_plugins.registry import get_plugin
        self.plugin = get_plugin(self.memory_type)
        
        # If prompt is None or default, use plugin's prompt
        # Note: SESSION_EXTRACTION_PROMPT is the default value in signature.
        # But if user doesn't pass it, it is equal.
        # We assume if it equals SESSION_EXTRACTION_PROMPT, we can override with plugin's default.
        if self.prompt == SESSION_EXTRACTION_PROMPT:
            self.prompt = self.plugin.get_extraction_prompt()

    def _resolve_llm_api_key(self) -> str:
        """Resolve API key for chat-based LLM calls used by Daft UDFs.

        Preference order:
        1) Explicit openai_api_key passed to DaftPromptRunner
        2) OPENAI_API_KEY
        3) VOLCENGINE_API_KEY
        4) ARK_API_KEY
        """
        return (
            self.openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("VOLCENGINE_API_KEY")
            or os.getenv("ARK_API_KEY")
            or ""
        )

    def _is_agent_mode(self, base_url: str) -> bool:
        return (self.mode or "").lower() == "agent"

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

        # Delegate to run_dataframe for unified processing
        processed_df = self.run_dataframe(
            df, 
            prompt_type="ingest", 
            text_col="text", 
            id_col="path"
        )

        collected = processed_df.collect()
        result = collected.to_pydict()

        effective_base_url = self.openai_base_url or os.getenv("OPENAI_BASE_URL") or ""
        is_agent = self._is_agent_mode(effective_base_url)

        all_entries: List[Dict[str, Any]] = []
        
        if is_agent:
            # Agent mode returns strings in "ingest_result"
            results = result.get("ingest_result", [])
            for r in results:
                # Wrap simple status in a structure consistent enough for reporting
                all_entries.append({"agent_response": r})
        else:
            # Direct mode returns list of dicts in "entries"
            entries_col = result.get("entries", [])
            for entries in entries_col:
                if not entries:
                    continue
                if isinstance(entries, list):
                    all_entries.extend(entries)

        return all_entries

    def run_dataframe(
        self,
        df: Any,  # daft.DataFrame
        prompt_type: str = "qa",  # "qa" or "ingest"
        text_col: str = "text",
        id_col: Optional[str] = None,
    ) -> Any:  # daft.DataFrame
        """
        Run LLM processing on a Daft DataFrame directly.
        Supports 'qa' and 'ingest' modes, switching between local and OpenClaw implementations.
        """
        daft = self._daft
        col = self._daft_col
        
        effective_api_key = self._resolve_llm_api_key()
        effective_base_url = self.openai_base_url or os.getenv("OPENAI_BASE_URL") or ""
        
        # Detect mode
        is_openclaw = self._is_agent_mode(effective_base_url)

        # Optimization: Repartition for parallelism if needed
        if self.parallelism > 1:
            df = df.repartition(self.parallelism)

        if prompt_type == "ingest":
            if is_openclaw:
                # OpenClaw Ingest Mode
                # For ingest, we need an ID column to potentially derive user_id
                if not id_col:
                    # Fallback if no ID column provided
                    id_col_expr = daft.lit("unknown_source")
                else:
                    id_col_expr = col(id_col)
                
                return df.with_column(
                    "ingest_result",
                    OpenClawIngestUDF.with_init_args(
                        base_url=effective_base_url,
                        api_key=effective_api_key,
                        model=self.model,
                        dry_run=self.dry_run
                    )(col(text_col), id_col_expr)
                )
            else:
                # Local Extraction Mode
                # This usually expects file content in text_col
                
                user_id_expr = col("user_id") if "user_id" in df.column_names else daft.lit(None).cast(DataType.string())

                from .udfs import make_extraction_udf
                extract_memories = make_extraction_udf(
                    api_key=effective_api_key,
                    base_url=effective_base_url,
                    model=self.model,
                    prompt=self.prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    dry_run=self.dry_run,
                )

                return df.with_column(
                    "entries",
                    extract_memories(
                        col(text_col), 
                        col(id_col) if id_col else daft.lit("unknown"), 
                        user_id_expr
                    ),
                )

        elif prompt_type == "qa":
            if is_openclaw:
                # OpenClaw QA Mode
                return df.with_column(
                    "answer",
                    OpenClawQAUDF.with_init_args(
                        base_url=effective_base_url,
                        api_key=effective_api_key,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        dry_run=self.dry_run
                    )(col(text_col))
                )
            else:
                # Local QA Mode
                # QAUDF expects (question, context)
                if "context" not in df.column_names:
                    df = df.with_column("context", daft.lit(""))
                
                from .udfs import make_qa_udf
                qa_udf = make_qa_udf(
                    api_key=effective_api_key,
                    base_url=effective_base_url,
                    model=self.model,
                    prompt=self.prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    dry_run=self.dry_run,
                )

                return df.with_column(
                    "answer",
                    qa_udf(col(text_col), col("context")),
                )
        
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

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
                # The output of ExtractionUDF is now a list of dicts (from JSON parsing)
                # But daft might return it as a list of structs or similar
                # We expect dicts here
                if isinstance(item, dict):
                    # Use Memory Plugin to normalize entry
                    normalized = self.plugin.normalize_entry(item)
                    
                    text = str(normalized.get("text", "")).strip()
                    if not text:
                        continue
                        
                    category = normalized.get("category", "other")
                    scope = normalized.get("scope", "global")
                    importance = normalized.get("importance", 0.7)
                    metadata = normalized.get("metadata", {})
                        
                    payloads.append(
                        MemoryEntryPayload(
                            text=text,
                            category=category, # type: ignore
                            scope=scope,
                            importance=importance,
                            metadata=metadata
                        )
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

        # Use large batch size for efficient writing
        batch_size = 2048
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
        
        # Use large batch size
        batch_size = 2048
        
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
        user_ids = [item.get("user_id") for item in qa_items]
        
        df = daft.from_pydict({
            "question": questions,
            "context": contexts,
            "user_id": user_ids,
            "original_item": qa_items # Daft handles list of dicts
        })

        # Use QAUDF (Class UDF)
        # We need to call .with_init_args() on the UDF wrapper first
        effective_api_key = self._resolve_llm_api_key()
        effective_base_url = self.openai_base_url or os.getenv("OPENAI_BASE_URL") or ""

        is_openclaw = self._is_agent_mode(effective_base_url)

        if is_openclaw:
            logger.info("Using OpenClawQAUDF for QA generation")
            from .udfs import make_openclaw_qa_udf
            openclaw_qa = make_openclaw_qa_udf(
                api_key=effective_api_key,
                base_url=effective_base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                dry_run=self.dry_run,
            )
            df = df.with_column(
                "answer",
                openclaw_qa(col("question"), col("user_id")),
            )
        else:
            from .udfs import make_qa_udf
            qa_udf = make_qa_udf(
                api_key=effective_api_key,
                base_url=effective_base_url,
                model=self.model,
                prompt=self.prompt,  # This will be the QA system prompt
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                dry_run=self.dry_run,
            )
            df = df.with_column(
                "answer",
                qa_udf(col("question"), col("context")),
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

        # Use JudgeUDF (Function UDF)
        # Prioritize JUDGE_ specific env vars, then fallback to standard
        effective_api_key = os.getenv("JUDGE_API_KEY") or self._resolve_llm_api_key()
        effective_base_url = os.getenv("JUDGE_BASE_URL") or self.openai_base_url or os.getenv("OPENAI_BASE_URL") or ""
        effective_model = os.getenv("JUDGE_MODEL") or self.model

        from .udfs import make_judge_udf
        judge_udf = make_judge_udf(
            api_key=effective_api_key,
            base_url=effective_base_url,
            model=effective_model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_template=JUDGE_USER_TEMPLATE,
            dry_run=self.dry_run,
        )

        df = df.with_column(
            "judge_result",
            judge_udf(col("question"), col("answer"), col("ground_truth")),
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