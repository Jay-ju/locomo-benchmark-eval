from __future__ import annotations

"""Typer-based CLI for the locomo_eval tool.

Example usage (dry-run, no API key required)::

    python -m locomo_eval.cli add \
        data/locomo/locomo10.json \
        --store-type lancedb \
        --db-path ./tmp_lancedb \
        --vector-dim 64 \
        --dry-run

After installing this package with ``pip install -e .``, the same
command is available as a console script::

    locomo_eval add ...
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv

# Try to load .env file from current directory
load_dotenv()

from ..core.daft_runner import DaftPromptRunner
from ..adapters.embedder import (
    OpenAICompatibleConfig,
    OpenAICompatibleEmbedder,
    RandomEmbedder,
)
from ..core.pipeline import ImportPipeline, PipelineConfig
from ..core.search import SearchRunner
from ..adapters.vector_store import create_vector_store

app = typer.Typer(help="Import OpenClaw-style memories into configurable vector stores.")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_SUPPORTED_STORE_TYPES = ("lancedb", "faiss", "milvus", "viking")

from ..core.prompts.qa import QA_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_store_type(store_type: str) -> str:
    normalized = store_type.lower()
    if normalized not in _SUPPORTED_STORE_TYPES:
        allowed = " | ".join(_SUPPORTED_STORE_TYPES)
        raise typer.BadParameter(f"Unsupported store-type '{store_type}'. Expected one of: {allowed}.")
    if normalized != "lancedb":
        # Only LanceDB has a working implementation in this version.
        typer.echo(
            f"[warning] store_type='{normalized}' adapter is not implemented in this example yet, "
            "it will raise NotImplementedError at runtime.",
            err=True,
        )
    return normalized


def _create_search_runner(
    *,
    store_type: str,
    db_path: Path,
    table_name: str,
    vector_dim: int,
    embed_base_url: Optional[str],
    embed_model: str,
    embed_api_key: Optional[str],
    dry_run: bool,
    parallelism: int,
) -> tuple[SearchRunner, str]:
    """Create a SearchRunner + normalised store type for search/qa commands."""

    store_type_normalized = _normalize_store_type(store_type)

    if dry_run:
        embedder = RandomEmbedder(dimensions=vector_dim)
    else:
        embed_cfg = OpenAICompatibleConfig(
            model=embed_model,
            dimensions=vector_dim,
            api_key=embed_api_key,
            base_url=embed_base_url,
            dry_run=False,
        )
        embedder = OpenAICompatibleEmbedder(embed_cfg)

    store = create_vector_store(
        store_type=store_type_normalized,
        db_path=str(db_path),
        table_name=table_name,
        vector_dim=vector_dim,
    )

    runner = SearchRunner(store=store, embedder=embedder, parallelism=parallelism)
    return runner, store_type_normalized


# ---------------------------------------------------------------------------
# add (formerly direct-import)
# ---------------------------------------------------------------------------


@app.command("add")
def add(
    input_paths: List[Path] = typer.Argument(
        ..., help="Input files or directories. Supports Markdown / JSONL / LoCoMo JSON.",
    ),
    store_type: str = typer.Option(
        "lancedb",
        "--store-type",
        help=(
            "Vector store backend type. Currently only 'lancedb' is implemented; "
            "'faiss' / 'milvus' / 'viking' are reserved placeholders."
        ),
    ),
    db_path: Path = typer.Option(
        Path("./tmp_lancedb"),
        "--db-path",
        help="LanceDB directory path (for --store-type lancedb; will be created if not exists).",
    ),
    table_name: str = typer.Option(
        "memories",
        "--table-name",
        help="Table / collection name in the target vector store (LanceDB: table name).",
    ),
    vector_dim: int = typer.Option(
        1024,
        "--vector-dim",
        envvar="VECTOR_DIM",
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        envvar="EMBEDDING_BASE_URL",
        help="OpenAI-compatible embeddings base URL (e.g. https://api.openai.com/v1).",
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        "--model",
        envvar="EMBEDDING_MODEL",
        help="Embedding model name for the provider.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="EMBEDDING_API_KEY",
        help="Embedding API key. Can also come from EMBEDDING_API_KEY environment variable.",
    ),
    task_query: Optional[str] = typer.Option(
        None,
        "--task-query",
        help="Optional provider-specific task hint for query embeddings.",
    ),
    task_passage: Optional[str] = typer.Option(
        None,
        "--task-passage",
        help="Optional provider-specific task hint for passage embeddings.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Generate deterministic random vectors instead of calling the remote embedding provider. "
            "Useful for local tests without any API key."
        ),
    ),
    input_format: str = typer.Option(
        "auto",
        "--input-format",
        help="Input format: auto | markdown | jsonl | locomo-result | text",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size for embedding + insert operations.",
    ),
    use_daft: bool = typer.Option(
        True,
        "--use-daft/--no-daft",
        help="Whether to try using Daft for larger batch orchestration.",
    ),
    parallelism: int = typer.Option(
        4,
        "--parallelism",
        help="Logical parallelism hint when Daft is enabled.",
    ),
) -> None:
    """Directly import structured ``MemoryEntry`` objects into a vector store.

    This mode assumes input files are already relatively well-structured
    (Markdown snippets, JSONL with memory fields, or LoCoMo result
    JSONs) and does **not** call an LLM to distill conversation logs.
    """

    store_type_normalized = _normalize_store_type(store_type)

    embed_cfg = OpenAICompatibleConfig(
        model=model,
        dimensions=vector_dim,
        api_key=api_key,
        base_url=base_url,
        task_query=task_query,
        task_passage=task_passage,
        dry_run=dry_run,
    )
    embedder = OpenAICompatibleEmbedder(embed_cfg)

    store = create_vector_store(
        store_type=store_type_normalized,
        db_path=str(db_path),
        table_name=table_name,
        vector_dim=vector_dim,
    )

    pipeline_cfg = PipelineConfig(
        batch_size=batch_size,
        use_daft=use_daft,
        parallelism=parallelism,
    )
    pipeline = ImportPipeline(embedder=embedder, store=store, config=pipeline_cfg)

    inserted = pipeline.run_direct_import(input_paths, input_format=input_format)

    if store_type_normalized == "lancedb":
        typer.echo(
            f"Imported {inserted} memories into LanceDB at {db_path} (table={table_name}).",
        )
    else:
        typer.echo(
            f"Imported {inserted} memories into vector store (type={store_type_normalized}).",
        )


# ---------------------------------------------------------------------------
# session-distill (DaftPromptRunner)
# ---------------------------------------------------------------------------


@app.command("ingest")
def ingest(
    input_paths: List[Path] = typer.Argument(
        ..., help="Session transcripts or long text files to distill from.",
    ),
    store_type: str = typer.Option(
        "lancedb",
        "--store-type",
        help=(
            "Vector store backend type. Currently only 'lancedb' is implemented; "
            "'faiss' / 'milvus' / 'viking' are reserved placeholders."
        ),
    ),
    db_path: Path = typer.Option(
        Path("./tmp_lancedb"),
        "--db-path",
        help="LanceDB directory path (for --store-type lancedb; will be created if not exists).",
    ),
    table_name: str = typer.Option(
        "memories",
        "--table-name",
        help="Table / collection name in the target vector store (LanceDB: table name).",
    ),
    vector_dim: int = typer.Option(
        1024,
        "--vector-dim",
        envvar="VECTOR_DIM",
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    # OpenAI-compatible parameters for the *distillation LLM*.
    base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--base-url",
        envvar="OPENAI_BASE_URL",
        help=(
            "OpenAI-compatible chat/responses base URL for distillation LLM "
            "(e.g. https://api.openai.com/v1 or OpenClaw Gateway)."
        ),
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        envvar="OPENAI_MODEL",
        help="LLM model name used for session distillation (e.g. gpt-4o-mini, openclaw, etc).",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help=(
            "LLM API key for session distillation. If omitted, falls back to OPENAI_API_KEY "
            "environment variable."
        ),
    ),
    # Optional embedding provider configuration; defaults reuse LLM base URL / api key.
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        envvar="EMBEDDING_BASE_URL",
        help="Optional OpenAI-compatible base URL for embeddings (defaults to --base-url if omitted).",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
        envvar="EMBEDDING_MODEL",
        help="Embedding model name used when writing distilled memories into the vector store.",
    ),
    embed_api_key: Optional[str] = typer.Option(
        None,
        "--embed-api-key",
        envvar="EMBEDDING_API_KEY",
        help="Optional embedding API key (defaults to --api-key / OPENAI_API_KEY when omitted).",
    ),
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens",
        help="Maximum number of tokens generated by the distillation LLM per file.",
    ),
    temperature: float = typer.Option(
        0.2,
        "--temperature",
        help="Sampling temperature for the distillation LLM.",
    ),
    parallelism: int = typer.Option(
        4,
        "--parallelism",
        help="Logical parallelism hint used by Daft and the embedding+insert loop.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Use Daft to fan out tasks but return fixed example JSON memories instead of "
            "calling any remote LLM or embedding provider. Useful for local smoke tests."
        ),
    ),
) -> None:
    """Ingest memories from session logs (session distillation).

    This command (formerly session-distill) uses an LLM to extract atomic memories from conversation logs
    and stores them in the vector database.
    """

    store_type_normalized = _normalize_store_type(store_type)

    if not dry_run and not api_key:
        raise typer.BadParameter("ingest requires --api-key or OPENAI_API_KEY in non-dry-run mode.")

    # 1) Construct DaftPromptRunner (will raise exception with prompt if daft/openai missing).
    try:
        runner = DaftPromptRunner(
            openai_base_url=base_url,
            openai_api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallelism=parallelism,
            dry_run=dry_run,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    # 2) Execute distillation (each file -> multiple JSON memory entries).
    entries = runner.run([str(p) for p in input_paths])
    if not entries:
        typer.echo("ingest: No memory entries obtained from input files, ending.")
        raise typer.Exit(code=0)

    # 3) Create vector store adapter.
    store = create_vector_store(
        store_type=store_type_normalized,
        db_path=str(db_path),
        table_name=table_name,
        vector_dim=vector_dim,
    )

    # 4) Create embedding adapter: dry-run uses RandomEmbedder, others use OpenAI-compatible Embeddings.
    if dry_run:
        embedder = RandomEmbedder(dimensions=vector_dim)
    else:
        effective_embed_base_url = embed_base_url or base_url
        effective_embed_api_key = embed_api_key or api_key
        embed_cfg = OpenAICompatibleConfig(
            model=embed_model,
            dimensions=vector_dim,
            api_key=effective_embed_api_key,
            base_url=effective_embed_base_url,
            dry_run=False,
        )
        embedder = OpenAICompatibleEmbedder(embed_cfg)

    # 5) Write to vector store.
    inserted = runner.store_to_vector(store, embedder, entries)

    if store_type_normalized == "lancedb":
        typer.echo(
            f"Ingest completed: {inserted} memories imported into LanceDB at {db_path} (table={table_name}).",
        )
    else:
        typer.echo(
            f"Ingest completed: {inserted} memories imported into vector store (type={store_type_normalized}).",
        )


# ---------------------------------------------------------------------------
# search / search-batch
# ---------------------------------------------------------------------------


@app.command("search")
def search(
    query: Optional[str] = typer.Argument(
        None,
        help="Single query text. If omitted, you must provide --input-file.",
    ),
    store_type: str = typer.Option(
        "lancedb",
        "--store-type",
        help=(
            "Vector store backend type. Currently only 'lancedb' is implemented; "
            "'faiss' / 'milvus' / 'viking' are reserved placeholders."
        ),
    ),
    db_path: Path = typer.Option(
        Path("./tmp_lancedb"),
        "--db-path",
        help="LanceDB directory path (for --store-type lancedb; will be created if not exists).",
    ),
    table_name: str = typer.Option(
        "memories",
        "--table-name",
        help="Table / collection name in the target vector store (LanceDB: table name).",
    ),
    vector_dim: int = typer.Option(
        1024,
        "--vector-dim",
        envvar="VECTOR_DIM",
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        envvar="EMBEDDING_BASE_URL",
        help="OpenAI-compatible embeddings base URL (e.g. https://api.openai.com/v1).",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
        envvar="EMBEDDING_MODEL",
        help="Embedding model name for the provider.",
    ),
    embed_api_key: Optional[str] = typer.Option(
        None,
        "--embed-api-key",
        envvar="EMBEDDING_API_KEY",
        help="Embedding API key. Can also come from EMBEDDING_API_KEY environment variable.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Use RandomEmbedder for search (no remote embedding calls).",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Maximum number of results per query."),
    min_score: float = typer.Option(
        0.3,
        "--min-score",
        help="Minimum similarity score (0-1) to keep a result.",
    ),
    parallelism: int = typer.Option(
        4,
        "--parallelism",
        help="Logical parallelism hint when running batch search.",
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file",
        help="If provided, run batch search for each non-empty line as a query.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional output JSON file. Defaults to stdout.",
    ),
) -> None:
    """Search memories using vector similarity.

    - Defaults to single query execution (``query`` positional argument).
    - When ``--input-file`` is provided, runs batch search line by line, using Daft for parallelism."""

    if input_file is None and (query is None or not str(query).strip()):
        raise typer.BadParameter("search requires a query argument or --input-file.")

    # Read query set
    queries: List[str] = []
    is_batch = input_file is not None

    if input_file is not None:
        try:
            text = input_file.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - IO path
            raise typer.BadParameter(f"Cannot read --input-file: {exc}") from exc
        for line in text.splitlines():
            line = line.strip()
            if line:
                queries.append(line)
        if not queries:
            typer.echo("search: No valid query lines in input file.")
            raise typer.Exit(code=0)
    else:
        assert query is not None
        queries = [str(query).strip()]

    runner, store_type_normalized = _create_search_runner(
        store_type=store_type,
        db_path=db_path,
        table_name=table_name,
        vector_dim=vector_dim,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_api_key=embed_api_key,
        dry_run=dry_run,
        parallelism=parallelism,
    )

    if is_batch:
        try:
            batch_results = runner.search_batch(queries, top_k=top_k, min_score=min_score)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)
        payload = [
            {"query": q, "results": res}
            for q, res in zip(queries, batch_results)
        ]
    else:
        results = runner.search_one(queries[0], top_k=top_k, min_score=min_score)
        payload = {"query": queries[0], "results": results}

    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.write_text(json_str, encoding="utf-8")
    else:
        typer.echo(json_str)


@app.command("search-batch")
def search_batch(
    queries_file: Path = typer.Argument(..., help="Text file with one query per line."),
    store_type: str = typer.Option(
        "lancedb",
        "--store-type",
        help=(
            "Vector store backend type. Currently only 'lancedb' is implemented; "
            "'faiss' / 'milvus' / 'viking' are reserved placeholders."
        ),
    ),
    db_path: Path = typer.Option(
        Path("./tmp_lancedb"),
        "--db-path",
        help="LanceDB directory path (for --store-type lancedb; will be created if not exists).",
    ),
    table_name: str = typer.Option(
        "memories",
        "--table-name",
        help="Table / collection name in the target vector store (LanceDB: table name).",
    ),
    vector_dim: int = typer.Option(
        1024,
        "--vector-dim",
        envvar="VECTOR_DIM",
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        envvar="EMBEDDING_BASE_URL",
        help="OpenAI-compatible embeddings base URL (e.g. https://api.openai.com/v1).",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
        envvar="EMBEDDING_MODEL",
        help="Embedding model name for the provider.",
    ),
    embed_api_key: Optional[str] = typer.Option(
        None,
        "--embed-api-key",
        envvar="EMBEDDING_API_KEY",
        help="Embedding API key. Can also come from EMBEDDING_API_KEY environment variable.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Use RandomEmbedder for search (no remote embedding calls).",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Maximum number of results per query."),
    min_score: float = typer.Option(
        0.3,
        "--min-score",
        help="Minimum similarity score (0-1) to keep a result.",
    ),
    parallelism: int = typer.Option(
        4,
        "--parallelism",
        help="Logical parallelism hint when running batch search.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional output JSON file. Defaults to stdout.",
    ),
) -> None:
    """Batch search wrapper around :func:`search`.

    Equivalent to ``search --input-file <queries_file>``, keeping the interface semantic clear."""

    try:
        text = queries_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - IO path
        raise typer.BadParameter(f"Cannot read queries_file: {exc}") from exc

    queries: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            queries.append(line)

    if not queries:
        typer.echo("search-batch: No valid query lines in input file.")
        raise typer.Exit(code=0)

    runner, store_type_normalized = _create_search_runner(
        store_type=store_type,
        db_path=db_path,
        table_name=table_name,
        vector_dim=vector_dim,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_api_key=embed_api_key,
        dry_run=dry_run,
        parallelism=parallelism,
    )

    try:
        batch_results = runner.search_batch(queries, top_k=top_k, min_score=min_score)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    payload = [
        {"query": q, "results": res}
        for q, res in zip(queries, batch_results)
    ]

    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.write_text(json_str, encoding="utf-8")
    else:
        typer.echo(json_str)


# ---------------------------------------------------------------------------
# qa: retrieval + DaftPromptRunner answer
# ---------------------------------------------------------------------------


@app.command("eval")
def eval(
    queries_file: Path = typer.Argument(
        ..., help="Text file with one question per line (LoCoMo-style queries).",
    ),
    # Retrieval options
    store_type: str = typer.Option(
        "lancedb",
        "--store-type",
        help=(
            "Vector store backend type. Currently only 'lancedb' is implemented; "
            "'faiss' / 'milvus' / 'viking' are reserved placeholders."
        ),
    ),
    db_path: Path = typer.Option(
        Path("./tmp_lancedb"),
        "--db-path",
        help="LanceDB directory path (for --store-type lancedb; will be created if not exists).",
    ),
    table_name: str = typer.Option(
        "memories",
        "--table-name",
        help="Table / collection name in the target vector store (LanceDB: table name).",
    ),
    vector_dim: int = typer.Option(
        1024,
        "--vector-dim",
        envvar="VECTOR_DIM",
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        envvar="EMBEDDING_BASE_URL",
        help="OpenAI-compatible embeddings base URL for retrieval embeddings.",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
        envvar="EMBEDDING_MODEL",
        help="Embedding model name used for retrieval.",
    ),
    embed_api_key: Optional[str] = typer.Option(
        None,
        "--embed-api-key",
        envvar="EMBEDDING_API_KEY",
        help="Embedding API key for retrieval embeddings.",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Maximum number of memories retrieved per question."),
    min_score: float = typer.Option(
        0.3,
        "--min-score",
        help="Minimum similarity score (0-1) to keep a retrieved memory.",
    ),
    # LLM options for answering
    base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--base-url",
        envvar="OPENAI_BASE_URL",
        help="OpenAI-compatible chat base URL for QA answering.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        envvar="OPENAI_MODEL",
        help="LLM model name used for answering (e.g. gpt-4o-mini).",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="LLM API key for answering. If omitted, falls back to OPENAI_API_KEY.",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum tokens for each answer.",
    ),
    temperature: float = typer.Option(
        0.2,
        "--temperature",
        help="Sampling temperature for QA answering.",
    ),
    parallelism: int = typer.Option(
        4,
        "--parallelism",
        help="Logical parallelism hint used by Daft for per-question answering.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Use Daft but return fixed dry-run answers without calling any remote LLM.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional output JSON file. Defaults to stdout.",
    ),
) -> None:
    """Evaluate QA performance: retrieve memories then answer via DaftPromptRunner.

    This command (formerly qa) runs an end-to-end evaluation:
    1. Retrieves relevant memories for each question in the input file.
    2. Uses an LLM to generate an answer based on the retrieved context.
    3. Outputs the result (question, answer, context, context_ids).

    Output is a JSON array, where each element looks like::

        {
          "question": "...",
          "answer": "...",
          "context_ids": ["uuid-1", "uuid-2", ...]
        }
    """

    # Read questions
    try:
        text = queries_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - IO path
        raise typer.BadParameter(f"Cannot read queries_file: {exc}") from exc

    questions: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            questions.append(line)

    if not questions:
        typer.echo("eval: No valid question lines in input file.")
        raise typer.Exit(code=0)

    # 1) Retrieval phase: Use SearchRunner + Daft (batch) to get context
    search_runner, store_type_normalized = _create_search_runner(
        store_type=store_type,
        db_path=db_path,
        table_name=table_name,
        vector_dim=vector_dim,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_api_key=embed_api_key,
        dry_run=dry_run,
        parallelism=parallelism,
    )

    try:
        search_results = search_runner.search_batch(questions, top_k=top_k, min_score=min_score)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    qa_items: List[dict] = []
    for q, results in zip(questions, search_results):
        # Assemble context text
        lines: List[str] = []
        context_ids: List[str] = []
        for idx, r in enumerate(results):
            mid = str(r.get("id", ""))
            context_ids.append(mid)
            cat = r.get("category", "other")
            scope = r.get("scope", "global")
            score = r.get("score", 0.0)
            text_snippet = str(r.get("text", ""))
            lines.append(
                f"[{idx + 1}] (id={mid}, category={cat}, scope={scope}, score={score:.3f}) {text_snippet}"
            )

        context_text = "\n".join(lines)

        qa_items.append(
            {
                "question": q,
                "context": context_text,
                "context_ids": context_ids,
            },
        )

    # 2) Answering phase: DaftPromptRunner.answer_batch
    if not dry_run and not api_key:
        raise typer.BadParameter("eval requires --api-key or OPENAI_API_KEY in non-dry-run mode.")

    try:
        qa_runner = DaftPromptRunner(
            prompt=QA_SYSTEM_PROMPT,
            openai_base_url=base_url,
            openai_api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            parallelism=parallelism,
            dry_run=dry_run,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    qa_results = qa_runner.answer_batch(qa_items)

    json_str = json.dumps(qa_results, ensure_ascii=False, indent=2)
    if output is not None:
        output.write_text(json_str, encoding="utf-8")
    else:
        typer.echo(json_str)


@app.command("judge")
def judge(
    results_file: Path = typer.Argument(
        ..., help="JSON file containing QA results (question, answer, optional ground_truth).",
    ),
    # LLM options for judging
    base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--base-url",
        envvar="OPENAI_BASE_URL",
        help="OpenAI-compatible chat base URL for judging.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        envvar="OPENAI_MODEL",
        help="LLM model name used for judging (e.g. gpt-4o).",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="LLM API key for judging. If omitted, falls back to OPENAI_API_KEY.",
    ),
    parallelism: int = typer.Option(
        4,
        "--parallelism",
        help="Logical parallelism hint used by Daft for per-question judging.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Use Daft but return fixed dry-run scores without calling any remote LLM.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional output JSON file. Defaults to overwriting the input file if not provided.",
    ),
) -> None:
    """Evaluate QA results using LLM-as-a-Judge.

    Input file should be a JSON array of objects with "question" and "answer" fields.
    Output will add a "judge_result" field to each object:
      "judge_result": { "score": 1-5, "reasoning": "..." }
    """
    
    try:
        text = results_file.read_text(encoding="utf-8")
        qa_results = json.loads(text)
    except Exception as exc:
        raise typer.BadParameter(f"Cannot read results_file: {exc}") from exc
    
    if not isinstance(qa_results, list):
        raise typer.BadParameter("Input file must contain a JSON array.")

    if not dry_run and not api_key:
        raise typer.BadParameter("judge requires --api-key or OPENAI_API_KEY in non-dry-run mode.")

    try:
        runner = DaftPromptRunner(
            openai_base_url=base_url,
            openai_api_key=api_key,
            model=model,
            parallelism=parallelism,
            dry_run=dry_run,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    judged_results = runner.judge_batch(qa_results)
    
    # Calculate average score
    scores = []
    for item in judged_results:
        res = item.get("judge_result", {})
        if isinstance(res, dict):
            s = res.get("score")
            if isinstance(s, (int, float)):
                scores.append(s)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    typer.echo(f"Judge completed. Average Score: {avg_score:.2f} / 5.0")

    json_str = json.dumps(judged_results, ensure_ascii=False, indent=2)
    target_path = output or results_file
    target_path.write_text(json_str, encoding="utf-8")
    typer.echo(f"Results saved to {target_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
