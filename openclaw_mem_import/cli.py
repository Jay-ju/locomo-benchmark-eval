from __future__ import annotations

"""Typer-based CLI for the add_lance_memory tool.

Example usage (dry-run, no API key required)::

    python -m openclaw_mem_import.cli direct-import \
        openclaw-eval/data/viking/user/memories \
        --store-type lancedb \
        --db-path ./tmp_lancedb \
        --vector-dim 64 \
        --dry-run

After installing this package with ``pip install -e .``, the same
command is available as a console script::

    add_lance_memory direct-import ...
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

from .daft_prompt import DaftPromptRunner
from .distill_prompts import SESSION_DISTILL_PROMPT_ZH
from .embedder import (
    OpenAICompatibleConfig,
    OpenAICompatibleEmbedder,
    RandomEmbedder,
)
from .pipeline import ImportPipeline, PipelineConfig
from .search import SearchRunner
from .vector_store import create_vector_store

app = typer.Typer(help="Import OpenClaw-style memories into configurable vector stores.")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

_SUPPORTED_STORE_TYPES = ("lancedb", "faiss", "milvus", "viking")

# 简单的 QA system prompt，用于在检索上下文基础上回答问题。
_QA_SYSTEM_PROMPT_ZH = (
    "你是一个长期记忆评测用的问答助手。" "你将收到用户问题，以及若干与问题相关的记忆片段。" "\n\n"
    "原则：\n"
    "1. 只能基于提供的记忆内容回答问题，不要编造事实；\n"
    "2. 如果记忆不足以回答，请明确回答：‘我在当前记忆中找不到足够的信息来回答这个问题。’；\n"
    "3. 回答尽量简洁、直接，不重复整段记忆文本；\n"
    "4. 如有时间信息，请注意时间顺序与相对时间表达（昨天/上周等）。"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_store_type(store_type: str) -> str:
    normalized = store_type.lower()
    if normalized not in _SUPPORTED_STORE_TYPES:
        allowed = " | ".join(_SUPPORTED_STORE_TYPES)
        raise typer.BadParameter(f"Unsupported store-type '{store_type}'. Expected one of: {allowed}.")
    if normalized != "lancedb":
        # 当前版本中只有 LanceDB 有可用实现，其它类型作为占位符提前提示。
        typer.echo(
            f"[warning] store_type='{normalized}' 适配器尚未在此示例中实现, "
            "运行时会抛出 NotImplementedError。",
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
# direct-import
# ---------------------------------------------------------------------------


@app.command("direct-import")
def direct_import(
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
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="OpenAI-compatible embeddings base URL (e.g. https://api.openai.com/v1).",
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        "--model",
        help="Embedding model name for the provider.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="Embedding API key. Can also come from OPENAI_API_KEY environment variable.",
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


@app.command("session-distill")
def session_distill(
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
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    # OpenAI-compatible parameters for the *distillation LLM*.
    base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--base-url",
        help=(
            "OpenAI-compatible chat/responses base URL for distillation LLM "
            "(e.g. https://api.openai.com/v1 或 OpenClaw Gateway /v1)."
        ),
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
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
        help="Optional OpenAI-compatible base URL for embeddings (defaults to --base-url if omitted).",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
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
    """Run Daft-based session distillation and write memories into a vector store.

    Flow:

    1. 构造 :class:`DaftPromptRunner`，对每个输入文件执行中文会话蒸馏 Prompt；
    2. 收集所有 JSON 记忆条目；
    3. 通过 ``create_vector_store(...)`` 创建向量库适配器；
    4. 通过 ``RandomEmbedder``（dry-run）或 ``OpenAICompatibleEmbedder`` 生成向量；
    5. 调用 ``runner.store_to_vector(...)`` 批量写入。

    当 ``dry_run=True`` 时仍然会使用 Daft 做并行 map，但每个文件只生成一条
    示例记忆，不会访问任何网络。这样可以在本地快速验证端到端流程。
    """

    store_type_normalized = _normalize_store_type(store_type)

    if not dry_run and not api_key:
        raise typer.BadParameter("session-distill 在非 dry-run 模式下需要提供 --api-key 或设置 OPENAI_API_KEY。")

    # 1) 构造 DaftPromptRunner（会在缺少 daft/openai 时抛出带中文提示的异常）。
    try:
        runner = DaftPromptRunner(
            prompt=SESSION_DISTILL_PROMPT_ZH,
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

    # 2) 执行蒸馏（每个文件 → 若干 JSON 记忆条目）。
    entries = runner.run([str(p) for p in input_paths])
    if not entries:
        typer.echo("session-distill: 未从输入文件中得到任何记忆条目，结束。")
        raise typer.Exit(code=0)

    # 3) 创建向量库适配器。
    store = create_vector_store(
        store_type=store_type_normalized,
        db_path=str(db_path),
        table_name=table_name,
        vector_dim=vector_dim,
    )

    # 4) 创建嵌入适配器：dry-run 使用 RandomEmbedder，其它情况使用 OpenAI-compatible Embeddings。
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

    # 5) 写入向量库。
    inserted = runner.store_to_vector(store, embedder, entries)

    if store_type_normalized == "lancedb":
        typer.echo(
            f"Session distill imported {inserted} memories into LanceDB at {db_path} (table={table_name}).",
        )
    else:
        typer.echo(
            f"Session distill imported {inserted} memories into vector store (type={store_type_normalized}).",
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
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        help="OpenAI-compatible embeddings base URL (e.g. https://api.openai.com/v1).",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
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

    - 默认执行单条查询（``query`` 位置参数）。
    - 当提供 ``--input-file`` 时，按行批量查询，并使用 Daft 做并行。"""

    if input_file is None and (query is None or not str(query).strip()):
        raise typer.BadParameter("search 需要提供 query 参数或 --input-file 其一。")

    # 读取查询集合
    queries: List[str] = []
    is_batch = input_file is not None

    if input_file is not None:
        try:
            text = input_file.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - IO path
            raise typer.BadParameter(f"无法读取 --input-file: {exc}") from exc
        for line in text.splitlines():
            line = line.strip()
            if line:
                queries.append(line)
        if not queries:
            typer.echo("search: 输入文件中没有有效的查询行。")
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
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        help="OpenAI-compatible embeddings base URL (e.g. https://api.openai.com/v1).",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
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

    行为等价于 ``search --input-file <queries_file>``，保持接口语义清晰。"""

    try:
        text = queries_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - IO path
        raise typer.BadParameter(f"无法读取 queries_file: {exc}") from exc

    queries: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            queries.append(line)

    if not queries:
        typer.echo("search-batch: 输入文件中没有有效的查询行。")
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


@app.command("qa")
def qa(
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
        help="Embedding vector dimension. Must match the embedding model / provider.",
    ),
    embed_base_url: Optional[str] = typer.Option(
        None,
        "--embed-base-url",
        help="OpenAI-compatible embeddings base URL for retrieval embeddings.",
    ),
    embed_model: str = typer.Option(
        "text-embedding-3-small",
        "--embed-model",
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
        help="OpenAI-compatible chat base URL for QA answering.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
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
    """End-to-end QA: retrieve memories then answer via DaftPromptRunner.

    输出为 JSON 数组，每个元素形如::

        {
          "question": "...",
          "answer": "...",
          "context_ids": ["uuid-1", "uuid-2", ...]
        }
    """

    # 读取问题
    try:
        text = queries_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - IO path
        raise typer.BadParameter(f"无法读取 queries_file: {exc}") from exc

    questions: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            questions.append(line)

    if not questions:
        typer.echo("qa: 输入文件中没有有效的问题行。")
        raise typer.Exit(code=0)

    # 1) 检索阶段：使用 SearchRunner + Daft（批量）获取 context
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
        # 拼装上下文文本
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

    # 2) 回答阶段：DaftPromptRunner.answer_batch
    if not dry_run and not api_key:
        raise typer.BadParameter("qa 在非 dry-run 模式下需要提供 --api-key 或设置 OPENAI_API_KEY。")

    try:
        qa_runner = DaftPromptRunner(
            prompt=_QA_SYSTEM_PROMPT_ZH,
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


if __name__ == "__main__":  # pragma: no cover
    app()
