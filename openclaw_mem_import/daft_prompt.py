from __future__ import annotations

"""Daft-based prompt runner for session/text distillation.

This module provides :class:`DaftPromptRunner`, which uses a Daft DataFrame to
fan out LLM distillation jobs across files and then stores the distilled
memories into a vector store via the existing ``EmbedderAdapter`` and
``VectorStoreAdapter`` interfaces.

Design notes
------------
- Each *file* is treated as one distillation unit. The entire file content
  (txt / md / jsonl) is sent to the LLM together with
  ``SESSION_DISTILL_PROMPT_ZH``.
- The LLM is expected to return a JSON array of memory dicts, following the
  schema described in the prompt (text/category/scope/importance/metadata).
- Daft is used to parallelise per-file distillation via a
  ``from_pydict(...).with_column(col("payload").apply(...))`` pattern.
- ``dry_run=True`` keeps the same Daft flow but returns a deterministic
  example JSON for each file without making any network calls.

If Daft is not installed, this module raises a clear runtime error asking the
user to install it.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .distill_prompts import SESSION_DISTILL_PROMPT_ZH
from .embedder import EmbedderAdapter
from .models import MemoryEntryPayload, StoredMemoryEntry
from .vector_store import VectorStoreAdapter

logger = logging.getLogger(__name__)


@dataclass
class DaftPromptRunner:
    """Run LLM-based distillation over files using Daft for parallelism.

    Parameters
    ----------
    prompt:
        System prompt to use for distillation. Defaults to
        :data:`SESSION_DISTILL_PROMPT_ZH`.
    openai_base_url:
        Base URL of the OpenAI-compatible HTTP endpoint (e.g.
        ``https://api.openai.com/v1`` 或 OpenClaw Gateway 的 ``http://127.0.0.1:18789/v1``)。
    openai_api_key:
        API key for the endpoint. When omitted, falls back to
        ``OPENAI_API_KEY`` 环境变量（仅在 ``dry_run=False`` 时必需）。
    model:
        Chat / Responses 模型名称，用于执行蒸馏 Prompt（如 ``gpt-4o-mini`` 或
        OpenClaw 的 ``openclaw`` 等）。
    max_tokens:
        单次蒸馏调用的最大生成 token 数。
    temperature:
        采样温度。
    parallelism:
        逻辑并行度 hint，用于控制 Daft 的执行规模（目前主要用于控制批大小）。
    dry_run:
        若为 True，则不会调用任何远程 LLM，而是为每个文件生成一条示例记忆，
        方便在无网络 / 无密钥环境中验证 Daft 流程与向量库写入。
    """

    prompt: str = SESSION_DISTILL_PROMPT_ZH
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
            raise RuntimeError("DaftPromptRunner 需要 daft 库，请安装 daft 并重试。") from exc

        self._daft = daft
        self._daft_col = col
        self._daft_DataType = DataType

        self._client = None
        if not self.dry_run:
            try:  # pragma: no cover - network dependency
                from openai import OpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "DaftPromptRunner 需要 python-openai 库以调用 OpenAI 兼容接口，"
                    "请先安装 openai 并重试。",
                ) from exc

            api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("请通过 --api-key 或环境变量 OPENAI_API_KEY 提供 LLM API Key。")

            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            self._client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_paths: Sequence[str]) -> List[Dict[str, Any]]:
        """Run distillation over the given input paths.

        Each *file* (txt / md / markdown / jsonl) becomes one distillation
        task. The raw file content is fed into the LLM with the configured
        prompt, and the JSON array output is parsed into a flat list of
        memory dicts.
        """

        files = self._expand_input_paths(input_paths)
        if not files:
            logger.warning("DaftPromptRunner: 未找到任何输入文件，跳过蒸馏。")
            return []

        records: List[Dict[str, str]] = []
        for path in files:
            p = Path(path)
            try:
                text = p.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - IO path
                logger.warning("DaftPromptRunner: 无法读取文件 %s: %s", p, exc)
                continue

            text = text.strip()
            if not text:
                continue

            records.append({"path": str(p), "text": text})

        if not records:
            logger.warning("DaftPromptRunner: 所有输入文件内容为空，未生成蒸馏任务。")
            return []

        daft = self._daft
        col = self._daft_col
        DataType = self._daft_DataType

        df = daft.from_pydict({"payload": records})

        def _apply(payload: Dict[str, str]) -> List[Dict[str, Any]]:
            path = payload.get("path", "")
            text = payload.get("text", "")
            if not text:
                return []
            if self.dry_run:
                return self._fake_distill_one(path, text)
            return self._distill_one(path, text)

        df_with = df.with_column(
            "entries",
            col("payload").apply(_apply, return_dtype=DataType.python()),
        )

        collected = df_with.collect()
        result = collected.to_pydict()

        all_entries: List[Dict[str, Any]] = []
        payload_col = result.get("payload", [])
        entries_col = result.get("entries", [])
        for payload, entries in zip(payload_col, entries_col):
            if not entries:
                continue
            if not isinstance(entries, list):
                logger.warning("DaftPromptRunner: 单条结果不是列表，跳过: %r", entries)
                continue
            for item in entries:
                if isinstance(item, dict):
                    all_entries.append(item)

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
                logger.warning("DaftPromptRunner: 输入路径不存在: %s", p)

        return sorted(files)

    def _fake_distill_one(self, path: str, text: str) -> List[Dict[str, Any]]:
        """Return a deterministic example memory for dry-run mode.

        This keeps the Daft pipeline exercised without calling any remote
        LLM service.
        """

        filename = Path(path).name or "<unknown>"
        mem_text = f"示例记忆：文件 {filename} 已被标记为一条 dry-run 演示记忆。"
        metadata = {
            "source": "session-distill-dry-run",
            "source_path": path,
            "keywords_zh_line": "Keywords (zh): 示例; dry-run; session-distill",
        }
        return [
            {
                "text": mem_text,
                "category": "fact",
                "scope": "global",
                "importance": 0.5,
                "metadata": metadata,
            },
        ]

    def _distill_one(self, path: str, text: str) -> List[Dict[str, Any]]:
        assert self._client is not None, "LLM client must be initialised when dry_run=False"

        try:  # pragma: no cover - network path
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network path
            logger.error("DaftPromptRunner: 调用 LLM 失败（path=%s）: %s", path, exc)
            return []

        choice = response.choices[0]
        raw_content = getattr(choice.message, "content", "")  # type: ignore[attr-defined]
        if not raw_content:
            return []

        content = self._strip_code_fences(str(raw_content))

        try:
            parsed = json.loads(content)
        except Exception as exc:  # pragma: no cover - parse path
            logger.error(
                "DaftPromptRunner: 解析 LLM 输出 JSON 失败（path=%s）: %s\n原始内容: %s",
                path,
                exc,
                content,
            )
            return []

        if not isinstance(parsed, list):
            logger.warning("DaftPromptRunner: LLM 输出顶层不是数组（path=%s），忽略。", path)
            return []

        entries: List[Dict[str, Any]] = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            cleaned = self._normalize_entry(item, path=path, index=idx)
            if cleaned is not None:
                entries.append(cleaned)

        return entries

    def _strip_code_fences(self, text: str) -> str:
        t = text.strip()
        if not t.startswith("```"):
            return t
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
        return t

    def _normalize_entry(
        self,
        item: Dict[str, Any],
        *,
        path: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        text = str(item.get("text", "")).strip()
        if not text:
            return None

        category = str(item.get("category", "other")).strip() or "other"
        scope = str(item.get("scope", "global")).strip() or "global"

        try:
            importance = float(item.get("importance", 0.7))
        except Exception:
            importance = 0.7

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"metadata_raw": metadata}

        metadata.setdefault("source", "session-distill")
        metadata.setdefault("source_path", path)
        if "keywords_zh_line" not in metadata:
            metadata["keywords_zh_line"] = "Keywords (zh): 会话; 蒸馏; 记忆"

        return {
            "text": text,
            "category": category,
            "scope": scope,
            "importance": importance,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Storage helper
    # ------------------------------------------------------------------

    def store_to_vector(
        self,
        db_adapter: VectorStoreAdapter,
        embedder: EmbedderAdapter,
        entries_json: List[Dict[str, Any]],
    ) -> int:
        """Embed distilled entries and write them into the vector store.

        This mirrors the behaviour of :class:`ImportPipeline`'s internal
        embedding + storage logic, but starts from already-distilled JSON
        entries instead of raw files.
        """

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
                logger.warning("DaftPromptRunner: 跳过无效条目 index=%d: %s", idx, exc)

        if not payloads:
            return 0

        batch_size = max(1, int(self.parallelism) * 4)
        total_inserted = 0

        for i in range(0, len(payloads), batch_size):
            batch = payloads[i : i + batch_size]
            texts = [p.text for p in batch]

            vectors = embedder.embed_passages(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedding 返回向量数 {len(vectors)} 与 payload 数 {len(batch)} 不一致",
                )

            entries: List[StoredMemoryEntry] = [
                StoredMemoryEntry.from_payload(payload, vector)
                for payload, vector in zip(batch, vectors)
            ]

            inserted = db_adapter.insert_memories(entries)
            total_inserted += inserted

        db_adapter.flush()
        return total_inserted

    def answer_batch(
        self,
        qa_items: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use Daft + LLM to answer questions with retrieved context.

        Each item in ``qa_items`` should be a dict with at least:

        - ``question``: 用户问题
        - ``context``: 已经拼装好的检索上下文文本
        - ``context_ids``: 上下文对应的记忆 id 列表（可选）

        返回值中的每个元素同样是一个 dict，包含
        ``question`` / ``answer`` / ``context_ids``，方便直接序列化为 JSON。
        """

        if not qa_items:
            return []

        daft = self._daft
        col = self._daft_col
        DataType = self._daft_DataType

        records: List[Dict[str, Any]] = [dict(item) for item in qa_items]

        df = daft.from_pydict({"payload": records})

        def _apply(payload: Dict[str, Any]) -> Dict[str, Any]:
            question = str(payload.get("question", "")).strip()
            context = str(payload.get("context", "")).strip()
            context_ids = payload.get("context_ids") or []
            if not isinstance(context_ids, list):
                try:
                    context_ids = list(context_ids)  # type: ignore[arg-type]
                except Exception:
                    context_ids = []

            base: Dict[str, Any] = {
                "question": question,
                "context_ids": context_ids,
            }

            if not question:
                base["answer"] = ""
                return base

            if self.dry_run:
                base["answer"] = "这是一个 dry-run 答案示例（未调用任何远程 LLM）。"
                return base

            assert self._client is not None, "LLM client must be initialised when dry_run=False"

            user_content = (
                "问题：" + question + "\n\n" +
                "以下是与该问题相关的记忆片段：\n" + context + "\n\n" +
                "请仅依据上述记忆内容回答问题，若记忆不足以回答，请明确说明。"
            )

            try:  # pragma: no cover - network path
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                choice = response.choices[0]
                answer_text = getattr(choice.message, "content", "")  # type: ignore[attr-defined]
            except Exception as exc:
                logger.error("DaftPromptRunner.answer_batch: 调用 LLM 失败: %s", exc)
                answer_text = ""

            base["answer"] = str(answer_text or "").strip()
            return base

        df_with = df.with_column(
            "qa",
            col("payload").apply(_apply, return_dtype=DataType.python()),
        )

        collected = df_with.collect().to_pydict()
        qa_col = collected.get("qa", [])

        outputs: List[Dict[str, Any]] = []
        for item in qa_col:
            if isinstance(item, dict):
                outputs.append(item)

        return outputs
