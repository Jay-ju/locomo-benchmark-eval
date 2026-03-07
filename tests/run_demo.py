from __future__ import annotations

"""最小 E2E 演示脚本。

- 从 openclaw-eval 的 Markdown 记忆做 direct-import（dry-run）；
- 从 memorylake-locomo-benchmark 的 LoCoMo 结果 JSON 做 direct-import（dry-run）；
- 输出最终插入条数与 LanceDB 目录路径。
"""

import sys
from pathlib import Path

# 将 add-lance-memory 根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openclaw_mem_import.embedder import OpenAICompatibleConfig, OpenAICompatibleEmbedder
from openclaw_mem_import.pipeline import ImportPipeline, PipelineConfig
from openclaw_mem_import.vector_store import create_vector_store


def main() -> None:
    db_path = Path("./tmp_lancedb")

    # --- 构造 dry-run Embedder 与向量库适配器（显式 store_type="lancedb"） ---
    embed_cfg = OpenAICompatibleConfig(
        model="dry-run",
        dimensions=64,
        api_key=None,
        base_url=None,
        dry_run=True,
    )
    embedder = OpenAICompatibleEmbedder(embed_cfg)

    store = create_vector_store(
        store_type="lancedb",
        db_path=str(db_path),
        table_name="memories",
        vector_dim=64,
    )

    pipeline_cfg = PipelineConfig(batch_size=16, use_daft=False)
    pipeline = ImportPipeline(embedder=embedder, store=store, config=pipeline_cfg)

    # --- 1) 从 openclaw-eval 的用户记忆 Markdown 导入 ---
    viking_mem_dir = Path("../openclaw-eval/data/viking/user/memories")
    print(f"[demo] importing Markdown memories from {viking_mem_dir} ...")
    inserted_md = pipeline.run_direct_import([viking_mem_dir], input_format="auto")

    # --- 2) 从 LoCoMo 结果 JSON 导入若干事实 ---
    locomo_file = Path("../memorylake-locomo-benchmark/locomo_result_1.json")
    print(f"[demo] importing LoCoMo facts from {locomo_file} ...")
    inserted_locomo = pipeline.run_direct_import([locomo_file], input_format="auto")

    total = inserted_md + inserted_locomo
    print("[demo] total inserted memories:", total)
    print("[demo] LanceDB directory:", db_path.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()
