# add-lance-memory · OpenClaw 记忆导入 CLI

面向 OpenClaw 生态的**通用记忆导入工具**，可以在不依赖 Gateway 的情况下，将结构化记忆批量写入向量库，并保持与 **memory-lancedb-pro** 的表结构兼容。

- **向量库不是固定 LanceDB**：通过 `--store-type` 选择后端，LanceDB 只是默认实现；FAISS / Milvus / Viking 预留接口不改上层代码。
- **嵌入也可插拔**：默认使用 OpenAI-compatible Embedding，可轻松切换到 Jina / Voyage / Ollama / 内部网关，或使用纯本地 dry-run 随机向量完成测试。
- **Two modes**：`direct-import` 负责“已有结构化记忆”的高速导入；`session-distill` 预留“会话蒸馏入口”，由业务自定义 LLM Agent。

包名：`openclaw_mem_import` · CLI 名称：`add_lance_memory`

---

## Run

### 本地安装（可选）

```bash
cd add-lance-memory
pip install -e .
# 之后可以直接用 add_lance_memory 命令
```

也可以直接使用模块入口，无需安装：

```bash
cd add-lance-memory
python -m openclaw_mem_import.cli --help
```

### 示例 1：从 openclaw-eval Markdown 记忆导入（dry-run，无需 API Key）

```bash
cd add-lance-memory
python -m openclaw_mem_import.cli direct-import \
  ../openclaw-eval/data/viking/user/memories \
  --store-type lancedb \
  --db-path ./tmp_lancedb \
  --vector-dim 64 \
  --dry-run
```

- 使用 **随机向量**（`--dry-run`），不会访问外部 Embedding 服务；
- 在 `./tmp_lancedb` 下创建 LanceDB 数据库与 `memories` 表；
- 表结构与 `memory-lancedb-pro` 对齐，可直接被其检索链路消费。

### 示例 2：从 LoCoMo 评测结果导入事实记忆（dry-run）

```bash
cd add-lance-memory
python -m openclaw_mem_import.cli direct-import \
  ../memorylake-locomo-benchmark/locomo_result_1.json \
  --store-type lancedb \
  --db-path ./tmp_lancedb \
  --vector-dim 64 \
  --input-format auto \
  --dry-run
```

- 自动识别 `locomo_result_*.json` 为 LoCoMo 结果文件；
- 为前若干条样本生成 `fact` 类别的评测记忆，附带中文关键词行 `Keywords (zh): ...`；
- 与上一命令写入同一 LanceDB 目录，可被 OpenClaw 的 memory 插件统一检索。

### 示例 3：启用真实 Embedding Provider（OpenAI-compatible）

```bash
cd add-lance-memory
export OPENAI_API_KEY=sk-...
python -m openclaw_mem_import.cli direct-import \
  ../openclaw-eval/data/viking/user/memories \
  --store-type lancedb \
  --db-path ~/.openclaw/memory/lancedb-pro \
  --vector-dim 1536 \
  --base-url https://api.openai.com/v1 \
  --model text-embedding-3-small
```

- `--vector-dim` 必须与模型输出维度一致（如 `text-embedding-3-small` 为 1536）；
- 推荐把 `--db-path` 指向 `memory-lancedb-pro` 所使用的 LanceDB 目录，使导入的记忆直接进入生产检索路径。

> 其它向量库（`--store-type faiss|milvus|viking`）在当前版本为占位实现，接口已预留，但会在运行时抛出 `NotImplementedError`；用来锁定上层接口，不影响后续扩展。

---

## Two Modes

### direct-import：结构化记忆导入

**输入已经是 MemoryEntry 粒度**（Markdown 片段 / JSONL / LoCoMo 结果 / 纯文本），Pipeline 做三件事：

1. 文件解析 → `MemoryEntryPayload`：
   - Markdown：每个 `.md` 文件生成一条记忆；根据路径推断 `category`（preferences/events/entities/other）。
   - JSONL：每行一个 JSON 记忆对象，字段对齐 memory-lancedb-pro（`text/category/scope/importance/metadata/id/timestamp`）。
   - LoCoMo JSON：`locomo_result_*.json` 解析为若干评测事实记忆，并写入中文关键词行。
   - 纯文本：每个文件作为一条 `other` 记忆。
2. 嵌入：通过 `EmbedderAdapter` 得到向量；支持 OpenAI-compatible / dry-run 随机向量等。
3. 写入：通过 `VectorStoreAdapter` 将 `StoredMemoryEntry` 批量写入向量库。

### session-distill：会话蒸馏（仅 Daft 版本）

`session-distill` 子命令使用 **Daft DataFrame + OpenAI 兼容 LLM** 对每个输入文件做会话蒸馏：

1. 每个 `.txt` / `.md` / `.markdown` / `.jsonl` 文件作为一条蒸馏任务，整文件内容送入 LLM；
2. 使用中文 Prompt `SESSION_DISTILL_PROMPT_ZH` 要求 LLM 输出 JSON 数组，每个元素是一条记忆：
   `text/category/scope/importance/metadata`；
3. 用 Daft 在单机上并行执行多文件蒸馏；
4. 将得到的 JSON 记忆通过嵌入（Embedding）写入向量库，表结构与 `memory-lancedb-pro` 兼容。

> 当前仅提供 **Daft 版本**，不再提供非 Daft 实现；缺少 `daft` 时命令会直接报错提示安装。

#### 示例：在线 LLM + LanceDB

```bash
python -m openclaw_mem_import.cli session-distill \
  ./samples/sessions \
  --store-type lancedb --db-path ./tmp_lancedb --table-name memories --vector-dim 1024 \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY \
  --parallelism 8
```

- `--base-url` / `--model` / `--api-key`：用于调用会话蒸馏 LLM（可指向 OpenAI 或 OpenClaw Gateway）；
- `--embed-model` / `--embed-base-url` / `--embed-api-key`：可选，用于单独指定 Embedding Provider（默认复用 LLM 的 base URL 与 key，模型默认为 `text-embedding-3-small`）；
- `--vector-dim`：必须与 Embedding 模型的输出维度一致（如 OpenAI `text-embedding-3-small` 为 1536）。

#### 示例：dry-run（不联网，仅验证 Daft + 向量库链路）

```bash
python -m openclaw_mem_import.cli session-distill \
  ./samples/sessions \
  --store-type lancedb --db-path ./tmp_lancedb --vector-dim 64 \
  --dry-run --parallelism 8
```

- 仍然使用 Daft 做并行 map，但每个文件只生成一条示例记忆；
- 使用 `RandomEmbedder` 生成随机向量，不访问任何远程 Embedding 服务；
- 适合在本地验证 CLI/Daft/向量库写入是否畅通。

如果你需要完全自定义 LLM 蒸馏逻辑，也可以在自己的 Agent 中直接使用
`SESSION_DISTILL_PROMPT_ZH`，然后把 JSON 输出交给 `DaftPromptRunner.store_to_vector(...)` 或
`ImportPipeline`，复用同一写入链路。
---

## CLI Options

### direct-import 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_paths` (位置参数) | 必填 | 输入文件或目录，可多选；支持 Markdown / JSONL / LoCoMo JSON / 纯文本 |
| `--store-type` | `lancedb` | 向量库类型：`lancedb`（默认实现）；`faiss`/`milvus`/`viking` 为预留占位，当前会抛出 `NotImplementedError` |
| `--db-path` | `./tmp_lancedb` | **LanceDB 专用**：数据库目录路径，可指向 `~/.openclaw/memory/lancedb-pro` |
| `--table-name` | `memories` | 表/集合名；LanceDB 下为表名 |
| `--vector-dim` | `1024` | 向量维度，需与嵌入模型输出一致 |
| `--base-url` | `None` | OpenAI-compatible Embedding 的 base URL（OpenAI/Jina/Voyage/Ollama 网关等） |
| `--model` | `text-embedding-3-small` | Embedding 模型名称 |
| `--api-key` | 环境变量 `OPENAI_API_KEY` | Embedding API Key，可从环境变量读取 |
| `--task-query` | `None` | Provider 特定的 query 任务标签（如 Jina 的 `retrieval.query`） |
| `--task-passage` | `None` | Provider 特定的 passage 任务标签（如 `retrieval.passage`） |
| `--dry-run` | `False` | 使用确定性随机向量代替真实 Embedding，用于无密钥环境测试 |
| `--input-format` | `auto` | `auto` / `markdown` / `jsonl` / `locomo-result` / `text` |
| `--batch-size` | `32` | 每批嵌入+写入的条数 |
| `--use-daft/--no-daft` | `--use-daft` | 是否尝试加载 Daft，用于后续分布式批处理扩展；当前即使失败也会回退到本地循环 |
| `--parallelism` | `4` | Daft 可用时的逻辑并行度 hint |

> 设计要点：
>
> - **向量库是参数**：`--store-type` 控制后端，而 Pipeline 只依赖 `VectorStoreAdapter`，导入逻辑与具体向量库解耦；
> - **Embedding 也是参数**：`--base-url` / `--model` / `--api-key` 三元组抽象为 OpenAI-compatible Provider，可映射到 OpenAI / Jina / Voyage / Ollama / 内部网关；
> - 未来为 Milvus/Viking 补全实现时，仅需实现对应 Adapter 和 `create_vector_store` 分支，不需要改 CLI 或 Pipeline。

### 最佳实践导入接口草案

统一 CLI 入口（已在当前实现中采用）：

```bash
add_lance_memory direct-import \
  <INPUT_PATHS...> \
  --store-type <lancedb|faiss|milvus|viking> \
  --db-path <DB_PATH_FOR_LANCEDB> \
  --table-name memories \
  --vector-dim <D> \
  --base-url <EMBED_BASE_URL> \
  --model <EMBED_MODEL> \
  --api-key <EMBED_API_KEY> \
  --dry-run \
  --input-format auto \
  --batch-size 32 \
  --use-daft \
  --parallelism 4
```

典型组合：

- **本地 smoke test（无 API Key）**：`--store-type lancedb` + `--dry-run`；
- **生产环境（LanceDB + Jina Embedding）**：`--store-type lancedb` + `--base-url https://api.jina.ai/v1` + `--model jina-embeddings-v5-text-small` + `--vector-dim 1024`；
- **规划中的 Milvus/Viking**：上层 CLI 与 Pipeline 不变，只切换 `--store-type` 并补充相应后端配置。

---

## Typical Workflow

### 步骤一：导入记忆

1. 选择数据源：
   - LoCoMo 评测结果：`memorylake-locomo-benchmark/locomo_result_*.json`；
   - 手工整理的 Markdown 记忆：`openclaw-eval/data/viking/user/memories/**/*.md`；
   - 其它系统导出的 JSONL 记忆。
2. 运行 `direct-import`：

   ```bash
   cd add-lance-memory
   python -m openclaw_mem_import.cli direct-import \
     ../openclaw-eval/data/viking/user/memories \
     --store-type lancedb \
     --db-path ~/.openclaw/memory/lancedb-pro \
     --vector-dim 1536 \
     --base-url https://api.openai.com/v1 \
     --model text-embedding-3-small
   ```

3. 确认 LanceDB 目录下已经生成/更新 `memories` 表。

### 步骤二：复查与检索

1. 若你在 OpenClaw 中使用 `memory-lancedb-pro`：
   - 确保插件配置的 `dbPath` 指向与 `--db-path` 相同的目录；
   - 使用 `openclaw memory-pro search/list/stats` 验证导入的记忆是否可见。
2. 若你有自定义检索脚本：
   - 可直接使用 Python `lancedb` 打开 `db_path`，查询 `memories` 表进行手工抽样；
   - 或用其它向量库适配器（后续 Milvus/Viking）读取同一 schema。

---

## Search / Batch Search / QA

在导入（add）之外，本工具还提供 **search / batch-search / qa** 三类接口，方便对接 EverMemOS 风格的 `add → search → answer` 评测流程。

所有批量与 QA 并发均通过 **Daft UDF** 实现；当缺少 `daft` 时，对应子命令会给出明确错误提示。

### Search：单条检索

单条查询，直接在 `memories` 表上做向量检索并返回 JSON 结果：

```bash
python -m openclaw_mem_import.cli search "用户喜欢的食物是什么？" \
  --store-type lancedb \
  --db-path ./tmp_lancedb \
  --vector-dim 64 \
  --embed-base-url https://api.openai.com/v1 \
  --embed-model text-embedding-3-small \
  --embed-api-key $OPENAI_API_KEY \
  --top-k 10 --min-score 0.3
```

输出示例（简化）：

```json
{
  "query": "用户喜欢的食物是什么？",
  "results": [
    {
      "id": "...",
      "text": "用户最喜欢吃披萨。",
      "category": "preference",
      "scope": "global",
      "importance": 0.9,
      "timestamp": 1730000000000,
      "metadata": {"source": "markdown"},
      "score": 0.87
    }
  ]
}
```

> 若使用 `--dry-run`，则检索会基于随机向量，仅用于链路 smoke test。

### Batch Search：批量检索

从文本文件中批量读取查询（每行一个），使用 Daft 并行执行检索：

```bash
python -m openclaw_mem_import.cli search-batch \
  ./samples/queries.txt \
  --store-type lancedb --db-path ./tmp_lancedb --vector-dim 64 \
  --embed-base-url https://api.openai.com/v1 \
  --embed-model text-embedding-3-small --embed-api-key $OPENAI_API_KEY \
  --top-k 10 --min-score 0.3 --parallelism 8 \
  --output batch_search.json
```

输出为 JSON 数组，每个元素形如：

```json
{
  "query": "...",
  "results": [ {"id": "...", "text": "...", "score": 0.8, ...}, ... ]
}
```

### QA：检索 + 答案生成

`qa` 子命令按 **检索 → 组装上下文 → DaftPromptRunner 并发回答** 的流程运行：

1. 使用 SearchRunner + 嵌入，在 `memories` 表上为每个问题检索 top_k 条记忆；
2. 将检索结果拼接为上下文文本；
3. 通过 `DaftPromptRunner.answer_batch(...)` 调用 OpenAI 兼容 LLM 输出答案；
4. 输出 `question/answer/context_ids` JSON 数组。

#### 示例：在线 LLM + LanceDB

```bash
python -m openclaw_mem_import.cli qa \
  ./samples/questions.txt \
  --store-type lancedb --db-path ./tmp_lancedb --table-name memories --vector-dim 1024 \
  --embed-base-url https://api.openai.com/v1 --embed-model text-embedding-3-small --embed-api-key $OPENAI_API_KEY \
  --top-k 10 --min-score 0.3 \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY \
  --parallelism 8 --max-tokens 512 \
  --output qa_results.json
```

输出示例（结构）：

```json
[
  {
    "question": "用户最喜欢的食物是什么？",
    "answer": "根据当前记忆，用户最喜欢吃披萨。",
    "context_ids": ["uuid-1", "uuid-2", "..."]
  },
  ...
]
```

#### 示例：dry-run（不访问外部 LLM）

```bash
python -m openclaw_mem_import.cli qa \
  ./samples/questions.txt \
  --store-type lancedb --db-path ./tmp_lancedb --vector-dim 64 \
  --dry-run --parallelism 8
```

- 检索阶段仍然使用 Daft 并发；
- 回答阶段返回固定的 dry-run 答案文本，仅用于验证整条 `search → answer` 管线是否跑通；
- 方便在无网络 / 无 API Key 的环境下做本地调试。
