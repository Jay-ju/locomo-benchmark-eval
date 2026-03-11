# Locomo Benchmark Eval (locomo_eval)

[English](README.md) | [中文](README_zh.md)

---

**Locomo Benchmark Eval (locomo_eval)** 是一个面向 AI 记忆系统（特别是 **OpenClaw** 和 **LoCoMo Benchmark**）的稳健评估框架。它支持双模式评估架构：

1.  **Direct Pipeline Mode（直接模式）**：使用内置的 RAG 流水线（Ingest -> VectorDB -> Retrieval -> LLM QA）。完全在评估工具内部运行，直接连接 LLM 和向量库。适用于隔离测试 Embedding 模型和检索算法。
2.  **Agent Mode（Agent 模式）**：通过 Chat API 与运行中的 Agent（如 OpenClaw）交互（通过 "Remember this" 进行摄入，通过对话进行问答）。适用于端到端系统评估。

### 功能特性

- **双模式评估**：通过 `--mode agent` (默认) 或 `--mode direct` 切换。
- **LoCoMo Benchmark 支持**：内置 LoCoMo 数据集转换器。
- **存储无关性**：默认支持 **LanceDB**。代码结构支持扩展至 Milvus/FAISS/Viking 等其他向量库。
- **可插拔 Embedding**：支持 OpenAI 兼容的提供商。
- **分布式处理**: 使用 **Daft**。

### 核心优势

1.  **高性能离线摄入 (Offline Ingestion)**:
    通过本地并行抽取并将记忆直接存入向量库 (LanceDB)，绕过了 OpenClaw Gateway API 的并发限制瓶颈。该方案将 LoCoMo 数据集的摄入时间从 **约 5 小时** (API 串行) 缩减至 **约 10 分钟** (本地并行)，极大提升了实验效率。

2.  **稳健的 QA 评估**:
    内置 **受控并发 (Semaphore)** 和 **自适应重试 (Adaptive Retry)** 机制，有效规避了 OpenClaw Gateway 在高并发下的不稳定性（如 500 错误）。即使在服务端抖动的情况下，也能保证评估任务的顺利完成。

3.  **深度评分报告**:
    LLM-as-a-Judge 不仅给出评分，还会生成详细的 **判分理由 (Reasoning)**，并自动汇总 **失败案例分析**，帮助开发者快速定位系统缺陷，而不仅仅是一个准确率数字。

4.  **精细化记忆抽取策略**:
    不同于将整个会话作为单一 Chunk 进行 Embedding（导致检索噪音大、效率低），本框架支持自定义的 LLM 抽取策略 (`ExtractionUDF`)，能够将长对话蒸馏为原子化的核心事实。这种细粒度的存储方式显著提升了检索的准确性和 RAG 的推理效果。

### 功能支持矩阵 (Feature Matrix)

我们正在持续扩展对更多后端和组件的支持。

| 组件类型 (Component) | 实现 (Implementation) | 状态 (Status) | 说明 (Note) |
| :--- | :--- | :--- | :--- |
| **Vector Store** | **LanceDB** | ✅ 已支持 | 默认后端 |
| | **Mem0** | 🚧 计划中 | **Next Step** |
| | Milvus | 🚧 计划中 | 欢迎社区贡献 |
| | FAISS | 🚧 计划中 | 欢迎社区贡献 |
| **Embedding** | OpenAI / Doubao | ✅ 已支持 | 通过 API 调用 |
| | Local HuggingFace (BGE) | ✅ 已支持 | 本地运行 |
| **Evaluation Mode** | Agent Mode (OpenClaw) | ✅ 已支持 | 端到端评估 |
| | Direct Mode (RAG Pipeline) | ✅ 已支持 | 组件级评估 |

### 向量存储支持

目前工具仅内置了 **LanceDB** 的实现。如果您需要对接其他向量数据库（如 Milvus, Elasticsearch 等），可以参考 `src/adapters/vector_store.py` 中的 `VectorStoreAdapter` 接口进行扩展。

**LanceDB 配置**:
*   `--store-type lancedb` (默认)
*   `--db-path <path>`: 指定 LanceDB 的存储目录 (默认为 `./tmp_lancedb`)。
*   表名默认为 `memories`，暂不支持自定义。

### 安装

```bash
cd locomo-benchmark-eval
pip install -e .
```

### 网络配置 (Proxy)

如果下载 HuggingFace 模型时遇到网络问题，请设置代理环境变量：

```bash
export http_proxy=http://your-proxy-host:port
export https_proxy=http://your-proxy-host:port
export no_proxy=localhost,127.0.0.1,your-internal-domain
```

### 快速开始

#### 1. 配置 (`.env`)

创建 `.env` 文件并填入密钥：

```bash
# === 通用配置 ===
# 用于 Agent 模式 (OpenClaw)
# OPENAI_BASE_URL=http://localhost:18789/v1
# OPENAI_API_KEY=openclaw-secret
# OPENAI_MODEL=ark/doubao-seed-2-0-pro-260215

# 用于 Direct 模式 (内部 Pipeline)
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_KEY=your-volcengine-key
OPENAI_MODEL=doubao-pro-32k

# === Proxy ===
http_proxy=http://your-proxy-host:port
https_proxy=http://your-proxy-host:port
no_proxy=localhost,127.0.0.1

# === Embedding (仅 Direct 模式需要) ===
# 选项 1: Doubao / OpenAI
EMBEDDING_PROVIDER=doubao
EMBEDDING_API_KEY=your-volcengine-key
EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal
EMBEDDING_MODEL=doubao-embedding-vision
VECTOR_DIM=2048

# 选项 2: 本地 HuggingFace (BGE)
# EMBEDDING_PROVIDER=local-huggingface
# EMBEDDING_MODEL=BAAI/bge-m3 (或本地路径)
# VECTOR_DIM=1024
```

#### 2. LoCoMo Benchmark 工作流

**步骤 1: 记忆摄入 (Ingest)**

您可以摄入整个数据集，或过滤特定的 Sample/Session。

```bash
# 摄入整个数据集 (默认)
python -m src.cli.main ingest data/locomo/locomo10.json

# 摄入特定 Sample 和 Sessions
# 例如: Sample 0, Sessions 1-4, 并指定特定的 User ID
python -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --sample 0 \
  --sessions 1-4 \
  --user my-user-uuid

# 使用本地 BGE 模型 (Direct Mode)
python -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --mode direct \
  --embed-provider local-huggingface \
  --embed-model BAAI/bge-m3 \
  --vector-dim 1024
```

**步骤 2: 评估 (QA)**

针对摄入的上下文运行 QA 测试。

```bash
# 评估整个数据集
python -m src.cli.main eval \
  data/locomo/locomo10.json \
  --output results/locomo_results.json

# 评估特定 Sample 并指定 User ID 上下文
python -m src.cli.main eval \
  data/locomo/locomo10.json \
  --sample 0 \
  --user my-user-uuid \
  --output results/sample0_results.json
```

**步骤 3: 评分 (Judge)**

使用 LLM-as-a-Judge 评估回答的准确性。

```bash
python -m src.cli.main judge \
  results/locomo_results.json \
  --output results/locomo_judged.json
```

**Judge 说明**:
*   **输入格式**: JSON 列表，每个对象需包含 `question`, `answer`, `ground_truth`。
*   **评分逻辑**: LLM 会对比 `answer` 和 `ground_truth`，判断是否正确 (`CORRECT` 或 `WRONG`)。
*   **输出格式**: 在原对象中追加 `judge_result` 字段：
    ```json
    "judge_result": {
      "label": "CORRECT",
      "reasoning": "回答提到了关键信息...",
      "score": 1.0
    }
    ```
*   **控制台报告**:
    *   **Task Completion Rate (Accuracy)**: 总体准确率 (e.g. 85.00%)。
    *   **Failure Report**: 列出所有失败任务的 User ID、问题摘要及失败原因，便于快速定位问题人物。
*   **最终得分**: 计算所有问题的平均分 (Accuracy, 0.0 - 1.0)。

### 架构说明

#### Direct Pipeline Mode (`--mode direct`)
- **Ingest**: 文件 -> `ExtractionUDF` (LLM) -> `LanceDB`
- **Eval**: 问题 -> `VectorSearch` (LanceDB) -> `QAUDF` (LLM) -> 答案

#### Agent Mode (`--mode agent`)
- **Ingest**: 文件 -> `OpenClawIngestUDF` -> HTTP POST `/v1/chat/completions` (OpenClaw)
- **Eval**: 问题 -> `OpenClawQAUDF` -> HTTP POST `/v1/chat/completions` (OpenClaw)


### 数据格式

**Ingest JSONL**:
```json
{
  "text": "User: I love coding.\nAI: That's great!",
  "path": "source/path/session_1",
  "user_id": "user_123",  // 用于 Scope 隔离
  "timestamp": "2023-10-01"
}
```

**Eval JSONL**:
```json
{
  "question": "What does the user love?",
  "ground_truth": "Coding",
  "user_id": "user_123"  // 确保 QA 针对正确的用户 Scope
}
```

### Agent Mode 评估流程 (OpenClaw)

此工作流用于评估运行中的 OpenClaw 实例在 LoCoMo Benchmark 上的表现。

1.  **配置环境**:
    编辑 `.env` 文件，填入您的 API Key 和 Proxy 设置：
    ```bash
    # Proxy
    http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    no_proxy=code.byted.org,localhost,127.0.0.1
    
    # Judge 配置 (Doubao)
    OPENAI_API_KEY=your-key
    OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/
    OPENAI_MODEL=doubao-seed-2-0-pro-260215
    ```

2.  **运行评估 (QA)**:
    对本地 OpenClaw 服务 (`http://localhost:18789`) 执行 1986 个问题的测试。
    
    ```bash
    ./scripts/eval_agent_openclaw.sh
    ```
    *   **注意**: 脚本默认使用 `parallelism=10` 以平衡速度和服务器稳定性，并包含重试机制。

3.  **运行评分 (Judge)**:
    使用 LLM-as-a-Judge 对回答进行打分。
    
    ```bash
    ./scripts/judge_openclaw.sh
    ```

4.  **查看统计**:
    生成准确率报告。
    
    ```bash
    python scripts/calc_score.py results/openclaw_eval_scored.json
    ```

### 项目结构

```
.
├── data/               # 数据集存储 (LoCoMo, LanceDB)
├── results/            # 评估结果输出
├── scripts/            # 自动化脚本 (ingest, eval, judge)
├── src/                # 源代码
│   ├── adapters/       # 向量库与 Embedding 适配器
│   ├── cli/            # CLI 入口 (Typer)
│   ├── core/           # 核心逻辑 (Pipeline, Search, Prompts)
│   └── ...
├── tests/              # 测试代码
├── .env                # 环境变量配置
└── README.md           # 英文文档
```

### CLI 参数说明

**`ingest` 命令**

| 参数 (Flag) | 默认值 (Default) | 说明 (Description) |
| :--- | :--- | :--- |
| `input_paths` | 必填 | 输入文件路径 (JSON/TXT) |
| `--store-type` | lancedb | 向量库后端类型 |
| `--db-path` | ./tmp_lancedb | 数据库路径 |
| `--embed-provider` | openai | Embedding 提供商 (openai/doubao/local-huggingface) |
| `--parallelism` | 4 | 并发度 |
| `--schema-mode` | pro | 抽取模式 (basic/pro) |

**`eval` 命令**

| 参数 (Flag) | 默认值 (Default) | 说明 (Description) |
| :--- | :--- | :--- |
| `input_path` | 必填 | QA 输入文件路径 |
| `--mode` | agent | 评估模式 (`agent` 或 `direct`) |
| `--base-url` | - | OpenClaw/LLM Base URL |
| `--api-key` | - | API Key |
| `--parallelism` | 4 | 并发度 (OpenClaw 推荐设为 10) |
| `--output` | stdout | 输出 JSON 文件路径 |

**`judge` 命令**

| 参数 (Flag) | 默认值 (Default) | 说明 (Description) |
| :--- | :--- | :--- |
| `results_file` | 必填 | 包含 QA 结果的 JSON 文件 |
| `--output` | 覆盖原文件 | 包含评分结果的输出文件 |
| `--parallelism` | 4 | 并发度 |
