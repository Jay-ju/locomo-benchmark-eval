# Locomo Benchmark Eval (locomo_eval)

[English](README.md) | [中文](README_zh.md)

---

**Locomo Benchmark Eval (locomo_eval)** 是一个面向 AI 记忆系统（特别是 **OpenClaw** 和 **LoCoMo Benchmark**）的稳健评估框架。它支持双模式评估架构：

1.  **Direct Pipeline Mode（直接模式）**：使用内置的 RAG 流水线（Ingest -> VectorDB -> Retrieval -> LLM QA）。完全在评估工具内部运行，直接连接 LLM 和向量库。适用于隔离测试 Embedding 模型和检索算法。
2.  **Agent Mode（Agent 模式）**：通过 Chat API 与运行中的 Agent（如 OpenClaw）交互（通过 "Remember this" 进行摄入，通过对话进行问答）。适用于端到端系统评估。

### 功能特性

- **双模式评估**：通过 `--mode agent` (默认) 或 `--mode direct` 切换。
- **LoCoMo Benchmark 支持**：内置 LoCoMo 数据集转换器。
- **存储无关性**：默认支持 **LanceDB**。
- **可插拔 Embedding**：支持 OpenAI 兼容的提供商。
- **分布式处理**：使用 **Daft**。

### 安装

```bash
cd locomo-benchmark-eval
pip install -e .
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

```bash
python -m src.cli.main judge \
  results/locomo_results.json \
  --output results/locomo_judged.json
```

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
