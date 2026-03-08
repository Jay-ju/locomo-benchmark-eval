# Locomo Benchmark Eval (locomo_eval)

[English](README.md) | [中文](README_zh.md)

---

**Locomo Benchmark Eval (locomo_eval)** 是面向 OpenClaw 生态的 CLI 工具，旨在促进记忆导入和评测。它参考并支持 [EverMemOS](https://github.com/EverMind-AI/EverMemOS/tree/main/evaluation) 的评测数据集，提供了一套从记忆蒸馏、检索到问答评测的流畅工作流。

### 功能特性

- **多向量库支持**：默认支持 **LanceDB**，预留 FAISS / Milvus / Viking 接口。
- **嵌入模型可插拔**：支持 OpenAI 兼容接口（OpenAI, Jina, Voyage, Ollama, Doubao/Volcano 等），支持本地 **dry-run**（随机向量）测试。
- **记忆蒸馏**：`session-distill` 命令利用 LLM 从对话日志中提取原子化记忆，模拟 EverMemOS 的记忆形成过程。
- **评测工具**：提供 `search`, `search-batch`, `qa` 命令，用于验证记忆检索和问答效果。

### 安装

```bash
cd locomo-benchmark-eval
pip install -e .
# 安装后可使用 `locomo_eval` 命令或 `python -m src.cli.main`
```

### 使用指南

#### 1. 记忆摄入 (`ingest`)

使用 LLM 从对话日志中提取原子化记忆 (Session Distillation)。这是从原始聊天记录构建记忆的主要方式。

```bash
# 请确保 .env 文件中已配置 LLM/Embedding 密钥
python -m src.cli.main ingest \
  ./sessions/chat.txt \
  --db-path ./tmp_lancedb \
  --model gpt-4o-mini
```

#### 2. 问答评测 (`eval`)

端到端检索增强生成（RAG）评测：检索相关记忆并生成答案。

```bash
python -m src.cli.main eval \
  ./questions.txt \
  --db-path ./tmp_lancedb \
  --output qa_results.json
```

##### 使用 Doubao (Volcengine/Ark) 作为蒸馏与评测 LLM

- 使用 LiteLLM 的 Doubao 原生集成时：
  - 聊天模型名形如：`volcengine/<YOUR_ENDPOINT_ID>`。
  - 鉴权优先从 `VOLCENGINE_API_KEY` 或 `ARK_API_KEY` 环境变量读取（也可以显式传入 `--api-key`）。
  - 本项目内部不会为 Doubao 传递 `base_url`，因此 **不需要** 在命令行中显式设置 `--base-url`，保持默认即可；即使传入，Doubao 路径也会被忽略。

- 如果通过 Doubao 的 OpenAI-Compatible 网关调用：
  - 模型名需要加前缀：`openai/<your-model-name>`。
  - `--base-url` 必须是形如 `https://your-gateway-host/v1` 的根路径，**不要额外拼接** `/chat/completions` 等子路径。
  - API 密钥建议放在 `OPENAI_API_KEY` 环境变量中。

示例（使用 Doubao 做蒸馏 + 评测，dry-run 时无需任何密钥）：

```bash
# 蒸馏（ingest），使用 Doubao Endpoint ID（原生 Provider）
export VOLCENGINE_API_KEY="YOUR_DOUBAO_KEY"

python -m src.cli.main ingest \
  ./data/chat_logs/sample_chat.txt \
  --db-path ./tmp_lancedb_eval \
  --model volcengine/<YOUR_ENDPOINT_ID>

# 评测（eval），复用同一模型
python -m src.cli.main eval \
  ./data/questions.txt \
  --db-path ./tmp_lancedb_eval \
  --model volcengine/<YOUR_ENDPOINT_ID>
```

```bash
# 通过 OpenAI-Compatible 网关调用 Doubao
export OPENAI_API_KEY="YOUR_PROXY_KEY"

python -m src.cli.main ingest \
  ./data/chat_logs/sample_chat.txt \
  --db-path ./tmp_lancedb_eval \
  --model openai/<YOUR_MODEL_NAME> \
  --base-url https://your-gateway-host/v1
```

常见错误与排查：

- **模型前缀错误**：例如在 Doubao 原生模式下使用 `gpt-4o` 或缺少 `volcengine/` 前缀，或在 OpenAI-Compatible 模式下忘记加 `openai/` 前缀。
- **base_url 缺少 `/v1`**：对于 OpenAI-Compatible 网关，`--base-url` 未以 `/v1` 结尾时，常见报错为 `Not Found` 或 404。
- **密钥未设置**：`VOLCENGINE_API_KEY` / `ARK_API_KEY` / `OPENAI_API_KEY` 未配置或配置错误，导致鉴权失败。

#### 3. 添加结构化记忆 (`add`)

将预先结构化的文件（Markdown, JSONL, LoCoMo JSON）直接导入 LanceDB，不进行蒸馏。

```bash
python -m src.cli.main add \
  ./data/memories \
  --store-type lancedb \
  --db-path ./tmp_lancedb
```

#### 4. 调试检索 (`search` / `search-batch`)

手动检查检索结果，用于调试记忆质量。

```bash
# 单条查询
python -m src.cli.main search "用户喜欢什么？" --db-path ./tmp_lancedb

# 批量查询
python -m src.cli.main search-batch ./queries.txt --output results.json
```

#### 5. 结果评分 (`judge`)

使用 LLM-as-a-Judge 对生成的答案质量进行评分（基于事实一致性或合理性）。

```bash
python -m src.cli.main judge \
  ./qa_results.json \
  --output judged_results.json \
  --model gpt-4o-mini
```

---

## 评测 Benchmark 示例

我们在 `data/` 目录下提供了一个样本数据集，用于演示完整的评测流程（摄入 -> 评测 -> 评分）。

### 步骤 1：摄入样本聊天记录

使用 `ingest` 从对话日志中提取记忆。您可以提供任何包含对话记录的文本文件（如 `.txt`、`.md`、`.jsonl`）。

*注意：`data/chat_logs/sample_chat.txt` 仅为示例路径。您可以将其指向任何包含对话数据的文件或目录。*

```bash
python -m src.cli.main ingest \
  ./data/chat_logs/sample_chat.txt \
  --db-path ./tmp_lancedb_eval
```

### 步骤 2：运行评测

针对蒸馏出的记忆运行样本问题。系统将检索相关上下文并生成答案。

```bash
python -m src.cli.main eval \
  ./data/questions.txt \
  --db-path ./tmp_lancedb_eval \
  --output ./data/qa_results.json
```

### 步骤 3：运行评分 (可选)

使用 LLM Judge 对答案质量进行评估。

```bash
python -m src.cli.main judge \
  ./data/qa_results.json \
  --output ./data/judged_results.json
```

### 步骤 4：查看结果

检查输出文件 `data/judged_results.json`，查看问题、答案以及它们的评分 (1-5分) 和理由。

---

## 官方 Benchmark 数据集

我们在 `data/` 目录下包含了 `EverMemOS` 的官方评测数据集，包括：

- **LoCoMo**: 长上下文记忆 Benchmark (`data/locomo/locomo10.json`)
- **LongMemEval**: 多会话对话评测 (`data/longmemeval/`)
- **PersonaMem**: 角色一致性评测 (`data/personamem/`)

这些数据集包含丰富的多会话聊天记录和问答对，适合对长时记忆系统进行进阶评测。

> **注意**: 这些文件为 JSON 格式。如需使用 `session-distill`，可能需要先提取对话文本，或使用自定义脚本进行处理。
