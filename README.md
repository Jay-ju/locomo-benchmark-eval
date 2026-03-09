# Locomo Benchmark Eval (locomo_eval)

[English](README.md) | [中文](README_zh.md)

---

**Locomo Benchmark Eval (locomo_eval)** is a robust evaluation framework designed for AI Memory systems, specifically targeting **OpenClaw** and the **LoCoMo Benchmark**. It supports a dual-mode evaluation architecture:

1.  **Direct Pipeline Mode**: Uses the internal RAG pipeline (Ingest -> VectorDB -> Retrieval -> LLM QA). This runs entirely within the evaluation tool, connecting directly to LLMs and Vector Stores. Ideal for testing embedding models and retrieval algorithms in isolation.
2.  **Agent Mode**: Evaluates a running Agent (e.g., OpenClaw) by interacting with it via chat APIs (Ingest via "Remember this", QA via Chat). Ideal for end-to-end system evaluation.

### Features

- **Dual-Mode Evaluation**: Switch between Agent Mode (Default) and Direct Pipeline via `--mode agent` or `--mode direct`.
- **LoCoMo Benchmark Support**: Built-in converter for LoCoMo dataset.
- **Store Agnostic**: Defaults to **LanceDB**. Architecture supports adding adapters for Milvus, FAISS, etc.
- **Pluggable Embedding**: Supports OpenAI-compatible providers.
- **Distributed Processing**: Uses **Daft**.

### Feature Matrix

We are continuously expanding support for more backends.

| Component | Implementation | Status | Note |
| :--- | :--- | :--- | :--- |
| **Vector Store** | **LanceDB** | ✅ Supported | Default backend |
| | **Mem0** | 🚧 Planned | **Next Step** |
| | Milvus | 🚧 Planned | Contributions welcome |
| | FAISS | 🚧 Planned | Contributions welcome |
| **Embedding** | OpenAI / Doubao | ✅ Supported | API |
| | Local HuggingFace (BGE) | ✅ Supported | Local execution |
| **Mode** | Agent Mode (OpenClaw) | ✅ Supported | End-to-end |
| | Direct Mode (RAG Pipeline) | ✅ Supported | Component-level |

### Vector Store Support

Currently, only **LanceDB** is implemented out-of-the-box. To add support for other vector databases (e.g., Milvus, Elasticsearch), implement the `VectorStoreAdapter` interface in `src/adapters/vector_store.py`.

**LanceDB Configuration**:
*   `--store-type lancedb` (Default)
*   `--db-path <path>`: Directory for LanceDB storage (Default: `./tmp_lancedb`).
*   Table name is fixed to `memories`.

### Installation

```bash
cd locomo-benchmark-eval
pip install -e .
```

### Network Configuration (Proxy)

If you encounter network issues when downloading HuggingFace models, set proxy environment variables:

```bash
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=code.byted.org
```

### Quick Start

#### 1. Configuration (`.env`)

Create a `.env` file with your credentials:

```bash
# === Common ===
# For Agent Mode (OpenClaw)
# OPENAI_BASE_URL=http://localhost:18789/v1
# OPENAI_API_KEY=openclaw-secret
# OPENAI_MODEL=ark/doubao-seed-2-0-pro-260215

# For Direct Mode (Internal Pipeline)
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_KEY=your-volcengine-key
OPENAI_MODEL=doubao-pro-32k

# === Embedding (Direct Mode Only) ===
# Option 1: Doubao / OpenAI
EMBEDDING_PROVIDER=doubao
EMBEDDING_API_KEY=your-volcengine-key
EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal
EMBEDDING_MODEL=doubao-embedding-vision
VECTOR_DIM=2048

# Option 2: Local HuggingFace (BGE)
# EMBEDDING_PROVIDER=local-huggingface
# EMBEDDING_MODEL=BAAI/bge-m3 (or local path)
# VECTOR_DIM=1024
```

#### 2. LoCoMo Benchmark Workflow

**Step 1: Ingest Memories**

You can ingest the entire dataset or filter specific samples/sessions.

```bash
# Ingest entire dataset (Default)
python -m src.cli.main ingest data/locomo/locomo10.json

# Ingest specific sample and sessions
# e.g., Sample 0, Sessions 1-4, assigning a specific User ID
python -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --sample 0 \
  --sessions 1-4 \
  --user my-user-uuid

# Using Local BGE Model (Direct Mode)
python -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --mode direct \
  --embed-provider local-huggingface \
  --embed-model BAAI/bge-m3 \
  --vector-dim 1024
```

**Step 2: Evaluate (QA)**

Run QA against the ingested context.

```bash
# Evaluate entire dataset
python -m src.cli.main eval \
  data/locomo/locomo10.json \
  --output results/locomo_results.json

# Evaluate specific sample with specific User ID context
python -m src.cli.main eval \
  data/locomo/locomo10.json \
  --sample 0 \
  --user my-user-uuid \
  --output results/sample0_results.json
```

**Step 3: Judge**

Evaluate the accuracy of the answers using an LLM-as-a-Judge.

```bash
python -m src.cli.main judge \
  results/locomo_results.json \
  --output results/locomo_judged.json
```

**Judge Details**:
*   **Input Format**: JSON list of objects containing `question`, `answer`, `ground_truth`.
*   **Logic**: LLM compares `answer` vs `ground_truth` and labels it `CORRECT` or `WRONG`.
*   **Output Format**: Appends `judge_result` to each object:
    ```json
    "judge_result": {
      "label": "CORRECT",
      "reasoning": "The answer mentions key details...",
      "score": 1.0
    }
    ```
*   **Console Report**:
    *   **Task Completion Rate (Accuracy)**: Overall accuracy percentage.
    *   **Failure Report**: Lists all failed tasks with User ID, Question snippet, and Reasoning for quick review.
*   **Final Score**: Average Accuracy (0.0 - 1.0).

### Architecture

#### Direct Pipeline Mode (`--mode direct`)
- **Ingest**: File -> `ExtractionUDF` (LLM) -> `LanceDB`
- **Eval**: Question -> `VectorSearch` (LanceDB) -> `QAUDF` (LLM) -> Answer

#### Agent Mode (`--mode agent`)
- **Ingest**: File -> `OpenClawIngestUDF` -> HTTP POST `/v1/chat/completions` (OpenClaw)
- **Eval**: Question -> `OpenClawQAUDF` -> HTTP POST `/v1/chat/completions` (OpenClaw)


### Data Format

**Ingest JSONL**:
```json
{
  "text": "User: I love coding.\nAI: That's great!",
  "path": "source/path/session_1",
  "user_id": "user_123",  // Used for scoping
  "timestamp": "2023-10-01"
}
```

**Eval JSONL**:
```json
{
  "question": "What does the user love?",
  "ground_truth": "Coding",
  "user_id": "user_123"  // Ensures QA is scoped to the correct user
}
```
