# Locomo Benchmark Eval (locomo_eval)

[English](README.md) | [中文](README_zh.md)

---

**Locomo Benchmark Eval (locomo_eval)** is a CLI tool designed for the OpenClaw ecosystem to facilitate memory import and evaluation. It draws inspiration from and supports the evaluation datasets of [EverMemOS](https://github.com/EverMind-AI/EverMemOS/tree/main/evaluation), providing a streamlined workflow for memory distillation, retrieval, and QA evaluation.

### Features

- **Store Agnostic**: Defaults to **LanceDB**, with interfaces reserved for FAISS / Milvus / Viking.
- **Pluggable Embedding**: Supports OpenAI-compatible providers (OpenAI, Jina, Voyage, Ollama, Doubao/Volcano, etc.) or local **dry-run** (random vectors) for testing.
- **Memory Distillation**: `session-distill` command uses LLM to extract atomic memories from conversation logs, similar to the memory formation process in EverMemOS.
- **Evaluation**: `search`, `search-batch`, and `qa` commands for verifying memory retrieval and answering capabilities.

### Installation

```bash
cd locomo-benchmark-eval
pip install -e .
# Now you can use the `locomo_eval` command or `python -m src.cli.main`
```

### Usage

#### 1. Ingest Memories (`ingest`)

Extract atomic memories from conversation logs using an LLM (Session Distillation). This is the primary way to build memory from raw chat logs.

```bash
# Ensure your .env file is configured with LLM/Embedding keys
python -m src.cli.main ingest \
  ./sessions/chat.txt \
  --db-path ./tmp_lancedb \
  --model gpt-4o-mini
```

#### 2. Evaluate QA Performance (`eval`)

End-to-end Retrieval-Augmented Generation (RAG) evaluation: Retrieve relevant memories and generate answers.

```bash
python -m src.cli.main eval \
  ./questions.txt \
  --db-path ./tmp_lancedb \
  --output qa_results.json
```

#### 3. Add Structured Memories (`add`)

Import pre-structured files (Markdown, JSONL, LoCoMo JSON) directly into LanceDB without distillation.

```bash
python -m src.cli.main add \
  ./data/memories \
  --store-type lancedb \
  --db-path ./tmp_lancedb
```

#### 4. Debug Search (`search` / `search-batch`)

Manually inspect retrieval results to debug memory quality.

```bash
# Single query
python -m src.cli.main search "What does the user like?" --db-path ./tmp_lancedb

# Batch search
python -m src.cli.main search-batch ./queries.txt --output results.json
```

#### 5. Judge QA Results (`judge`)

Use an LLM-as-a-Judge to evaluate the quality of the generated answers against ground truth (or plausibility).

```bash
python -m src.cli.main judge \
  ./qa_results.json \
  --output judged_results.json \
  --model gpt-4o-mini
```

---

## Evaluation Benchmark Workflow

We provide a sample dataset in `data/` to demonstrate the full evaluation lifecycle (Ingest -> Eval -> Judge).

### Step 1: Ingest Sample Chat Logs

Use `ingest` to extract memories from conversation logs. You can provide any text file (e.g., `.txt`, `.md`, `.jsonl`) containing conversation transcripts.

*Note: `data/chat_logs/sample_chat.txt` is just an example path. You can point this to any file or directory containing your conversation data.*

```bash
python -m src.cli.main ingest \
  ./data/chat_logs/sample_chat.txt \
  --db-path ./tmp_lancedb_eval
```

### Step 2: Run Evaluation

Run the sample questions against the distilled memories. The system will retrieve relevant context and generate answers.

```bash
python -m src.cli.main eval \
  ./data/questions.txt \
  --db-path ./tmp_lancedb_eval \
  --output ./data/qa_results.json
```

### Step 3: Run Judge (Optional)

Evaluate the quality of the answers using an LLM Judge.

```bash
python -m src.cli.main judge \
  ./data/qa_results.json \
  --output ./data/judged_results.json
```

### Step 4: Review Results

Check the output file `data/judged_results.json` to see the questions, answers, and their scores (1-5) with reasoning.

---

## Official Benchmark Data

We also include the official benchmark datasets from `EverMemOS` in `data/`. This includes:

- **LoCoMo**: Long-Context Memory Benchmark (`data/locomo/locomo10.json`)
- **LongMemEval**: Multi-session conversation evaluation (`data/longmemeval/`)
- **PersonaMem**: Persona consistency evaluation (`data/personamem/`)

These datasets contain rich, multi-session conversation logs and QA pairs, suitable for advanced evaluation of long-term memory systems.

> **Note**: These files are in JSON format. You may need to extract the conversation text for `session-distill` or use custom scripts to process them.
