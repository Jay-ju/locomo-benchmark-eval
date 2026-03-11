#!/bin/bash
# Ingest FULL LoCoMo dataset using Doubao Embedding + Doubao Extraction (From .env)

PYTHON=/data00/miniconda3/envs/daft_env/bin/python

# 1. Load Config from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Ensure variables are set
if [ -z "$EMBEDDING_API_KEY" ]; then
    echo "Error: EMBEDDING_API_KEY not set in .env"
    exit 1
fi
EMBED_BASE_URL=${EMBEDDING_BASE_URL:-"https://ark.cn-beijing.volces.com/api/v3/"}
EMBED_MODEL=${EMBEDDING_MODEL:-"doubao-embedding-large-text-250515"}
VECTOR_DIM=${VECTOR_DIM:-2048}

DB_PATH="data/lancedb_doubao_full"

# Clean up existing
rm -rf $DB_PATH

echo "Ingesting FULL dataset..."
echo "  Embedding: $EMBED_MODEL ($VECTOR_DIM)"
echo "  Extraction LLM: $OPENAI_MODEL"

$PYTHON -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --mode direct \
  --embed-provider doubao \
  --embed-base-url "$EMBED_BASE_URL" \
  --embed-model "$EMBED_MODEL" \
  --vector-dim "$VECTOR_DIM" \
  --db-path "$DB_PATH" \
  --parallelism 16

echo "Done. DB saved to $DB_PATH"
