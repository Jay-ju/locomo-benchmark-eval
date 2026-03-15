#!/bin/bash
# Ingest FULL LoCoMo dataset using BAAI/bge-small-zh

PYTHON=/data00/miniconda3/envs/daft_env/bin/python

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Allow overriding model via env var (e.g. for local path)
# Use HF_EMBEDDING_MODEL to avoid conflict with .env's EMBEDDING_MODEL (usually for Doubao)
EMBED_MODEL=${HF_EMBEDDING_MODEL:-"BAAI/bge-small-zh-v1.5"}

# FORCE VECTOR_DIM to 512 for BGE-Small (overriding .env which might have 2048 for Doubao)
# Unless HF_VECTOR_DIM is set explicitly
VECTOR_DIM=${HF_VECTOR_DIM:-512}

DB_PATH="data/lancedb_bge_small_full"

# Clean up existing DB to avoid duplication
if [ -d "$DB_PATH" ]; then
    echo "Removing existing DB at $DB_PATH..."
    rm -rf "$DB_PATH"
fi

echo "Ingesting FULL dataset using $EMBED_MODEL..."

# Use high parallelism for full dataset
$PYTHON -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --mode direct \
  --embed-provider local-huggingface \
  --embed-model "$EMBED_MODEL" \
  --vector-dim "$VECTOR_DIM" \
  --db-path $DB_PATH \
  --schema-mode basic \
  --parallelism 16

echo "Done. Full DB saved to $DB_PATH"
