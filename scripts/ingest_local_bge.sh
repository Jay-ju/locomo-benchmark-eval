#!/bin/bash
# Ingest FULL LoCoMo dataset using BAAI/bge-small-zh

PYTHON=/data00/miniconda3/envs/daft_env/bin/python

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

DB_PATH="data/lancedb_bge_small_full"

# Clean up existing DB to avoid duplication
if [ -d "$DB_PATH" ]; then
    echo "Removing existing DB at $DB_PATH..."
    rm -rf "$DB_PATH"
fi

echo "Ingesting FULL dataset (all samples) using BAAI/bge-small-zh..."

# Use high parallelism for full dataset
$PYTHON -m src.cli.main ingest \
  data/locomo/locomo10.json \
  --mode direct \
  --embed-provider local-huggingface \
  --embed-model BAAI/bge-small-zh \
  --vector-dim 512 \
  --db-path $DB_PATH \
  --parallelism 16

echo "Done. Full DB saved to $DB_PATH"
