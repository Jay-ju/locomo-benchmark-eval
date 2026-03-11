#!/bin/bash
# Run LLM-as-a-Judge on OpenClaw QA Results

# Load config from .env (includes Proxy and API Keys)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PYTHON=/data00/miniconda3/envs/daft_env/bin/python

# Judge Config (Relies on OPENAI_* env vars loaded from .env)
# Or user can override here if needed.

INPUT_FILE="results/openclaw_eval_full.json"
OUTPUT_FILE="results/openclaw_eval_scored.json"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

echo "Judging QA results in $INPUT_FILE..."
echo "  Judge Model: $OPENAI_MODEL"

$PYTHON -m src.cli.main judge \
  $INPUT_FILE \
  --output $OUTPUT_FILE \
  --parallelism 16

echo "Done. Scored results: $OUTPUT_FILE"
