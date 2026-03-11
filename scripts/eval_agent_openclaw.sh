#!/bin/bash
# Eval using Agent Mode (OpenClaw for QA)

# Load config from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PYTHON=/data00/miniconda3/envs/daft_env/bin/python

# OpenClaw Config (Local Service)
OPENCLAW_URL="http://localhost:18789"
OPENCLAW_TOKEN="openclaw-secret"

echo "Running Full Eval (Agent Mode -> OpenClaw)..."

$PYTHON -m src.cli.main eval \
  data/locomo/locomo10.json \
  --mode agent \
  --base-url $OPENCLAW_URL \
  --api-key $OPENCLAW_TOKEN \
  --model doubao-pro-32k \
  --output results/openclaw_eval_full.json \
  --parallelism 10  # Use parallelism=10 to balance speed and server stability

echo "Done. Results saved to results/openclaw_eval_full.json"
