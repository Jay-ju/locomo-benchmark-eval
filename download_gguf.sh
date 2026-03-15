#!/bin/bash
# Load .env for proxy
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

mkdir -p data/models
TARGET="data/models/bge-small-zh-v1.5-f16.gguf"
URL="https://huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf/resolve/main/bge-small-zh-v1.5-f16.gguf"

echo "Downloading GGUF model to $TARGET..."
wget -O "$TARGET" "$URL"

if [ $? -eq 0 ]; then
    echo "Download success."
else
    echo "Download failed."
    exit 1
fi
