#!/bin/bash
set -e

MODEL_DIR="/models"
HF_BASE="https://huggingface.co/Lightricks/LTX-2/resolve/main"

echo "=== LTX-2 Model Download ==="

# FP8 distilled model (~10GB) - best for 3090
if [ ! -f "$MODEL_DIR/ltx-2-19b-distilled-fp8.safetensors" ]; then
    echo "Downloading LTX-2 distilled FP8 model..."
    curl -L -o "$MODEL_DIR/ltx-2-19b-distilled-fp8.safetensors" \
        "$HF_BASE/ltx-2-19b-distilled-fp8.safetensors"
else
    echo "LTX-2 distilled FP8 model already present."
fi

# Spatial upscaler
if [ ! -f "$MODEL_DIR/ltx-2-spatial-upscaler-x2-1.0.safetensors" ]; then
    echo "Downloading spatial upscaler..."
    curl -L -o "$MODEL_DIR/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
        "$HF_BASE/ltx-2-spatial-upscaler-x2-1.0.safetensors"
else
    echo "Spatial upscaler already present."
fi

# Distilled LoRA
if [ ! -f "$MODEL_DIR/ltx-2-19b-distilled-lora-384.safetensors" ]; then
    echo "Downloading distilled LoRA..."
    curl -L -o "$MODEL_DIR/ltx-2-19b-distilled-lora-384.safetensors" \
        "$HF_BASE/ltx-2-19b-distilled-lora-384.safetensors"
else
    echo "Distilled LoRA already present."
fi

# Gemma 3 text encoder (quantized)
GEMMA_DIR="$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
if [ ! -d "$GEMMA_DIR" ]; then
    echo "Downloading Gemma 3 text encoder..."
    mkdir -p "$GEMMA_DIR"
    # Download key files from HF
    HF_GEMMA="https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main"
    for f in config.json tokenizer_config.json tokenizer.json special_tokens_map.json \
             model.safetensors.index.json generation_config.json; do
        curl -L -o "$GEMMA_DIR/$f" "$HF_GEMMA/$f" 2>/dev/null || true
    done
    # Download model shards
    for i in $(seq -w 1 5); do
        SHARD="model-0000${i}-of-00005.safetensors"
        if [ ! -f "$GEMMA_DIR/$SHARD" ]; then
            echo "  Downloading $SHARD..."
            curl -L -o "$GEMMA_DIR/$SHARD" "$HF_GEMMA/$SHARD"
        fi
    done
else
    echo "Gemma 3 text encoder already present."
fi

echo "=== All models ready ==="
ls -lh $MODEL_DIR/*.safetensors 2>/dev/null
