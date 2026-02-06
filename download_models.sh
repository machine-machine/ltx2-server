#!/bin/bash
set -e

MODEL_DIR="/models"
HF_BASE="https://huggingface.co/Lightricks/LTX-2/resolve/main"

echo "=== LTX-2 Model Download ==="
echo "Model directory: $MODEL_DIR"
df -h /models 2>/dev/null || true

# FP8 distilled model (~10GB) - best for 3090
if [ ! -f "$MODEL_DIR/ltx-2-19b-distilled-fp8.safetensors" ]; then
    echo "Downloading LTX-2 distilled FP8 model..."
    curl -L --retry 3 --progress-bar -o "$MODEL_DIR/ltx-2-19b-distilled-fp8.safetensors" \
        "$HF_BASE/ltx-2-19b-distilled-fp8.safetensors"
else
    echo "✅ LTX-2 distilled FP8 model already present."
fi

# Spatial upscaler
if [ ! -f "$MODEL_DIR/ltx-2-spatial-upscaler-x2-1.0.safetensors" ]; then
    echo "Downloading spatial upscaler..."
    curl -L --retry 3 --progress-bar -o "$MODEL_DIR/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
        "$HF_BASE/ltx-2-spatial-upscaler-x2-1.0.safetensors"
else
    echo "✅ Spatial upscaler already present."
fi

# Distilled LoRA
if [ ! -f "$MODEL_DIR/ltx-2-19b-distilled-lora-384.safetensors" ]; then
    echo "Downloading distilled LoRA..."
    curl -L --retry 3 --progress-bar -o "$MODEL_DIR/ltx-2-19b-distilled-lora-384.safetensors" \
        "$HF_BASE/ltx-2-19b-distilled-lora-384.safetensors"
else
    echo "✅ Distilled LoRA already present."
fi

# Auth header for gated models (Gemma requires HF login)
HF_AUTH=""
if [ -n "$HF_TOKEN" ]; then
    HF_AUTH="-H \"Authorization: Bearer $HF_TOKEN\""
    echo "🔑 HuggingFace token found"
fi

# Helper: download with optional auth
hf_download() {
    local url="$1" dest="$2" show_progress="${3:-false}"
    if [ "$show_progress" = "true" ]; then
        eval curl -L --retry 3 --progress-bar $HF_AUTH -o "\"$dest\"" "\"$url\""
    else
        eval curl -L --retry 3 -sS $HF_AUTH -o "\"$dest\"" "\"$url\""
    fi
    # Verify it's not an HTML error page
    if head -c 20 "$dest" 2>/dev/null | grep -qi "access\|<!DOCTYPE\|<html"; then
        echo "  ❌ Download failed (gated model? check HF_TOKEN): $dest"
        rm -f "$dest"
        return 1
    fi
    return 0
}

# Gemma 3 text encoder (quantized) — GATED MODEL, needs HF_TOKEN
GEMMA_DIR="$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
HF_GEMMA="https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/resolve/main"

# Always re-validate config.json (previous downloads may have saved HTML error pages)
if [ -f "$GEMMA_DIR/config.json" ]; then
    if head -c 5 "$GEMMA_DIR/config.json" | grep -q "^{"; then
        echo "✅ Gemma 3 text encoder already present and valid."
    else
        echo "⚠️  Gemma files are corrupt (HTML error pages from gated download). Re-downloading..."
        rm -rf "$GEMMA_DIR"
    fi
fi

if [ ! -d "$GEMMA_DIR" ] || [ ! -f "$GEMMA_DIR/config.json" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "❌ HF_TOKEN not set! Gemma 3 is a gated model and requires authentication."
        echo "   Set HF_TOKEN environment variable with a HuggingFace token that has Gemma access."
        exit 1
    fi
    echo "Downloading Gemma 3 text encoder (gated model, using HF_TOKEN)..."
    mkdir -p "$GEMMA_DIR"
    
    # All config/tokenizer/processor files needed by the pipeline
    for f in config.json tokenizer_config.json tokenizer.json tokenizer.model special_tokens_map.json \
             model.safetensors.index.json generation_config.json preprocessor_config.json \
             processor_config.json added_tokens.json chat_template.json; do
        if [ ! -f "$GEMMA_DIR/$f" ]; then
            echo "  Downloading $f..."
            hf_download "$HF_GEMMA/$f" "$GEMMA_DIR/$f" || echo "  Warning: $f not available (may be optional)"
        fi
    done
    
    # Download model shards
    for i in $(seq -w 1 5); do
        SHARD="model-0000${i}-of-00005.safetensors"
        if [ ! -f "$GEMMA_DIR/$SHARD" ]; then
            echo "  Downloading $SHARD..."
            hf_download "$HF_GEMMA/$SHARD" "$GEMMA_DIR/$SHARD" true
        else
            echo "  ✅ $SHARD already present."
        fi
    done
else
    # Ensure all config files exist and are valid
    for f in tokenizer.model preprocessor_config.json processor_config.json added_tokens.json chat_template.json; do
        if [ ! -f "$GEMMA_DIR/$f" ]; then
            echo "  Downloading missing $f..."
            hf_download "$HF_GEMMA/$f" "$GEMMA_DIR/$f" || echo "  Warning: $f not available"
        fi
    done
fi

echo ""
echo "=== All models ready ==="
ls -lh $MODEL_DIR/*.safetensors 2>/dev/null || true
echo ""
du -sh $MODEL_DIR 2>/dev/null || true
