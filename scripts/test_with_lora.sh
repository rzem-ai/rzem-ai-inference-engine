#!/usr/bin/env bash
# Test LoRA application with any pipeline.
#
# This script demonstrates running a FLUX.1 Dev generation with LoRAs applied.
# Modify for other model types as needed.
#
# Required:
#   LORA_PATH  - Path to LoRA safetensors file
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

FLUX1_TRANSFORMER="${FLUX1_TRANSFORMER:-black-forest-labs/FLUX.1-dev}"
CLIP_TOKENIZER="${CLIP_TOKENIZER:-openai/clip-vit-large-patch14}"
CLIP_ENCODER="${CLIP_ENCODER:-openai/clip-vit-large-patch14}"
T5_TOKENIZER="${T5_TOKENIZER:-google/t5-v1_1-xxl}"
T5_ENCODER="${T5_ENCODER:-google/t5-v1_1-xxl}"
VAE_MODEL="${VAE_MODEL:-black-forest-labs/FLUX.1-dev}"
LORA_PATH="${LORA_PATH:?Set LORA_PATH to a .safetensors LoRA file}"
LORA_STRENGTH="${LORA_STRENGTH:-0.8}"
OUTPUT="${OUTPUT:-output_lora_$(date +%s).png}"

echo "=== Testing LoRA Application ==="
echo "Transformer: $FLUX1_TRANSFORMER"
echo "LoRA:        $LORA_PATH (strength=$LORA_STRENGTH)"
echo "Output:      $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting" \
    --transformer-model "$FLUX1_TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$CLIP_TOKENIZER" \
    --clip-encoder "$CLIP_ENCODER" \
    --t5-tokenizer "$T5_TOKENIZER" \
    --t5-encoder "$T5_ENCODER" \
    --vae-model "$VAE_MODEL" \
    --lora "${LORA_PATH}:${LORA_STRENGTH}" \
    --steps 20 \
    --cfg-scale 3.5 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
