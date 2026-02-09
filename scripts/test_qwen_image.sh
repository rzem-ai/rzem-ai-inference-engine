#!/usr/bin/env bash
# Test Qwen-Image pipeline via CLI.
#
# Required models (set via environment variables or edit the defaults below):
#   QWEN_IMAGE_TRANSFORMER - Path to Qwen-Image 20B transformer
#   QWEN3_MODEL            - Qwen3 tokenizer + encoder
#   VAE_MODEL              - FLUX-style VAE (16 channels)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

QWEN_IMAGE_TRANSFORMER="${QWEN_IMAGE_TRANSFORMER:-Qwen/Qwen-Image}"
QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3-4B}"
VAE_MODEL="${VAE_MODEL:-black-forest-labs/FLUX.1-dev}"
OUTPUT="${OUTPUT:-output_qwen_image_$(date +%s).png}"

echo "=== Testing Qwen-Image Pipeline ==="
echo "Transformer: $QWEN_IMAGE_TRANSFORMER"
echo "Qwen3:       $QWEN3_MODEL"
echo "Output:      $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a photorealistic portrait of a woman reading a book in a cozy library" \
    --transformer-model "$QWEN_IMAGE_TRANSFORMER" \
    --transformer-type qwen_image \
    --qwen3-tokenizer "$QWEN3_MODEL" \
    --qwen3-encoder "$QWEN3_MODEL" \
    --vae-model "$VAE_MODEL" \
    --steps 50 \
    --cfg-scale 4.0 \
    --width 1328 \
    --height 1328 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
