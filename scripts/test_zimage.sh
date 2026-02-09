#!/usr/bin/env bash
# Test Z-Image pipeline via CLI.
#
# Required models (set via environment variables or edit the defaults below):
#   TRANSFORMER - Path to Z-Image transformer (e.g. Tongyi-MAI/Z-Image-Turbo)
#   QWEN3_MODEL        - Qwen3-4B tokenizer + encoder
#   VAE_MODEL          - FLUX-style VAE (16 channels)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

TRANSFORMER="${TRANSFORMER:-Tongyi-MAI/Z-Image-Turbo}"
QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3-4B}"
VAE_MODEL="${VAE_MODEL:-black-forest-labs/FLUX.1-dev}"
OUTPUT="${OUTPUT:-output_zimage_$(date +%s).png}"

echo "=== Testing Z-Image Pipeline ==="
echo "Transformer: $TRANSFORMER"
echo "Qwen3:       $QWEN3_MODEL"
echo "Output:      $OUTPUT"
echo ""

inference-engine generate \
    --prompt "mountain landscape at sunset, dramatic clouds, warm colors" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type z_image \
    --qwen3-tokenizer "$QWEN3_MODEL" \
    --qwen3-encoder "$QWEN3_MODEL" \
    --vae-model "$VAE_MODEL" \
    --steps 5 \
    --cfg-scale 1.0 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
