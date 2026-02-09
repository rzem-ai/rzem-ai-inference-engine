#!/usr/bin/env bash
# Test Qwen-Image pipeline via CLI.
#
# The Qwen/Qwen-Image repo bundles all components (transformer, text encoder,
# tokenizer, VAE) in subfolders. The pipeline resolves each automatically.
#
# Text encoder is Qwen2.5-VL-7B-Instruct (not Qwen3).
# VAE is AutoencoderKLQwenImage (not standard AutoencoderKL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

TRANSFORMER="${TRANSFORMER:-Qwen/Qwen-Image}"
OUTPUT="${OUTPUT:-output_qwen_image_$(date +%s).png}"

echo "=== Testing Qwen-Image Pipeline ==="
echo "Repo:   $TRANSFORMER"
echo "Output: $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a photorealistic portrait of a woman reading a book in a cozy library" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type qwen_image \
    --qwen3-tokenizer "$TRANSFORMER" \
    --qwen3-encoder "$TRANSFORMER" \
    --vae-model "$TRANSFORMER" \
    --steps 50 \
    --cfg-scale 4.0 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
