#!/usr/bin/env bash
# Test FLUX.1 Dev pipeline with a GGUF-quantized transformer.
#
# Demonstrates the repo/file syntax for single-file downloads:
#   city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf
#
# Text encoders and VAE come from the BFL repo (safetensors).
#
#
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

TRANSFORMER="${TRANSFORMER:-city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf}"
TEXT_ENCODER="${TEXT_ENCODER:-black-forest-labs/FLUX.1-dev}"
VAE_MODEL="${VAE_MODEL:-black-forest-labs/FLUX.1-dev}"
OUTPUT="${OUTPUT:-output_flux1_gguf_$(date +%s).png}"

echo "=== Testing FLUX.1 Dev Pipeline (GGUF transformer) ==="
echo "Transformer:  $TRANSFORMER"
echo "Text encoder: $TEXT_ENCODER"
echo "VAE:          $VAE_MODEL"
echo "Output:       $OUTPUT"
echo ""

uv run rzem-ai-inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting, photorealistic" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$TEXT_ENCODER" \
    --clip-encoder "$TEXT_ENCODER" \
    --t5-tokenizer "$TEXT_ENCODER" \
    --t5-encoder "$TEXT_ENCODER" \
    --vae-model "$VAE_MODEL" \
    --steps 20 \
    --cfg-scale 3.5 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
