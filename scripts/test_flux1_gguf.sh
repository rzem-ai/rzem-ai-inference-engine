#!/usr/bin/env bash
# Test FLUX.1 Dev pipeline with a GGUF-quantized transformer.
#
# Demonstrates the repo/file syntax for single-file downloads:
#   city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf
#
# Text encoders and VAE come from the BFL repo (safetensors).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

FLUX1_TRANSFORMER="${FLUX1_TRANSFORMER:-city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf}"
BFL_REPO="${BFL_REPO:-black-forest-labs/FLUX.1-dev}"
OUTPUT="${OUTPUT:-output_flux1_gguf_$(date +%s).png}"

echo "=== Testing FLUX.1 Dev Pipeline (GGUF transformer) ==="
echo "Transformer: $FLUX1_TRANSFORMER"
echo "Text/VAE:    $BFL_REPO"
echo "Output:      $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting, photorealistic" \
    --transformer-model "$FLUX1_TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$BFL_REPO" \
    --clip-encoder "$BFL_REPO" \
    --t5-tokenizer "$BFL_REPO" \
    --t5-encoder "$BFL_REPO" \
    --vae-model "$BFL_REPO" \
    --steps 20 \
    --cfg-scale 3.5 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
