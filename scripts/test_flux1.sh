#!/usr/bin/env bash
# Test FLUX.1 Dev pipeline — all models from black-forest-labs/FLUX.1-dev.
#
# Uses the single HF repo for every component. The engine auto-resolves
# each to the correct subfolder (transformer/, text_encoder/, etc.).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

BFL_REPO="${BFL_REPO:-black-forest-labs/FLUX.1-dev}"
OUTPUT="${OUTPUT:-output_flux1_$(date +%s).png}"

echo "=== Testing FLUX.1 Dev Pipeline (BFL repo) ==="
echo "Repo:   $BFL_REPO"
echo "Output: $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting, photorealistic" \
    --transformer-model "$BFL_REPO" \
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
