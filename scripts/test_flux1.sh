#!/usr/bin/env bash
# Test FLUX.1 Dev pipeline — all models from black-forest-labs/FLUX.1-dev.
#
# Uses the single HF repo for every component. The engine auto-resolves
# each to the correct subfolder (transformer/, text_encoder/, etc.).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

TRANSFORMER="${TRANSFORMER:-black-forest-labs/FLUX.1-dev}"
OUTPUT="${OUTPUT:-output_flux1_$(date +%s).png}"

echo "=== Testing FLUX.1 Dev Pipeline (BFL repo) ==="
echo "Repo:   $TRANSFORMER"
echo "Output: $OUTPUT"
echo ""

uv run rzem-ai-inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting, photorealistic" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$TRANSFORMER" \
    --clip-encoder "$TRANSFORMER" \
    --t5-tokenizer "$TRANSFORMER" \
    --t5-encoder "$TRANSFORMER" \
    --vae-model "$TRANSFORMER" \
    --steps 20 \
    --cfg-scale 3.5 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
