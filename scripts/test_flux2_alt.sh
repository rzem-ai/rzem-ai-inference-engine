#!/usr/bin/env bash
# Test FLUX.2 Dev pipeline via CLI.
#
# Required models (set via environment variables or edit the defaults below):
#   TRANSFORMER  - Path to FLUX.2 Dev transformer
#   QWEN3_MODEL        - Qwen3 tokenizer + encoder (e.g. Qwen/Qwen3-8B)
#   VAE_MODEL          - FLUX.2 VAE (32 channels)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

TRANSFORMER="${TRANSFORMER:-black-forest-labs/FLUX.2-dev}"
TEXT_ENCODER="${QWEN3_MODEL:-Qwen/Qwen3-4B}"
VAE_MODEL="${VAE_MODEL:-black-forest-labs/FLUX.2-dev}"
OUTPUT="${OUTPUT:-output_flux2_$(date +%s).png}"

echo "=== Testing FLUX.2 Dev Pipeline ==="
echo "Transformer:  $TRANSFORMER"
echo "Qwen3:        $TEXT_ENCODER"
echo "VAE:          $VAE_MODEL"
echo "Output:       $OUTPUT"
echo ""

uv run rzem-ai-inference-engine generate \
    --prompt "a serene Japanese garden with cherry blossoms and a koi pond" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type flux2_dev \
    --qwen3-tokenizer "$TEXT_ENCODER" \
    --qwen3-encoder "$TEXT_ENCODER" \
    --vae-model "$VAE_MODEL" \
    --steps 4 \
    --cfg-scale 1.0 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
