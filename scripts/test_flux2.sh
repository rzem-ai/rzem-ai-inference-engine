#!/usr/bin/env bash
# Test FLUX.2 Dev pipeline via CLI.
#
# The black-forest-labs/FLUX.2-dev repo bundles all components (transformer,
# text encoder [Mistral 3], tokenizer, VAE) in subfolders.
#
# Text encoder is Mistral 3 (not Qwen3).
# VAE is AutoencoderKLFlux2 with batch normalization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

FLUX2_REPO="${FLUX2_REPO:-black-forest-labs/FLUX.2-dev}"
OUTPUT="${OUTPUT:-output_flux2_$(date +%s).png}"

echo "=== Testing FLUX.2 Dev Pipeline ==="
echo "Repo:   $FLUX2_REPO"
echo "Output: $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a serene Japanese garden with cherry blossoms and a koi pond" \
    --transformer-model "$FLUX2_REPO" \
    --transformer-type flux2_dev \
    --qwen3-tokenizer "$FLUX2_REPO" \
    --qwen3-encoder "$FLUX2_REPO" \
    --vae-model "$FLUX2_REPO" \
    --steps 50 \
    --cfg-scale 4.0 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
