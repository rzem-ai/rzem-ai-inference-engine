#!/usr/bin/env bash
# Test FLUX.1 Dev pipeline — models from separate repos.
#
# Uses openai/clip-vit-large-patch14 for CLIP, google/t5-v1_1-xxl for T5,
# and black-forest-labs/FLUX.1-dev for the transformer and VAE.
# This also demonstrates the repo/file syntax for single-file downloads.
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
OUTPUT="${OUTPUT:-output_flux1_alt_$(date +%s).png}"

echo "=== Testing FLUX.1 Dev Pipeline (separate repos) ==="
echo "Transformer: $FLUX1_TRANSFORMER"
echo "CLIP:        $CLIP_ENCODER"
echo "T5:          $T5_ENCODER"
echo "VAE:         $VAE_MODEL"
echo "Output:      $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting, photorealistic" \
    --transformer-model "$FLUX1_TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$CLIP_TOKENIZER" \
    --clip-encoder "$CLIP_ENCODER" \
    --t5-tokenizer "$T5_TOKENIZER" \
    --t5-encoder "$T5_ENCODER" \
    --vae-model "$VAE_MODEL" \
    --steps 20 \
    --cfg-scale 3.5 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "$OUTPUT"

echo ""
echo "Done. Output saved to $OUTPUT"
