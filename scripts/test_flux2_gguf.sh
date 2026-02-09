#!/usr/bin/env bash
# Test FLUX.2 Dev pipeline with a GGUF-quantized transformer.
#
# Uses Mistral3 for text encoding (passed via --qwen3-* CLI params).
# GGUF transformer from city96, Mistral3 text encoder from unsloth.
#
# Required models (set via environment variables or edit the defaults below):
#   FLUX2_TRANSFORMER  - GGUF quantized FLUX.2 transformer
#   TEXT_ENCODER       - Mistral3 text encoder (HF repo or GGUF)
#   VAE_MODEL          - FLUX.2 VAE (32 channels with BatchNorm)
set -euo pipefail

FLUX2_TRANSFORMER="${FLUX2_TRANSFORMER:-city96/FLUX.2-dev-gguf/flux2-dev-Q8_0.gguf}"
TEXT_ENCODER="${TEXT_ENCODER:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
VAE_MODEL="${VAE_MODEL:-Comfy-Org/flux2-dev/split_files/vae/flux2-vae.safetensors}"
OUTPUT="${OUTPUT:-output_flux2_gguf_$(date +%s).png}"

echo "=== Testing FLUX.2 Dev Pipeline (GGUF transformer) ==="
echo "Transformer:  $FLUX2_TRANSFORMER"
echo "Text encoder: $TEXT_ENCODER"
echo "VAE:          $VAE_MODEL"
echo "Output:       $OUTPUT"
echo ""

inference-engine generate \
    --prompt "a serene Japanese garden with cherry blossoms and a koi pond" \
    --transformer-model "$FLUX2_TRANSFORMER" \
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
