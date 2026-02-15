#!/usr/bin/env bash
# Test LoRA application with any pipeline.
#
# This script demonstrates running a FLUX.1 Dev generation with LoRAs applied.
# Modify for other model types as needed.
#
#
#
# Required:
#   LORA_PATH  - Path to LoRA safetensors file
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

TRANSFORMER="${TRANSFORMER:-city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf}"
TEXT_ENCODER="${TEXT_ENCODER:-black-forest-labs/FLUX.1-dev}"
VAE_MODEL="${VAE_MODEL:-black-forest-labs/FLUX.1-dev}"
LORA_PATH="${LORA_PATH:-/home/alex/.cache/huggingface/hub/models--alexrzem--flux-loras/snapshots/7ddcce1925b55ccbed32322bf617e5ad5df8bd2f/dev/Moebius_-_Corruptedcupid.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-0.8}"
OUTPUT_W="${OUTPUT:-output_lora_with_$(date +%s).png}"
OUTPUT_WO="${OUTPUT:-output_lora_without_$(date +%s).png}"

PROMPT="Moebiusstyle, a planet with rings, space dust, psychedelic art"

echo "=== Testing LoRA Application ==="
echo "Transformer:  $TRANSFORMER"
echo "Text encoder: $TEXT_ENCODER"
echo "LoRA:         $LORA_PATH (strength=$LORA_STRENGTH)"
echo "Output:       $OUTPUT_W"
echo ""

uv run rzem-ai-inference-engine generate \
    --prompt "$PROMPT" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$TEXT_ENCODER" \
    --clip-encoder "$TEXT_ENCODER" \
    --t5-tokenizer "$TEXT_ENCODER" \
    --t5-encoder "$TEXT_ENCODER" \
    --vae-model "$VAE_MODEL" \
    --steps 30 \
    --cfg-scale 3.5 \
    --seed 65611195 \
    --output "$OUTPUT_WO"

uv run rzem-ai-inference-engine generate \
    --prompt "$PROMPT" \
    --transformer-model "$TRANSFORMER" \
    --transformer-type flux1_dev \
    --clip-tokenizer "$TEXT_ENCODER" \
    --clip-encoder "$TEXT_ENCODER" \
    --t5-tokenizer "$TEXT_ENCODER" \
    --t5-encoder "$TEXT_ENCODER" \
    --vae-model "$VAE_MODEL" \
    --lora "${LORA_PATH}:${LORA_STRENGTH}" \
    --steps 30 \
    --cfg-scale 3.5 \
    --seed 65611195 \
    --output "$OUTPUT_W"

echo ""
echo "Done. Output saved to $OUTPUT_W"
