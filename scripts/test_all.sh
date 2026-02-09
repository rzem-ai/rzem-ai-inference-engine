#!/usr/bin/env bash
# Run all pipeline tests sequentially.
#
# Set model paths via environment variables before running, or edit the
# individual test scripts to point to your local model files.
#
# Usage:
#   # With HuggingFace models (will download on first run):
#   ./scripts/test_all.sh
#
#   # With local model paths:
#   FLUX1_TRANSFORMER=./models/flux1-dev.safetensors \
#   VAE_MODEL=./models/ae.safetensors \
#   ./scripts/test_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_activate.sh"

echo "========================================"
echo "  Inference Engine — Full Test Suite"
echo "========================================"
echo ""

run_test() {
    local name="$1"
    local script="$2"
    echo "────────────────────────────────────────"
    echo "  $name"
    echo "────────────────────────────────────────"
    if bash "$script"; then
        echo "  PASSED"
    else
        echo "  FAILED (exit code $?)"
    fi
    echo ""
}

run_test "FLUX.1 Dev"   "$SCRIPT_DIR/test_flux1.sh"
run_test "FLUX.1 Dev"   "$SCRIPT_DIR/test_flux1_gguf.sh"
run_test "FLUX.1 Dev"   "$SCRIPT_DIR/test_flux1_alt.sh"
run_test "FLUX.2 Dev"   "$SCRIPT_DIR/test_flux2.sh"
run_test "FLUX.2 Dev"   "$SCRIPT_DIR/test_flux2_gguf.sh"
run_test "FLUX.2 Dev"   "$SCRIPT_DIR/test_flux2_alt.sh"
run_test "Z-Image"      "$SCRIPT_DIR/test_zimage.sh"
run_test "Qwen-Image"   "$SCRIPT_DIR/test_qwen_image.sh"

echo "========================================"
echo "  All tests complete."
echo "========================================"
