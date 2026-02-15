#!/usr/bin/env bash
# Setup script for the inference engine.
# Creates a virtual environment, installs dependencies, and verifies the install.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Inference Engine Setup ==="
echo "Project: $PROJECT_DIR"

# Create venv + install dependencies
echo "Creating virtual environment and installing dependencies..."
cd "$PROJECT_DIR"
uv venv
uv sync

# Verify
echo ""
echo "=== Verification ==="
uv run rzem-ai-inference-engine --help
echo ""
echo "Setup complete. Run any script directly — uv manages the environment automatically:"
echo "  bash scripts/test_zimage.sh"
