#!/usr/bin/env bash
# Setup script for the inference engine.
# Creates a virtual environment, installs the package, and verifies the install.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Inference Engine Setup ==="
echo "Project: $PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/.venv"
fi

# Activate
source "$PROJECT_DIR/.venv/bin/activate"

# Install in editable mode
echo "Installing rzem-ai-inference-engine..."
pip install -e "$PROJECT_DIR" --quiet

# Verify
echo ""
echo "=== Verification ==="
rzem-ai-inference-engine --help
echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $PROJECT_DIR/.venv/bin/activate"
echo ""
echo "Or just run any test script directly — it activates the venv automatically:"
echo "  ./scripts/test_zimage.sh"
