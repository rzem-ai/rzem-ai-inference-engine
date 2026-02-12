#!/usr/bin/env bash
# Cleans PyInstaller build artifacts

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/scripts/_activate.sh"

echo "Cleaning PyInstaller build artifacts..."

cd "$PROJECT_DIR"

# Remove build directories
if [ -d "build" ]; then
    echo "  Removing build/"
    rm -rf build
fi

if [ -d "dist" ]; then
    echo "  Removing dist/"
    rm -rf dist
fi

# Remove any spec files in root (should only be in packaging/pyinstaller/)
if ls *.spec 1> /dev/null 2>&1; then
    echo "  Removing *.spec files from root"
    rm -f *.spec
fi

echo "✓ Clean complete"
