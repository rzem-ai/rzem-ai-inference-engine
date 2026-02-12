#!/usr/bin/env bash
# Cleans PyInstaller build artifacts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Cleaning PyInstaller build artifacts..."

cd "$PROJECT_ROOT"

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
