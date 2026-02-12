#!/usr/bin/env bash
# Builds standalone executables for RZEM AI Inference Engine
# Auto-detects platform (CUDA on Linux/Windows, MPS on macOS, CPU fallback)
#
# Usage:
#   bash build.sh server    # Build server variant (generate + serve)
#   bash build.sh cli       # Build CLI variant (generate only)
#   bash build.sh all       # Build both variants (default)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/scripts/_activate.sh"

# Check PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: PyInstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

# Detect platform and PyTorch variant
detect_platform() {
    local os_type=$(uname -s)
    local platform=""
    local has_cuda="false"

    # Detect CUDA availability via PyTorch (more accurate than nvidia-smi)
    has_cuda=$(python3 -c "import torch; print('true' if torch.cuda.is_available() else 'false')" 2>/dev/null || echo "false")

    case "$os_type" in
        Linux)
            if [ "$has_cuda" = "true" ]; then
                platform="Linux-CUDA"
            else
                platform="Linux-CPU"
            fi
            ;;
        Darwin)
            # macOS - check for MPS (Apple Silicon M1+)
            local has_mps=$(python3 -c "import torch; print('true' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'false')" 2>/dev/null || echo "false")
            if [ "$has_mps" = "true" ]; then
                platform="macOS-MPS"
            else
                platform="macOS-CPU"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            # Windows
            if [ "$has_cuda" = "true" ]; then
                platform="Windows-CUDA"
            else
                platform="Windows-CPU"
            fi
            ;;
        *)
            platform="Unknown"
            ;;
    esac

    echo "$platform"
}

# Build function
build_variant() {
    local variant=$1
    local spec_file="$PROJECT_DIR/packaging/pyinstaller/rzem-ai-inference-engine-${variant}.spec"
    local platform=$(detect_platform)

    if [ ! -f "$spec_file" ]; then
        echo "Error: Spec file not found: $spec_file"
        exit 1
    fi

    echo "=========================================="
    echo "Building: $variant"
    echo "Platform: $platform"
    echo "Spec file: $spec_file"
    echo "=========================================="

    cd "$PROJECT_DIR"
    pyinstaller "$spec_file" --clean --noconfirm

    # Check for executable (Unix vs Windows)
    if [ -f "dist/rzem-ai-inference-engine-${variant}/rzem-ai-inference-engine-${variant}" ] || \
       [ -f "dist/rzem-ai-inference-engine-${variant}/rzem-ai-inference-engine-${variant}.exe" ]; then
        echo "✓ Build successful: dist/rzem-ai-inference-engine-${variant}/"
        echo ""
    else
        echo "✗ Build failed for ${variant}"
        exit 1
    fi
}

# Parse arguments
BUILD_TYPE="${1:-all}"

case "$BUILD_TYPE" in
    server)
        build_variant "server"
        ;;
    cli)
        build_variant "cli"
        ;;
    all)
        build_variant "server"
        echo ""
        build_variant "cli"
        ;;
    *)
        echo "Usage: $0 {server|cli|all}"
        echo "  server - Build server variant (generate + serve commands)"
        echo "  cli    - Build CLI variant (generate command only)"
        echo "  all    - Build both variants (default)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Platform detected: $(detect_platform)"
echo "Build complete!"
echo "=========================================="
echo ""
echo "To test the executable(s):"
if [ "$BUILD_TYPE" = "server" ] || [ "$BUILD_TYPE" = "all" ]; then
    echo "  ./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server --help"
fi
if [ "$BUILD_TYPE" = "cli" ] || [ "$BUILD_TYPE" = "all" ]; then
    echo "  ./dist/rzem-ai-inference-engine-cli/rzem-ai-inference-engine-cli --help"
fi
