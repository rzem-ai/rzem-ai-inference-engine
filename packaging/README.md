# RZEM AI Inference Engine - Standalone Executables

This directory contains PyInstaller configuration for building standalone executables of the RZEM AI Inference Engine. These executables bundle all dependencies (except model weights) into a single distributable package.

## Quick Start

```bash
# Build both variants (default)
bash build.sh all

# Build only server variant (generate + serve)
bash build.sh server

# Build only CLI variant (generate only)
bash build.sh cli

# Run the executable
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server --help
```

## Why Use PyInstaller?

**Advantages:**

- No Python installation required on target machines
- All dependencies bundled (PyTorch, diffusers, transformers)
- Simplified deployment in restricted environments
- Consistent Python environment across systems

**Trade-offs:**

- Large executable size (~3-4 GB for server, ~2.5-3.5 GB for CLI)
- Platform-specific builds (Linux builds won't run on Windows/macOS)
- Model weights NOT included (still downloaded from HuggingFace Hub)
- Slower startup than pip-installed version (one-time extraction)

## Build Variants

| Variant | Size | Commands | Use Case |
|---------|------|----------|----------|
| **Server** | ~3-4 GB | `generate`, `serve` | Full deployment with REST API |
| **CLI** | ~2.5-3.5 GB | `generate` only | Image generation without web server |

Both variants include:

- PyTorch (CUDA, MPS, or CPU based on installed version)
- All four model pipelines (FLUX.1, FLUX.2, Z-Image, Qwen-Image)
- LoRA support
- GGUF quantization support

## Build Requirements

- Python 3.10+
- PyInstaller >= 5.0: `pip install pyinstaller`
- All project dependencies installed: `pip install -e .`
- 10+ GB free disk space for build artifacts

## Platform Detection

The build script automatically detects your platform and PyTorch variant:

| Platform | Detection Method | Result |
|----------|------------------|--------|
| **Linux** | `torch.cuda.is_available()` | Linux-CUDA or Linux-CPU |
| **macOS** | `torch.backends.mps.is_available()` | macOS-MPS or macOS-CPU |
| **Windows** | `torch.cuda.is_available()` | Windows-CUDA or Windows-CPU |

**Important:** The executable will only work on the same platform and accelerator type it was built on. Build separate executables for each target platform.

## Building for Different Platforms

### Linux with CUDA

```bash
# Ensure CUDA PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build
bash build.sh all

# Result: dist/rzem-ai-inference-engine-{server,cli}/ (Linux-CUDA)
```

### Linux CPU-only

```bash
# Ensure CPU PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Build
bash build.sh all

# Result: dist/rzem-ai-inference-engine-{server,cli}/ (Linux-CPU)
```

### macOS (Apple Silicon)

```bash
# PyTorch with MPS support (PyTorch 2.3+)
pip install torch

# Build
bash build.sh all

# Result: dist/rzem-ai-inference-engine-{server,cli}/ (macOS-MPS)
```

### Windows with CUDA

```bash
# Ensure CUDA PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build
bash build.sh all

# Result: dist/rzem-ai-inference-engine-{server,cli}/ (Windows-CUDA)
```

### Windows CPU-only

```bash
# Ensure CPU PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Build
bash build.sh all

# Result: dist/rzem-ai-inference-engine-{server,cli}/ (Windows-CPU)
```

## Usage

### Server Variant (Full)

```bash
# Help
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server --help

# Generate image
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server generate \
    --prompt "a cat sitting on a windowsill" \
    --transformer-model black-forest-labs/FLUX.1-dev \
    --transformer-type flux1_dev \
    --clip-tokenizer black-forest-labs/FLUX.1-dev \
    --clip-encoder black-forest-labs/FLUX.1-dev \
    --t5-tokenizer black-forest-labs/FLUX.1-dev \
    --t5-encoder black-forest-labs/FLUX.1-dev \
    --vae-model black-forest-labs/FLUX.1-dev \
    --steps 20 \
    --output output.png

# Start REST API server
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server serve \
    --host 127.0.0.1 \
    --port 8000 \
    --device auto
```

### CLI Variant (Generate-only)

```bash
# Help
./dist/rzem-ai-inference-engine-cli/rzem-ai-inference-engine-cli --help

# Generate image (same as server variant)
./dist/rzem-ai-inference-engine-cli/rzem-ai-inference-engine-cli generate \
    --prompt "mountain landscape at sunset" \
    --transformer-model Tongyi-MAI/Z-Image-Turbo \
    --transformer-type z_image \
    --qwen3-tokenizer Qwen/Qwen3-4B \
    --qwen3-encoder Qwen/Qwen3-4B \
    --vae-model black-forest-labs/FLUX.1-dev \
    --steps 9 \
    --output output.png

# Note: 'serve' command is NOT available in CLI variant
```

## Model Weights

**Important:** Model weights are NOT bundled in the executable. You must:

1. **Download models from HuggingFace Hub** (automatic on first use):

   ```bash
   # First run will download models to ~/.cache/huggingface/hub/
   ./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server generate \
       --transformer-model black-forest-labs/FLUX.1-dev \
       ...
   ```

2. **Or use local model paths**:

   ```bash
   ./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server generate \
       --transformer-model /path/to/local/flux.safetensors \
       ...
   ```

3. **Set HuggingFace token** (for private models):

   ```bash
   export HF_TOKEN=your_token_here
   # Or place in ~/.huggingface/token
   ```

## Distribution

### Packaging for End Users

1. **Zip the distribution folder**:

   ```bash
   cd dist
   zip -r rzem-ai-inference-engine-server-linux-cuda.zip rzem-ai-inference-engine-server/
   ```

2. **Include README** with:
   - System requirements (platform, GPU)
   - Model download instructions
   - Example usage commands

3. **Upload** to GitHub Releases, cloud storage, or internal deployment system

### System Requirements

Inform users of the minimum requirements:

| Variant | Platform | Memory | GPU | Storage |
|---------|----------|--------|-----|---------|
| Server (CUDA) | Linux x64 | 32+ GB RAM | NVIDIA GPU (24+ GB VRAM) | 50+ GB |
| Server (MPS) | macOS ARM64 | 32+ GB Unified Memory | Apple Silicon M3+ | 50+ GB |
| CLI (CUDA) | Linux x64 | 32+ GB RAM | NVIDIA GPU (24+ GB VRAM) | 50+ GB |
| CLI (CPU) | Any | 64+ GB RAM | None | 50+ GB (slow) |

Storage includes:

- Executable: ~3-4 GB
- Model cache: 10-50 GB (varies by model)
- Output images: varies

## Customization

### Excluding GGUF Support

To reduce executable size by ~100 MB, remove GGUF from spec files:

1. Edit `packaging/pyinstaller/rzem-ai-inference-engine-{server,cli}.spec`
2. Remove `'gguf'` from `hiddenimports` list
3. Rebuild

**Note:** GGUF-quantized models will not work in builds without GGUF support.

### Modifying Hidden Imports

If you add new dependencies or pipelines, update the spec files:

1. Edit `packaging/pyinstaller/rzem-ai-inference-engine-{server,cli}.spec`
2. Add new modules to `hiddenimports` list
3. Rebuild and test

See `packaging/pyinstaller/README.md` for technical details.

## Troubleshooting

### Build Fails with "Module Not Found"

**Cause:** Missing dependency or incorrect spec file hiddenimports.

**Fix:**

1. Check PyInstaller build output for warnings
2. Add missing module to spec file's `hiddenimports`
3. Or add to `packaging/pyinstaller/hooks/hook-*.py`
4. Rebuild

### Executable Runs but "No module named X"

**Cause:** Dynamic import not discovered by PyInstaller.

**Fix:**

1. Identify the missing module
2. Add to spec file's `hiddenimports` list
3. Rebuild

### CUDA Not Available in Built Executable

**Cause:** Built with CPU-only PyTorch.

**Fix:**

1. Uninstall PyTorch: `pip uninstall torch`
2. Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Rebuild

Verify before building:

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Build is Too Large

**Causes:**

- PyTorch CUDA builds include large CUDA libraries (~2 GB)
- All four pipelines are included (even if you only use one)

**Mitigations:**

1. Use CPU-only PyTorch for smaller builds (CPU inference only)
2. Remove unused dependencies from spec file (advanced)
3. Use one-file mode with UPX compression (slower startup)

### Models Not Found After Distribution

**Cause:** Model paths are system-specific.

**Fix:** Instruct users to:

1. Use HuggingFace Hub model names (e.g., `black-forest-labs/FLUX.1-dev`)
2. Or provide absolute local paths
3. Ensure `HF_TOKEN` is set for private models

## Performance Notes

- **Startup time:** 5-10 seconds slower than pip-installed version (one-time)
- **Inference speed:** Identical to pip-installed version
- **Memory usage:** Identical to pip-installed version

## Getting Help

- **Build issues:** See `packaging/pyinstaller/README.md` for technical details
- **Runtime issues:** Same as pip-installed version - check main README.md
- **PyInstaller docs:** <https://pyinstaller.org/en/stable/>

## Alternatives to PyInstaller

Consider these alternatives for different use cases:

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **pip install** | Small, fast, updatable | Requires Python | Development, Python users |
| **PyInstaller** | Standalone, no Python needed | Large, platform-specific | End users, restricted envs |
| **Docker** | Cross-platform, reproducible | Requires Docker runtime | Servers, cloud deployment |
| **Conda** | Environment management | Requires Conda | Data scientists, ML teams |
