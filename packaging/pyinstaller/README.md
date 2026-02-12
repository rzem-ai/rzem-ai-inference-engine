# PyInstaller Technical Reference

This document provides technical details about the PyInstaller configuration for RZEM AI Inference Engine.

## Spec File Anatomy

### Entry Point

Both spec files use the same entry point:

```python
[str(src_path / 'rzem_ai_inference_engine' / '__main__.py')]
```

This is equivalent to `python -m rzem_ai_inference_engine`, which imports `cli:main()` and starts the Click CLI.

### Analysis Phase

```python
a = Analysis(
    ['path/to/__main__.py'],  # Entry point script
    pathex=[],                 # Additional import paths
    binaries=[],               # Binary dependencies (native libs)
    datas=[],                  # Data files to bundle
    hiddenimports=[...],       # Modules not detected by static analysis
    hookspath=['./hooks'],     # Custom PyInstaller hooks
    excludes=[],               # Modules to explicitly exclude
    ...
)
```

**Key parameters:**

- **`hiddenimports`**: Critical for dynamic imports. PyInstaller's static analysis can't detect:
  - Imports inside function bodies (`cli.py` imports engine only when commands run)
  - Conditional imports (pipeline files import models based on type)
  - String-based imports (`importlib.import_module()`)

- **`hookspath`**: Points to `packaging/pyinstaller/hooks/` for custom hooks that collect submodules and data files

- **`excludes`**: Used in CLI variant to explicitly exclude web stack (fastapi, uvicorn)

### Build Modes

**One-Folder Mode** (current configuration):

```python
exe = EXE(
    ...
    exclude_binaries=True,  # Binaries go in COLLECT step
    ...
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    ...
    name='rzem-ai-inference-engine-server',
)
```

**Result:** `dist/rzem-ai-inference-engine-server/` directory with executable and `_internal/` subfolder.

**Pros:**

- Faster startup (no extraction to temp dir)
- Easier debugging (can inspect bundled files)
- Better for large dependencies like PyTorch

**Cons:**

- Multiple files to distribute (use zip for distribution)

**One-File Mode** (not used, shown for reference):

```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # Include binaries in EXE
    a.zipfiles,
    a.datas,
    [],
    name='rzem-ai-inference-engine-server',
    ...
    exclude_binaries=False,  # Embed everything
)
```

**Result:** Single `rzem-ai-inference-engine-server` executable file.

**Pros:**

- Single file to distribute

**Cons:**

- Slower startup (extracts to temp on each run)
- Large file (3-4 GB)
- Harder to debug

## Hidden Imports Breakdown

### Core ML Stack

```python
'torch',
'torch.distributed',
'torch.distributed.nn',
'torch.distributed.distributed_c10d',
```

**Why:** PyTorch has complex internal imports that aren't always detected. The distributed modules are loaded lazily.

### Diffusers

```python
'diffusers.models.transformers.flux',
'diffusers.models.transformers.flux2',
'diffusers.models.transformers.z_image',
'diffusers.models.transformers.qwen_image',
```

**Why:** Pipeline files import these conditionally based on `transformer_type`. Example from `flux1.py:75-77`:

```python
from diffusers import FluxTransformer2DModel  # Inside run() method
```

### Transformers

```python
'transformers.models.clip.modeling_clip',
'transformers.models.t5.modeling_t5',
'transformers.models.qwen3.modeling_qwen3',
```

**Why:** Same as diffusers - conditional imports in pipeline files.

### Web Stack (Server Variant Only)

```python
'uvicorn.loops.auto',
'uvicorn.protocols.http.auto',
'uvicorn.protocols.websockets.auto',
```

**Why:** Uvicorn uses runtime discovery for protocol implementations. The `.auto` modules select the best available implementation (asyncio, uvloop, etc.).

### Package Modules

```python
'rzem_ai_inference_engine.pipeline.flux1',
'rzem_ai_inference_engine.pipeline.flux2',
'rzem_ai_inference_engine.pipeline.z_image',
'rzem_ai_inference_engine.pipeline.qwen_image',
```

**Why:** All four pipelines are registered in `engine.py:64-69`:

```python
self._pipelines: dict[TransformerType, Any] = {
    TransformerType.FLUX1_DEV: Flux1DevPipeline(),
    TransformerType.FLUX2_DEV: Flux2DevPipeline(),
    TransformerType.Z_IMAGE: ZImagePipeline(),
    TransformerType.QWEN_IMAGE: QwenImagePipeline(),
}
```

Even if a user only uses FLUX.1, all four are instantiated, so all must be included.

## Custom Hooks

Hooks are Python files in `packaging/pyinstaller/hooks/` that tell PyInstaller how to collect a package.

### Hook Structure

```python
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Hidden imports
hiddenimports = collect_submodules('package_name')
hiddenimports += ['specific.module', 'another.module']

# Data files (configs, tokenizers, etc.)
datas = collect_data_files('package_name', include_py_files=True)
```

### When to Update Hooks

| Hook | Update Trigger | Example |
|------|---------------|---------|
| `hook-diffusers.py` | New diffusers model class | Adding support for Stable Diffusion 3 |
| `hook-transformers.py` | New transformers model | Adding Llama 3 text encoder |
| `hook-rzem_ai_inference_engine.py` | New package module | Adding `pipeline/stable_diffusion3.py` |

### Testing Hooks

After modifying hooks:

1. **Clean build:**

   ```bash
   bash clean.sh
   ```

2. **Rebuild with verbose output:**

   ```bash
   cd /path/to/project
   pyinstaller packaging/pyinstaller/rzem-ai-inference-engine-server.spec \
       --clean --noconfirm --log-level DEBUG
   ```

3. **Check for warnings:**
   Look for "WARNING: Hidden import '...' not found" in build output.

4. **Test executable:**

   ```bash
   ./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server --help
   ```

## Platform-Specific Notes

### Linux CUDA

**What's included:**

- PyTorch with CUDA binaries (~2 GB)
- cuDNN, cuBLAS, NCCL libraries
- NVIDIA runtime libraries

**Requirements on target system:**

- NVIDIA drivers (no CUDA toolkit needed)
- libc >= 2.27 (Ubuntu 18.04+, CentOS 8+)

**Testing:**

```bash
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server generate \
    --device cuda:0 \
    ...
```

### macOS MPS (Apple Silicon)

**What's included:**

- PyTorch with MPS backend
- Accelerate framework bindings

**Requirements on target system:**

- macOS 13.0+ (Ventura or later)
- Apple Silicon M1/M2/M3 series

**Testing:**

```bash
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server generate \
    --device mps \
    ...
```

### Windows CUDA

**What's included:**

- PyTorch with CUDA binaries
- cuDNN, cuBLAS DLLs
- Microsoft Visual C++ redistributables

**Requirements on target system:**

- NVIDIA drivers
- Windows 10/11

**Testing:**

```powershell
.\dist\rzem-ai-inference-engine-server\rzem-ai-inference-engine-server.exe generate `
    --device cuda:0 `
    ...
```

### Windows CPU

**What's included:**

- PyTorch CPU-only
- MKL (Math Kernel Library) for optimized CPU inference

**Requirements on target system:**

- Windows 10/11
- 64+ GB RAM (models are large)

**Testing:**

```powershell
.\dist\rzem-ai-inference-engine-server\rzem-ai-inference-engine-server.exe generate `
    --device cpu `
    ...
```

## Advanced Configuration

### UPX Compression

PyInstaller can compress binaries with UPX to reduce size:

```python
exe = EXE(
    ...
    upx=True,
    upx_exclude=['torch', 'diffusers'],  # Don't compress ML libs (causes issues)
)

coll = COLLECT(
    ...
    upx=True,
    upx_exclude=['torch', 'diffusers'],
)
```

**Trade-offs:**

- ~10-20% size reduction
- Slower startup
- Can break CUDA libraries
- Not recommended for this project

### Code Signing (macOS/Windows)

**macOS:**

```python
exe = EXE(
    ...
    codesign_identity='Developer ID Application: Your Name (TEAM_ID)',
    entitlements_file='path/to/entitlements.plist',
)
```

**Windows:**
After building, sign with signtool:

```powershell
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com `
    dist\rzem-ai-inference-engine-server\rzem-ai-inference-engine-server.exe
```

### Excluding Unused Submodules

To reduce size, exclude unused transformers model classes:

1. **Identify unused models** (example: if you never use Qwen-Image):

   ```python
   excludes=[
       'transformers.models.qwen2',
       'transformers.models.qwen3',
   ]
   ```

2. **Remove from hiddenimports:**

   ```python
   # hiddenimports = [
   #     ...
   #     # 'transformers.models.qwen3',  # REMOVED
   # ]
   ```

3. **Update package code** to handle missing imports gracefully (advanced).

**Warning:** This can break runtime if any code path tries to import the excluded module.

## Debugging Build Issues

### Enable Debug Output

```bash
pyinstaller packaging/pyinstaller/rzem-ai-inference-engine-server.spec \
    --clean --noconfirm --log-level DEBUG --debug all
```

### Common Errors

#### "ImportError: No module named X"

**Cause:** Missing hidden import.

**Fix:**

1. Add `'X'` to spec file's `hiddenimports` list
2. Or add to appropriate hook file
3. Rebuild

#### "OSError: Could not find library X"

**Cause:** Native library not bundled.

**Fix:**

1. Find library path: `python -c "import X; print(X.__file__)"`
2. Add to spec file's `binaries`:

   ```python
   binaries=[
       ('/path/to/libX.so', 'lib'),
   ],
   ```

#### "RuntimeError: CUDA not available"

**Cause:** Built with CPU-only PyTorch.

**Fix:** Rebuild with CUDA PyTorch installed.

#### Build hangs during Analysis phase

**Cause:** PyInstaller analyzing too many files.

**Fix:**

1. Check `hiddenimports` for overly broad patterns
2. Use `--exclude-module` for large unused packages:

   ```bash
   pyinstaller ... --exclude-module matplotlib --exclude-module pandas
   ```

## File Size Breakdown

Typical build sizes (Linux CUDA, server variant):

| Component | Size | Notes |
|-----------|------|-------|
| PyTorch | ~2.0 GB | Includes CUDA libs |
| diffusers | ~50 MB | Model configs and base classes |
| transformers | ~500 MB | Model configs and tokenizers |
| Other deps | ~500 MB | accelerate, safetensors, etc. |
| Package code | ~10 MB | rzem_ai_inference_engine modules |
| **Total** | **~3-4 GB** | |

CLI variant is 500 MB smaller (no FastAPI/uvicorn).

## References

- [PyInstaller Manual](https://pyinstaller.org/en/stable/)
- [PyInstaller Hooks](https://pyinstaller.org/en/stable/hooks.html)
- [PyInstaller Spec Files](https://pyinstaller.org/en/stable/spec-files.html)
- [PyInstaller Runtime Information](https://pyinstaller.org/en/stable/runtime-information.html)
