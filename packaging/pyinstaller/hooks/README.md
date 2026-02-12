# PyInstaller Hooks

This directory contains custom PyInstaller hooks that ensure all dynamically imported modules are discovered during the bundling process.

## Hook Files

### `hook-diffusers.py`

Collects all diffusers model classes and submodules that are imported conditionally in pipeline files:
- Transformer models: FluxTransformer2DModel, Flux2Transformer2DModel, ZImageTransformer2DModel, QwenImageTransformer2DModel
- VAE models: AutoencoderKL, AutoencoderKLFlux2
- GGUF quantization support: GGUFQuantizationConfig

**Update when:**
- A new diffusers model class is used in any pipeline
- A new quantization format is supported

### `hook-transformers.py`

Collects transformers model classes, tokenizers, and data files:
- CLIP models and tokenizers (FLUX.1)
- T5 models and tokenizers (FLUX.1)
- Qwen3 models and tokenizers (Z-Image, FLUX.2, Qwen-Image)
- Auto classes for dynamic loading
- Tokenizer data files (vocab, merges, special tokens)

**Update when:**
- A new transformers model class is used in any pipeline
- A new tokenizer is required

### `hook-rzem_ai_inference_engine.py`

Ensures all package modules are collected:
- All four pipeline modules (flux1, flux2, z_image, qwen_image, lora_applicator, base)
- All API modules (app, routes, models, state, ws)
- All model management modules (cache, loader, memory)
- All queue modules (manager, processor)
- Core modules (engine, types, cli)

**Update when:**
- A new pipeline is added
- A new module is created in the package
- A module is restructured or renamed

## How PyInstaller Uses These Hooks

PyInstaller performs static analysis on your code to discover imports. However, it cannot detect:
- Imports inside function bodies (lazy imports)
- Conditional imports (if/else blocks)
- Dynamic imports (importlib, __import__)
- String-based imports

These hooks tell PyInstaller explicitly which modules to include using `hiddenimports` and `datas`.

## Testing Hooks

After modifying hooks, test with:

```bash
# Clean previous builds
bash clean.sh

# Build and check for missing imports
bash build.sh server

# If build succeeds, test the executable
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server --help
```

If you see "ModuleNotFoundError" when running the executable, the missing module should be added to the appropriate hook's `hiddenimports` list.

## References

- [PyInstaller Hooks Documentation](https://pyinstaller.org/en/stable/hooks.html)
- [PyInstaller Hook Utils API](https://pyinstaller.org/en/stable/hooks.html#module-PyInstaller.utils.hooks)
