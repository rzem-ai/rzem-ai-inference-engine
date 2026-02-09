# CLAUDE.md — Inference Engine

## Project Overview

Text-to-image inference engine with job queue, event system, and automatic VRAM management. Supports FLUX.1 Dev, FLUX.2 Dev, Z-Image, and Qwen-Image model families. Python package installed via `pip install -e .`.

## Quick Reference

```bash
# Install
pip install -e .

# CLI
inference-engine generate --help
inference-engine generate --prompt "..." --transformer-model <path> --transformer-type flux1_dev --vae-model <path> ...

# Test scripts
bash scripts/test_flux1.sh        # FLUX.1 Dev (all from BFL repo)
bash scripts/test_flux1_alt.sh    # FLUX.1 Dev (separate repos)
bash scripts/test_flux1_gguf.sh   # FLUX.1 Dev (GGUF Q8_0 transformer)
bash scripts/test_zimage.sh       # Z-Image Turbo
```

## Architecture

- `src/inference_engine/engine.py` — Main `InferenceEngine` class (public API, thread-safe)
- `src/inference_engine/types.py` — All types: `JobParams`, `TransformerType`, events, enums
- `src/inference_engine/pipeline/` — Pipeline implementations (one per model family)
  - `base.py` — `BasePipeline` ABC: `validate_params`, `get_required_models`, `run`
  - `flux1.py` — FLUX.1 Dev: CLIP + T5 encoding, rectified flow Euler, guidance-distilled
  - `flux2.py` — FLUX.2 Dev: Qwen3 multi-layer encoding, 32-ch VAE, BN normalization
  - `z_image.py` — Z-Image: S3-DiT single-stream, Qwen3-4B, v-prediction with negation
  - `qwen_image.py` — Qwen-Image: 20B MMDiT, true CFG, Qwen3 encoding
- `src/inference_engine/models/cache.py` — Two-tier VRAM/RAM cache, smallest-first eviction
- `src/inference_engine/models/loader.py` — Path resolution (local, HF repo, HF repo+file)
- `src/inference_engine/models/memory.py` — VRAM estimation and device detection
- `src/inference_engine/lora/applicator.py` — LoRA format detection and weight patching
- `src/inference_engine/queue/` — Job queue (`manager.py`) and processor (`processor.py`)
- `src/inference_engine/cli.py` — Click CLI

## Key Patterns

### Model path resolution
Three formats: local path, `org/repo` (snapshot_download), `org/repo/file.ext` (hf_hub_download). Pipelines use `_resolve_sub()` for repos with subfolders (transformer/, vae/, text_encoder/, etc.). Download filter: `*.safetensors, *.bin, *.gguf, *.json, *.txt, *.model, *.tiktoken, *.py`.

### GGUF quantization
FLUX.1 pipeline detects `.gguf` extension on the transformer path and passes `GGUFQuantizationConfig(compute_dtype=dtype)` to `from_single_file`. Requires the `gguf` pip package. Weights stay quantized (uint8) and are dequantized on-the-fly during forward passes.

### Cache keys
Composite: `"{path}::{role}"` — e.g. `"black-forest-labs/FLUX.1-dev::transformer"`.

### Sequential model offloading (FLUX.1)
Text encoders and transformer don't fit in VRAM simultaneously. Pipeline stages: encode text -> release encoders -> load transformer -> denoise -> release -> load VAE -> decode. Other pipelines lock all models at once (they're smaller).

### Denoising
- **Timestep range**: Transformers expect timesteps in **[0, 1]**, not [0, 1000]. FLUX.1 passes sigma directly. Z-Image uses `1 - sigma`.
- **Euler step**: `latents = latents + (sigma_next - sigma_curr) * noise_pred`
- **Time schedule**: Linear interpolation mu formula: `mu = m * seq_len + b` where `m = (1.15 - 0.5) / (4096 - 256)`. Applied via exponential time shift: `exp(mu) / (exp(mu) + (1/t - 1)^sigma)`.

### LoRA system
Format detection (Kohya, Diffusers, XLabs, AIToolkit, OneTrainer) based on key patterns. Apply as weight deltas: `weight += strength * (alpha/rank) * (up @ down)`. Snapshot/unapply for model reuse between jobs.

### Event system
Thread-safe callback registration on `InferenceEngine`. Events: JOB_QUEUED, JOB_STARTED, JOB_PROGRESS, JOB_COMPLETED, JOB_FAILED, JOB_CANCELLED, MODEL_LOADING, MODEL_LOADED, MODEL_UNLOADED.

## Common Pitfalls

- **Timestep scaling**: Never pass `sigma * 1000` to FluxTransformer2DModel. It expects [0, 1].
- **VRAM eviction order**: Always call `ensure_vram()` BEFORE `model.to(device)`, not after.
- **Cache thread safety**: Loading happens outside the lock (slow I/O), but VRAM placement happens inside the lock with double-check.
- **torch.cuda.empty_cache()**: Must be called after moving models to CPU during eviction (guarded by `device.type == "cuda"`).
- **Z-Image text encoder**: Uses `Qwen3Model` (not `AutoModelForCausalLM`), `hidden_states[-2]`, mask-filtered to 2D.
- **Z-Image v-prediction**: `noise_pred = -model_output` (negate the output).
- **Z-Image `_time_shift` boundary**: Python floats raise on `1/0`. The function has explicit boundary checks for `t <= 0` (returns 0) and `t >= 1` (returns 1). The FLUX.1 version uses torch tensors which handle inf gracefully.
- **MPS/Apple Silicon**: Device detection, cache eviction, and generator all handle MPS. bfloat16 requires PyTorch 2.3+. No `empty_cache()` equivalent for MPS. Use `--vram-limit` to constrain unified memory usage.

## Dependencies

torch, diffusers (>=0.36), transformers, accelerate, safetensors, huggingface-hub, pydantic, Pillow, click, einops, sentencepiece, loguru. Optional: gguf (for GGUF quantized models). Build system: hatchling.

## Testing Status

- FLUX.1 Dev: Working (verified with BFL repo models and GGUF Q8_0, seed 42)
- Z-Image Turbo: Working (verified with Tongyi-MAI/Z-Image-Turbo + Qwen3-4B, seed 42)
- FLUX.2 Dev: Untested
- Qwen-Image: Untested
- LoRA: Untested
