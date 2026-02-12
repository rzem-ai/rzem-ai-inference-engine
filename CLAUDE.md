# CLAUDE.md — RZEM AI Inference Engine

## Project Overview

Text-to-image inference engine with job queue, event system, and automatic VRAM management. Supports FLUX.1 Dev, FLUX.2 Dev, Z-Image, and Qwen-Image model families. Python package installed via `pip install -e .`.

## Quick Reference

```bash
# Install
pip install -e .

# CLI — generate
rzem-ai-inference-engine generate --help
rzem-ai-inference-engine generate --prompt "..." --transformer-model <path> --transformer-type flux1_dev --vae-model <path> ...

# CLI — REST API server
rzem-ai-inference-engine serve --host 127.0.0.1 --port 8000 --device auto --output-dir ./output
bash server.sh start               # Background with PID management

# Test scripts
bash scripts/test_flux1.sh        # FLUX.1 Dev (all from BFL repo)
bash scripts/test_flux1_alt.sh    # FLUX.1 Dev (separate repos)
bash scripts/test_flux1_gguf.sh   # FLUX.1 Dev (GGUF Q8_0 transformer)
bash scripts/test_zimage.sh       # Z-Image Turbo
bash scripts/test_flux1_lora.sh   # FLUX.1 Dev + LoRA (bf16)
bash scripts/test_flux1_gguf_lora.sh  # FLUX.1 Dev + LoRA (GGUF Q8_0)
```

## Architecture

- `src/rzem_ai_inference_engine/engine.py` — Main `InferenceEngine` class (public API, thread-safe)
- `src/rzem_ai_inference_engine/types.py` — All types: `JobParams`, `TransformerType`, events, enums
- `src/rzem_ai_inference_engine/pipeline/` — Pipeline implementations (one per model family)
  - `base.py` — `BasePipeline` ABC: `validate_params`, `get_required_models`, `run`
  - `flux1.py` — FLUX.1 Dev: CLIP + T5 encoding, rectified flow Euler, guidance-distilled
  - `flux2.py` — FLUX.2 Dev: Qwen3 multi-layer encoding, 32-ch VAE, BN normalization
  - `z_image.py` — Z-Image: S3-DiT single-stream, Qwen3-4B, v-prediction with negation
  - `qwen_image.py` — Qwen-Image: 20B MMDiT, true CFG, Qwen3 encoding
- `src/rzem_ai_inference_engine/models/cache.py` — Two-tier VRAM/RAM cache, smallest-first eviction
- `src/rzem_ai_inference_engine/models/loader.py` — Path resolution (local, HF repo, HF repo+file)
- `src/rzem_ai_inference_engine/models/memory.py` — VRAM estimation, device detection, `preferred_dtype()`
  - `lora_applicator.py` — LoRA format detection and forward-hook application
- `src/rzem_ai_inference_engine/queue/` — Job queue (`manager.py`) and processor (`processor.py`)
- `src/rzem_ai_inference_engine/api/` — REST API (FastAPI)
  - `app.py` — App factory with lifespan, WebSocket endpoint
  - `routes.py` — HTTP routes (jobs CRUD, models listing, health)
  - `models.py` — Pydantic response models
  - `state.py` — `JobStateStore` (engine event → async WebSocket bridge)
  - `ws.py` — `ConnectionManager` (WebSocket broadcast)
- `src/rzem_ai_inference_engine/cli.py` — Click CLI (`generate` + `serve` commands)

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

Format detection (Kohya, Diffusers, XLabs, AIToolkit, OneTrainer) based on key patterns. Forward-hook approach: `output += F.linear(F.linear(input, down), up) * (strength * alpha/rank)`. No weight modification, works with quantized (GGUF) models. Hooks are removed after inference. Kohya/XLabs keys use BFL naming and require conversion via `_flux1_bfl_to_diffusers()` (fused QKV/linear1 splitting).

### Dtype selection

All pipelines and the LoRA applicator resolve dtype via `preferred_dtype(device)` in `models/memory.py`. Returns bfloat16 for CUDA and MPS (M3+, PyTorch 2.3+), float32 for CPU. Never hardcode `torch.bfloat16` directly — use the helper so dtype logic stays centralized.

### Event system

Thread-safe callback registration on `InferenceEngine`. Events: JOB_QUEUED, JOB_STARTED, JOB_PROGRESS, JOB_COMPLETED, JOB_FAILED, JOB_CANCELLED, MODEL_LOADING, MODEL_LOADED, MODEL_UNLOADED.

### REST API

FastAPI app factory in `api/app.py` → `create_app(device, vram_limit_gb, output_dir)`. WebSocket at `/ws` broadcasts all job events as JSON. Threading bridge: engine callbacks fire on processor thread, `JobStateStore` uses `loop.call_soon_threadsafe` to schedule async broadcasts. Images saved as `{job_id}.png` in output_dir, served via `GET /jobs/{id}/image`. `GET /models` and `GET /models/all` use `huggingface_hub.scan_cache_dir()` to list locally cached models.

## Common Pitfalls

- **Timestep scaling**: Never pass `sigma * 1000` to FluxTransformer2DModel. It expects [0, 1].
- **VRAM eviction order**: Always call `ensure_vram()` BEFORE `model.to(device)`, not after.
- **Cache thread safety**: Loading happens outside the lock (slow I/O), but VRAM placement happens inside the lock with double-check.
- **torch.cuda.empty_cache()**: Must be called after moving models to CPU during eviction (guarded by `device.type == "cuda"`).
- **Z-Image text encoder**: Uses `Qwen3Model` (not `AutoModelForCausalLM`), `hidden_states[-2]`, mask-filtered to 2D.
- **Z-Image v-prediction**: `noise_pred = -model_output` (negate the output).
- **Z-Image `_time_shift` boundary**: Python floats raise on `1/0`. The function has explicit boundary checks for `t <= 0` (returns 0) and `t >= 1` (returns 1). The FLUX.1 version uses torch tensors which handle inf gracefully.
- **MPS/Apple Silicon**: Device detection, cache eviction, and generator all handle MPS. bfloat16 requires PyTorch 2.3+ (native M3+ GPU ALU support). No `empty_cache()` equivalent for MPS. Use `--vram-limit` to constrain unified memory usage. GGUF quantization is slower on MPS than full-precision bfloat16 due to unoptimized dequantization kernels.

## Dependencies

torch, diffusers (>=0.32), transformers, accelerate, safetensors, huggingface-hub, pydantic, Pillow, click, einops, sentencepiece, loguru, fastapi (>=0.110), uvicorn[standard] (>=0.27), python-multipart. Optional: gguf (for GGUF quantized models). Build system: hatchling.

## Testing Status

- FLUX.1 Dev: Working (verified with BFL repo models and GGUF Q8_0, seed 42)
- Z-Image Turbo: Working (verified with Tongyi-MAI/Z-Image-Turbo + Qwen3-4B, seed 42)
- FLUX.2 Dev: Untested
- Qwen-Image: Untested
- LoRA: Working (verified Kohya format with FLUX.1 Dev bf16 + GGUF Q8_0, Moebius style LoRA)
