# Progress — RZEM AI Inference Engine

Last updated: 2026-02-09

## What's Done

### Infrastructure (all complete)
- Package scaffold: `pyproject.toml` (hatchling), `src/rzem_ai_inference_engine/`, `pip install -e .`
- Type system: `types.py` — `JobParams`, `TransformerType`, `LoraParams`, all event dataclasses
- Model cache: `models/cache.py` — two-tier VRAM/RAM, smallest-first eviction, lock/unlock, `torch.cuda.empty_cache()` after eviction
- Model loader: `models/loader.py` — local path, `org/repo` (snapshot_download), `org/repo/file.ext` (hf_hub_download)
- Memory utils: `models/memory.py` — `estimate_model_size()`, `get_free_vram()`, `detect_device()` (cuda > mps > cpu)
- LoRA system: `lora/applicator.py` — format detection (Kohya, Diffusers, XLabs, AIToolkit, OneTrainer), weight patching, snapshot/restore
- Job queue: `queue/manager.py` — thread-safe FIFO, cancel support
- Job processor: `queue/processor.py` — background thread, sequential execution
- Engine: `engine.py` — public API, event system (on/off/emit), submit/cancel/shutdown
- CLI: `cli.py` — Click-based `rzem-ai-inference-engine generate` and `rzem-ai-inference-engine serve`
- REST API: `api/` — FastAPI with WebSocket event broadcasting
  - `POST /jobs` — submit generation jobs
  - `GET /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}` — job CRUD
  - `GET /jobs/{id}/image` — download generated PNG
  - `GET /models` — list locally cached HF models (summary)
  - `GET /models/all` — list cached models with full revision/file tree
  - `GET /health` — health check with device and queue stats
  - `WS /ws` — real-time job event broadcasting
  - `JobStateStore` bridges engine thread callbacks to async WebSocket via `call_soon_threadsafe`
  - `scripts/server.sh` — start/stop/restart/status with PID management
- Docs: `CLAUDE.md`, `README.md`, `ARCHITECTURE.md`

### Pipelines

| Pipeline | File | Status | Notes |
|---|---|---|---|
| FLUX.1 Dev | `pipeline/flux1.py` | **Working** | Sequential offloading (text encode → denoise → VAE decode). Tested with BFL repo and GGUF Q8_0. |
| Z-Image Turbo | `pipeline/z_image.py` | **Working** | S3-DiT + Qwen3-4B. 9 steps, ~5s denoising. Tested with Tongyi-MAI/Z-Image-Turbo. |
| FLUX.2 Dev | `pipeline/flux2.py` | Written, **untested** | Qwen3 multi-layer encoding, 32-ch VAE, BN normalization, 4D position IDs. |
| Qwen-Image | `pipeline/qwen_image.py` | Written, **untested** | 20B MMDiT, true CFG with negative prompts. Uses Z-Image text encoding helpers. |

### GGUF Support
- `flux1.py` detects `.gguf` extension and passes `GGUFQuantizationConfig(compute_dtype=dtype)` to `from_single_file`
- `loader.py` allows `*.gguf` and `*.bin` in `snapshot_download` patterns
- Tested with `city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf` — works, visually identical to bf16

### Bugs Fixed During Development
1. **CUDA OOM**: `model.to(device)` was called before `ensure_vram()` — reordered in cache
2. **Device mismatch**: Text encoders evicted to CPU while pipeline expected them on GPU — restructured FLUX.1 into sequential stages
3. **Missing guidance param**: FLUX.1 Dev is guidance-distilled, needs `guidance` tensor in transformer call
4. **Timestep scaling**: Transformer expects [0, 1] range, not [0, 1000] — removed `* 1000.0`
5. **Mu calculation**: Changed from log-base-2 formula to linear interpolation matching diffusers
6. **Division by zero**: Z-Image `_time_shift()` used Python floats which raise on `1/0` — added boundary checks

## What's Next

### Immediate: Test remaining pipelines
1. **FLUX.2 Dev** — needs `Flux2Transformer2DModel` (check if available in diffusers 0.36), Qwen3 encoder, 32-ch VAE
2. **Qwen-Image** — needs `QwenImageTransformer2DModel` (check diffusers availability), 20B model (~40GB, may need offloading or quantization)
3. **LoRA** — test with a real LoRA file on FLUX.1 Dev

### Known Issues to Watch
- **FLUX.2**: `flux2.py` still has `timestep * 1000.0` on line 235 — needs the same fix as FLUX.1 (remove the multiplication, or verify what Flux2Transformer2DModel expects)
- **Qwen-Image**: `qwen_image.py` uses `AutoModelForCausalLM` for text encoder but Z-Image found that `Qwen3Model` is correct — may need the same fix
- **FLUX.2 BN normalization**: The `_get_bn_stats()` approach is speculative, needs real model testing
- **LoRA Kohya key conversion**: `_kohya_key_to_model_key()` uses heuristic string replacements, may fail on some LoRA files

## Apple Silicon (M3) Notes

### Environment Setup
```bash
# Install PyTorch for MPS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or use conda:
# conda install pytorch torchvision -c pytorch-nightly

pip install -e .
```

### What Should Work Out of the Box
- Device detection: `detect_device()` in `memory.py` already handles MPS (`torch.backends.mps.is_available()`)
- Model loading: All HuggingFace downloads are platform-independent
- Pipeline math: All tensor operations use standard PyTorch (no CUDA kernels)

### Potential MPS Issues
1. **bfloat16**: All pipelines hardcode `dtype = torch.bfloat16`. MPS added bfloat16 support in PyTorch 2.3+, but some ops may fall back to float32. If you see errors, try changing to `torch.float16` or `torch.float32`.
2. **`torch.cuda.empty_cache()`**: Called in `cache.py:_evict_to_ram()` — guarded by `if self._device.type == "cuda"` so it's safe, but MPS has no equivalent cache clearing.
3. **Memory reporting**: `get_free_vram()` returns a large sentinel (2^40) for MPS since Apple doesn't expose free memory. The `--vram-limit` CLI flag is useful here — set it to something like `--vram-limit 18` for a 36GB M3 Max.
4. **GGUF quantization**: `GGUFQuantizationConfig` may not support MPS. Dequantization happens on forward pass and may require CUDA. Test with full-precision models first.
5. **flash_attn**: Not available on MPS. Ensure it's not installed (`pip uninstall flash-attn`). Diffusers/transformers will fall back to standard attention automatically.
6. **Unified memory**: MPS uses unified memory so "evicting to RAM" doesn't actually free GPU memory. The two-tier cache still works but the eviction is less meaningful. Models just move between MPS and CPU address spaces.
7. **Generator device**: Noise is generated on CPU (`device="cpu"`) then moved to device — this is already MPS-compatible. MPS generators have quirks so CPU generation is correct.

### Models That Fit on M3 (36GB unified)
- **Z-Image Turbo**: ~19 GB total (Qwen3-4B 7.5GB + transformer 11.5GB + VAE 0.2GB) — fits easily
- **FLUX.1 Dev**: ~32 GB total but sequential offloading keeps peak at ~22GB — should fit with `--vram-limit 24`
- **FLUX.1 GGUF Q4**: ~6 GB transformer + 9 GB text encoders — fits comfortably
- **Qwen-Image**: 20B transformer (~40 GB) — won't fit even with offloading, needs GGUF quantization

### Quick Test on M3
```bash
# Z-Image (smallest, fastest, most likely to work)
bash scripts/test_zimage.sh

# FLUX.1 with VRAM limit
VRAM_LIMIT=24 rzem-ai-inference-engine generate \
    --prompt "a cat on a windowsill" \
    --transformer-model black-forest-labs/FLUX.1-dev \
    --transformer-type flux1_dev \
    --clip-tokenizer black-forest-labs/FLUX.1-dev \
    --clip-encoder black-forest-labs/FLUX.1-dev \
    --t5-tokenizer black-forest-labs/FLUX.1-dev \
    --t5-encoder black-forest-labs/FLUX.1-dev \
    --vae-model black-forest-labs/FLUX.1-dev \
    --vram-limit 24 \
    --steps 20 --cfg-scale 3.5 \
    --width 1024 --height 1024 \
    --seed 42 --output output.png
```

## Test Scripts
```
scripts/test_flux1.sh          # FLUX.1 Dev — all from BFL repo
scripts/test_flux1_alt.sh      # FLUX.1 Dev — separate repos (may fail: google/t5-v1_1-xxl has .bin only)
scripts/test_flux1_gguf.sh     # FLUX.1 Dev — GGUF Q8_0 transformer
scripts/test_zimage.sh         # Z-Image Turbo
scripts/test_flux2.sh          # FLUX.2 Dev (untested)
scripts/test_qwen_image.sh     # Qwen-Image (untested)
scripts/test_with_lora.sh      # LoRA application (untested)
scripts/test_all.sh            # Run all tests
scripts/server.sh              # REST API server management (start/stop/restart/status)
```

## Installed Versions (Linux, NVIDIA RTX 5090)
```
Python 3.13
torch 2.10.0+cu128
torchvision 0.25.0+cu128
diffusers 0.36.0
transformers (latest)
gguf 0.17.1
```
