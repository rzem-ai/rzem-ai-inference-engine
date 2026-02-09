# Architecture

## Design Goals

1. **Unified API** — Hide model family differences behind a single `InferenceEngine.submit(JobParams)` interface
2. **Automatic VRAM management** — Models stay loaded when they fit; smallest-first eviction when they don't
3. **Event-driven** — Callers get progress updates and results via callbacks, not polling
4. **Thread-safe** — Background processing thread handles jobs sequentially; callers interact from any thread

## Package Layout

```
src/inference_engine/
├── __init__.py              # Public exports
├── __main__.py              # python -m inference_engine
├── engine.py                # InferenceEngine (main entry point)
├── types.py                 # All shared types
├── cli.py                   # Click CLI (generate + serve commands)
├── api/
│   ├── __init__.py          # Exports create_app
│   ├── app.py               # FastAPI factory, lifespan, WebSocket endpoint
│   ├── routes.py            # HTTP routes (jobs, models, health)
│   ├── models.py            # Pydantic response models
│   ├── state.py             # JobStateStore (event → WebSocket bridge)
│   └── ws.py                # ConnectionManager (WebSocket broadcast)
├── pipeline/
│   ├── base.py              # BasePipeline ABC
│   ├── flux1.py             # FLUX.1 Dev
│   ├── flux2.py             # FLUX.2 Dev
│   ├── z_image.py           # Z-Image
│   └── qwen_image.py        # Qwen-Image
├── models/
│   ├── cache.py             # ModelCache (two-tier VRAM/RAM)
│   ├── loader.py            # ModelLoader (path resolution)
│   └── memory.py            # VRAM estimation, device detection
├── lora/
│   └── applicator.py        # LoRA loading, format detection, patching
└── queue/
    ├── manager.py            # JobQueue (FIFO with state tracking)
    └── processor.py          # JobProcessor (background loop)
```

## Component Relationships

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             REST API (FastAPI)                             │
│                                                                           │
│  POST /jobs  GET /jobs  DELETE /jobs/{id}  GET /jobs/{id}/image           │
│  GET /models  GET /models/all  GET /health  WS /ws                       │
│                                                                           │
│  ┌─────────────────┐  ┌────────────────────────────────────┐              │
│  │ JobStateStore    │  │ ConnectionManager (WebSocket)       │              │
│  │ (event→state)   │──│ broadcasts JSON to all clients      │              │
│  └────────┬────────┘  └────────────────────────────────────┘              │
│           │ loop.call_soon_threadsafe                                      │
├───────────▼───────────────────────────────────────────────────────────────┤
│                        InferenceEngine                                     │
│                                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐           │
│  │ JobQueue  │──│ JobProcessor │──│ Pipeline (flux1/z_image/…) │           │
│  │          │  │  (bg thread) │  │                            │           │
│  └──────────┘  └──────────────┘  └─────────────┬──────────────┘           │
│                                                │                          │
│                                   ┌────────────▼──────────────┐           │
│                                   │       ModelCache          │           │
│                                   │  ┌──────┐  ┌───────────┐ │           │
│                                   │  │ VRAM │◄─│    RAM     │ │           │
│                                   │  │ (hot)│  │  (warm)    │ │           │
│                                   │  └──────┘  └───────────┘ │           │
│                                   └───────────────────────────┘           │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │  Event System: on()/off()/_emit() with thread-safe cbs  │             │
│  └──────────────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Job Lifecycle

```
submit(JobParams) ──► JobQueue.add() ──► JOB_QUEUED event
                          │
                          ▼
                   JobProcessor.run()
                          │
                  ┌───────▼────────┐
                  │ Pipeline.run() │
                  │                │
                  │ 1. Validate    │──► ValueError if missing fields
                  │ 2. Text encode │──► Load encoders via cache
                  │ 3. Denoise     │──► Load transformer, emit JOB_PROGRESS
                  │ 4. VAE decode  │──► Load VAE
                  │ 5. Return img  │
                  └───────┬────────┘
                          │
                  JOB_COMPLETED ──► CompletedEvent(image, seed)
                  or JOB_FAILED ──► FailedEvent(error, traceback)
```

### VRAM Management Flow

```
Pipeline needs a model
        │
        ▼
ModelCache.get_or_load(key, loader)
        │
   ┌────▼─────────────────┐
   │ In VRAM? ──► Return  │
   │ In RAM?  ──► Move to │──► ensure_vram() ──► evict smallest unlocked
   │              VRAM     │
   │ Neither? ──► Load    │──► ensure_vram() ──► model.to(device)
   └──────────────────────┘
```

Eviction strategy:
1. Sort unlocked VRAM entries by size (ascending)
2. Move smallest to RAM via `.to("cpu")` + `torch.cuda.empty_cache()` (CUDA only; no-op on MPS/CPU)
3. Repeat until enough space is available (including 1 GB working buffer)

## Pipeline Implementations

### BasePipeline ABC

Every pipeline implements three methods:

```python
class BasePipeline(ABC):
    def validate_params(self, params: JobParams) -> None: ...
    def get_required_models(self, params: JobParams) -> list[ModelSpec]: ...
    def run(self, params, cache, progress_cb) -> tuple[PIL.Image.Image, int]: ...
```

### FLUX.1 Dev (`flux1.py`)

**Architecture**: Dual-stream MMDiT (19 double blocks + 38 single blocks)

**Text encoding**: CLIP (768-dim pooled) + T5-XXL (4096-dim sequence, 512 tokens)

**Denoising**: Rectified flow with guidance-distilled Euler sampling. The `guidance` embedding replaces traditional CFG — a scalar `cfg_scale` value is embedded and passed to the transformer, eliminating the need for two forward passes.

**Sequential offloading**: Text encoders (~9 GB total) and transformer (~22 GB) don't fit simultaneously in 32 GB VRAM. The pipeline runs in three isolated stages:

```
Stage 1: Load CLIP + T5 → encode text → embeddings stay as GPU tensors → encoders can be evicted
Stage 2: Load transformer → lock → denoise (20 steps) → unlock
Stage 3: Load VAE → lock → decode → unlock
```

**Sigma schedule**: Linear sigmas [1.0 → 0.0] with exponential time shift. Mu computed via linear interpolation based on image patch count:
```
mu = m * seq_len + b
m = (1.15 - 0.5) / (4096 - 256)
b = 0.5 - m * 256
```

**Timestep convention**: Sigmas passed directly to the transformer in [0, 1] range.

**GGUF support**: The `_load_transformer` function detects `.gguf` file extension and passes `GGUFQuantizationConfig(compute_dtype=dtype)` to `from_single_file`. Quantized weights stay as uint8 and are dequantized on-the-fly during each forward pass. Example: Q8_0 reduces the transformer from 22 GB to 12 GB with no visible quality loss.

### Z-Image (`z_image.py`)

**Architecture**: S3-DiT single-stream transformer (6B params)

**Text encoding**: Qwen3-4B with `hidden_states[-2]`, attention mask filtering (non-padding tokens only), returns 2D tensor `[seq_len, hidden_dim]`.

**Denoising**: V-prediction with negation (`noise_pred = -model_output`). Transformer API uses `(x=list_of_tensors, t=timestep, cap_feats=list_of_embeds)` with an added frame dimension.

**Timestep convention**: `model_t = 1 - sigma` (0 = noise, 1 = clean).

**Turbo defaults**: 9 steps, cfg_scale=1.0 (no CFG needed).

### FLUX.2 Dev (`flux2.py`)

**Architecture**: Similar to FLUX.1 but with 32-channel VAE

**Text encoding**: Qwen3 with multi-layer extraction — stacks hidden states from layers (9, 18, 27) into `(batch, seq_len, hidden_dim * 3)`. Mean-pooled last hidden state for global conditioning.

**VAE**: 32 latent channels with batch normalization on the packed representation. BN statistics are extracted and applied/inverted around the denoising loop.

**Position IDs**: 4D `(T, H, W, L)` with int64 dtype (float causes NaN in rotary embeddings).

**Schedule**: Empirical mu computation with piecewise linear interpolation based on image sequence length and step count.

### Qwen-Image (`qwen_image.py`)

**Architecture**: 20B MMDiT with FLUX-style transformer API

**Text encoding**: Same as Z-Image (Qwen3 `hidden_states[-2]`, mask-filtered).

**Denoising**: True classifier-free guidance with negative prompts — two forward passes per step. `noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)`.

**Defaults**: 50 steps, cfg_scale=4.0. Supports specific resolutions (1328x1328, 1664x928, etc.).

## Model Cache (`cache.py`)

### Two-Tier Design

```
┌─────────────────────────┐     ┌─────────────────────────┐
│         VRAM             │     │          RAM             │
│   (GPU, fast access)     │     │   (CPU, warm standby)    │
│                          │     │                          │
│  transformer  22.2 GB 🔒 │────►│  t5_encoder    8.9 GB   │
│  vae           0.2 GB 🔒 │     │  clip_encoder  0.2 GB   │
│                          │     │  tokenizers    ~0 GB     │
└─────────────────────────┘     └─────────────────────────┘
         🔒 = locked (in use, cannot be evicted)
```

### Cache Entry

```python
@dataclass
class CacheEntry:
    key: str           # "path::role" composite key
    model: Any         # torch.nn.Module or tokenizer
    size_bytes: int    # Estimated from parameters + buffers
    device: str        # "cuda" or "cpu"
    last_used: float   # timestamp for LRU
    _locks: int        # Re-entrant lock count (>0 = cannot evict)
```

### Thread Safety

- The `_lock` mutex protects all dict mutations (add, remove, move between tiers)
- Model loading (I/O-bound) happens **outside** the lock
- After loading, a double-check inside the lock prevents duplicate entries
- Event emission happens outside the lock to avoid deadlocks with listener callbacks

## LoRA System (`applicator.py`)

### Format Detection

Checks key name patterns in the state dict to identify the source:

| Format | Key Pattern | Example |
|---|---|---|
| Kohya | `lora_unet_*` | `lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight` |
| Diffusers/PEFT | `*.lora_A.weight` | `double_blocks.0.img_attn.qkv.lora_A.weight` |
| XLabs | `*processor*double_blocks*` | `double_blocks.0.processor.qkv_lora.lora_A` |
| AIToolkit | `transformer.*.lora_*` | `transformer.double_blocks.0.img_attn.qkv.lora_A.weight` |
| OneTrainer | `bundle_emb*` or `lora_te*` | `lora_te_text_projection.lora_down.weight` |

### Weight Patching

```
weight += strength * (alpha / rank) * (up @ down)
```

- `up` and `down` are the low-rank matrices
- `alpha / rank` is the LoRA scaling factor
- `strength` is the user-specified blending weight
- Applied in-place to model parameters

For model reuse across jobs with different LoRAs, the applicator supports snapshot/restore:
1. `snapshot_weights(model)` — clone all parameters before patching
2. `unapply(model, snapshot)` — restore original weights after the job

## Model Path Resolution (`loader.py`)

```
Input                                    Resolution
─────────────────────────────────────    ──────────────────────────────────
./models/flux.safetensors                Local file → return Path
black-forest-labs/FLUX.1-dev             HF repo → snapshot_download()
city96/FLUX.1-dev-gguf/flux1-Q8.gguf    HF repo+file → hf_hub_download()
```

For HF repos, `snapshot_download` filters to relevant file types: `*.safetensors`, `*.json`, `*.txt`, `*.model`, `*.tiktoken`, `*.py`.

Pipelines use `_resolve_sub(path, subfolder)` to handle multi-component repos where models live in subdirectories (`transformer/`, `vae/`, `text_encoder/`, `tokenizer/`, `text_encoder_2/`, `tokenizer_2/`).

## Event System

The engine uses a simple observer pattern:

```python
# Registration (thread-safe via _listener_lock)
engine.on(EventType.JOB_PROGRESS, callback)

# Emission (from processor thread)
# 1. Copy listener list under lock
# 2. Call each listener outside lock (prevents deadlocks)
# 3. Catch and log exceptions from listeners (never crash the processor)
```

Events are emitted synchronously on the processor thread. Listeners that do heavy work should offload to their own threads.

## REST API (`api/`)

### Application Factory

`create_app(device, vram_limit_gb, output_dir)` returns a FastAPI app. The lifespan context manager creates the `InferenceEngine`, `ConnectionManager`, and `JobStateStore`, attaching them to `app.state`.

### Routes (`routes.py`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs` | Submit a generation job (body: `JobParams`) |
| `GET` | `/jobs` | List all tracked jobs |
| `GET` | `/jobs/{id}` | Get job details |
| `DELETE` | `/jobs/{id}` | Cancel a queued job |
| `GET` | `/jobs/{id}/image` | Download the generated PNG |
| `GET` | `/models` | List locally cached HF models (summary) |
| `GET` | `/models/all` | List cached models with full file tree |
| `GET` | `/health` | Health check with device and queue stats |

### WebSocket (`/ws`)

All job lifecycle events are broadcast as JSON to connected WebSocket clients. The `ConnectionManager` tracks active connections and handles dead socket cleanup.

### Threading Bridge (`state.py`)

Engine callbacks fire on the processor thread. `JobStateStore` bridges to the async event loop via `loop.call_soon_threadsafe(asyncio.ensure_future, ws.broadcast(msg))`. This keeps the processor thread non-blocking while delivering real-time updates over WebSocket.

### CLI

`inference-engine serve --host --port --device --vram-limit --output-dir` starts a uvicorn server. The `scripts/server.sh` helper supports start/stop/restart/status with PID file management.

## Queue and Processing

`JobQueue` is a thread-safe FIFO backed by `collections.deque`. It uses `threading.Event` for efficient blocking (`get_next` blocks until a job is available or timeout expires).

`JobProcessor` runs a simple loop in a daemon thread:
1. `queue.get_next(timeout=0.5)` — blocks briefly, returns `None` on timeout
2. Look up the pipeline for `params.transformer_type`
3. `pipeline.validate_params()` — fail fast on missing fields
4. Emit `JOB_STARTED`
5. `pipeline.run()` — the actual generation
6. Emit `JOB_COMPLETED` or `JOB_FAILED`

Jobs are processed sequentially. The queue supports cancellation of queued (not running) jobs.
