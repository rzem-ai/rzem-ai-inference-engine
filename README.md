# RZEM AI Inference Engine

A Python text-to-image inference engine with job queue, event callbacks, and automatic VRAM management. Supports multiple model families with a unified API that hides architectural differences while exposing full parameter control.

## Supported Models

| Model Family | Transformer | Text Encoder | VAE | Default Steps |
|---|---|---|---|---|
| **FLUX.1 Dev** | `FluxTransformer2DModel` | CLIP + T5-XXL | 16-ch AutoencoderKL | 20 |
| **FLUX.2 Dev** | `Flux2Transformer2DModel` | Qwen3 (multi-layer) | 32-ch AutoencoderKL + BN | 20 |
| **Z-Image** | `ZImageTransformer2DModel` (S3-DiT) | Qwen3-4B | 16-ch AutoencoderKL | 9 |
| **Qwen-Image** | `QwenImageTransformer2DModel` (20B MMDiT) | Qwen3 | 16-ch AutoencoderKL | 50 |

All models support LoRA weight patching with automatic format detection (Kohya, Diffusers/PEFT, XLabs, AIToolkit, OneTrainer).

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.0+. GPU with 24+ GB VRAM recommended. Apple Silicon M3+ with PyTorch 2.3+ is supported (MPS backend).

## Distribution

### Standalone Executable

Build a standalone executable that bundles all dependencies (models are still user-provided):

```bash
# Build server variant (generate + serve commands, ~3-4 GB)
bash scripts/build_executable.sh server

# Build CLI variant (generate only, ~2.5-3.5 GB)
bash scripts/build_executable.sh cli

# Build both variants (default)
bash scripts/build_executable.sh all

# Run the executables
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server generate --help
./dist/rzem-ai-inference-engine-server/rzem-ai-inference-engine-server serve --help
./dist/rzem-ai-inference-engine-cli/rzem-ai-inference-engine-cli generate --help
```

See [packaging/README.md](packaging/README.md) for detailed build instructions, platform requirements, and distribution options.

**Note:** Executables do not include model weights. Models must be downloaded separately via HuggingFace Hub or provided as local paths.

## Usage

### Python API

```python
from rzem_ai_inference_engine import InferenceEngine, JobParams, EventType, TransformerType

engine = InferenceEngine()

# Listen for events
engine.on(EventType.JOB_PROGRESS, lambda e: print(f"Step {e.step}/{e.total_steps}"))
engine.on(EventType.JOB_COMPLETED, lambda e: e.image.save("output.png"))

# Submit a job
job_id = engine.submit(JobParams(
    prompt="a cat sitting on a windowsill, golden hour lighting",
    transformer_model="black-forest-labs/FLUX.1-dev",
    transformer_type=TransformerType.FLUX1_DEV,
    clip_tokenizer="black-forest-labs/FLUX.1-dev",
    clip_encoder="black-forest-labs/FLUX.1-dev",
    t5_tokenizer="black-forest-labs/FLUX.1-dev",
    t5_encoder="black-forest-labs/FLUX.1-dev",
    vae_model="black-forest-labs/FLUX.1-dev",
    steps=20,
    cfg_scale=3.5,
    width=1024,
    height=1024,
    seed=42,
))

# ... engine processes the job in a background thread ...
# Call engine.shutdown() when done
```

### CLI

```bash
# FLUX.1 Dev — all models from Black Forest Labs
rzem-ai-inference-engine generate \
    --prompt "a cat sitting on a windowsill, golden hour lighting" \
    --transformer-model black-forest-labs/FLUX.1-dev \
    --transformer-type flux1_dev \
    --clip-tokenizer black-forest-labs/FLUX.1-dev \
    --clip-encoder black-forest-labs/FLUX.1-dev \
    --t5-tokenizer black-forest-labs/FLUX.1-dev \
    --t5-encoder black-forest-labs/FLUX.1-dev \
    --vae-model black-forest-labs/FLUX.1-dev \
    --steps 20 --cfg-scale 3.5 \
    --width 1024 --height 1024 \
    --seed 42 \
    --output output.png

# Z-Image Turbo
rzem-ai-inference-engine generate \
    --prompt "mountain landscape at sunset" \
    --transformer-model Tongyi-MAI/Z-Image-Turbo \
    --transformer-type z_image \
    --qwen3-tokenizer Qwen/Qwen3-4B \
    --qwen3-encoder Qwen/Qwen3-4B \
    --vae-model black-forest-labs/FLUX.1-dev \
    --steps 9 --cfg-scale 1.0 \
    --output output.png

# With LoRAs (format: path:strength)
rzem-ai-inference-engine generate \
    ... \
    --lora ./models/anime.safetensors:0.8 \
    --lora ./models/detail.safetensors:0.5 \
    --output output.png
```

### Model Paths

Model path arguments accept three formats:

| Format | Example | Behavior |
|---|---|---|
| Local path | `./models/flux.safetensors` | Used directly |
| HuggingFace repo | `black-forest-labs/FLUX.1-dev` | Downloads full repo, auto-resolves subfolders |
| HuggingFace repo + file | `city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf` | Downloads only the specified file |

Multi-component HF repos (like `black-forest-labs/FLUX.1-dev`) are handled automatically — the engine resolves subfolders like `transformer/`, `text_encoder/`, `vae/`, etc.

## REST API Server

Start the server to accept jobs over HTTP and receive real-time updates via WebSocket:

```bash
# Listen on all interfaces (enables LAN discovery)
rzem-ai-inference-engine serve --host 0.0.0.0 --port 8000 --device auto --output-dir ./output

# Or run as a background daemon
bash scripts/server.sh start --port 8000 --device cuda
bash scripts/server.sh status
bash scripts/server.sh stop
```

| Endpoint | Description |
|---|---|
| `POST /jobs` | Submit a generation job |
| `GET /jobs` | List all jobs |
| `GET /jobs/{id}` | Get job details |
| `GET /jobs/{id}/image` | Download generated image |
| `DELETE /jobs/{id}` | Cancel a job |
| `GET /models` | List locally cached HuggingFace models |
| `GET /health` | Health check with queue stats |
| `WS /ws` | WebSocket — real-time job event broadcasts (JSON) |

### Network Announcement (LAN Discovery)

When the server binds to a non-localhost address (e.g. `--host 0.0.0.0`), it automatically announces itself on the local network via **mDNS/DNS-SD** — the same protocol used by AirPlay, Chromecast, and network printers. Client applications can discover running servers without manual configuration.

- **Service type**: `_rzem-ai._tcp.local.`
- **TXT record**: `version`, `device`, `api=rest`, `ws=/ws`
- **Disable**: `--no-announce`

When bound to `127.0.0.1` (the default), announcement is skipped since the server is only reachable locally.

**Client-side discovery example:**

```python
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

class Listener(ServiceListener):
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addr = info.parsed_addresses()[0]
            port = info.port
            print(f"Found server at {addr}:{port}")

zc = Zeroconf()
browser = ServiceBrowser(zc, "_rzem-ai._tcp.local.", Listener())
```

## VRAM Management

The engine manages a two-tier model cache:

- **VRAM (hot)**: Models actively on the GPU for fast inference
- **RAM (warm)**: Models offloaded to CPU, moved back to GPU on demand

When VRAM is insufficient for the next model, the cache evicts the **smallest unlocked** model first. A 1 GB working memory buffer is reserved for intermediate tensors.

For FLUX.1 Dev on GPUs with ~32 GB VRAM, the pipeline automatically sequences model loading: text encoders are used then released before the 22 GB transformer is loaded.

### Dtype Selection

All pipelines resolve dtype automatically via `preferred_dtype(device)`:

| Device | Dtype | Notes |
|---|---|---|
| CUDA (Ampere+) | bfloat16 | Native tensor-core support |
| MPS (M3+) | bfloat16 | Native GPU ALU support (PyTorch 2.3+) |
| CPU | float32 | No reduced-precision benefit |

### Apple Silicon Notes

- On MPS with unified memory, cache eviction is effectively a no-op — all models share the same physical RAM. Use `--vram-limit` to constrain memory if needed.
- GGUF quantized models are slower on MPS than full-precision bfloat16 due to unoptimized dequantization kernels. Use full-precision models for best performance.

## Events

| Event | Payload | When |
|---|---|---|
| `JOB_QUEUED` | `QueuedEvent` | Job added to queue |
| `JOB_STARTED` | `StartedEvent` | Processing begins |
| `JOB_PROGRESS` | `ProgressEvent` | Each denoising step (includes step/total_steps) |
| `JOB_COMPLETED` | `CompletedEvent` | Success (includes PIL Image and seed) |
| `JOB_FAILED` | `FailedEvent` | Error (includes message and traceback) |
| `JOB_CANCELLED` | `CancelledEvent` | Job cancelled before processing |
| `MODEL_LOADING` | `ModelLoadingEvent` | Model load starting |
| `MODEL_LOADED` | `ModelLoadedEvent` | Model loaded (includes size and device) |
| `MODEL_UNLOADED` | `ModelUnloadedEvent` | Model evicted from VRAM |

## Test Scripts

```bash
bash scripts/test_flux1.sh            # FLUX.1 Dev (BFL repo)
bash scripts/test_flux1_alt.sh        # FLUX.1 Dev (separate repos)
bash scripts/test_flux1_gguf.sh       # FLUX.1 Dev (GGUF Q8_0 transformer)
bash scripts/test_zimage.sh           # Z-Image Turbo
bash scripts/test_flux1_lora.sh       # FLUX.1 Dev + LoRA (bf16)
bash scripts/test_flux1_gguf_lora.sh  # FLUX.1 Dev + LoRA (GGUF Q8_0)
```

## Dependencies

- **PyTorch** >= 2.0 (CUDA recommended, MPS supported on M3+ with PyTorch 2.3+)
- **diffusers** >= 0.36
- **transformers** >= 4.40
- accelerate, safetensors, huggingface-hub
- pydantic >= 2.0, Pillow, click, einops, sentencepiece, loguru
- **zeroconf** >= 0.131 (mDNS network announcement)
- fastapi >= 0.110, uvicorn[standard] >= 0.27 (REST API server)
