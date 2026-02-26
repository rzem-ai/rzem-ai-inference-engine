from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import PIL.Image
    import torch


class TransformerType(str, Enum):
    FLUX1_DEV = "flux1_dev"
    FLUX1_KONTEXT = "flux1_kontext"
    FLUX2_DEV = "flux2_dev"
    Z_IMAGE = "z_image"
    QWEN_IMAGE = "qwen_image"
    FAL_CLOUD = "fal_cloud"


class EventType(str, Enum):
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PreviewConfig:
    """Configuration for in-progress preview image generation.

    When enabled, pipelines decode intermediate latents at the given
    step interval and attach a small PIL Image to ProgressEvent.
    """

    enabled: bool = False
    interval: int = 5  # generate preview every N steps
    max_size: int = 256  # max dimension (width or height) for preview


class LoraFormat(str, Enum):
    KOHYA = "kohya"
    DIFFUSERS = "diffusers"
    XLABS = "xlabs"
    ONETRAINER = "onetrainer"
    AITOOLKIT = "aitoolkit"
    UNKNOWN = "unknown"


class LoraParams(BaseModel):
    model_file: str
    strength: float = 1.0


class JobParams(BaseModel):
    prompt: str
    input_image_path: str | None = None
    steps: int = 20
    cfg_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    seed: int = -1  # -1 = random

    # Transformer
    transformer_model: str
    transformer_type: TransformerType
    transformer_config: str | dict | None = None

    # CLIP text encoder (FLUX.1 only)
    clip_tokenizer: str | None = None
    clip_tokenizer_config: str | dict | None = None
    clip_encoder: str | None = None
    clip_encoder_config: str | dict | None = None

    # T5 text encoder (FLUX.1 only)
    t5_tokenizer: str | None = None
    t5_tokenizer_config: str | dict | None = None
    t5_encoder: str | None = None
    t5_encoder_config: str | dict | None = None

    # Qwen3 text encoder (FLUX.2, Z-Image, Qwen-Image)
    qwen3_tokenizer: str | None = None
    qwen3_tokenizer_config: str | dict | None = None
    qwen3_encoder: str | None = None
    qwen3_encoder_config: str | dict | None = None

    # VAE
    vae_model: str
    vae_config: str | dict | None = None

    # Sampling
    sampler: str = "euler"
    scheduler: str = "normal"

    # LoRAs
    loras: list[LoraParams] = Field(default_factory=list)

    # FAL.ai cloud
    fal_endpoint: str | None = None
    fal_api_key: str | None = None
    fal_aspectratio: list[str] | None = None


@dataclass
class ModelSpec:
    """Describes a model that a pipeline needs loaded."""
    key: str
    loader: object  # Callable[[], Any] — typed loosely to avoid import issues
    estimated_size_bytes: int


@dataclass
class Job:
    job_id: str
    params: JobParams
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)


# ── Event payloads ──────────────────────────────────────────────────────


@dataclass
class QueuedEvent:
    job_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class StartedEvent:
    job_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProgressEvent:
    job_id: str
    step: int
    total_steps: int
    timestamp: float = field(default_factory=time.time)
    preview_image: PIL.Image.Image | None = None


@dataclass
class CompletedEvent:
    job_id: str
    image: PIL.Image.Image
    seed: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailedEvent:
    job_id: str
    error: str
    traceback: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CancelledEvent:
    job_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelLoadingEvent:
    key: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelLoadedEvent:
    key: str
    size_bytes: int
    device: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelUnloadedEvent:
    key: str
    from_device: str
    timestamp: float = field(default_factory=time.time)
