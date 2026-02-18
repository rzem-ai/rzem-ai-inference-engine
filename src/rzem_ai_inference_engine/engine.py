"""InferenceEngine — main public API."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

from loguru import logger

from rzem_ai_inference_engine.models.cache import ModelCache
from rzem_ai_inference_engine.models.memory import detect_device
from rzem_ai_inference_engine.pipeline.flux1 import Flux1DevPipeline
from rzem_ai_inference_engine.pipeline.flux2 import Flux2DevPipeline
from rzem_ai_inference_engine.pipeline.qwen_image import QwenImagePipeline
from rzem_ai_inference_engine.pipeline.z_image import ZImagePipeline
from rzem_ai_inference_engine.queue.manager import JobQueue
from rzem_ai_inference_engine.queue.processor import JobProcessor
from rzem_ai_inference_engine.types import EventType, JobParams, PreviewConfig, TransformerType


class InferenceEngine:
    """Text-to-image inference engine with job queue and VRAM management.

    Thread-safe. Manages a background processing thread that pulls jobs
    from the queue and executes them sequentially.

    Usage::

        engine = InferenceEngine()
        engine.on(EventType.JOB_COMPLETED, lambda e: save(e.image))
        engine.on(EventType.JOB_PROGRESS, lambda e: print(f"{e.step}/{e.total_steps}"))
        job_id = engine.submit(JobParams(...))
        # ... later ...
        engine.shutdown()
    """

    def __init__(
        self,
        device: str = "auto",
        vram_limit_gb: float | None = None,
        preview_config: PreviewConfig | None = None,
    ) -> None:
        import torch

        # Device selection
        if device == "auto":
            self._device = detect_device()
        else:
            self._device = torch.device(device)

        # Event system
        self._listeners: dict[EventType, list[Callable]] = defaultdict(list)
        self._listener_lock = threading.Lock()

        # Model cache
        self._cache = ModelCache(
            device=self._device,
            vram_limit_gb=vram_limit_gb,
            emit=self._emit,
        )

        # Pipelines
        self._pipelines: dict[TransformerType, Any] = {
            TransformerType.FLUX1_DEV: Flux1DevPipeline(),
            TransformerType.FLUX2_DEV: Flux2DevPipeline(),
            TransformerType.Z_IMAGE: ZImagePipeline(),
            TransformerType.QWEN_IMAGE: QwenImagePipeline(),
        }

        # FAL cloud pipeline
        from rzem_ai_inference_engine.pipeline.fal import FalPipeline
        self._pipelines[TransformerType.FAL_CLOUD] = FalPipeline()

        # Job queue
        self._queue = JobQueue(emit=self._emit)

        # Processor (background thread)
        self._processor = JobProcessor(
            queue=self._queue,
            cache=self._cache,
            pipelines=self._pipelines,
            emit=self._emit,
            preview_config=preview_config,
        )
        self._thread = threading.Thread(
            target=self._processor.run,
            name="inference-processor",
            daemon=True,
        )
        self._thread.start()
        logger.info("InferenceEngine initialized")

    # ── Event API ────────────────────────────────────────────────────

    def on(self, event: EventType, callback: Callable) -> None:
        """Register an event listener."""
        with self._listener_lock:
            self._listeners[event].append(callback)

    def off(self, event: EventType, callback: Callable) -> None:
        """Remove an event listener."""
        with self._listener_lock:
            try:
                self._listeners[event].remove(callback)
            except ValueError:
                pass

    def _emit(self, event: EventType, data: Any = None) -> None:
        """Emit an event to all registered listeners."""
        with self._listener_lock:
            listeners = list(self._listeners.get(event, []))
        for cb in listeners:
            try:
                cb(data)
            except Exception:
                logger.opt(exception=True).warning(f"Error in event listener for {event.value}")

    # ── Job API ──────────────────────────────────────────────────────

    def submit(self, params: JobParams) -> str:
        """Submit a generation job. Returns the job ID."""
        return self._queue.add(params)

    def cancel(self, job_id: str) -> None:
        """Cancel a queued job."""
        self._queue.cancel(job_id)

    # ── Lifecycle ────────────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        """Stop the processor thread and clean up."""
        logger.info("Shutting down InferenceEngine")
        self._processor.stop()
        self._queue.shutdown()
        if wait:
            self._thread.join(timeout=30)
        self._cache.clear()
        logger.info("InferenceEngine shut down")
