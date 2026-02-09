"""Sequential job processing loop."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Callable

from loguru import logger

from inference_engine.types import (
    CompletedEvent,
    EventType,
    FailedEvent,
    PreviewConfig,
    ProgressEvent,
    StartedEvent,
)

if TYPE_CHECKING:
    from inference_engine.models.cache import ModelCache
    from inference_engine.pipeline.base import BasePipeline
    from inference_engine.queue.manager import JobQueue
    from inference_engine.types import TransformerType


class JobProcessor:
    """Pulls jobs from the queue and runs them sequentially."""

    def __init__(
        self,
        queue: JobQueue,
        cache: ModelCache,
        pipelines: dict[TransformerType, BasePipeline],
        emit: Callable,
        preview_config: PreviewConfig | None = None,
    ) -> None:
        self._queue = queue
        self._cache = cache
        self._pipelines = pipelines
        self._emit = emit
        self._preview_config = preview_config
        self._running = True

    def run(self) -> None:
        """Main processing loop. Runs until ``stop()`` is called."""
        logger.info("Job processor started")
        while self._running:
            job = self._queue.get_next(timeout=0.5)
            if job is None:
                continue

            job_id = job.job_id
            params = job.params
            logger.info(f"Processing job {job_id}: {params.transformer_type.value}")

            try:
                pipeline = self._pipelines.get(params.transformer_type)
                if pipeline is None:
                    raise ValueError(
                        f"No pipeline registered for transformer type: {params.transformer_type.value}"
                    )

                # Validate params for this pipeline
                pipeline.validate_params(params)

                self._emit(EventType.JOB_STARTED, StartedEvent(job_id=job_id))

                # Build progress callback
                def progress_cb(event: ProgressEvent) -> None:
                    event.job_id = job_id
                    self._emit(EventType.JOB_PROGRESS, event)

                # Run the pipeline
                image, seed = pipeline.run(params, self._cache, progress_cb, self._preview_config)

                self._queue.mark_completed(job_id)
                self._emit(
                    EventType.JOB_COMPLETED,
                    CompletedEvent(job_id=job_id, image=image, seed=seed),
                )
                logger.info(f"Job completed: {job_id}")

            except Exception as e:
                self._queue.mark_failed(job_id)
                tb = traceback.format_exc()
                self._emit(
                    EventType.JOB_FAILED,
                    FailedEvent(job_id=job_id, error=str(e), traceback=tb),
                )
                logger.error(f"Job failed: {job_id}: {e}")

        logger.info("Job processor stopped")

    def stop(self) -> None:
        self._running = False
