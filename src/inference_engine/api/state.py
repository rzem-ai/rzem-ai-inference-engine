"""In-memory job state tracking and engine event → WebSocket bridge."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from inference_engine.types import (
    CancelledEvent,
    CompletedEvent,
    EventType,
    FailedEvent,
    JobStatus,
    ProgressEvent,
    QueuedEvent,
    StartedEvent,
)

if TYPE_CHECKING:
    from inference_engine.api.ws import ConnectionManager
    from inference_engine.engine import InferenceEngine


@dataclass
class TrackedJob:
    """Minimal per-job state kept by the API layer."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    step: int | None = None
    total_steps: int | None = None
    seed: int | None = None
    error: str | None = None


class JobStateStore:
    """Tracks job state and bridges engine events to the async WebSocket layer.

    Engine callbacks fire on the processor thread. This class uses
    ``loop.call_soon_threadsafe`` to schedule async broadcasts on the
    event loop that owns the WebSocket connections.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        ws: ConnectionManager,
        output_dir: Path,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._engine = engine
        self._ws = ws
        self._output_dir = output_dir
        self._loop = loop
        self._lock = threading.Lock()
        self._jobs: dict[str, TrackedJob] = {}

        # Register engine event callbacks
        engine.on(EventType.JOB_QUEUED, self._on_queued)
        engine.on(EventType.JOB_STARTED, self._on_started)
        engine.on(EventType.JOB_PROGRESS, self._on_progress)
        engine.on(EventType.JOB_COMPLETED, self._on_completed)
        engine.on(EventType.JOB_FAILED, self._on_failed)
        engine.on(EventType.JOB_CANCELLED, self._on_cancelled)

    # ── Public API (called from async route handlers) ────────────

    def get_job(self, job_id: str) -> TrackedJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrackedJob]:
        with self._lock:
            return list(self._jobs.values())

    def image_path(self, job_id: str) -> Path:
        return self._output_dir / f"{job_id}.png"

    # ── Engine event callbacks (called on processor thread) ──────

    def _on_queued(self, event: QueuedEvent) -> None:
        with self._lock:
            self._jobs[event.job_id] = TrackedJob(job_id=event.job_id)
        self._broadcast({"event": "job_queued", "job_id": event.job_id, "data": {}})

    def _on_started(self, event: StartedEvent) -> None:
        with self._lock:
            job = self._jobs.get(event.job_id)
            if job:
                job.status = JobStatus.RUNNING
        self._broadcast({"event": "job_started", "job_id": event.job_id, "data": {}})

    def _on_progress(self, event: ProgressEvent) -> None:
        with self._lock:
            job = self._jobs.get(event.job_id)
            if job:
                job.step = event.step
                job.total_steps = event.total_steps
        self._broadcast({
            "event": "job_progress",
            "job_id": event.job_id,
            "data": {"step": event.step, "total_steps": event.total_steps},
        })

    def _on_completed(self, event: CompletedEvent) -> None:
        # Save image to disk before broadcasting (file ready when client gets event)
        out_path = self._output_dir / f"{event.job_id}.png"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            event.image.save(str(out_path))
        except Exception:
            logger.opt(exception=True).error(f"Failed to save image for {event.job_id}")

        with self._lock:
            job = self._jobs.get(event.job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.seed = event.seed

        image_url = f"/jobs/{event.job_id}/image"
        self._broadcast({
            "event": "job_completed",
            "job_id": event.job_id,
            "data": {"seed": event.seed, "image_url": image_url},
        })

    def _on_failed(self, event: FailedEvent) -> None:
        with self._lock:
            job = self._jobs.get(event.job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = event.error
        self._broadcast({
            "event": "job_failed",
            "job_id": event.job_id,
            "data": {"error": event.error},
        })

    def _on_cancelled(self, event: CancelledEvent) -> None:
        with self._lock:
            job = self._jobs.get(event.job_id)
            if job:
                job.status = JobStatus.CANCELLED
        self._broadcast({
            "event": "job_cancelled",
            "job_id": event.job_id,
            "data": {},
        })

    # ── Internal ────────────────────────────────────────────────

    def _broadcast(self, message: dict) -> None:
        """Schedule an async broadcast from the sync callback thread."""
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._ws.broadcast(message),
        )
