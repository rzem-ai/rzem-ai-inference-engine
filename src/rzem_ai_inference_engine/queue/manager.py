"""Job queue with state tracking."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from typing import Callable

from loguru import logger

from rzem_ai_inference_engine.types import (
    CancelledEvent,
    EventType,
    Job,
    JobParams,
    JobStatus,
    QueuedEvent,
)

_CANCELLABLE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


class JobQueue:
    """Thread-safe FIFO job queue with status tracking."""

    def __init__(self, emit: Callable) -> None:
        self._queue: deque[Job] = deque()
        self._jobs: dict[str, Job] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._not_empty = threading.Event()
        self._emit = emit

    def add(self, params: JobParams) -> str:
        """Enqueue a job and return its ID."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, params=params)

        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job)
            self._not_empty.set()

        logger.info(f"Job queued: {job_id}")
        self._emit(EventType.JOB_QUEUED, QueuedEvent(job_id=job_id))
        return job_id

    def get_next(self, timeout: float | None = None) -> Job | None:
        """Block until a job is available, then dequeue and return it.

        Returns ``None`` if *timeout* expires or the queue is shut down.
        """
        if not self._not_empty.wait(timeout=timeout):
            return None

        with self._lock:
            # Skip cancelled jobs
            while self._queue:
                job = self._queue.popleft()
                if job.status == JobStatus.CANCELLED:
                    continue
                job.status = JobStatus.RUNNING
                self._cancel_events[job.job_id] = threading.Event()
                if not self._queue:
                    self._not_empty.clear()
                return job

            self._not_empty.clear()
            return None

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns True if the job was actually cancelled."""
        cancel_event = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in _CANCELLABLE_STATUSES:
                return False
            job.status = JobStatus.CANCELLED
            if job_id in self._cancel_events:
                cancel_event = self._cancel_events[job_id]

        if cancel_event is not None:
            cancel_event.set()

        self._emit(EventType.JOB_CANCELLED, CancelledEvent(job_id=job_id))
        logger.info(f"Job cancelled: {job_id}")
        return True

    def get_cancel_event(self, job_id: str) -> threading.Event | None:
        """Return the cancel event for a running job, or None if not found."""
        return self._cancel_events.get(job_id)

    def mark_completed(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
            self._cancel_events.pop(job_id, None)

    def mark_failed(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
            self._cancel_events.pop(job_id, None)

    def get_status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.status if job else None

    def shutdown(self) -> None:
        """Wake up any threads waiting on get_next."""
        self._not_empty.set()
