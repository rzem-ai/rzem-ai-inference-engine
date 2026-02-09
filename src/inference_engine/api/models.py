"""Pydantic response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel

from inference_engine.types import JobStatus


class ProgressInfo(BaseModel):
    step: int
    total_steps: int


class ResultInfo(BaseModel):
    seed: int
    image_url: str


class ErrorInfo(BaseModel):
    error: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: ProgressInfo | None = None
    result: ResultInfo | None = None
    error: ErrorInfo | None = None


class CachedModelResponse(BaseModel):
    repo_id: str
    size_on_disk: int
    nb_files: int
    last_modified: float


class CachedFileResponse(BaseModel):
    file_name: str
    size_on_disk: int


class CachedRevisionResponse(BaseModel):
    commit_hash: str
    ref: str | None
    size_on_disk: int
    files: list[CachedFileResponse]


class CachedModelDetailResponse(BaseModel):
    repo_id: str
    size_on_disk: int
    nb_files: int
    revisions: list[CachedRevisionResponse]


class HealthResponse(BaseModel):
    status: str
    device: str
    jobs_queued: int
    jobs_running: int
    jobs_completed: int


def build_job_response(
    job_id: str,
    status: JobStatus,
    *,
    step: int | None = None,
    total_steps: int | None = None,
    seed: int | None = None,
    image_url: str | None = None,
    error: str | None = None,
) -> JobResponse:
    """Build a JobResponse from tracked job state."""
    progress = None
    if step is not None and total_steps is not None:
        progress = ProgressInfo(step=step, total_steps=total_steps)

    result = None
    if seed is not None and image_url is not None:
        result = ResultInfo(seed=seed, image_url=image_url)

    error_info = None
    if error is not None:
        error_info = ErrorInfo(error=error)

    return JobResponse(
        job_id=job_id,
        status=status,
        progress=progress,
        result=result,
        error=error_info,
    )
