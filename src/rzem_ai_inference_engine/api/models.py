"""Pydantic response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel

from rzem_ai_inference_engine.types import JobStatus


class ProgressInfo(BaseModel):
    step: int
    total_steps: int
    preview: str | None = None


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


class VramResponse(BaseModel):
    available: bool
    allocated: int = 0
    reserved: int = 0
    free: int = 0
    total: int = 0


class GpuInfoResponse(BaseModel):
    device_type: str
    device_name: str | None = None
    total_vram_gb: float | None = None


class CudaVersionResponse(BaseModel):
    cuda_version: str | None = None


class StatusResponse(BaseModel):
    status: str
    message: str | None = None


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
