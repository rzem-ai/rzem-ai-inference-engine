"""HTTP route definitions for the REST API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from rzem_ai_inference_engine.api.models import (
    CachedFileResponse,
    CachedModelDetailResponse,
    CachedModelResponse,
    CachedRevisionResponse,
    CudaVersionResponse,
    GpuInfoResponse,
    HealthResponse,
    JobResponse,
    StatusResponse,
    VramResponse,
    build_job_response,
)
from rzem_ai_inference_engine.types import JobParams, JobStatus

router = APIRouter()


@router.post("/jobs", status_code=201, response_model=JobResponse)
async def submit_job(params: JobParams, request: Request) -> JobResponse:
    """Submit a new generation job."""
    engine = request.app.state.engine
    store = request.app.state.store

    job_id = engine.submit(params)

    # The job is now tracked via the engine's JOB_QUEUED event callback.
    # Return the initial queued state.
    return build_job_response(job_id, JobStatus.QUEUED)


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(request: Request) -> list[JobResponse]:
    """List all tracked jobs."""
    store = request.app.state.store
    responses = []
    for job in store.list_jobs():
        image_url = f"/jobs/{job.job_id}/image" if job.status == JobStatus.COMPLETED else None
        responses.append(build_job_response(
            job.job_id,
            job.status,
            step=job.step,
            total_steps=job.total_steps,
            seed=job.seed,
            image_url=image_url,
            error=job.error,
        ))
    return responses


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request) -> JobResponse:
    """Get details for a specific job."""
    store = request.app.state.store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    image_url = f"/jobs/{job_id}/image" if job.status == JobStatus.COMPLETED else None
    return build_job_response(
        job.job_id,
        job.status,
        step=job.step,
        total_steps=job.total_steps,
        seed=job.seed,
        image_url=image_url,
        error=job.error,
    )


@router.get("/jobs/{job_id}/image")
async def get_job_image(job_id: str, request: Request) -> FileResponse:
    """Download the generated image for a completed job."""
    store = request.app.state.store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job has not completed")

    path = store.image_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(path, media_type="image/png", filename=f"{job_id}.png")


@router.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str, request: Request) -> JobResponse:
    """Cancel a queued or running job."""
    store = request.app.state.store
    engine = request.app.state.engine

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job with status '{job.status.value}'",
        )

    engine.cancel(job_id)

    # Re-fetch after cancellation
    job = store.get_job(job_id)
    return build_job_response(job.job_id, job.status)


@router.get("/models", response_model=list[CachedModelResponse])
async def list_cached_models() -> list[CachedModelResponse]:
    """List HuggingFace-cached models available locally."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    return [
        CachedModelResponse(
            repo_id=r.repo_id,
            size_on_disk=r.size_on_disk,
            nb_files=r.nb_files,
            last_modified=r.last_modified,
        )
        for r in info.repos
        if r.repo_type == "model"
    ]


@router.get("/models/all", response_model=list[CachedModelDetailResponse])
async def list_cached_models_detail() -> list[CachedModelDetailResponse]:
    """List HuggingFace-cached models with full revision and file tree."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    return [
        CachedModelDetailResponse(
            repo_id=r.repo_id,
            size_on_disk=r.size_on_disk,
            nb_files=r.nb_files,
            revisions=[
                CachedRevisionResponse(
                    commit_hash=rev.commit_hash,
                    ref="main" if "main" in rev.refs else next(iter(sorted(rev.refs))) if rev.refs else None,
                    size_on_disk=rev.size_on_disk,
                    files=sorted(
                        [
                            CachedFileResponse(
                                file_name=f.file_name,
                                size_on_disk=f.size_on_disk,
                            )
                            for f in rev.files
                        ],
                        key=lambda f: f.file_name,
                    ),
                )
                for rev in r.revisions
            ],
        )
        for r in info.repos
        if r.repo_type == "model"
    ]


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check with device and queue stats."""
    store = request.app.state.store
    device = str(request.app.state.engine._device)

    jobs = store.list_jobs()
    queued = sum(1 for j in jobs if j.status == JobStatus.QUEUED)
    running = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
    completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)

    return HealthResponse(
        status="ok",
        device=device,
        jobs_queued=queued,
        jobs_running=running,
        jobs_completed=completed,
    )


# ── GPU / VRAM / CUDA endpoints (sidecar support) ────────────────


@router.get("/gpu-info", response_model=GpuInfoResponse)
async def gpu_info(request: Request) -> GpuInfoResponse:
    """Return GPU device type, name, and total VRAM."""
    import torch

    device = request.app.state.engine._device
    device_type = device.type

    device_name = None
    total_vram_gb = None

    if device_type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        _, total = torch.cuda.mem_get_info(device)
        total_vram_gb = round(total / (1024 ** 3), 1)
    elif device_type == "mps":
        device_name = "Apple Silicon (MPS)"

    return GpuInfoResponse(
        device_type=device_type,
        device_name=device_name,
        total_vram_gb=total_vram_gb,
    )


@router.get("/vram", response_model=VramResponse)
async def vram_usage(request: Request) -> VramResponse:
    """Return current VRAM allocation stats."""
    import torch

    device = request.app.state.engine._device

    if device.type != "cuda":
        return VramResponse(available=False)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)

    return VramResponse(
        available=True,
        allocated=allocated,
        reserved=reserved,
        free=free,
        total=total,
    )


@router.post("/vram/clear", response_model=StatusResponse)
async def clear_vram(request: Request) -> StatusResponse:
    """Clear the VRAM model cache and run torch.cuda.empty_cache()."""
    import torch

    engine = request.app.state.engine
    engine._cache.clear()

    if engine._device.type == "cuda":
        torch.cuda.empty_cache()

    return StatusResponse(status="ok", message="VRAM cache cleared")


@router.get("/cuda-version", response_model=CudaVersionResponse)
async def cuda_version() -> CudaVersionResponse:
    """Return the CUDA runtime version if available."""
    import torch

    version = torch.version.cuda if torch.cuda.is_available() else None
    return CudaVersionResponse(cuda_version=version)


@router.delete("/models/{repo_id:path}", response_model=StatusResponse)
async def delete_cached_model(repo_id: str) -> StatusResponse:
    """Delete a HuggingFace-cached model by repo_id."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()

    # Find the repo
    target = None
    for repo in info.repos:
        if repo.repo_id == repo_id:
            target = repo
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Model '{repo_id}' not found in cache")

    # Collect all revision hashes for deletion
    revision_hashes = [rev.commit_hash for rev in target.revisions]
    delete_strategy = info.delete_revisions(*revision_hashes)
    delete_strategy.execute()

    return StatusResponse(status="ok", message=f"Deleted {repo_id} ({len(revision_hashes)} revisions)")
