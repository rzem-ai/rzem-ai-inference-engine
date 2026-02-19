"""FAL.ai cloud pipeline — delegates image generation to FAL endpoints."""

from __future__ import annotations

import os
import random
from io import BytesIO
from typing import TYPE_CHECKING, Callable

from loguru import logger

from rzem_ai_inference_engine.pipeline.base import BasePipeline
from rzem_ai_inference_engine.types import ModelSpec, ProgressEvent

if TYPE_CHECKING:
    import PIL.Image

    from rzem_ai_inference_engine.models.cache import ModelCache
    from rzem_ai_inference_engine.types import JobParams, PreviewConfig

# Endpoints that use `aspect_ratio` (string) instead of `image_size` (object)
_ASPECT_RATIO_ENDPOINTS = frozenset({
    "fal-ai/nano-banana",
})

# Supported aspect ratios for aspect_ratio-style endpoints
_SUPPORTED_RATIOS = [
    (21, 9), (16, 9), (3, 2), (4, 3), (5, 4),
    (1, 1),
    (4, 5), (3, 4), (2, 3), (9, 16),
]


def _closest_aspect_ratio(width: int, height: int) -> str:
    """Map a width×height to the closest supported aspect ratio string."""
    target = width / height
    best_ratio = "1:1"
    best_diff = float("inf")
    for w, h in _SUPPORTED_RATIOS:
        diff = abs(target - w / h)
        if diff < best_diff:
            best_diff = diff
            best_ratio = f"{w}:{h}"
    return best_ratio


class FalPipeline(BasePipeline):
    """Pipeline that calls FAL.ai cloud endpoints for image generation.

    No local models are loaded — all inference happens server-side.
    """

    def validate_params(self, params: JobParams) -> None:
        if not params.fal_endpoint:
            raise ValueError("fal_endpoint is required for FAL cloud pipeline")
        if not params.fal_api_key:
            raise ValueError("fal_api_key is required for FAL cloud pipeline")

    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        return []

    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
        preview_config: PreviewConfig | None = None,
    ) -> tuple[PIL.Image.Image, int]:
        import fal_client
        import httpx
        import PIL.Image

        # Set the API key for fal-client
        os.environ["FAL_KEY"] = params.fal_api_key

        seed = params.seed if params.seed >= 0 else random.randint(0, 2**31 - 1)
        endpoint = params.fal_endpoint

        # Build arguments — image size format varies by endpoint
        arguments: dict = {
            "prompt": params.prompt,
            "seed": seed,
        }

        if endpoint in _ASPECT_RATIO_ENDPOINTS:
            arguments["aspect_ratio"] = _closest_aspect_ratio(params.width, params.height)
        else:
            arguments["image_size"] = {
                "width": int(params.width),
                "height": int(params.height),
            }

        # Map params to endpoint-specific argument names
        if "schnell" not in endpoint:
            arguments["guidance_scale"] = params.cfg_scale
        if params.steps > 0:
            arguments["num_inference_steps"] = params.steps

        # Emit initial progress (FAL has no intermediate progress)
        progress_cb(ProgressEvent(
            job_id="",  # filled by processor
            step=0,
            total_steps=params.steps,
        ))

        logger.info("Calling FAL endpoint: {} with arguments: {}", endpoint, arguments)
        result = fal_client.run(endpoint, arguments=arguments)

        # Download the generated image
        image_url = result["images"][0]["url"]
        logger.info("Downloading FAL result from: {}", image_url)
        response = httpx.get(image_url, timeout=60)
        response.raise_for_status()
        image = PIL.Image.open(BytesIO(response.content)).convert("RGB")

        # Emit final progress
        progress_cb(ProgressEvent(
            job_id="",
            step=params.steps,
            total_steps=params.steps,
        ))

        return image, seed
