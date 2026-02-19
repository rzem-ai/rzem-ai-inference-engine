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


# Named aspect ratio strings used by some FAL endpoints (e.g. fal-ai/flux-2-pro)
# mapped to their numeric W/H equivalents for closest-match comparison.
_NAMED_RATIO_VALUES: dict[str, tuple[int, int]] = {
    "square_hd":      (1, 1),
    "square":         (1, 1),
    "portrait_4_3":   (3, 4),
    "portrait_16_9":  (9, 16),
    "landscape_4_3":  (4, 3),
    "landscape_16_9": (16, 9),
}


def _ratio_value(ratio_str: str) -> tuple[int, int]:
    """Return the (w, h) numeric pair for a ratio string.

    Accepts both colon format ("16:9") and FAL named format ("landscape_16_9").
    """
    if ratio_str in _NAMED_RATIO_VALUES:
        return _NAMED_RATIO_VALUES[ratio_str]
    w_str, h_str = ratio_str.split(":")
    return int(w_str), int(h_str)


def _closest_aspect_ratio(width: int, height: int, supported: list[str]) -> str:
    """Map a width×height to the closest ratio string from the supported list.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        supported: List of ratio strings — either colon format ("16:9") or
                   FAL named format ("landscape_16_9"). Both are accepted.

    Returns:
        The closest matching ratio string from the supported list.
    """
    target = width / height
    best_ratio = supported[0]
    best_diff = float("inf")
    for ratio_str in supported:
        w, h = _ratio_value(ratio_str)
        diff = abs(target - w / h)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio_str
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

        if params.fal_aspectratio:
            arguments["aspect_ratio"] = _closest_aspect_ratio(params.width, params.height, params.fal_aspectratio)
            arguments["image_size"] = _closest_aspect_ratio(params.width, params.height, params.fal_aspectratio)
        else:
            arguments["image_size"] = {
                "width": int(params.width),
                "height": int(params.height),
            }

        # Map params to endpoint-specific argument names
        if params.cfg_scale > 0:
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
