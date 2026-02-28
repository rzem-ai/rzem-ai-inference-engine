"""Abstract base class for all image generation pipelines."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import PIL.Image

    from rzem_ai_inference_engine.models.cache import ModelCache
    from rzem_ai_inference_engine.types import JobParams, ModelSpec, PreviewConfig, ProgressEvent


class BasePipeline(ABC):
    """Interface that all pipeline implementations must satisfy."""

    @abstractmethod
    def validate_params(self, params: JobParams) -> None:
        """Raise ``ValueError`` if *params* is missing required fields
        for this pipeline type.
        """

    @abstractmethod
    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        """Return specs for every model this pipeline needs to run *params*.

        Used by the processor to pre-check VRAM availability.
        """

    @abstractmethod
    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
        preview_config: PreviewConfig | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[PIL.Image.Image, int]:
        """Execute the full generation pipeline.

        Returns ``(image, seed)`` where *seed* is the actual seed used
        (resolved from -1 if random).
        """

    @staticmethod
    def _generate_preview(
        latents: torch.Tensor,
        vae,
        max_size: int = 256,
    ) -> PIL.Image.Image:
        """Decode latents into a small preview image.

        Spatially downsamples latents via bilinear interpolation before
        VAE decode to keep the operation fast (~15-50ms).
        """
        import PIL.Image

        # Determine downscale factor from max_size and VAE spatial ratio (8x)
        _, _, lh, lw = latents.shape
        pixel_h, pixel_w = lh * 8, lw * 8
        scale = min(max_size / pixel_h, max_size / pixel_w, 1.0)

        if scale < 1.0:
            target_h = max(1, round(lh * scale))
            target_w = max(1, round(lw * scale))
            latents = F.interpolate(
                latents.float(), size=(target_h, target_w), mode="bilinear", align_corners=False,
            ).to(latents.dtype)

        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)
