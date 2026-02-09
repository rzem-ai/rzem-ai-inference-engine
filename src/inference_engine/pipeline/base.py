"""Abstract base class for all image generation pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import PIL.Image

    from inference_engine.models.cache import ModelCache
    from inference_engine.types import JobParams, ModelSpec, ProgressEvent


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
    ) -> tuple[PIL.Image.Image, int]:
        """Execute the full generation pipeline.

        Returns ``(image, seed)`` where *seed* is the actual seed used
        (resolved from -1 if random).
        """
