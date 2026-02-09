"""FLUX.1 Dev pipeline — CLIP + T5 text encoding, rectified flow denoising."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
from loguru import logger

from inference_engine.pipeline.lora_applicator import LoraApplicator
from inference_engine.models.loader import ModelLoader
from inference_engine.pipeline.base import BasePipeline
from inference_engine.types import ModelSpec, ProgressEvent, TransformerType

if TYPE_CHECKING:
    import PIL.Image

    from inference_engine.models.cache import ModelCache
    from inference_engine.types import JobParams, PreviewConfig


def _cache_key(path: str, role: str) -> str:
    return f"{path}::{role}"


def _resolve_sub(path_or_repo: str, subfolder: str):
    """Resolve a path, checking for a component subfolder in directories."""
    path = ModelLoader.resolve_path(path_or_repo)
    if path.is_dir():
        sub = path / subfolder
        if sub.exists():
            return sub
    return path


class Flux1DevPipeline(BasePipeline):
    """Full FLUX.1 Dev inference: CLIP + T5 text encoding, rectified flow Euler denoising, VAE decode.

    Models are loaded and released in stages to fit within VRAM:
    1. Text encoding (CLIP + T5) → produce embeddings → release encoders
    2. Denoising (transformer) → produce denoised latents → release transformer
    3. VAE decode → produce image → release VAE
    """

    def validate_params(self, params: JobParams) -> None:
        required = {
            "clip_tokenizer": params.clip_tokenizer,
            "clip_encoder": params.clip_encoder,
            "t5_tokenizer": params.t5_tokenizer,
            "t5_encoder": params.t5_encoder,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"FLUX.1 Dev requires: {', '.join(missing)}"
            )

    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        return [
            ModelSpec(key=_cache_key(params.clip_encoder, "clip_encoder"), loader=lambda: None, estimated_size_bytes=500_000_000),
            ModelSpec(key=_cache_key(params.t5_encoder, "t5_encoder"), loader=lambda: None, estimated_size_bytes=10_000_000_000),
            ModelSpec(key=_cache_key(params.transformer_model, "transformer"), loader=lambda: None, estimated_size_bytes=23_000_000_000),
            ModelSpec(key=_cache_key(params.vae_model, "vae"), loader=lambda: None, estimated_size_bytes=200_000_000),
        ]

    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
        preview_config: PreviewConfig | None = None,
    ) -> tuple[PIL.Image.Image, int]:
        import PIL.Image
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        device = cache._device
        dtype = torch.bfloat16

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Stage 1: Text encoding ────────────────────────────────────
        # Load text encoders, encode prompt, then release so cache can
        # evict them to make room for the transformer.

        clip_tok = cache.get_or_load(
            _cache_key(params.clip_tokenizer, "clip_tokenizer"),
            lambda: CLIPTokenizer.from_pretrained(
                _resolve_sub(params.clip_tokenizer, "tokenizer"),
            ),
        )
        clip_enc = cache.get_or_load(
            _cache_key(params.clip_encoder, "clip_encoder"),
            lambda: CLIPTextModel.from_pretrained(
                _resolve_sub(params.clip_encoder, "text_encoder"),
                torch_dtype=dtype,
            ),
        )

        clip_inputs = clip_tok(
            params.prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            clip_out = clip_enc(**clip_inputs)
        pooled_prompt_embeds = clip_out.pooler_output.to(dtype)  # (1, 768)
        del clip_inputs, clip_out

        t5_tok = cache.get_or_load(
            _cache_key(params.t5_tokenizer, "t5_tokenizer"),
            lambda: T5TokenizerFast.from_pretrained(
                _resolve_sub(params.t5_tokenizer, "tokenizer_2"),
            ),
        )
        t5_enc = cache.get_or_load(
            _cache_key(params.t5_encoder, "t5_encoder"),
            lambda: T5EncoderModel.from_pretrained(
                _resolve_sub(params.t5_encoder, "text_encoder_2"),
                torch_dtype=dtype,
            ),
        )

        t5_inputs = t5_tok(
            params.prompt, padding="max_length", max_length=512,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            t5_out = t5_enc(**t5_inputs)
        prompt_embeds = t5_out.last_hidden_state.to(dtype)  # (1, 512, 4096)
        del t5_inputs, t5_out

        # Text embeddings are now GPU tensors; encoders can be evicted.

        # ── Stage 2: Denoising ────────────────────────────────────────

        def _load_transformer():
            path = ModelLoader.resolve_path(params.transformer_model)
            if path.is_file():
                kwargs = {"torch_dtype": dtype}
                config = ModelLoader.resolve_config(params.transformer_config, "flux_transformer")
                if config:
                    kwargs["config"] = config
                if path.suffix == ".gguf":
                    from diffusers import GGUFQuantizationConfig
                    kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
                return FluxTransformer2DModel.from_single_file(str(path), **kwargs)
            sub = path / "transformer"
            if sub.exists():
                path = sub
            return FluxTransformer2DModel.from_pretrained(str(path), torch_dtype=dtype)

        transformer = cache.get_or_load(
            _cache_key(params.transformer_model, "transformer"),
            _load_transformer,
        )
        cache.lock(_cache_key(params.transformer_model, "transformer"))

        # Pre-load VAE for previews if enabled (VAE is ~200MB, fits alongside transformer)
        preview_vae = None
        want_previews = preview_config is not None and preview_config.enabled
        if want_previews:
            def _load_vae():
                path = ModelLoader.resolve_path(params.vae_model)
                if path.is_file():
                    config = ModelLoader.resolve_config(params.vae_config, "vae")
                    if config:
                        return AutoencoderKL.from_single_file(str(path), config=config, torch_dtype=dtype)
                    return AutoencoderKL.from_single_file(str(path), torch_dtype=dtype)
                sub = path / "vae"
                if sub.exists():
                    path = sub
                return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype)

            preview_vae = cache.get_or_load(
                _cache_key(params.vae_model, "vae"),
                _load_vae,
            )
            cache.lock(_cache_key(params.vae_model, "vae"))

        try:
            lora_hooks = []
            if params.loras:
                lora_specs = [(lp.model_file, lp.strength) for lp in params.loras]
                lora_hooks = LoraApplicator.load_and_apply(transformer, lora_specs, TransformerType.FLUX1_DEV)

            latents = self._denoise(
                params=params,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                transformer=transformer,
                device=device,
                dtype=dtype,
                generator=generator,
                progress_cb=progress_cb,
                preview_vae=preview_vae,
                preview_config=preview_config,
            )

            LoraApplicator.remove_hooks(lora_hooks)
        finally:
            cache.unlock(_cache_key(params.transformer_model, "transformer"))
            if want_previews:
                cache.unlock(_cache_key(params.vae_model, "vae"))

        del prompt_embeds, pooled_prompt_embeds

        # ── Stage 3: VAE decode ───────────────────────────────────────
        # If VAE was pre-loaded for previews, cache.get_or_load is a cache hit.

        def _load_vae():
            path = ModelLoader.resolve_path(params.vae_model)
            if path.is_file():
                config = ModelLoader.resolve_config(params.vae_config, "vae")
                if config:
                    return AutoencoderKL.from_single_file(str(path), config=config, torch_dtype=dtype)
                return AutoencoderKL.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "vae"
            if sub.exists():
                path = sub
            return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype)

        vae = cache.get_or_load(
            _cache_key(params.vae_model, "vae"),
            _load_vae,
        )
        cache.lock(_cache_key(params.vae_model, "vae"))

        try:
            image = self._decode(latents, vae)
        finally:
            cache.unlock(_cache_key(params.vae_model, "vae"))

        return image, seed

    # ── Denoising ─────────────────────────────────────────────────────

    def _denoise(
        self,
        *,
        params: JobParams,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        transformer,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
        progress_cb: Callable[[ProgressEvent], None],
        preview_vae=None,
        preview_config: PreviewConfig | None = None,
    ) -> torch.Tensor:
        """Run the rectified flow Euler denoising loop. Returns unpacked latents."""

        latent_h = math.ceil(params.height / 16) * 2
        latent_w = math.ceil(params.width / 16) * 2
        latent_channels = 16

        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator, dtype=dtype, device="cpu",
        ).to(device)

        # Time schedule
        num_steps = params.steps
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = self._calculate_shift(image_seq_len)
        sigmas = self._time_shift(mu, 1.0, sigmas)

        # Pack latents
        latents = self._pack_latents(latents)

        # Position IDs
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_ids = self._generate_image_ids(packed_h, packed_w, device, dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

        # Guidance embedding (used by guidance-distilled FLUX.1 Dev)
        guidance = torch.tensor([params.cfg_scale], device=device, dtype=dtype)
        guidance = guidance.expand(latents.shape[0])

        # Preview settings
        want_previews = preview_vae is not None and preview_config is not None and preview_config.enabled
        preview_interval = preview_config.interval if preview_config else 5
        preview_max_size = preview_config.max_size if preview_config else 256

        # Denoising loop
        with torch.no_grad():
            for i in range(num_steps):
                t_curr = sigmas[i]
                t_next = sigmas[i + 1]
                timestep = t_curr.expand(latents.shape[0])

                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    return_dict=False,
                )[0]

                dt = t_next - t_curr
                latents = latents + dt * noise_pred

                # Build progress event, optionally with preview
                step = i + 1
                preview_image = None
                if want_previews and step % preview_interval == 0 and step < num_steps:
                    try:
                        spatial_latents = self._unpack_latents(latents, latent_h, latent_w)
                        preview_image = self._generate_preview(spatial_latents, preview_vae, preview_max_size)
                    except Exception:
                        logger.opt(exception=True).debug("Preview generation failed")

                progress_cb(ProgressEvent(
                    job_id="", step=step, total_steps=num_steps,
                    preview_image=preview_image,
                ))

        # Unpack
        return self._unpack_latents(latents, latent_h, latent_w)

    # ── VAE decode ────────────────────────────────────────────────────

    @staticmethod
    def _decode(latents: torch.Tensor, vae) -> PIL.Image.Image:
        import PIL.Image

        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        """Linear interpolation of time shift mu based on image sequence length."""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    @staticmethod
    def _time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """Apply exponential time shift used by FLUX dev models."""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pixel-unshuffle by 2, then reshape to sequence format.

        (B, C, H, W) → (B, C*4, H/2, W/2) → (B, H/2*W/2, C*4)
        """
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H/2, W/2, C, 2, 2)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)  # (B, seq, C*4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reverse of _pack_latents.

        (B, seq, C*4) → (B, C, H, W)
        """
        b, _seq, d = latents.shape
        c = d // 4
        packed_h = h // 2
        packed_w = w // 2
        latents = latents.reshape(b, packed_h, packed_w, c, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, pH, 2, pW, 2)
        latents = latents.reshape(b, c, h, w)
        return latents

    @staticmethod
    def _generate_image_ids(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate positional IDs for the image patches.

        Returns tensor of shape (h*w, 3) where columns are (batch_id=0, y, x).
        """
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        ids = torch.stack([
            torch.zeros_like(grid_y),  # batch dimension (always 0)
            grid_y,
            grid_x,
        ], dim=-1)
        return ids.reshape(-1, 3)
