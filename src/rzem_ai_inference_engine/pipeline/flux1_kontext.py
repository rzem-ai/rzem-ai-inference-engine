"""FLUX.1 Kontext [dev] pipeline — image-to-image editing via input image conditioning.

Uses the same model components as Flux1DevPipeline (CLIP + T5 + FluxTransformer2DModel +
AutoencoderKL) but adds a VAE-encoded input image as conditioning during denoising.

Stages:
1. Text encoding (CLIP + T5) -> produce embeddings -> release encoders
2. Image encoding (VAE) -> produce image latents -> release VAE (reloaded for decode)
3. Denoising (transformer + concatenated image latents) -> produce denoised latents -> release transformer
4. VAE decode -> produce image -> release VAE
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from loguru import logger

from rzem_ai_inference_engine.models.loader import ModelLoader
from rzem_ai_inference_engine.models.memory import preferred_dtype
from rzem_ai_inference_engine.pipeline.flux1 import (
    Flux1DevPipeline,
    _cache_key,
    _resolve_sub,
    _resolve_t5_quantization,
)
from rzem_ai_inference_engine.pipeline.lora_applicator import LoraApplicator
from rzem_ai_inference_engine.types import ModelSpec, ProgressEvent, TransformerType

if TYPE_CHECKING:
    import PIL.Image

    from rzem_ai_inference_engine.models.cache import ModelCache
    from rzem_ai_inference_engine.types import JobParams, PreviewConfig

# Preferred resolutions for Kontext models (sorted by aspect ratio, landscape last).
# Input images are resized to the nearest resolution from this list.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800),
    (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]


def _find_nearest_resolution(width: int, height: int) -> tuple[int, int]:
    """Find the preferred Kontext resolution closest to the given dimensions.

    Picks the resolution whose aspect ratio is closest to the input image's
    aspect ratio, breaking ties by total pixel count difference.
    """
    input_aspect = width / height
    best = PREFERRED_KONTEXT_RESOLUTIONS[0]
    best_diff = float("inf")

    for w, h in PREFERRED_KONTEXT_RESOLUTIONS:
        aspect_diff = abs((w / h) - input_aspect)
        if aspect_diff < best_diff:
            best_diff = aspect_diff
            best = (w, h)
        elif aspect_diff == best_diff:
            # Tie-break by total pixel count similarity
            size_diff = abs(w * h - width * height)
            best_size_diff = abs(best[0] * best[1] - width * height)
            if size_diff < best_size_diff:
                best = (w, h)

    return best


class Flux1KontextPipeline(Flux1DevPipeline):
    """FLUX.1 Kontext [dev] image-to-image editing pipeline.

    Extends Flux1DevPipeline with input image conditioning. The input image
    is VAE-encoded and concatenated with noise latents at each denoising step
    so the transformer can attend to both the reference image and the text prompt.
    """

    def validate_params(self, params: JobParams) -> None:
        # Validate text encoder params (same as Flux1Dev)
        super().validate_params(params)

        # Kontext additionally requires an input image
        if not params.input_image_path:
            raise ValueError("FLUX.1 Kontext requires input_image_path")

        image_path = Path(params.input_image_path)
        if not image_path.exists():
            raise ValueError(
                f"FLUX.1 Kontext input image not found: {params.input_image_path}"
            )

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
        dtype = preferred_dtype(device)

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Load and resize input image ──────────────────────────────
        input_image = PIL.Image.open(params.input_image_path).convert("RGB")
        target_w, target_h = _find_nearest_resolution(input_image.width, input_image.height)
        logger.info(
            "Kontext input image {}x{} -> resized to {}x{}",
            input_image.width, input_image.height, target_w, target_h,
        )
        input_image = input_image.resize((target_w, target_h), PIL.Image.LANCZOS)

        # ── Stage 1: Text encoding ───────────────────────────────────

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
        t5_suffix, _ = _resolve_t5_quantization(params.t5_encoder_config)

        def _load_t5():
            path = _resolve_sub(params.t5_encoder, "text_encoder_2")
            load_kwargs: dict = {"torch_dtype": dtype}
            t5_config = ModelLoader.resolve_config(params.t5_encoder_config, "t5_encoder")
            if t5_config and (t5_config.get("load_in_4bit") or t5_config.get("load_in_8bit")):
                from transformers import BitsAndBytesConfig

                bnb_kwargs = dict(t5_config)
                for key in ("bnb_4bit_compute_dtype", "bnb_8bit_compute_dtype"):
                    if key in bnb_kwargs and isinstance(bnb_kwargs[key], str):
                        bnb_kwargs[key] = getattr(torch, bnb_kwargs[key])
                if t5_config.get("load_in_4bit") and "bnb_4bit_compute_dtype" not in bnb_kwargs:
                    bnb_kwargs["bnb_4bit_compute_dtype"] = dtype
                load_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
                logger.info("Loading T5 encoder with BitsAndBytes quantization")
            return T5EncoderModel.from_pretrained(path, **load_kwargs)

        t5_enc = cache.get_or_load(
            _cache_key(params.t5_encoder, "t5_encoder") + t5_suffix,
            _load_t5,
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

        # ── Stage 2: Image encoding (VAE) ────────────────────────────

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
            # Convert PIL image to tensor: [0,255] -> [0,1] -> [-1,1], shape (1, 3, H, W)
            image_np = np.array(input_image).astype(np.float32) / 255.0  # (H, W, 3)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            image_tensor = image_tensor.to(device, dtype=dtype)
            image_tensor = image_tensor * 2.0 - 1.0

            with torch.no_grad():
                image_latents = vae.encode(image_tensor).latent_dist.mode()

            # Apply VAE scaling
            image_latents = (
                (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
            )
            del image_tensor
        finally:
            cache.unlock(_cache_key(params.vae_model, "vae"))

        # Pack image latents to sequence format for the transformer
        image_latents = self._pack_latents(image_latents)

        # Generate position IDs for the reference image latents
        latent_h = math.ceil(target_h / 16) * 2
        latent_w = math.ceil(target_w / 16) * 2
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        ref_image_ids = self._generate_image_ids(packed_h, packed_w, device, dtype)

        logger.info(
            "Kontext image encoded: latent shape {}, packed seq_len {}",
            image_latents.shape, image_latents.shape[1],
        )

        # ── Stage 3: Denoising ───────────────────────────────────────

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

        # Pre-load VAE for previews if enabled
        preview_vae = None
        want_previews = preview_config is not None and preview_config.enabled
        if want_previews:
            preview_vae = cache.get_or_load(
                _cache_key(params.vae_model, "vae"),
                _load_vae,
            )
            cache.lock(_cache_key(params.vae_model, "vae"))

        try:
            lora_hooks = []
            if params.loras:
                lora_specs = [(lp.model_file, lp.strength) for lp in params.loras]
                # Kontext uses the same transformer architecture as Flux1 Dev
                lora_hooks = LoraApplicator.load_and_apply(
                    transformer, lora_specs, TransformerType.FLUX1_DEV,
                )

            latents = self._denoise_kontext(
                params=params,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                transformer=transformer,
                image_latents=image_latents,
                ref_image_ids=ref_image_ids,
                target_h=target_h,
                target_w=target_w,
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

        del prompt_embeds, pooled_prompt_embeds, image_latents, ref_image_ids

        # ── Stage 4: VAE decode ──────────────────────────────────────

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

    # ── Kontext denoising with image conditioning ────────────────────

    def _denoise_kontext(
        self,
        *,
        params: JobParams,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        transformer,
        image_latents: torch.Tensor,
        ref_image_ids: torch.Tensor,
        target_h: int,
        target_w: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
        progress_cb: Callable[[ProgressEvent], None],
        preview_vae=None,
        preview_config: PreviewConfig | None = None,
    ) -> torch.Tensor:
        """Rectified flow Euler denoising with input image conditioning.

        At each step, noise latents and reference image latents are concatenated
        along the sequence dimension before being passed to the transformer. The
        output is then sliced back to the noise portion only.
        """

        latent_h = math.ceil(target_h / 16) * 2
        latent_w = math.ceil(target_w / 16) * 2
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

        # Pack noise latents
        latents = self._pack_latents(latents)

        # Position IDs for the noise latents
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        noise_image_ids = self._generate_image_ids(packed_h, packed_w, device, dtype)

        # Text position IDs
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

        # Guidance embedding (used by guidance-distilled FLUX.1 Dev / Kontext)
        guidance = torch.tensor([params.cfg_scale], device=device, dtype=dtype)
        guidance = guidance.expand(latents.shape[0])

        # Combined image position IDs: noise + reference
        combined_ids = torch.cat([noise_image_ids, ref_image_ids], dim=0)

        # Preview settings
        want_previews = preview_vae is not None and preview_config is not None and preview_config.enabled
        preview_interval = preview_config.interval if preview_config else 5
        preview_max_size = preview_config.max_size if preview_config else 256

        # Emit initial noise preview (step 0)
        if want_previews:
            try:
                spatial_latents = self._unpack_latents(latents, latent_h, latent_w)
                noise_preview = self._generate_preview(spatial_latents, preview_vae, preview_max_size)
            except Exception:
                noise_preview = None
            progress_cb(ProgressEvent(
                job_id="", step=0, total_steps=num_steps,
                preview_image=noise_preview,
            ))

        # Denoising loop with image conditioning
        noise_seq_len = latents.shape[1]

        with torch.no_grad():
            for i in range(num_steps):
                t_curr = sigmas[i]
                t_next = sigmas[i + 1]
                timestep = t_curr.expand(latents.shape[0])

                # Concatenate noise latents and reference image latents along sequence dim
                latent_model_input = torch.cat([latents, image_latents], dim=1)

                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0]

                # Slice to noise portion only (discard reference image output)
                noise_pred = noise_pred[:, :noise_seq_len]

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

        # Unpack to spatial format for VAE decode
        return self._unpack_latents(latents, latent_h, latent_w)
