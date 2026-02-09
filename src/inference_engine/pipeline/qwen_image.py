"""Qwen-Image pipeline — 20B MMDiT with true CFG and negative prompts.

Uses Z-Image-style text encoding (hidden_states[-2], filtered by mask)
but with a larger transformer and true classifier-free guidance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
from loguru import logger

from inference_engine.lora.applicator import LoraApplicator
from inference_engine.models.loader import ModelLoader
from inference_engine.pipeline.base import BasePipeline
from inference_engine.pipeline.z_image import ZImagePipeline
from inference_engine.types import ModelSpec, ProgressEvent, TransformerType

if TYPE_CHECKING:
    import PIL.Image

    from inference_engine.models.cache import ModelCache
    from inference_engine.types import JobParams


def _cache_key(path: str, role: str) -> str:
    return f"{path}::{role}"


class QwenImagePipeline(BasePipeline):
    """Qwen-Image: 20B MMDiT, true classifier-free guidance, Qwen3 encoder.

    Key differences from Z-Image:
    - Larger model (20B vs 6B)
    - Supports true CFG with negative prompts (default cfg_scale 4.0)
    - Higher default step count (50)
    - Uses FLUX-style transformer API (hidden_states/encoder_hidden_states)
    """

    SUPPORTED_RESOLUTIONS = [
        (1328, 1328), (1664, 928), (928, 1664),
        (1472, 1140), (1140, 1472), (1824, 768), (768, 1824),
        (1024, 1024), (1280, 720), (720, 1280),
    ]

    def validate_params(self, params: JobParams) -> None:
        required = {
            "qwen3_tokenizer": params.qwen3_tokenizer,
            "qwen3_encoder": params.qwen3_encoder,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Qwen-Image requires: {', '.join(missing)}")

    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        return [
            ModelSpec(key=_cache_key(params.qwen3_encoder, "qwen3"), loader=lambda: None, estimated_size_bytes=8_000_000_000),
            ModelSpec(key=_cache_key(params.transformer_model, "transformer"), loader=lambda: None, estimated_size_bytes=40_000_000_000),
            ModelSpec(key=_cache_key(params.vae_model, "vae"), loader=lambda: None, estimated_size_bytes=200_000_000),
        ]

    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> tuple[PIL.Image.Image, int]:
        import PIL.Image
        from transformers import AutoTokenizer, Qwen3Model

        device = cache._device
        dtype = torch.bfloat16

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Load Qwen3 tokenizer + encoder ───────────────────────────
        def _resolve_sub(path_or_repo: str, subfolder: str):
            path = ModelLoader.resolve_path(path_or_repo)
            if path.is_dir():
                sub = path / subfolder
                if sub.exists():
                    return sub
            return path

        qwen3_tok = cache.get_or_load(
            _cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"),
            lambda: AutoTokenizer.from_pretrained(
                _resolve_sub(params.qwen3_tokenizer, "tokenizer"),
            ),
        )
        qwen3_enc = cache.get_or_load(
            _cache_key(params.qwen3_encoder, "qwen3_encoder"),
            lambda: Qwen3Model.from_pretrained(
                _resolve_sub(params.qwen3_encoder, "text_encoder"),
                torch_dtype=dtype,
            ),
        )

        # ── Load Qwen-Image Transformer ──────────────────────────────
        def _load_transformer():
            from diffusers.models import QwenImageTransformer2DModel
            path = ModelLoader.resolve_path(params.transformer_model)
            if path.is_file():
                return QwenImageTransformer2DModel.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "transformer"
            if sub.exists():
                path = sub
            return QwenImageTransformer2DModel.from_pretrained(str(path), torch_dtype=dtype)

        transformer = cache.get_or_load(
            _cache_key(params.transformer_model, "transformer"),
            _load_transformer,
        )

        # ── Load VAE (FLUX-derived, 16 channels) ────────────────────
        def _load_vae():
            from diffusers import AutoencoderKL
            path = ModelLoader.resolve_path(params.vae_model)
            if path.is_file():
                return AutoencoderKL.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "vae"
            if sub.exists():
                path = sub
            return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype)

        vae = cache.get_or_load(
            _cache_key(params.vae_model, "vae"),
            _load_vae,
        )

        # Lock models
        keys = [
            _cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"),
            _cache_key(params.qwen3_encoder, "qwen3_encoder"),
            _cache_key(params.transformer_model, "transformer"),
            _cache_key(params.vae_model, "vae"),
        ]
        for k in keys:
            cache.lock(k)

        try:
            original_state = None
            if params.loras:
                lora_specs = [(lp.model_file, lp.strength) for lp in params.loras]
                original_state = LoraApplicator.snapshot_weights(transformer)
                LoraApplicator.load_and_apply(transformer, lora_specs, TransformerType.QWEN_IMAGE)

            image = self._generate(
                params=params,
                qwen3_tok=qwen3_tok,
                qwen3_enc=qwen3_enc,
                transformer=transformer,
                vae=vae,
                device=device,
                dtype=dtype,
                seed=seed,
                generator=generator,
                progress_cb=progress_cb,
            )

            if original_state is not None:
                LoraApplicator.unapply(transformer, original_state)
        finally:
            for k in keys:
                cache.unlock(k)

        return image, seed

    def _generate(
        self,
        *,
        params: JobParams,
        qwen3_tok,
        qwen3_enc,
        transformer,
        vae,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        generator: torch.Generator,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> PIL.Image.Image:
        import PIL.Image

        # ── 1. Text encoding (Z-Image style: hidden_states[-2], filtered by mask)
        prompt_embeds = ZImagePipeline._encode_qwen3_zimage(
            params.prompt, qwen3_tok, qwen3_enc, device, dtype
        )

        # Negative prompt for true CFG
        do_cfg = params.cfg_scale > 1.0
        neg_prompt_embeds = None
        if do_cfg:
            neg_prompt_embeds = ZImagePipeline._encode_qwen3_zimage(
                "", qwen3_tok, qwen3_enc, device, dtype
            )

        # ── 2. Prepare latent noise ──────────────────────────────────
        latent_channels = 16
        latent_h = math.ceil(params.height / 16) * 2
        latent_w = math.ceil(params.width / 16) * 2

        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator,
            dtype=dtype,
            device="cpu",
        ).to(device)

        # ── 3. Time schedule ─────────────────────────────────────────
        num_steps = params.steps
        patch_size = 2
        img_token_h = latent_h // patch_size
        img_token_w = latent_w // patch_size
        img_seq_len = img_token_h * img_token_w

        mu = ZImagePipeline._calculate_shift(img_seq_len)
        sigmas = ZImagePipeline._get_sigmas(mu, num_steps)

        # ── 4. Pack latents ──────────────────────────────────────────
        latents = self._pack_latents(latents)

        # ── 5. Position IDs ──────────────────────────────────────────
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_ids = self._generate_image_ids(packed_h, packed_w, device, dtype)
        text_ids = torch.zeros(prompt_embeds.shape[0], 3, device=device, dtype=dtype)

        # Add batch dim to prompt_embeds for transformer: [seq, dim] → [1, seq, dim]
        prompt_embeds_batched = prompt_embeds.unsqueeze(0)

        # ── 6. Denoising with true CFG ───────────────────────────────
        with torch.no_grad():
            for i in range(num_steps):
                sigma_curr = sigmas[i]
                sigma_prev = sigmas[i + 1]
                timestep = torch.tensor([sigma_curr], device=device, dtype=dtype).expand(latents.shape[0]) * 1000.0

                # Pooled projection (mean of prompt embeds)
                pooled = prompt_embeds.mean(dim=0, keepdim=True)  # [1, dim]

                # Conditional prediction
                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_batched,
                    pooled_projections=pooled,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    return_dict=False,
                )[0]

                # True CFG: pred = neg + cfg_scale * (pos - neg)
                if do_cfg and neg_prompt_embeds is not None:
                    neg_batched = neg_prompt_embeds.unsqueeze(0)
                    neg_text_ids = torch.zeros(neg_prompt_embeds.shape[0], 3, device=device, dtype=dtype)
                    neg_pooled = neg_prompt_embeds.mean(dim=0, keepdim=True)

                    noise_pred_neg = transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=neg_batched,
                        pooled_projections=neg_pooled,
                        txt_ids=neg_text_ids,
                        img_ids=image_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_neg + params.cfg_scale * (noise_pred - noise_pred_neg)

                dt = sigma_prev - sigma_curr
                latents = latents + dt * noise_pred

                progress_cb(ProgressEvent(
                    job_id="",
                    step=i + 1,
                    total_steps=num_steps,
                ))

        # ── 7. Unpack and decode ─────────────────────────────────────
        latents = self._unpack_latents(latents, latent_h, latent_w)

        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _seq, d = latents.shape
        c = d // 4
        packed_h = h // 2
        packed_w = w // 2
        latents = latents.reshape(b, packed_h, packed_w, c, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(b, c, h, w)
        return latents

    @staticmethod
    def _generate_image_ids(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        ids = torch.stack([torch.zeros_like(grid_y), grid_y, grid_x], dim=-1)
        return ids.reshape(-1, 3)
