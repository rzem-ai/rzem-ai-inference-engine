"""Qwen-Image pipeline — 20B MMDiT with true CFG.

Uses Qwen2.5-VL-7B-Instruct as text encoder with prompt template encoding.
Supports true classifier-free guidance with negative prompts and norm rescaling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from loguru import logger

from inference_engine.lora.applicator import LoraApplicator
from inference_engine.models.loader import ModelLoader
from inference_engine.pipeline.base import BasePipeline
from inference_engine.types import ModelSpec, ProgressEvent, TransformerType

if TYPE_CHECKING:
    import PIL.Image

    from inference_engine.models.cache import ModelCache
    from inference_engine.types import JobParams


def _cache_key(path: str, role: str) -> str:
    return f"{path}::{role}"


class QwenImagePipeline(BasePipeline):
    """Qwen-Image: 20B MMDiT, true classifier-free guidance, Qwen2.5-VL encoder.

    Key differences from FLUX / Z-Image:
    - Text encoder is Qwen2.5-VL-7B-Instruct (vision-language model, text-only mode)
    - Prompt template wrapping with system prefix drop
    - Transformer takes img_shapes, txt_seq_lens, encoder_hidden_states_mask
    - VAE is AutoencoderKLQwenImage with mean/std normalization and 5D tensors
    - CFG uses norm rescaling to prevent artifact blowup
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        "size, texture, quantity, text, spatial relationships of the objects and "
        "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    PROMPT_TEMPLATE_DROP_IDX = 34
    TOKENIZER_MAX_LENGTH = 1024

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
            ModelSpec(key=_cache_key(params.qwen3_encoder, "qwen3"), loader=lambda: None, estimated_size_bytes=15_000_000_000),
            ModelSpec(key=_cache_key(params.transformer_model, "transformer"), loader=lambda: None, estimated_size_bytes=40_000_000_000),
            ModelSpec(key=_cache_key(params.vae_model, "vae"), loader=lambda: None, estimated_size_bytes=500_000_000),
        ]

    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> tuple[PIL.Image.Image, int]:
        import PIL.Image
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        device = cache._device
        dtype = torch.bfloat16

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Load tokenizer + text encoder (Qwen2.5-VL) ───────────────
        def _resolve_sub(path_or_repo: str, subfolder: str):
            path = ModelLoader.resolve_path(path_or_repo)
            if path.is_dir():
                sub = path / subfolder
                if sub.exists():
                    return sub
            return path

        tokenizer = cache.get_or_load(
            _cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"),
            lambda: AutoTokenizer.from_pretrained(
                _resolve_sub(params.qwen3_tokenizer, "tokenizer"),
            ),
        )

        text_encoder = cache.get_or_load(
            _cache_key(params.qwen3_encoder, "qwen3_encoder"),
            lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        # ── Load VAE (QwenImage-specific, mean/std normalization) ────
        def _load_vae():
            from diffusers import AutoencoderKLQwenImage
            path = ModelLoader.resolve_path(params.vae_model)
            if path.is_file():
                return AutoencoderKLQwenImage.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "vae"
            if sub.exists():
                path = sub
            return AutoencoderKLQwenImage.from_pretrained(str(path), torch_dtype=dtype)

        vae = cache.get_or_load(
            _cache_key(params.vae_model, "vae"),
            _load_vae,
        )

        # Lock all models
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
                tokenizer=tokenizer,
                text_encoder=text_encoder,
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

    def _encode_prompt(
        self,
        prompt: str,
        tokenizer,
        text_encoder,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text using Qwen2.5-VL with prompt template.

        Returns (prompt_embeds, attention_mask) both with batch dim.
        """
        txt = self.PROMPT_TEMPLATE.format(prompt)
        drop_idx = self.PROMPT_TEMPLATE_DROP_IDX

        tokens = tokenizer(
            [txt],
            max_length=self.TOKENIZER_MAX_LENGTH + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = output.hidden_states[-1]

        # Extract masked hidden states and drop system prompt prefix
        mask = tokens.attention_mask.bool()
        valid_lengths = mask.sum(dim=1)
        selected = hidden_states[mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        split_result = [e[drop_idx:] for e in split_result]

        # Build padded tensors with attention masks
        attn_masks = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in split_result]
        max_seq_len = max(e.size(0) for e in split_result)

        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_result
        ])
        encoder_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_masks
        ])

        return prompt_embeds.to(dtype=dtype), encoder_mask

    def _generate(
        self,
        *,
        params: JobParams,
        tokenizer,
        text_encoder,
        transformer,
        vae,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        generator: torch.Generator,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> PIL.Image.Image:
        import PIL.Image

        # ── 1. Text encoding ─────────────────────────────────────────
        prompt_embeds, prompt_mask = self._encode_prompt(
            params.prompt, tokenizer, text_encoder, device, dtype
        )

        # Negative prompt for true CFG
        do_cfg = params.cfg_scale > 1.0
        neg_prompt_embeds = None
        neg_prompt_mask = None
        if do_cfg:
            neg_prompt_embeds, neg_prompt_mask = self._encode_prompt(
                "", tokenizer, text_encoder, device, dtype
            )

        # ── 2. Prepare latent noise ──────────────────────────────────
        vae_scale_factor = 2 ** len(vae.temperal_downsample) if hasattr(vae, "temperal_downsample") else 8
        latent_channels = transformer.config.in_channels // 4

        latent_h = 2 * (params.height // (vae_scale_factor * 2))
        latent_w = 2 * (params.width // (vae_scale_factor * 2))

        # 5D shape: (batch, temporal=1, channels, height, width)
        latents = torch.randn(
            (1, 1, latent_channels, latent_h, latent_w),
            generator=generator,
            dtype=dtype,
            device="cpu",
        ).to(device)

        # Pack: (B, 1, C, H, W) → (B, seq, C*4)
        latents = self._pack_latents(latents, 1, latent_channels, latent_h, latent_w)

        # ── 3. Image shape info for transformer ──────────────────────
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]]

        # ── 4. Time schedule ─────────────────────────────────────────
        num_steps = params.steps
        image_seq_len = latents.shape[1]
        mu = self._calculate_shift(image_seq_len)
        sigmas = self._get_sigmas(mu, num_steps)

        # ── 5. Guidance embedding (for guidance-distilled variants) ──
        guidance = None
        if getattr(transformer.config, "guidance_embeds", False):
            guidance = torch.full([1], params.cfg_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])

        # ── 6. Text sequence lengths ─────────────────────────────────
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()
        neg_txt_seq_lens = neg_prompt_mask.sum(dim=1).tolist() if neg_prompt_mask is not None else None

        # ── 7. Denoising loop ────────────────────────────────────────
        with torch.no_grad():
            for i in range(num_steps):
                sigma_curr = sigmas[i]
                sigma_next = sigmas[i + 1]
                timestep = torch.tensor([sigma_curr], device=device, dtype=dtype)

                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                # True CFG with norm rescaling
                if do_cfg and neg_prompt_embeds is not None:
                    neg_noise_pred = transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        guidance=guidance,
                        encoder_hidden_states=neg_prompt_embeds,
                        encoder_hidden_states_mask=neg_prompt_mask,
                        img_shapes=img_shapes,
                        txt_seq_lens=neg_txt_seq_lens,
                        return_dict=False,
                    )[0]

                    combined = neg_noise_pred + params.cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
                    noise_pred = combined * (cond_norm / combined_norm)

                # Euler step
                dt = sigma_next - sigma_curr
                latents = latents + dt * noise_pred

                progress_cb(ProgressEvent(
                    job_id="",
                    step=i + 1,
                    total_steps=num_steps,
                ))

        # ── 8. Unpack and decode ─────────────────────────────────────
        latents = self._unpack_latents(latents, params.height, params.width, vae_scale_factor)
        latents = latents.to(vae.dtype)

        # QwenImage VAE: denormalize with mean/std
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(vae.config.latents_std)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean

        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]  # drop temporal dim

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    @staticmethod
    def _get_sigmas(mu: float, num_steps: int) -> list[float]:
        """Generate time-shifted sigma schedule matching FlowMatchEulerDiscreteScheduler."""
        raw = np.linspace(1.0, 1.0 / num_steps, num_steps)
        shifted = []
        for t in raw:
            if t <= 0:
                shifted.append(0.0)
            elif t >= 1:
                shifted.append(1.0)
            else:
                shifted.append(math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)))
        shifted.append(0.0)  # final sigma
        return shifted

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        batch_size, _seq, channels = latents.shape
        h = 2 * (int(height) // (vae_scale_factor * 2))
        w = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, h, w)
        return latents
