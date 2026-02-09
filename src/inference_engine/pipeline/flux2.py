"""FLUX.2 Dev pipeline — Mistral 3 text encoding, Flux2Transformer2DModel, AutoencoderKLFlux2.

Uses Mistral 3 (not Qwen3) as text encoder with multi-layer hidden state extraction
from layers (10, 20, 30). VAE is AutoencoderKLFlux2 with batch normalization.
Guidance-distilled (guidance scale as embedding, no true CFG).
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


# System message from the official FLUX.2 reference
SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object attribution and actions "
    "without speculation."
)

# Hidden state layers to extract from Mistral 3
HIDDEN_STATE_LAYERS = (10, 20, 30)


def _cache_key(path: str, role: str) -> str:
    return f"{path}::{role}"


class Flux2DevPipeline(BasePipeline):
    """FLUX.2 Dev: Mistral 3 encoder, Flux2Transformer2DModel, AutoencoderKLFlux2 with BN.

    Key differences from FLUX.1:
    - Text encoder is Mistral 3 (not CLIP+T5)
    - Multi-layer hidden state extraction from layers (10, 20, 30)
    - VAE is AutoencoderKLFlux2 with batch normalization on patchified latents
    - Patchify/unpatchify instead of FLUX.1-style 2x2 pack/unpack
    - Guidance-distilled (guidance scale as embedding, no true CFG)
    - 4D position IDs (T, H, W, L) via cartesian product
    """

    def validate_params(self, params: JobParams) -> None:
        required = {
            "qwen3_tokenizer": params.qwen3_tokenizer,
            "qwen3_encoder": params.qwen3_encoder,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"FLUX.2 Dev requires: {', '.join(missing)}")

    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        return [
            ModelSpec(key=_cache_key(params.qwen3_encoder, "qwen3"), loader=lambda: None, estimated_size_bytes=15_000_000_000),
            ModelSpec(key=_cache_key(params.transformer_model, "transformer"), loader=lambda: None, estimated_size_bytes=23_000_000_000),
            ModelSpec(key=_cache_key(params.vae_model, "vae"), loader=lambda: None, estimated_size_bytes=500_000_000),
        ]

    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> tuple[PIL.Image.Image, int]:
        import PIL.Image
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        device = cache._device
        dtype = torch.bfloat16

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Load processor (tokenizer) and text encoder (Mistral 3) ──
        def _resolve_sub(path_or_repo: str, subfolder: str):
            path = ModelLoader.resolve_path(path_or_repo)
            if path.is_dir():
                sub = path / subfolder
                if sub.exists():
                    return sub
            return path

        processor = cache.get_or_load(
            _cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"),
            lambda: AutoProcessor.from_pretrained(
                _resolve_sub(params.qwen3_tokenizer, "tokenizer"),
            ),
        )

        text_encoder = cache.get_or_load(
            _cache_key(params.qwen3_encoder, "qwen3_encoder"),
            lambda: Mistral3ForConditionalGeneration.from_pretrained(
                _resolve_sub(params.qwen3_encoder, "text_encoder"),
                torch_dtype=dtype,
            ),
        )

        # ── Load Flux2 Transformer ───────────────────────────────────
        def _load_transformer():
            from diffusers import Flux2Transformer2DModel
            path = ModelLoader.resolve_path(params.transformer_model)
            if path.is_file():
                return Flux2Transformer2DModel.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "transformer"
            if sub.exists():
                path = sub
            return Flux2Transformer2DModel.from_pretrained(str(path), torch_dtype=dtype)

        transformer = cache.get_or_load(
            _cache_key(params.transformer_model, "transformer"),
            _load_transformer,
        )

        # ── Load VAE (FLUX.2-specific with batch norm) ───────────────
        def _load_vae():
            from diffusers import AutoencoderKLFlux2
            path = ModelLoader.resolve_path(params.vae_model)
            if path.is_file():
                return AutoencoderKLFlux2.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "vae"
            if sub.exists():
                path = sub
            return AutoencoderKLFlux2.from_pretrained(str(path), torch_dtype=dtype)

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
                LoraApplicator.load_and_apply(transformer, lora_specs, TransformerType.FLUX2_DEV)

            image = self._generate(
                params=params,
                processor=processor,
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
        processor,
        text_encoder,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int = 512,
    ) -> torch.Tensor:
        """Encode text using Mistral 3 with multi-layer hidden state extraction.

        Returns prompt_embeds of shape (B, seq_len, num_layers * hidden_dim).
        """
        messages = [[
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            output = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Stack hidden states from extraction layers
        out = torch.stack([output.hidden_states[k] for k in HIDDEN_STATE_LAYERS], dim=1)
        out = out.to(dtype=dtype, device=device)

        # Reshape: (B, num_layers, seq, dim) → (B, seq, num_layers * dim)
        batch_size, num_layers, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )

        return prompt_embeds

    def _generate(
        self,
        *,
        params: JobParams,
        processor,
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
        prompt_embeds = self._encode_prompt(
            params.prompt, processor, text_encoder, device, dtype
        )

        # Text position IDs: (B, seq_len, 4) — T=0, H=0, W=0, L=0..seq_len-1
        text_ids = self._prepare_text_ids(prompt_embeds, device)

        # ── 2. Prepare latent noise in patchified space ──────────────
        vae_scale_factor = (
            2 ** (len(vae.config.block_out_channels) - 1)
            if hasattr(vae.config, "block_out_channels")
            else 8
        )
        num_channels = transformer.config.in_channels // 4

        latent_h = 2 * (params.height // (vae_scale_factor * 2))
        latent_w = 2 * (params.width // (vae_scale_factor * 2))

        # Noise in patchified space: (B, C*4, H//2, W//2)
        latents = torch.randn(
            (1, num_channels * 4, latent_h // 2, latent_w // 2),
            generator=generator,
            dtype=dtype,
            device="cpu",
        ).to(device)

        # Position IDs for the patchified grid
        latent_ids = self._prepare_latent_ids(latents, device)

        # Pack: (B, C, H, W) → (B, H*W, C)
        latents = self._pack_latents(latents)

        # ── 3. Time schedule with empirical mu ───────────────────────
        num_steps = params.steps
        image_seq_len = latents.shape[1]
        mu = self._compute_empirical_mu(image_seq_len, num_steps)
        sigmas = self._get_sigmas(mu, num_steps)

        # ── 4. Guidance embedding ────────────────────────────────────
        guidance = torch.full([1], params.cfg_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # ── 5. Denoising loop ────────────────────────────────────────
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
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

                # Euler step
                dt = sigma_next - sigma_curr
                latents = latents + dt * noise_pred

                progress_cb(ProgressEvent(
                    job_id="",
                    step=i + 1,
                    total_steps=num_steps,
                ))

        # ── 6. Unpack → BN denormalize → unpatchify → decode ────────
        latents = self._unpack_latents(latents, latent_h // 2, latent_w // 2)

        # BN denormalization using VAE batch norm stats
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean

        # Unpatchify: (B, C*4, H//2, W//2) → (B, C, H, W)
        latents = self._unpatchify_latents(latents)

        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _prepare_text_ids(
        prompt_embeds: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate 4D text position IDs: T=0, H=0, W=0, L=0..seq_len-1."""
        B, L, _ = prompt_embeds.shape
        out_ids = []
        for _ in range(B):
            coords = torch.cartesian_prod(
                torch.arange(1), torch.arange(1),
                torch.arange(1), torch.arange(L),
            )
            out_ids.append(coords)
        return torch.stack(out_ids).to(device)

    @staticmethod
    def _prepare_latent_ids(
        latents: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate 4D latent position IDs: T=0, H=0..h-1, W=0..w-1, L=0."""
        B, _, H, W = latents.shape
        latent_ids = torch.cartesian_prod(
            torch.arange(1), torch.arange(H),
            torch.arange(W), torch.arange(1),
        )
        return latent_ids.unsqueeze(0).expand(B, -1, -1).to(device)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack: (B, C, H, W) → (B, H*W, C)."""
        B, C, H, W = latents.shape
        return latents.reshape(B, C, H * W).permute(0, 2, 1)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Unpack: (B, H*W, C) → (B, C, H, W)."""
        B, _seq, C = latents.shape
        return latents.permute(0, 2, 1).reshape(B, C, h, w)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Unpatchify: (B, C*4, H//2, W//2) → (B, C, H, W)."""
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(B, C // 4, H * 2, W * 2)
        return latents

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 schedule shifting."""
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

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
