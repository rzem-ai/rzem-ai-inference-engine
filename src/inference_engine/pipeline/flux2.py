"""FLUX.2 Dev pipeline — Qwen3 multi-layer encoding, 32-channel VAE, BN normalization.

FLUX.2 Klein differs from FLUX.1 in several key ways:
- Qwen3 replaces CLIP+T5; hidden states from layers (9, 18, 27) are stacked
- VAE has 32 latent channels with batch normalization on the packed representation
- 4D position IDs (T, H, W, L) with int64 dtype
- Empirical mu computation for schedule shifting
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

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

# Hidden state layers to extract and stack for Klein-style encoding
KLEIN_EXTRACTION_LAYERS = (9, 18, 27)


def _cache_key(path: str, role: str) -> str:
    return f"{path}::{role}"


class Flux2DevPipeline(BasePipeline):
    """FLUX.2 Dev: Qwen3 encoder, Flux2Transformer2DModel, 32-ch VAE with BN.

    Differences from FLUX.1:
    - Qwen3 replaces CLIP+T5 — stacks layers (9, 18, 27) for richer encoding
    - VAE has 32 latent channels with batch normalization
    - 4D position IDs (T, H, W, L) with int64 dtype
    - Empirical mu for schedule shifting
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
            ModelSpec(key=_cache_key(params.qwen3_encoder, "qwen3"), loader=lambda: None, estimated_size_bytes=8_000_000_000),
            ModelSpec(key=_cache_key(params.transformer_model, "transformer"), loader=lambda: None, estimated_size_bytes=23_000_000_000),
            ModelSpec(key=_cache_key(params.vae_model, "vae"), loader=lambda: None, estimated_size_bytes=200_000_000),
        ]

    def run(
        self,
        params: JobParams,
        cache: ModelCache,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> tuple[PIL.Image.Image, int]:
        import PIL.Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            lambda: AutoModelForCausalLM.from_pretrained(
                _resolve_sub(params.qwen3_encoder, "text_encoder"),
                torch_dtype=dtype,
            ),
        )

        # ── Load Flux2 Transformer ───────────────────────────────────
        def _load_transformer():
            from diffusers import FluxTransformer2DModel
            path = ModelLoader.resolve_path(params.transformer_model)
            if path.is_file():
                return FluxTransformer2DModel.from_single_file(str(path), torch_dtype=dtype)
            sub = path / "transformer"
            if sub.exists():
                path = sub
            return FluxTransformer2DModel.from_pretrained(str(path), torch_dtype=dtype)

        transformer = cache.get_or_load(
            _cache_key(params.transformer_model, "transformer"),
            _load_transformer,
        )

        # ── Load VAE (32 latent channels) ────────────────────────────
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
                LoraApplicator.load_and_apply(transformer, lora_specs, TransformerType.FLUX2_DEV)

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

        # ── 1. Text encoding via Qwen3 (Klein-style multi-layer) ────
        prompt_embeds, pooled_prompt_embeds = self._encode_qwen3_klein(
            params.prompt, qwen3_tok, qwen3_enc, device, dtype
        )

        # ── 2. Prepare latent noise (32 channels for FLUX.2) ────────
        latent_channels = 32
        latent_h = math.ceil(params.height / 16) * 2
        latent_w = math.ceil(params.width / 16) * 2

        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator,
            dtype=dtype,
            device="cpu",
        ).to(device)

        # ── 3. Time schedule with empirical mu ───────────────────────
        num_steps = params.steps
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        mu = self._compute_empirical_mu(image_seq_len, num_steps)
        sigmas = self._get_schedule(num_steps)

        # ── 4. Pack latents ──────────────────────────────────────────
        latents = self._pack_latents(latents)

        # ── 5. BN normalization (if VAE has batch norm) ──────────────
        bn_mean, bn_std = self._get_bn_stats(vae)
        if bn_mean is not None and bn_std is not None:
            latents = self._bn_normalize(latents, bn_mean, bn_std)

        # ── 6. Position IDs (4D: T, H, W, L — int64) ────────────────
        img_ids = self._generate_image_ids_4d(latent_h, latent_w, device)

        # Text position IDs: 4D, only L coordinate varies
        seq_len = prompt_embeds.shape[1]
        txt_ids = torch.zeros(1, seq_len, 4, device=device, dtype=torch.long)
        txt_ids[..., 3] = torch.arange(seq_len, device=device, dtype=torch.long)

        # ── 7. Denoising ─────────────────────────────────────────────
        with torch.no_grad():
            for i in range(num_steps):
                t_curr = sigmas[i]
                t_next = sigmas[i + 1]
                timestep = torch.tensor([t_curr], device=device, dtype=dtype).expand(latents.shape[0]) * 1000.0

                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                if not math.isclose(params.cfg_scale, 1.0):
                    noise_pred = noise_pred * params.cfg_scale

                dt = t_next - t_curr
                latents = latents + dt * noise_pred

                progress_cb(ProgressEvent(
                    job_id="",
                    step=i + 1,
                    total_steps=num_steps,
                ))

        # ── 8. BN denormalization ────────────────────────────────────
        if bn_mean is not None and bn_std is not None:
            latents = self._bn_denormalize(latents, bn_mean, bn_std)

        # ── 9. Unpack and decode ─────────────────────────────────────
        latents = self._unpack_latents(latents, latent_h, latent_w, latent_channels)

        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)

    # ── Text encoding ────────────────────────────────────────────────

    @staticmethod
    def _encode_qwen3_klein(
        prompt: str,
        tokenizer,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text using Qwen3 with Klein-style multi-layer extraction.

        Extracts hidden states from layers (9, 18, 27), stacks and reshapes
        to (batch, seq_len, hidden_size * 3). Also produces a mean-pooled
        embedding for global conditioning.

        Returns (stacked_embeds, pooled_embeds).
        """
        # Apply chat template — Klein disables thinking
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (AttributeError, TypeError):
            text = prompt

        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

        num_hidden_layers = len(outputs.hidden_states)

        # Stack hidden states from extraction layers
        hidden_states_list = []
        for layer_idx in KLEIN_EXTRACTION_LAYERS:
            idx = min(layer_idx, num_hidden_layers - 1)
            hidden_states_list.append(outputs.hidden_states[idx])

        # Stack: (batch, num_layers, seq_len, hidden_dim)
        out = torch.stack(hidden_states_list, dim=1)
        out = out.to(dtype=dtype, device=device)

        # Reshape: (batch, seq_len, hidden_dim * num_layers)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )

        # Pooled embedding: mean over non-padding tokens of last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        expanded_mask = attention_mask.expand_as(last_hidden_state)
        sum_embeds = (last_hidden_state * expanded_mask).sum(dim=1)
        num_tokens = expanded_mask.sum(dim=1).clamp(min=1)
        pooled_embeds = (sum_embeds / num_tokens).to(dtype)

        return prompt_embeds, pooled_embeds

    # ── BN normalization ─────────────────────────────────────────────

    @staticmethod
    def _get_bn_stats(vae) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extract batch norm statistics from the VAE if available."""
        bn_layer = None
        for attr in ("bn", "batch_norm"):
            if hasattr(vae, attr):
                bn_layer = getattr(vae, attr)
                break
            if hasattr(vae, "encoder") and hasattr(vae.encoder, attr):
                bn_layer = getattr(vae.encoder, attr)
                break

        if bn_layer is None or bn_layer.running_mean is None or bn_layer.running_var is None:
            return None, None

        bn_mean = bn_layer.running_mean.clone()
        bn_var = bn_layer.running_var.clone()
        bn_eps = getattr(bn_layer, "eps", 1e-4)
        bn_std = torch.sqrt(bn_var + bn_eps)
        return bn_mean, bn_std

    @staticmethod
    def _bn_normalize(x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
        """Apply BN normalization: y = (x - mean) / std."""
        bn_mean = bn_mean.to(x.device, x.dtype)
        bn_std = bn_std.to(x.device, x.dtype)
        return (x - bn_mean) / bn_std

    @staticmethod
    def _bn_denormalize(x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
        """Inverse BN: x = y * std + mean."""
        bn_mean = bn_mean.to(x.device, x.dtype)
        bn_std = bn_std.to(x.device, x.dtype)
        return x * bn_std + bn_mean

    # ── Schedule helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 schedule shifting.

        Matches diffusers' Flux2KleinPipeline implementation.
        """
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
    def _get_schedule(num_steps: int) -> list[float]:
        """Get linear timestep schedule for FLUX.2.

        Returns sigmas from 1.0 to 1/num_steps, plus final 0.0.
        The actual shifting is applied by the scheduler via mu.
        """
        import numpy as np
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        sigmas_list = [float(s) for s in sigmas]
        sigmas_list.append(0.0)
        return sigmas_list

    # ── Latent helpers ───────────────────────────────────────────────

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack: (B, C, H, W) → (B, H/2*W/2, C*4)."""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, h: int, w: int, c: int = 32) -> torch.Tensor:
        """Unpack: (B, seq, C*4) → (B, C, H, W)."""
        b, _seq, d = latents.shape
        packed_h = h // 2
        packed_w = w // 2
        latents = latents.reshape(b, packed_h, packed_w, c, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(b, c, h, w)
        return latents

    @staticmethod
    def _generate_image_ids_4d(
        latent_h: int, latent_w: int, device: torch.device,
    ) -> torch.Tensor:
        """Generate 4D position IDs (T, H, W, L) with int64 dtype for FLUX.2.

        IMPORTANT: Must be int64, not float. Using float causes NaN in rotary embeddings.
        """
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        img_ids = torch.zeros(packed_h, packed_w, 4, device=device, dtype=torch.long)
        img_ids[..., 1] = torch.arange(packed_h, device=device, dtype=torch.long)[:, None]
        img_ids[..., 2] = torch.arange(packed_w, device=device, dtype=torch.long)[None, :]

        return img_ids.reshape(1, packed_h * packed_w, 4)
