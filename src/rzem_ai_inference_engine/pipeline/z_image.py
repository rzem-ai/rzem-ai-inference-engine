"""Z-Image pipeline — S3-DiT single-stream architecture with Qwen3-4B encoder.

Z-Image uses a different transformer API from FLUX:
- Forward call: transformer(x=list_of_tensors, t=timestep, cap_feats=list_of_embeds)
- Input x: list of [C, 1, H, W] tensors (one per batch, with frame dim)
- Output: tuple where [0] is a list of [C, 1, H, W] tensors
- V-prediction with negation: noise_pred = -model_output
- Timestep convention: model_t = 1 - sigma (0=noise, 1=clean)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
from loguru import logger

from rzem_ai_inference_engine.pipeline.lora_applicator import LoraApplicator
from rzem_ai_inference_engine.models.loader import ModelLoader
from rzem_ai_inference_engine.models.memory import preferred_dtype
from rzem_ai_inference_engine.pipeline.base import BasePipeline
from rzem_ai_inference_engine.types import ModelSpec, ProgressEvent, TransformerType

if TYPE_CHECKING:
    import PIL.Image

    from rzem_ai_inference_engine.models.cache import ModelCache
    from rzem_ai_inference_engine.types import JobParams, PreviewConfig


def _cache_key(path: str, role: str) -> str:
    return f"{path}::{role}"


class ZImagePipeline(BasePipeline):
    """Z-Image: S3-DiT single-stream transformer, Qwen3-4B encoder, FLUX-style 16-ch VAE.

    Turbo defaults: 9 steps, cfg_scale=1.0.
    """

    def validate_params(self, params: JobParams) -> None:
        required = {
            "qwen3_tokenizer": params.qwen3_tokenizer,
            "qwen3_encoder": params.qwen3_encoder,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"Z-Image requires: {', '.join(missing)}")

    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        return [
            ModelSpec(key=_cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"), loader=lambda: None, estimated_size_bytes=0),
            ModelSpec(key=_cache_key(params.qwen3_encoder, "qwen3_encoder"), loader=lambda: None, estimated_size_bytes=8_000_000_000),
            ModelSpec(key=_cache_key(params.transformer_model, "transformer"), loader=lambda: None, estimated_size_bytes=12_000_000_000),
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
        from transformers import AutoTokenizer, Qwen3Model

        device = cache._device
        dtype = preferred_dtype(device)

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Load Qwen3 tokenizer + encoder ───────────────────────────
        def _resolve_sub(path_or_repo: str, subfolder: str):
            path = ModelLoader.resolve_path(path_or_repo, subfolder=subfolder)
            if path.is_dir():
                sub = path / subfolder
                if sub.exists():
                    return sub
            return path

        # Lock each model immediately after loading to prevent eviction
        # during subsequent loads (earlier models could be evicted to make
        # room for later ones, leaving stale CPU references).
        keys = []
        try:
            qwen3_tok = cache.get_or_load(
                _cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"),
                lambda: AutoTokenizer.from_pretrained(
                    _resolve_sub(params.qwen3_tokenizer, "tokenizer"),
                ),
            )
            keys.append(_cache_key(params.qwen3_tokenizer, "qwen3_tokenizer"))
            cache.lock(keys[-1])

            qwen3_enc = cache.get_or_load(
                _cache_key(params.qwen3_encoder, "qwen3_encoder"),
                lambda: Qwen3Model.from_pretrained(
                    _resolve_sub(params.qwen3_encoder, "text_encoder"),
                    dtype=dtype,
                ),
            )
            keys.append(_cache_key(params.qwen3_encoder, "qwen3_encoder"))
            cache.lock(keys[-1])

            # ── Load Z-Image Transformer ────────────────────────────
            def _load_transformer():
                from diffusers.models import ZImageTransformer2DModel
                path = ModelLoader.resolve_path(params.transformer_model, subfolder="transformer")
                if path.is_file():
                    return ZImageTransformer2DModel.from_single_file(str(path), torch_dtype=dtype)
                sub = path / "transformer"
                if sub.exists():
                    path = sub
                return ZImageTransformer2DModel.from_pretrained(str(path), torch_dtype=dtype)

            transformer = cache.get_or_load(
                _cache_key(params.transformer_model, "transformer"),
                _load_transformer,
            )
            keys.append(_cache_key(params.transformer_model, "transformer"))
            cache.lock(keys[-1])

            # ── Load VAE (FLUX-style 16 channels) ───────────────────
            def _load_vae():
                from diffusers import AutoencoderKL
                path = ModelLoader.resolve_path(params.vae_model, subfolder="vae")
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
            keys.append(_cache_key(params.vae_model, "vae"))
            cache.lock(keys[-1])

            lora_hooks = []
            if params.loras:
                lora_specs = [(lp.model_file, lp.strength) for lp in params.loras]
                lora_hooks = LoraApplicator.load_and_apply(transformer, lora_specs, TransformerType.Z_IMAGE)

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
                preview_config=preview_config,
            )

            LoraApplicator.remove_hooks(lora_hooks)
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
        preview_config: PreviewConfig | None = None,
    ) -> PIL.Image.Image:
        import PIL.Image

        # ── 1. Text encoding via Qwen3 (Z-Image style) ──────────────
        # Z-Image uses hidden_states[-2], filters by attention mask,
        # and returns a 2D tensor [seq_len, hidden_dim] (no batch dim)
        prompt_embeds = self._encode_qwen3_zimage(
            params.prompt, qwen3_tok, qwen3_enc, device, dtype
        )

        # Negative prompt for CFG
        do_cfg = not math.isclose(params.cfg_scale, 1.0)
        neg_prompt_embeds = None
        if do_cfg:
            neg_prompt_embeds = self._encode_qwen3_zimage(
                "", qwen3_tok, qwen3_enc, device, dtype
            )

        # ── 2. Prepare latent noise (16 channels) ───────────────────
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
        img_token_h = (latent_h) // patch_size
        img_token_w = (latent_w) // patch_size
        img_seq_len = img_token_h * img_token_w

        mu = self._calculate_shift(img_seq_len)
        sigmas = self._get_sigmas(mu, num_steps)

        # Preview settings
        want_previews = preview_config is not None and preview_config.enabled
        preview_interval = preview_config.interval if preview_config else 5
        preview_max_size = preview_config.max_size if preview_config else 256

        # Emit initial noise preview (step 0)
        if want_previews:
            try:
                noise_preview = self._generate_preview(latents, vae, preview_max_size)
            except Exception:
                noise_preview = None
            progress_cb(ProgressEvent(
                job_id="", step=0, total_steps=num_steps,
                preview_image=noise_preview,
            ))

        # ── 4. Denoising loop (Z-Image Euler) ────────────────────────
        with torch.no_grad():
            for step_idx in range(num_steps):
                sigma_curr = sigmas[step_idx]
                sigma_prev = sigmas[step_idx + 1]

                # Z-Image timestep convention: model expects t=0 at start (noise), t=1 at end (clean)
                # sigma goes from 1 (noise) to 0 (clean), so model_t = 1 - sigma
                model_t = 1.0 - sigma_curr
                timestep = torch.tensor([model_t], device=device, dtype=dtype).expand(latents.shape[0])

                # Prepare input: [B, C, H, W] → [B, C, 1, H, W] → list of [C, 1, H, W]
                latent_input = latents.to(transformer.dtype)
                latent_input = latent_input.unsqueeze(2)  # add frame dimension
                latent_input_list = list(latent_input.unbind(dim=0))

                # Forward pass: Z-Image API is (x, t, cap_feats)
                model_output = transformer(
                    x=latent_input_list,
                    t=timestep,
                    cap_feats=[prompt_embeds],
                )
                model_out_list = model_output[0]  # extract list of tensors

                # Stack outputs, remove frame dim, negate (v-prediction)
                noise_pred_cond = torch.stack([t.float() for t in model_out_list], dim=0)
                noise_pred_cond = noise_pred_cond.squeeze(2)  # remove frame dim
                noise_pred_cond = -noise_pred_cond  # v-prediction with negation

                # CFG
                if do_cfg and neg_prompt_embeds is not None:
                    model_output_uncond = transformer(
                        x=latent_input_list,
                        t=timestep,
                        cap_feats=[neg_prompt_embeds],
                    )
                    model_out_list_uncond = model_output_uncond[0]
                    noise_pred_uncond = torch.stack([t.float() for t in model_out_list_uncond], dim=0)
                    noise_pred_uncond = noise_pred_uncond.squeeze(2)
                    noise_pred_uncond = -noise_pred_uncond
                    noise_pred = noise_pred_uncond + params.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Euler step (in float32 for precision)
                latents_dtype = latents.dtype
                latents = latents.to(dtype=torch.float32)
                latents = latents + (sigma_prev - sigma_curr) * noise_pred
                latents = latents.to(dtype=latents_dtype)

                # Build progress event, optionally with preview
                step = step_idx + 1
                preview_image = None
                if want_previews and step % preview_interval == 0 and step < num_steps:
                    try:
                        preview_image = self._generate_preview(latents, vae, preview_max_size)
                    except Exception:
                        logger.opt(exception=True).debug("Preview generation failed")

                progress_cb(ProgressEvent(
                    job_id="",
                    step=step,
                    total_steps=num_steps,
                    preview_image=preview_image,
                ))

        # ── 5. VAE decode ────────────────────────────────────────────
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
    def _encode_qwen3_zimage(
        prompt: str,
        tokenizer,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode text using Qwen3 for Z-Image.

        Returns a 2D tensor [valid_seq_len, hidden_dim] — only non-padding tokens,
        with no batch dimension. Uses hidden_states[-2] and enable_thinking=True.
        """
        # Apply chat template with thinking enabled (Z-Image convention)
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except (AttributeError, TypeError):
            text = prompt

        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # Use second-to-last hidden state
        prompt_embeds = outputs.hidden_states[-2]

        # Filter to only valid (non-padding) tokens → 2D tensor [seq_len, hidden_dim]
        mask = inputs["attention_mask"][0].bool()
        prompt_embeds = prompt_embeds[0][mask]

        return prompt_embeds.to(dtype)

    # ── Schedule helpers ─────────────────────────────────────────────

    @staticmethod
    def _calculate_shift(image_seq_len: int) -> float:
        """Calculate mu for time shift based on image sequence length."""
        # Linear interpolation between two anchor points, matching InvokeAI
        base_shift = 0.5
        max_shift = 1.15
        m = (max_shift - base_shift) / (4096 - 256)
        b = base_shift - m * 256
        return m * image_seq_len + b

    @staticmethod
    def _time_shift(mu: float, sigma: float, t: float) -> float:
        """Apply exponential time shift. Returns 0 at t=0, 1 at t=1."""
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    @staticmethod
    def _get_sigmas(mu: float, num_steps: int) -> list[float]:
        """Generate time-shifted sigma schedule from 1 → 0."""
        sigmas = []
        for i in range(num_steps + 1):
            t = 1.0 - i / num_steps  # 1.0 → 0.0
            sigma = ZImagePipeline._time_shift(mu, 1.0, t)
            sigmas.append(sigma)
        return sigmas
