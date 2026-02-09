"""FLUX.2 Dev pipeline — Mistral3 text encoding, 32-channel VAE with BN, Flux2Transformer2DModel.

FLUX.2 differs from FLUX.1 in several key ways:
- Mistral3 replaces CLIP+T5; hidden states from layers (10, 20, 30) are stacked
- VAE has 32 latent channels with BatchNorm2d on the patchified representation
- 4D position IDs (T, H, W, L) via cartesian_prod, int64 dtype
- Noise is generated directly in patchified space (128 channels)
- BN denormalization replaces scaling_factor/shift_factor for VAE decode
- Empirical mu for schedule shifting (piecewise linear based on seq_len + steps)
"""

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
    from inference_engine.types import JobParams

# Hidden state layers to extract from Mistral3 and stack for conditioning
EXTRACTION_LAYERS = (10, 20, 30)


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


class Flux2DevPipeline(BasePipeline):
    """FLUX.2 Dev: Mistral3 encoder, Flux2Transformer2DModel, 32-ch VAE with BN.

    Models are loaded and released in stages to fit within VRAM:
    1. Text encoding (Mistral3) → produce embeddings → release encoder
    2. Denoising (transformer) → produce denoised latents → release transformer
    3. VAE decode → BN denorm + unpatchify + decode → release VAE
    """

    def validate_params(self, params: JobParams) -> None:
        # qwen3_* CLI params are reused for the Mistral3 text encoder
        required = {
            "qwen3_tokenizer": params.qwen3_tokenizer,
            "qwen3_encoder": params.qwen3_encoder,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"FLUX.2 Dev requires text encoder params: {', '.join(missing)}")

    def get_required_models(self, params: JobParams) -> list[ModelSpec]:
        return [
            ModelSpec(key=_cache_key(params.qwen3_encoder, "text_encoder"), loader=lambda: None, estimated_size_bytes=45_000_000_000),
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

        device = cache._device
        dtype = torch.bfloat16

        seed = params.seed if params.seed >= 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Stage 1: Text encoding (Mistral3) ────────────────────────
        # The Mistral3-24B text encoder is ~48 GB at bf16 — too large for VRAM.
        # We load it on CPU, encode on CPU, then move only the small embedding
        # tensors to GPU. The encoder is freed after encoding.
        logger.info("Loading text encoder on CPU (too large for VRAM)")
        tokenizer = self._load_tokenizer(params.qwen3_tokenizer)
        text_encoder = self._load_text_encoder(params.qwen3_encoder, dtype)

        prompt_embeds, pooled_embeds, text_ids = self._encode_text(
            params.prompt, tokenizer, text_encoder, torch.device("cpu"), dtype
        )
        # Free the encoder immediately — it's huge
        del text_encoder, tokenizer
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Move only the small embedding tensors to GPU
        prompt_embeds = prompt_embeds.to(device)
        pooled_embeds = pooled_embeds.to(device)
        text_ids = text_ids.to(device)

        # ── Prepare latent noise ──────────────────────────────────────
        latent_h = 2 * math.ceil(params.height / 16)
        latent_w = 2 * math.ceil(params.width / 16)
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        # Noise is generated directly in patchified space: (B, 128, packed_h, packed_w)
        latents = torch.randn(
            (1, 128, packed_h, packed_w),
            generator=generator, dtype=dtype, device="cpu",
        ).to(device)

        # Pack to sequence: (B, seq_len, 128)
        latents = self._pack(latents)

        # 4D position IDs for image tokens
        img_ids = self._generate_image_ids(packed_h, packed_w, device)

        # ── Stage 2: Denoising ────────────────────────────────────────
        def _load_transformer():
            from diffusers.models import Flux2Transformer2DModel
            path = ModelLoader.resolve_path(params.transformer_model)
            if path.is_file():
                kwargs = {"torch_dtype": dtype}
                config = ModelLoader.resolve_config(params.transformer_config, "flux2_transformer")
                if config:
                    kwargs["config"] = config
                if path.suffix == ".gguf":
                    from diffusers import GGUFQuantizationConfig
                    kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
                return Flux2Transformer2DModel.from_single_file(str(path), **kwargs)
            sub = path / "transformer"
            if sub.exists():
                path = sub
            return Flux2Transformer2DModel.from_pretrained(str(path), torch_dtype=dtype)

        transformer = cache.get_or_load(
            _cache_key(params.transformer_model, "transformer"),
            _load_transformer,
        )
        cache.lock(_cache_key(params.transformer_model, "transformer"))

        try:
            original_state = None
            if params.loras:
                lora_specs = [(lp.model_file, lp.strength) for lp in params.loras]
                original_state = LoraApplicator.snapshot_weights(transformer)
                LoraApplicator.load_and_apply(transformer, lora_specs, TransformerType.FLUX2_DEV)

            latents = self._denoise(
                latents=latents,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                img_ids=img_ids,
                transformer=transformer,
                image_seq_len=image_seq_len,
                params=params,
                device=device,
                dtype=dtype,
                progress_cb=progress_cb,
            )

            if original_state is not None:
                LoraApplicator.unapply(transformer, original_state)
        finally:
            cache.unlock(_cache_key(params.transformer_model, "transformer"))

        del prompt_embeds, pooled_embeds, text_ids, img_ids

        # ── Stage 3: BN denorm + unpatchify + VAE decode ──────────────
        def _load_vae():
            path = ModelLoader.resolve_path(params.vae_model)
            # Try Flux2 VAE (has BatchNorm) first
            try:
                from diffusers import AutoencoderKLFlux2
                if path.is_file():
                    return AutoencoderKLFlux2.from_single_file(str(path), torch_dtype=dtype)
                sub = path / "vae"
                if sub.exists():
                    path = sub
                return AutoencoderKLFlux2.from_pretrained(str(path), torch_dtype=dtype)
            except (ImportError, Exception) as e:
                logger.debug(f"AutoencoderKLFlux2 not available, trying AutoencoderKL: {e}")
            # Fallback to standard VAE
            from diffusers import AutoencoderKL
            path = ModelLoader.resolve_path(params.vae_model)
            if path.is_file():
                cfg = ModelLoader.resolve_config(params.vae_config, "vae")
                if cfg:
                    return AutoencoderKL.from_single_file(str(path), config=cfg, torch_dtype=dtype)
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
            image = self._decode(latents, vae, packed_h, packed_w)
        finally:
            cache.unlock(_cache_key(params.vae_model, "vae"))

        return image, seed

    # ── Denoising ─────────────────────────────────────────────────────

    def _denoise(
        self,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        transformer,
        image_seq_len: int,
        params: JobParams,
        device: torch.device,
        dtype: torch.dtype,
        progress_cb: Callable[[ProgressEvent], None],
    ) -> torch.Tensor:
        """Rectified flow Euler denoising loop."""
        num_steps = params.steps

        # Sigma schedule with empirical mu and time shifting
        mu = self._compute_empirical_mu(image_seq_len, num_steps)
        sigmas = self._get_sigmas(mu, num_steps)

        # Guidance embedding (FLUX.2 uses guidance-distilled approach like FLUX.1)
        guidance = torch.tensor([params.cfg_scale], device=device, dtype=dtype)
        guidance = guidance.expand(latents.shape[0])

        with torch.no_grad():
            for i in range(num_steps):
                sigma_curr = sigmas[i]
                sigma_next = sigmas[i + 1]

                # Pass sigma directly — transformer multiplies by 1000 internally
                timestep = torch.tensor([sigma_curr], device=device, dtype=dtype)
                timestep = timestep.expand(latents.shape[0])

                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                # Euler step
                dt = sigma_next - sigma_curr
                latents = latents + dt * noise_pred

                progress_cb(ProgressEvent(
                    job_id="", step=i + 1, total_steps=num_steps,
                ))

        return latents

    # ── VAE decode ────────────────────────────────────────────────────

    @staticmethod
    def _decode(
        latents: torch.Tensor,
        vae,
        packed_h: int,
        packed_w: int,
    ) -> PIL.Image.Image:
        """Unpack → BN denormalize → unpatchify → VAE decode."""
        import PIL.Image

        # Unpack from sequence: (B, seq, 128) → (B, 128, H, W)
        latents = Flux2DevPipeline._unpack(latents, packed_h, packed_w)

        # BN denormalization: reverse the VAE encoder's batch norm
        bn = getattr(vae, "bn", None)
        if bn is not None and bn.running_mean is not None:
            eps = getattr(bn, "eps", 1e-5)
            bn_mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            bn_std = torch.sqrt(bn.running_var + eps).view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents = latents * bn_std + bn_mean
        else:
            # Fallback: use scaling_factor/shift_factor if available
            logger.warning("VAE has no batch norm — falling back to scaling_factor/shift_factor")
            sf = getattr(vae.config, "scaling_factor", 1.0)
            sh = getattr(vae.config, "shift_factor", 0.0)
            latents = (latents / sf) + sh

        # Unpatchify: (B, 128, H/2, W/2) → (B, 32, H, W)
        latents = Flux2DevPipeline._unpatchify(latents)

        # VAE decode
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).round().astype("uint8")
        return PIL.Image.fromarray(decoded)

    # ── Text encoding ────────────────────────────────────────────────

    @staticmethod
    def _load_tokenizer(path_or_repo: str):
        """Load tokenizer. Handles HF repos, local dirs, and GGUF repo/file paths."""
        from transformers import AutoTokenizer

        # For repo/file.gguf format, load tokenizer from the repo (not the file)
        parts = path_or_repo.split("/")
        if len(parts) >= 3 and not path_or_repo.startswith("/"):
            repo_id = "/".join(parts[:2])
            try:
                return AutoTokenizer.from_pretrained(repo_id)
            except Exception:
                pass

        # For plain HF repos (org/repo), load directly — avoids downloading
        # the entire model just to get tokenizer files
        if len(parts) == 2 and not path_or_repo.startswith("/"):
            return AutoTokenizer.from_pretrained(path_or_repo)

        path = ModelLoader.resolve_path(path_or_repo)
        if path.is_dir():
            sub = path / "tokenizer"
            if sub.exists():
                return AutoTokenizer.from_pretrained(str(sub))
            return AutoTokenizer.from_pretrained(str(path))

        # Single file — try parent directory (may have tokenizer files)
        return AutoTokenizer.from_pretrained(str(path.parent))

    @staticmethod
    def _load_text_encoder(path_or_repo: str, dtype: torch.dtype):
        """Load text encoder (Mistral3 or compatible causal LM) on CPU.

        The 24B text encoder is too large for VRAM (~48 GB at bf16).
        Always loads to CPU; encoding runs on CPU, embeddings move to GPU.
        """
        from transformers import AutoModelForCausalLM

        path = ModelLoader.resolve_path(path_or_repo)

        if path.is_file() and path.suffix == ".gguf":
            # GGUF text encoder — load via repo + filename
            parts = path_or_repo.split("/")
            if len(parts) >= 3 and not path_or_repo.startswith("/"):
                repo_id = "/".join(parts[:2])
                filename = "/".join(parts[2:])
                return AutoModelForCausalLM.from_pretrained(
                    repo_id, gguf_file=filename,
                    torch_dtype=dtype, device_map="cpu",
                    low_cpu_mem_usage=True,
                )
            return AutoModelForCausalLM.from_pretrained(
                str(path.parent), gguf_file=path.name,
                torch_dtype=dtype, device_map="cpu",
                low_cpu_mem_usage=True,
            )

        if path.is_dir():
            sub = path / "text_encoder"
            if sub.exists():
                return AutoModelForCausalLM.from_pretrained(
                    str(sub), torch_dtype=dtype, device_map="cpu",
                    low_cpu_mem_usage=True,
                )
            return AutoModelForCausalLM.from_pretrained(
                str(path), torch_dtype=dtype, device_map="cpu",
                low_cpu_mem_usage=True,
            )

        # Single non-GGUF file — try parent directory
        return AutoModelForCausalLM.from_pretrained(
            str(path.parent), torch_dtype=dtype, device_map="cpu",
            low_cpu_mem_usage=True,
        )

    @staticmethod
    def _encode_text(
        prompt: str,
        tokenizer,
        model,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text using multi-layer extraction (layers 10, 20, 30).

        Returns:
            prompt_embeds: (1, seq_len, hidden_dim * 3) — stacked hidden states
            pooled_embeds: (1, hidden_dim) — mean-pooled last hidden state
            text_ids: (1, seq_len, 4) — 4D position IDs for text tokens
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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
                use_cache=False,
            )

        # Stack hidden states from extraction layers
        num_hidden = len(outputs.hidden_states)
        layers = []
        for idx in EXTRACTION_LAYERS:
            layer_idx = min(idx, num_hidden - 1)
            layers.append(outputs.hidden_states[layer_idx])

        # (B, 3, seq_len, hidden_dim) → (B, seq_len, 3 * hidden_dim)
        stacked = torch.stack(layers, dim=1).to(dtype)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )

        # Pooled embedding: mean over non-padding tokens of last hidden state
        last_hs = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.to(dtype)

        # Text position IDs: 4D (T=0, H=0, W=0, L=0..seq_len-1)
        text_ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=torch.long)
        text_ids[..., 3] = torch.arange(seq_len, device=device, dtype=torch.long)

        return prompt_embeds, pooled, text_ids

    # ── Patchify / Unpatchify ────────────────────────────────────────

    @staticmethod
    def _patchify(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C*4, H/2, W/2) — interleave spatial patches into channels."""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)  # (B, C, 2, 2, H/2, W/2)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
        """(B, C*4, H/2, W/2) → (B, C, H, W) — reverse of patchify."""
        b, c4, h, w = latents.shape
        c = c4 // 4
        latents = latents.reshape(b, c, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)  # (B, C, H, 2, W, 2)
        return latents.reshape(b, c, h * 2, w * 2)

    # ── Pack / Unpack ────────────────────────────────────────────────

    @staticmethod
    def _pack(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H*W, C) — flatten spatial to sequence."""
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).permute(0, 2, 1)

    @staticmethod
    def _unpack(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, H*W, C) → (B, C, H, W) — restore spatial layout."""
        b, _seq, c = latents.shape
        return latents.permute(0, 2, 1).reshape(b, c, h, w)

    # ── Position IDs ─────────────────────────────────────────────────

    @staticmethod
    def _generate_image_ids(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generate 4D position IDs (T, H, W, L) for image tokens.

        Must be int64 — float causes NaN in rotary embeddings.
        """
        t = torch.arange(1, device=device, dtype=torch.long)   # [0]
        ys = torch.arange(h, device=device, dtype=torch.long)
        xs = torch.arange(w, device=device, dtype=torch.long)
        l = torch.arange(1, device=device, dtype=torch.long)   # [0]
        ids = torch.cartesian_prod(t, ys, xs, l)  # (h*w, 4)
        return ids.unsqueeze(0)  # (1, h*w, 4)

    # ── Schedule helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 schedule shifting.

        Piecewise linear based on image sequence length and step count.
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
    def _time_shift(mu: float, sigma: float, t: float) -> float:
        """Apply exponential time shift. Returns 0 at t=0, 1 at t=1."""
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    @staticmethod
    def _get_sigmas(mu: float, num_steps: int) -> list[float]:
        """Generate time-shifted sigma schedule for FLUX.2.

        Base: linspace(1.0, 1/num_steps, num_steps) + final 0.
        Then apply exponential time shift with empirical mu.
        """
        import numpy as np
        raw = list(np.linspace(1.0, 1.0 / num_steps, num_steps)) + [0.0]
        return [Flux2DevPipeline._time_shift(mu, 1.0, s) for s in raw]
