"""LoRA loading, format detection, and weight patching."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors.torch import load_file

from inference_engine.types import LoraFormat, TransformerType


class LoraApplicator:
    """Load, detect, and apply LoRA weight patches to transformer models."""

    # ── Public API ───────────────────────────────────────────────────

    @staticmethod
    def load_and_apply(
        model: torch.nn.Module,
        lora_specs: list[tuple[str, float]],  # (path, strength)
        model_type: TransformerType,
    ) -> None:
        """Load and apply one or more LoRAs to a model in-place.

        Each LoRA's weight deltas are applied as:
            weight += strength * scale * (up @ down)
        where scale = alpha / rank.
        """
        for lora_path, strength in lora_specs:
            logger.info(f"Applying LoRA: {lora_path} (strength={strength})")
            state_dict = LoraApplicator._load_state_dict(lora_path)
            fmt = LoraApplicator.detect_format(state_dict)
            logger.info(f"  Detected format: {fmt.value}")

            # Normalize keys to a common format: layer_name -> (up, down, alpha)
            lora_layers = LoraApplicator._parse_lora_layers(state_dict, fmt, model_type)

            applied = 0
            for layer_key, layer_data in lora_layers.items():
                applied += LoraApplicator._apply_layer(model, layer_key, layer_data, strength)

            logger.info(f"  Applied {applied} LoRA layers")

    @staticmethod
    def snapshot_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Save a copy of all model weights for later restoration."""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    @staticmethod
    def unapply(model: torch.nn.Module, original_state: dict[str, torch.Tensor]) -> None:
        """Restore model weights from a snapshot."""
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

    # ── Format detection ─────────────────────────────────────────────

    @staticmethod
    def detect_format(state_dict: dict[str, torch.Tensor]) -> LoraFormat:
        """Detect the LoRA format from key patterns in the state dict."""
        keys = set(state_dict.keys())
        sample_keys = list(keys)[:50]  # Check a sample for efficiency
        key_str = " ".join(sample_keys)

        # Kohya: keys like "lora_unet_double_blocks_0_..." or "lora_unet_single_blocks_..."
        if any(k.startswith("lora_unet_") for k in sample_keys):
            return LoraFormat.KOHYA

        # XLabs: keys contain specific patterns like "double_blocks.0.processor"
        if any("processor" in k and "double_blocks" in k for k in sample_keys):
            return LoraFormat.XLABS

        # AI Toolkit: keys like "transformer.double_blocks.0.lora_A.weight"
        # but with "transformer." prefix and direct lora_A/lora_B
        if any(k.startswith("transformer.") and ".lora_" in k for k in sample_keys):
            return LoraFormat.AITOOLKIT

        # OneTrainer: keys like "bundle_emb.string_to_param..." or specific onetrainer patterns
        if any("bundle_emb" in k or "lora_te" in k for k in sample_keys):
            return LoraFormat.ONETRAINER

        # Diffusers/PEFT: keys contain "lora_A.weight" and "lora_B.weight"
        if any("lora_A" in k or "lora_B" in k or "lora_down" in k or "lora_up" in k for k in sample_keys):
            return LoraFormat.DIFFUSERS

        # Check for diffusion_model prefix (used by some Z-Image LoRAs)
        if any(k.startswith("diffusion_model.") for k in sample_keys):
            return LoraFormat.DIFFUSERS

        return LoraFormat.UNKNOWN

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
        """Load a LoRA state dict from a file."""
        p = Path(path)
        if not p.exists():
            from inference_engine.models.loader import ModelLoader
            p = ModelLoader.resolve_single_file(path)

        if p.suffix == ".safetensors":
            return load_file(str(p))
        return torch.load(str(p), map_location="cpu", weights_only=True)

    @staticmethod
    def _parse_lora_layers(
        state_dict: dict[str, torch.Tensor],
        fmt: LoraFormat,
        model_type: TransformerType,
    ) -> dict[str, dict[str, Any]]:
        """Parse a LoRA state dict into per-layer data.

        Returns a dict mapping target layer names to:
            {"up": Tensor, "down": Tensor, "alpha": float | None}
        """
        if fmt == LoraFormat.KOHYA:
            return LoraApplicator._parse_kohya(state_dict, model_type)
        elif fmt == LoraFormat.DIFFUSERS:
            return LoraApplicator._parse_diffusers(state_dict, model_type)
        elif fmt == LoraFormat.XLABS:
            return LoraApplicator._parse_xlabs(state_dict, model_type)
        elif fmt == LoraFormat.AITOOLKIT:
            return LoraApplicator._parse_aitoolkit(state_dict, model_type)
        elif fmt == LoraFormat.ONETRAINER:
            return LoraApplicator._parse_diffusers(state_dict, model_type)  # Similar structure
        else:
            logger.warning(f"Unknown LoRA format, attempting diffusers-style parse")
            return LoraApplicator._parse_diffusers(state_dict, model_type)

    @staticmethod
    def _parse_kohya(
        state_dict: dict[str, torch.Tensor],
        model_type: TransformerType,
    ) -> dict[str, dict[str, Any]]:
        """Parse Kohya-format LoRA keys.

        Kohya keys: lora_unet_<layer_path>.lora_down.weight / .lora_up.weight / .alpha
        The layer path uses underscores where the model uses dots.
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            if not key.startswith("lora_unet_"):
                continue

            # Strip prefix and extract parts
            rest = key[len("lora_unet_"):]

            if rest.endswith(".lora_down.weight"):
                layer_key = rest[: -len(".lora_down.weight")]
                layers.setdefault(layer_key, {})["down"] = tensor
            elif rest.endswith(".lora_up.weight"):
                layer_key = rest[: -len(".lora_up.weight")]
                layers.setdefault(layer_key, {})["up"] = tensor
            elif rest.endswith(".alpha"):
                layer_key = rest[: -len(".alpha")]
                layers.setdefault(layer_key, {})["alpha"] = tensor.item()

        # Convert Kohya layer names to model parameter names
        # Kohya uses underscores, model uses dots
        # e.g., "double_blocks_0_img_attn_qkv" → "double_blocks.0.img_attn.qkv"
        converted: dict[str, dict[str, Any]] = {}
        for kohya_key, data in layers.items():
            model_key = LoraApplicator._kohya_key_to_model_key(kohya_key)
            converted[model_key] = data

        return converted

    @staticmethod
    def _parse_diffusers(
        state_dict: dict[str, torch.Tensor],
        model_type: TransformerType,
    ) -> dict[str, dict[str, Any]]:
        """Parse Diffusers/PEFT format LoRA keys.

        Keys: <prefix>.<layer_path>.lora_A.weight / .lora_B.weight
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            # Strip common prefixes
            clean_key = key
            for prefix in ("transformer.", "diffusion_model.", "unet."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break

            if ".lora_A." in clean_key or ".lora_down." in clean_key:
                # Extract layer name
                layer_key = re.sub(r"\.(lora_A|lora_down)\.(weight|default\.weight)$", "", clean_key)
                layers.setdefault(layer_key, {})["down"] = tensor
            elif ".lora_B." in clean_key or ".lora_up." in clean_key:
                layer_key = re.sub(r"\.(lora_B|lora_up)\.(weight|default\.weight)$", "", clean_key)
                layers.setdefault(layer_key, {})["up"] = tensor
            elif clean_key.endswith(".alpha"):
                layer_key = clean_key[: -len(".alpha")]
                layers.setdefault(layer_key, {})["alpha"] = tensor.item()

        return layers

    @staticmethod
    def _parse_xlabs(
        state_dict: dict[str, torch.Tensor],
        model_type: TransformerType,
    ) -> dict[str, dict[str, Any]]:
        """Parse XLabs-format LoRA keys.

        XLabs keys use "processor" in the path and have lora_A/lora_B suffixes.
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            if ".lora_A" in key or ".lora_a" in key:
                layer_key = re.sub(r"\.lora_[Aa]\.?.*$", "", key)
                # Convert XLabs key to model key
                layer_key = layer_key.replace(".processor", "")
                layers.setdefault(layer_key, {})["down"] = tensor
            elif ".lora_B" in key or ".lora_b" in key:
                layer_key = re.sub(r"\.lora_[Bb]\.?.*$", "", key)
                layer_key = layer_key.replace(".processor", "")
                layers.setdefault(layer_key, {})["up"] = tensor

        return layers

    @staticmethod
    def _parse_aitoolkit(
        state_dict: dict[str, torch.Tensor],
        model_type: TransformerType,
    ) -> dict[str, dict[str, Any]]:
        """Parse AI Toolkit format LoRA keys.

        Keys: transformer.<layer_path>.lora_A.weight / .lora_B.weight
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            clean_key = key
            if clean_key.startswith("transformer."):
                clean_key = clean_key[len("transformer."):]

            if ".lora_A." in clean_key:
                layer_key = clean_key.split(".lora_A.")[0]
                layers.setdefault(layer_key, {})["down"] = tensor
            elif ".lora_B." in clean_key:
                layer_key = clean_key.split(".lora_B.")[0]
                layers.setdefault(layer_key, {})["up"] = tensor
            elif clean_key.endswith(".alpha"):
                layer_key = clean_key[: -len(".alpha")]
                layers.setdefault(layer_key, {})["alpha"] = tensor.item()

        return layers

    @staticmethod
    def _apply_layer(
        model: torch.nn.Module,
        layer_key: str,
        layer_data: dict[str, Any],
        strength: float,
    ) -> int:
        """Apply a single LoRA layer to the model. Returns 1 if applied, 0 if skipped."""
        up = layer_data.get("up")
        down = layer_data.get("down")
        alpha = layer_data.get("alpha")

        if up is None or down is None:
            return 0

        # Find the target parameter
        # Try "layer_key.weight" first, then just "layer_key"
        target_param = None
        target_name = None
        for candidate in (f"{layer_key}.weight", layer_key):
            parts = candidate.split(".")
            try:
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                target_param = getattr(obj, parts[-1])
                target_name = candidate
                break
            except (AttributeError, IndexError):
                continue

        if target_param is None:
            return 0

        if not isinstance(target_param, (torch.nn.Parameter, torch.Tensor)):
            return 0

        # Compute scale: alpha / rank
        rank = down.shape[0]
        scale = (alpha / rank) if alpha is not None else 1.0

        # Compute delta: up @ down
        up = up.to(device=target_param.device, dtype=target_param.dtype)
        down = down.to(device=target_param.device, dtype=target_param.dtype)

        if up.dim() == 2 and down.dim() == 2:
            delta = up @ down
        elif up.dim() == 4 and down.dim() == 4:
            # Conv layers
            delta = torch.nn.functional.conv2d(
                down.permute(1, 0, 2, 3), up
            ).permute(1, 0, 2, 3)
        else:
            # Try simple matmul with reshape
            delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1))

        delta = delta.reshape(target_param.shape)
        target_param.data.add_(delta * (strength * scale))
        return 1

    @staticmethod
    def _kohya_key_to_model_key(kohya_key: str) -> str:
        """Convert a Kohya-style layer key to a model parameter path.

        Kohya uses underscores where the model uses dots and numeric indices.
        e.g., "double_blocks_0_img_attn_qkv" → "double_blocks.0.img_attn.qkv"
        """
        # Split on underscores, but numbers should become index separators
        parts = kohya_key.split("_")
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if part.isdigit():
                # Numeric index — use dot separator
                result.append(part)
            else:
                result.append(part)
            i += 1

        # Reconstruct with dots
        key = ".".join(result)

        # Fix known patterns:
        # "double.blocks" → "double_blocks"
        # "single.blocks" → "single_blocks"
        # "img.attn" → "img_attn"
        # "txt.attn" → "txt_attn"
        # "img.mlp" → "img_mlp"
        # "txt.mlp" → "txt_mlp"
        # "img.mod" → "img_mod"
        # "txt.mod" → "txt_mod"
        key = key.replace("double.blocks", "double_blocks")
        key = key.replace("single.blocks", "single_blocks")
        key = key.replace("img.attn", "img_attn")
        key = key.replace("txt.attn", "txt_attn")
        key = key.replace("img.mlp", "img_mlp")
        key = key.replace("txt.mlp", "txt_mlp")
        key = key.replace("img.mod", "img_mod")
        key = key.replace("txt.mod", "txt_mod")
        key = key.replace("final.layer", "final_layer")
        key = key.replace("time.in", "time_in")
        key = key.replace("vector.in", "vector_in")
        key = key.replace("guidance.in", "guidance_in")
        key = key.replace("linear.1", "linear_1")
        key = key.replace("linear.2", "linear_2")

        return key
