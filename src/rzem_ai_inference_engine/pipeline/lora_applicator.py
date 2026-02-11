"""LoRA loading, format detection, and forward-hook application."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file
from torch.utils.hooks import RemovableHandle

from rzem_ai_inference_engine.models.memory import preferred_dtype
from rzem_ai_inference_engine.types import LoraFormat, TransformerType

# ── Parsing tree for OneTrainer underscore→dot conversion ─────────────
# Defines valid diffusers FluxTransformer2DModel parameter paths.
# _match_tree() greedily matches underscore-separated tokens against this.

_INDEX = "__INDEX__"

_FLUX1_DIFFUSERS_TREE: dict = {
    "transformer_blocks": {
        _INDEX: {
            "attn": {
                "to_q": {}, "to_k": {}, "to_v": {},
                "to_out": {_INDEX: {}},
                "add_q_proj": {}, "add_k_proj": {}, "add_v_proj": {},
                "to_add_out": {},
                "norm_q": {}, "norm_k": {},
                "norm_added_q": {}, "norm_added_k": {},
            },
            "ff": {"net": {_INDEX: {"proj": {}}}},
            "ff_context": {"net": {_INDEX: {"proj": {}}}},
            "norm1": {"linear": {}},
            "norm1_context": {"linear": {}},
        }
    },
    "single_transformer_blocks": {
        _INDEX: {
            "attn": {
                "to_q": {}, "to_k": {}, "to_v": {},
                "norm_q": {}, "norm_k": {},
            },
            "norm": {"linear": {}},
            "proj_mlp": {},
            "proj_out": {},
        }
    },
    "time_text_embed": {
        "timestep_embedder": {"linear_1": {}, "linear_2": {}},
        "text_embedder": {"linear_1": {}, "linear_2": {}},
        "guidance_embedder": {"linear_1": {}, "linear_2": {}},
    },
    "x_embedder": {},
    "context_embedder": {},
    "norm_out": {"linear": {}},
    "proj_out": {},
}


def _match_tree(parts: list[str], idx: int, tree: dict) -> str | None:
    """Greedily match underscore-separated parts against a parsing tree.

    Returns the dotted key if a complete match is found, None otherwise.
    """
    if idx >= len(parts):
        return "" if not tree else None

    # Try compound tokens, longest first (e.g. "single_transformer_blocks")
    max_len = min(5, len(parts) - idx)
    for length in range(max_len, 0, -1):
        candidate = "_".join(parts[idx : idx + length])

        if candidate in tree:
            rest = _match_tree(parts, idx + length, tree[candidate])
            if rest is not None:
                return candidate + ("." + rest if rest else "")

    # Try numeric index
    if parts[idx].isdigit() and _INDEX in tree:
        rest = _match_tree(parts, idx + 1, tree[_INDEX])
        if rest is not None:
            return parts[idx] + ("." + rest if rest else "")

    return None


class LoraApplicator:
    """Load, detect, and apply LoRA weight patches to transformer models."""

    # ── Public API ───────────────────────────────────────────────────

    @staticmethod
    def load_and_apply(
        model: torch.nn.Module,
        lora_specs: list[tuple[str, float]],  # (path, strength)
        model_type: TransformerType,
    ) -> list[RemovableHandle]:
        """Load and apply one or more LoRAs via forward hooks.

        Registers hooks that add LoRA residuals during the forward pass:
            output += F.linear(F.linear(input, down), up) * (strength * alpha/rank)

        Returns a list of hook handles — call remove_hooks() to detach them.
        Weights are never modified, so no snapshot/restore is needed.
        """
        all_handles: list[RemovableHandle] = []

        for lora_path, strength in lora_specs:
            logger.info(f"Applying LoRA: {lora_path} (strength={strength})")
            state_dict = LoraApplicator._load_state_dict(lora_path)
            fmt = LoraApplicator.detect_format(state_dict)
            logger.info(f"  Detected format: {fmt.value}")

            lora_layers = LoraApplicator._parse_lora_layers(state_dict, fmt, model_type)
            logger.debug(f"  Parsed {len(lora_layers)} layer groups")

            applied = 0
            skipped = []
            for layer_key, layer_data in lora_layers.items():
                handle = LoraApplicator._register_hook(model, layer_key, layer_data, strength)
                if handle is not None:
                    all_handles.append(handle)
                    applied += 1
                elif layer_data.get("up") is not None and layer_data.get("down") is not None:
                    skipped.append(layer_key)

            logger.info(f"  Applied {applied}/{len(lora_layers)} LoRA layers")
            if skipped:
                logger.warning(f"  Skipped {len(skipped)} layers (no matching param): {skipped[:5]}...")

        return all_handles

    @staticmethod
    def remove_hooks(handles: list[RemovableHandle]) -> None:
        """Remove all LoRA forward hooks."""
        for handle in handles:
            handle.remove()
        handles.clear()

    # ── Format detection ─────────────────────────────────────────────

    @staticmethod
    def detect_format(state_dict: dict[str, torch.Tensor]) -> LoraFormat:
        """Detect the LoRA format from key patterns in the state dict."""
        keys = list(state_dict.keys())
        sample_keys = keys[:50]

        # Kohya: keys like "lora_unet_double_blocks_0_..."
        if any(k.startswith("lora_unet_") for k in sample_keys):
            return LoraFormat.KOHYA

        # OneTrainer: keys like "lora_transformer_transformer_blocks_0_..."
        if any(k.startswith("lora_transformer_") for k in sample_keys):
            return LoraFormat.ONETRAINER

        # XLabs: keys like "double_blocks.0.processor.qkv_lora1.down.weight"
        if any("processor" in k and "double_blocks" in k for k in sample_keys):
            return LoraFormat.XLABS

        # AI Toolkit: "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
        if any(k.startswith("transformer.") and ".lora_" in k for k in sample_keys):
            return LoraFormat.AITOOLKIT

        # Diffusers/PEFT: keys contain "lora_A.weight" / "lora_B.weight"
        if any("lora_A" in k or "lora_B" in k or "lora_down" in k or "lora_up" in k for k in sample_keys):
            return LoraFormat.DIFFUSERS

        # Fallback: diffusion_model prefix (some Z-Image LoRAs)
        if any(k.startswith("diffusion_model.") for k in sample_keys):
            return LoraFormat.DIFFUSERS

        return LoraFormat.UNKNOWN

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
        """Load a LoRA state dict from a file."""
        p = Path(path)
        if not p.exists():
            from rzem_ai_inference_engine.models.loader import ModelLoader
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
            layers = LoraApplicator._parse_kohya(state_dict)
            if model_type == TransformerType.FLUX1_DEV:
                layers = LoraApplicator._flux1_bfl_to_diffusers(layers)
            return layers

        elif fmt == LoraFormat.XLABS:
            layers = LoraApplicator._parse_xlabs(state_dict)
            if model_type == TransformerType.FLUX1_DEV:
                layers = LoraApplicator._flux1_bfl_to_diffusers(layers)
            return layers

        elif fmt == LoraFormat.ONETRAINER:
            return LoraApplicator._parse_onetrainer(state_dict, model_type)

        elif fmt == LoraFormat.AITOOLKIT:
            return LoraApplicator._parse_aitoolkit(state_dict)

        elif fmt == LoraFormat.DIFFUSERS:
            return LoraApplicator._parse_diffusers(state_dict)

        else:
            logger.warning("Unknown LoRA format, attempting diffusers-style parse")
            return LoraApplicator._parse_diffusers(state_dict)

    # ── Format-specific parsers ──────────────────────────────────────

    @staticmethod
    def _parse_kohya(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
        """Parse Kohya-format LoRA keys.

        Kohya keys: lora_unet_<layer_path>.lora_down.weight / .lora_up.weight / .alpha
        The layer path uses underscores in place of dots (BFL naming convention).
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            if not key.startswith("lora_unet_"):
                continue

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

        # Convert Kohya underscored keys to BFL dotted keys
        # e.g. "double_blocks_0_img_attn_qkv" → "double_blocks.0.img_attn.qkv"
        converted: dict[str, dict[str, Any]] = {}
        for kohya_key, data in layers.items():
            model_key = LoraApplicator._kohya_key_to_bfl_key(kohya_key)
            converted[model_key] = data

        return converted

    @staticmethod
    def _parse_xlabs(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
        """Parse XLabs-format LoRA keys.

        XLabs FLUX keys:
            double_blocks.{i}.processor.qkv_lora1.{down,up}.weight  (image QKV)
            double_blocks.{i}.processor.qkv_lora2.{down,up}.weight  (text QKV)
            double_blocks.{i}.processor.proj_lora1.{down,up}.weight (image proj)
            double_blocks.{i}.processor.proj_lora2.{down,up}.weight (text proj)

        Produces BFL-style keys (e.g. double_blocks.0.img_attn.qkv).
        """
        layers: dict[str, dict[str, Any]] = {}

        xlabs_re = re.compile(
            r"double_blocks\.(\d+)\.processor\.(qkv|proj)_lora([12])\.(down|up)\.weight"
        )

        for key, tensor in state_dict.items():
            m = xlabs_re.match(key)
            if not m:
                # Fallback: try generic lora_A/lora_B pattern
                if ".lora_A" in key or ".lora_a" in key:
                    layer_key = re.sub(r"\.lora_[Aa]\.?.*$", "", key)
                    layer_key = layer_key.replace(".processor", "")
                    layers.setdefault(layer_key, {})["down"] = tensor
                elif ".lora_B" in key or ".lora_b" in key:
                    layer_key = re.sub(r"\.lora_[Bb]\.?.*$", "", key)
                    layer_key = layer_key.replace(".processor", "")
                    layers.setdefault(layer_key, {})["up"] = tensor
                continue

            block_idx = m.group(1)
            component = m.group(2)   # "qkv" or "proj"
            stream = m.group(3)      # "1" (img) or "2" (txt)
            direction = m.group(4)   # "down" or "up"

            attn_type = "img_attn" if stream == "1" else "txt_attn"
            bfl_key = f"double_blocks.{block_idx}.{attn_type}.{component}"
            layers.setdefault(bfl_key, {})[direction] = tensor

        return layers

    @staticmethod
    def _parse_diffusers(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
        """Parse Diffusers/PEFT format LoRA keys.

        Keys: <prefix>.<layer_path>.lora_A.weight / .lora_B.weight
        Produces keys that already match diffusers model parameter names.
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            clean_key = key
            for prefix in ("transformer.", "diffusion_model.", "unet."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break

            if ".lora_A." in clean_key or ".lora_down." in clean_key:
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
    def _parse_aitoolkit(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
        """Parse AI Toolkit format LoRA keys.

        Keys: transformer.<layer_path>.lora_A.weight / .lora_B.weight
        Produces keys that already match diffusers model parameter names.
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            clean_key = key
            if clean_key.startswith("transformer."):
                clean_key = clean_key[len("transformer.") :]

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
    def _parse_onetrainer(
        state_dict: dict[str, torch.Tensor],
        model_type: TransformerType,
    ) -> dict[str, dict[str, Any]]:
        """Parse OneTrainer format LoRA keys.

        OneTrainer keys use Kohya-style notation with diffusers naming:
            lora_transformer_<diffusers_path_with_underscores>.lora_down.weight
        Uses a parsing tree to correctly insert dots between compound tokens.
        """
        layers: dict[str, dict[str, Any]] = {}

        for key, tensor in state_dict.items():
            if not key.startswith("lora_transformer_"):
                continue  # Skip text encoder keys for now

            rest = key[len("lora_transformer_"):]

            if rest.endswith(".lora_down.weight"):
                underscored = rest[: -len(".lora_down.weight")]
                direction = "down"
            elif rest.endswith(".lora_up.weight"):
                underscored = rest[: -len(".lora_up.weight")]
                direction = "up"
            elif rest.endswith(".alpha"):
                underscored = rest[: -len(".alpha")]
                dotted = LoraApplicator._onetrainer_key_to_diffusers(underscored, model_type)
                if dotted:
                    layers.setdefault(dotted, {})["alpha"] = tensor.item()
                continue
            else:
                continue

            dotted = LoraApplicator._onetrainer_key_to_diffusers(underscored, model_type)
            if dotted:
                layers.setdefault(dotted, {})[direction] = tensor

        return layers

    # ── Key conversion ───────────────────────────────────────────────

    @staticmethod
    def _kohya_key_to_bfl_key(kohya_key: str) -> str:
        """Convert a Kohya underscored key to a BFL dotted key.

        e.g. "double_blocks_0_img_attn_qkv" → "double_blocks.0.img_attn.qkv"
        """
        parts = kohya_key.split("_")
        key = ".".join(parts)

        # Restore known compound tokens that were broken by the split
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

    @staticmethod
    def _onetrainer_key_to_diffusers(underscored: str, model_type: TransformerType) -> str | None:
        """Convert a OneTrainer underscored key to a diffusers dotted key.

        Uses the parsing tree for the model type to correctly handle compound tokens.
        e.g. "single_transformer_blocks_0_attn_to_k" → "single_transformer_blocks.0.attn.to_k"
        """
        if model_type == TransformerType.FLUX1_DEV:
            tree = _FLUX1_DIFFUSERS_TREE
        else:
            logger.warning(f"  No OneTrainer parsing tree for {model_type.value}")
            return None

        parts = underscored.split("_")
        result = _match_tree(parts, 0, tree)
        if result is None:
            logger.warning(f"  Could not parse OneTrainer key: {underscored}")
        return result

    # ── BFL → Diffusers conversion (FLUX.1) ──────────────────────────

    # Simple renames: (BFL regex, diffusers template)
    _FLUX1_RENAMES = [
        (r"^double_blocks\.(\d+)\.img_attn\.proj$", r"transformer_blocks.\1.attn.to_out.0"),
        (r"^double_blocks\.(\d+)\.img_mlp\.0$", r"transformer_blocks.\1.ff.net.0.proj"),
        (r"^double_blocks\.(\d+)\.img_mlp\.2$", r"transformer_blocks.\1.ff.net.2"),
        (r"^double_blocks\.(\d+)\.img_mod\.lin$", r"transformer_blocks.\1.norm1.linear"),
        (r"^double_blocks\.(\d+)\.txt_attn\.proj$", r"transformer_blocks.\1.attn.to_add_out"),
        (r"^double_blocks\.(\d+)\.txt_mlp\.0$", r"transformer_blocks.\1.ff_context.net.0.proj"),
        (r"^double_blocks\.(\d+)\.txt_mlp\.2$", r"transformer_blocks.\1.ff_context.net.2"),
        (r"^double_blocks\.(\d+)\.txt_mod\.lin$", r"transformer_blocks.\1.norm1_context.linear"),
        (r"^single_blocks\.(\d+)\.linear2$", r"single_transformer_blocks.\1.proj_out"),
        (r"^single_blocks\.(\d+)\.modulation\.lin$", r"single_transformer_blocks.\1.norm.linear"),
    ]

    @staticmethod
    def _flux1_bfl_to_diffusers(
        layers: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Convert BFL-naming FLUX.1 keys to diffusers FluxTransformer2DModel naming.

        Handles:
        - Fused QKV splitting (img_attn.qkv → to_q/to_k/to_v)
        - Fused linear1 splitting (single block → to_q/to_k/to_v/proj_mlp)
        - Block renaming (double_blocks → transformer_blocks, etc.)
        """
        converted: dict[str, dict[str, Any]] = {}

        for bfl_key, data in layers.items():
            # ── Fused QKV: double block image attention ──
            m = re.match(r"^double_blocks\.(\d+)\.img_attn\.qkv$", bfl_key)
            if m:
                i = m.group(1)
                LoraApplicator._split_fused(
                    data,
                    [
                        f"transformer_blocks.{i}.attn.to_q",
                        f"transformer_blocks.{i}.attn.to_k",
                        f"transformer_blocks.{i}.attn.to_v",
                    ],
                    converted,
                )
                continue

            # ── Fused QKV: double block text attention ──
            m = re.match(r"^double_blocks\.(\d+)\.txt_attn\.qkv$", bfl_key)
            if m:
                i = m.group(1)
                LoraApplicator._split_fused(
                    data,
                    [
                        f"transformer_blocks.{i}.attn.add_q_proj",
                        f"transformer_blocks.{i}.attn.add_k_proj",
                        f"transformer_blocks.{i}.attn.add_v_proj",
                    ],
                    converted,
                )
                continue

            # ── Fused linear1: single block (QKV + MLP) ──
            m = re.match(r"^single_blocks\.(\d+)\.linear1$", bfl_key)
            if m:
                i = m.group(1)
                up = data["up"]
                hidden_dim = data["down"].shape[1]
                # linear1 = [to_q, to_k, to_v, proj_mlp] along dim 0
                splits = [
                    (f"single_transformer_blocks.{i}.attn.to_q", 0, hidden_dim),
                    (f"single_transformer_blocks.{i}.attn.to_k", hidden_dim, 2 * hidden_dim),
                    (f"single_transformer_blocks.{i}.attn.to_v", 2 * hidden_dim, 3 * hidden_dim),
                    (f"single_transformer_blocks.{i}.proj_mlp", 3 * hidden_dim, up.shape[0]),
                ]
                for new_key, start, end in splits:
                    converted[new_key] = {
                        "up": up[start:end],
                        "down": data["down"],
                        "alpha": data.get("alpha"),
                    }
                continue

            # ── Simple renames ──
            matched = False
            for pattern, replacement in LoraApplicator._FLUX1_RENAMES:
                new_key = re.sub(pattern, replacement, bfl_key)
                if new_key != bfl_key:
                    converted[new_key] = data
                    matched = True
                    break

            if not matched:
                logger.warning(f"  Unmapped BFL key (keeping as-is): {bfl_key}")
                converted[bfl_key] = data

        return converted

    @staticmethod
    def _split_fused(
        data: dict[str, Any],
        target_keys: list[str],
        output: dict[str, dict[str, Any]],
    ) -> None:
        """Split a fused LoRA layer (e.g. QKV) into N equal parts along up's dim 0."""
        up = data["up"]
        n = len(target_keys)
        chunk = up.shape[0] // n
        for j, key in enumerate(target_keys):
            output[key] = {
                "up": up[j * chunk : (j + 1) * chunk],
                "down": data["down"],
                "alpha": data.get("alpha"),
            }

    # ── Hook registration ───────────────────────────────────────────

    @staticmethod
    def _register_hook(
        model: torch.nn.Module,
        layer_key: str,
        layer_data: dict[str, Any],
        strength: float,
    ) -> RemovableHandle | None:
        """Register a forward hook on the target module for one LoRA layer.

        Returns the hook handle, or None if the target module wasn't found.
        """
        up = layer_data.get("up")
        down = layer_data.get("down")
        alpha = layer_data.get("alpha")

        if up is None or down is None:
            return None

        # Find the target module (the parent of the weight parameter).
        # layer_key is like "transformer_blocks.0.attn.to_q" and we need
        # the nn.Module at that path (not its .weight attribute).
        module = None
        for candidate in (layer_key, layer_key.rsplit(".", 1)[0] if "." in layer_key else None):
            if candidate is None:
                continue
            try:
                obj = model
                for part in candidate.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, torch.nn.Module) and hasattr(obj, "weight"):
                    module = obj
                    break
            except (AttributeError, IndexError):
                continue

        if module is None:
            return None

        # Compute scale and move LoRA tensors to the module's device/dtype
        rank = down.shape[0]
        scale = (alpha / rank) if alpha is not None else 1.0
        combined_scale = strength * scale

        device = next(module.parameters()).device
        dtype = preferred_dtype(device)
        up_d = up.to(device=device, dtype=dtype)
        down_d = down.to(device=device, dtype=dtype)

        def hook(_module: torch.nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            x = input[0].to(dtype=dtype)
            residual = F.linear(F.linear(x, down_d), up_d)
            return output + residual * combined_scale

        return module.register_forward_hook(hook)
