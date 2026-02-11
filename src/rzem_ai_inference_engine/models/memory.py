"""VRAM estimation utilities."""

from __future__ import annotations

import torch
from loguru import logger


def estimate_model_size(model: torch.nn.Module) -> int:
    """Estimate the memory footprint of a model in bytes.

    Counts parameters and buffers. Does not account for optimizer state
    or activation memory during forward passes.
    """
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return param_bytes + buffer_bytes


def get_free_vram(device: torch.device) -> int:
    """Return free VRAM in bytes for the given device."""
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return free
    if device.type == "mps":
        # MPS doesn't expose free memory; return a large sentinel
        return 2**40
    # CPU — effectively unlimited for our purposes
    return 2**40


def get_total_vram(device: torch.device) -> int:
    """Return total VRAM in bytes for the given device."""
    if device.type == "cuda":
        _, total = torch.cuda.mem_get_info(device)
        return total
    if device.type == "mps":
        # Apple reports unified memory; use a conservative estimate
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip())
        except Exception:
            return 16 * (1024**3)  # 16 GB fallback
    return 2**40


def preferred_dtype(device: torch.device) -> torch.dtype:
    """Return the best reduced-precision dtype for *device*.

    - CUDA (Ampere+): bfloat16 — native tensor-core support, wide dynamic range.
    - MPS (M3+, PyTorch 2.3+): bfloat16 — native GPU ALU support.
    - CPU: float32 — no GPU to benefit from reduced precision.
    """
    if device.type in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def detect_device() -> torch.device:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    logger.info("Using CPU device")
    return torch.device("cpu")
