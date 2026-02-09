"""Two-tier model cache with VRAM/RAM and smallest-first eviction."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from loguru import logger

from inference_engine.models.memory import estimate_model_size, get_free_vram
from inference_engine.types import EventType, ModelLoadedEvent, ModelLoadingEvent, ModelUnloadedEvent

# Reserve 1 GB for intermediate tensors during inference
_WORKING_MEMORY_BUFFER = 1 * (1024**3)


@dataclass
class CacheEntry:
    """A single cached model."""

    key: str
    model: Any  # torch.nn.Module or tokenizer, etc.
    size_bytes: int
    device: str  # "cuda", "cpu", "mps"
    last_used: float = field(default_factory=time.time)
    _locks: int = 0

    def lock(self) -> None:
        self._locks += 1

    def unlock(self) -> None:
        self._locks -= 1
        assert self._locks >= 0

    @property
    def is_locked(self) -> bool:
        return self._locks > 0

    def touch(self) -> None:
        self.last_used = time.time()


class ModelCache:
    """Two-tier model cache: GPU VRAM (hot) and CPU RAM (warm).

    Models are identified by a string key (typically ``file_path::model_type``).
    Eviction follows a smallest-unlocked-first strategy on VRAM, and LRU on RAM.
    """

    def __init__(
        self,
        device: torch.device,
        vram_limit_gb: float | None = None,
        emit: Callable | None = None,
    ) -> None:
        self._device = device
        self._vram: dict[str, CacheEntry] = {}
        self._ram: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._emit = emit or (lambda *_a, **_kw: None)

        if vram_limit_gb is not None:
            self._vram_limit = int(vram_limit_gb * (1024**3))
        else:
            self._vram_limit: int | None = None

    # ── Public API ───────────────────────────────────────────────────

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        estimated_size: int = 0,
    ) -> Any:
        """Return a cached model, loading it if necessary.

        *loader* is called only on a cache miss. The loaded model is placed
        on the target device (VRAM).
        """
        with self._lock:
            # 1. Check VRAM
            if key in self._vram:
                entry = self._vram[key]
                entry.touch()
                logger.debug(f"Cache hit (VRAM): {key}")
                return entry.model

            # 2. Check RAM — move to VRAM
            if key in self._ram:
                entry = self._ram.pop(key)
                self._ensure_vram_unlocked(entry.size_bytes)
                self._move_to_device(entry, self._device)
                self._vram[key] = entry
                entry.touch()
                logger.debug(f"Cache hit (RAM→VRAM): {key}")
                self._emit(
                    EventType.MODEL_LOADED,
                    ModelLoadedEvent(key=key, size_bytes=entry.size_bytes, device=str(self._device)),
                )
                return entry.model

        # 3. Cache miss — load (outside the lock, loading can be slow)
        self._emit(EventType.MODEL_LOADING, ModelLoadingEvent(key=key))
        logger.info(f"Loading model: {key}")
        model = loader()

        # Estimate size
        if isinstance(model, torch.nn.Module):
            size_bytes = estimate_model_size(model)
        else:
            size_bytes = estimated_size or 0

        with self._lock:
            # Double-check: another thread may have loaded it while we were loading
            if key in self._vram:
                return self._vram[key].model

            # Ensure VRAM space BEFORE moving to device
            if isinstance(model, torch.nn.Module):
                self._ensure_vram_unlocked(size_bytes)
                model = model.to(self._device)

            entry = CacheEntry(
                key=key,
                model=model,
                size_bytes=size_bytes,
                device=str(self._device),
            )
            self._vram[key] = entry

        self._emit(
            EventType.MODEL_LOADED,
            ModelLoadedEvent(key=key, size_bytes=size_bytes, device=str(self._device)),
        )
        logger.info(f"Model loaded: {key} ({size_bytes / (1024**3):.2f} GB)")
        return model

    def ensure_vram(self, bytes_needed: int) -> None:
        """Ensure at least *bytes_needed* free VRAM by evicting models."""
        with self._lock:
            self._ensure_vram_unlocked(bytes_needed)

    def lock(self, key: str) -> None:
        """Prevent eviction of a model while it's in use."""
        with self._lock:
            if key in self._vram:
                self._vram[key].lock()
            elif key in self._ram:
                self._ram[key].lock()

    def unlock(self, key: str) -> None:
        """Allow eviction of a model after use."""
        with self._lock:
            if key in self._vram:
                self._vram[key].unlock()
            elif key in self._ram:
                self._ram[key].unlock()

    def clear(self) -> None:
        """Remove all models from cache."""
        with self._lock:
            for entry in list(self._vram.values()):
                self._evict_to_nowhere(entry)
            for entry in list(self._ram.values()):
                del self._ram[entry.key]
            self._vram.clear()
            self._ram.clear()

    @property
    def vram_used(self) -> int:
        """Total bytes used by models on the GPU."""
        return sum(e.size_bytes for e in self._vram.values())

    # ── Internal ─────────────────────────────────────────────────────

    def _ensure_vram_unlocked(self, bytes_needed: int) -> None:
        """Evict smallest unlocked VRAM entries until there's enough room.

        Must be called with ``self._lock`` held.
        """
        available = self._available_vram()
        while available < bytes_needed + _WORKING_MEMORY_BUFFER:
            # Find smallest unlocked entry
            candidates = [e for e in self._vram.values() if not e.is_locked]
            if not candidates:
                logger.warning(
                    f"Cannot free enough VRAM: need {bytes_needed / (1024**3):.2f} GB, "
                    f"available {available / (1024**3):.2f} GB, "
                    f"all {len(self._vram)} VRAM entries are locked"
                )
                break
            candidates.sort(key=lambda e: e.size_bytes)
            victim = candidates[0]
            self._evict_to_ram(victim)
            available = self._available_vram()

    def _available_vram(self) -> int:
        """Return available VRAM in bytes (respecting user limit)."""
        if self._vram_limit is not None:
            return self._vram_limit - self.vram_used
        return get_free_vram(self._device)

    def _evict_to_ram(self, entry: CacheEntry) -> None:
        """Move a model from VRAM to CPU RAM."""
        logger.info(f"Evicting to RAM: {entry.key} ({entry.size_bytes / (1024**3):.2f} GB)")
        self._vram.pop(entry.key, None)

        # Move model to CPU and free CUDA memory
        self._move_to_device(entry, torch.device("cpu"))
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._ram[entry.key] = entry

        self._emit(
            EventType.MODEL_UNLOADED,
            ModelUnloadedEvent(key=entry.key, from_device=str(self._device)),
        )

    def _evict_to_nowhere(self, entry: CacheEntry) -> None:
        """Fully remove a model from cache."""
        logger.info(f"Evicting entirely: {entry.key}")
        self._vram.pop(entry.key, None)
        self._ram.pop(entry.key, None)
        self._emit(
            EventType.MODEL_UNLOADED,
            ModelUnloadedEvent(key=entry.key, from_device=entry.device),
        )

    @staticmethod
    def _move_to_device(entry: CacheEntry, device: torch.device) -> None:
        """Move a model's tensors to the given device."""
        if isinstance(entry.model, torch.nn.Module):
            entry.model.to(device)
        entry.device = str(device)
