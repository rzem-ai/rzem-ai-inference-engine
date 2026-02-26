"""Model path resolution — local filesystem or HuggingFace Hub.

Supports three input formats:

1. **Local path** — ``./models/flux.safetensors`` or ``./models/flux-dir/``
   Returned directly if the path exists on disk.

2. **HuggingFace repo** — ``org/repo`` (exactly one slash, no file extension)
   Downloads the full repo via ``snapshot_download`` and returns the cache dir.

3. **HuggingFace repo + file** — ``org/repo/filename.ext`` (2+ slashes, last
   segment has a file extension)
   Downloads only that file via ``hf_hub_download`` and returns its local path.
   Useful for repos with multiple model variants (e.g. GGUF quants).

Examples::

    # Local file
    ModelLoader.resolve_path("./models/flux1-dev.safetensors")

    # Full HF repo (returns directory)
    ModelLoader.resolve_path("black-forest-labs/FLUX.1-dev")

    # Specific file from HF repo (returns file)
    ModelLoader.resolve_path("city96/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger


def _parse_hf_path(path_or_repo: str) -> tuple[str, str | None]:
    """Split an HF-style path into (repo_id, filename | None).

    ``"org/repo"``            → ``("org/repo", None)``
    ``"org/repo/file.gguf"``  → ``("org/repo", "file.gguf")``
    ``"org/repo/sub/f.safetensors"`` → ``("org/repo", "sub/f.safetensors")``
    """
    parts = path_or_repo.split("/")
    if len(parts) <= 2:
        return path_or_repo, None

    # 3+ segments: first two are the repo, rest is the file path within it
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    # Only treat as repo+file if the last segment looks like a file (has extension)
    if "." in parts[-1]:
        return repo_id, filename

    # No extension on last part — treat the whole thing as a repo ID
    # (unlikely, but safe fallback)
    return path_or_repo, None


class ModelLoader:
    """Resolves model paths and configuration dicts."""

    # Default patterns for snapshot_download — excludes docs, notebooks, images
    _DEFAULT_ALLOW = [
        "*.safetensors", "*.bin", "*.gguf",
        "*.json", "*.txt", "*.model",
        "*.tiktoken", "*.py",
    ]

    @staticmethod
    def resolve_path(
        path_or_repo: str,
        *,
        subfolder: str | None = None,
    ) -> Path:
        """Return a local Path for a model.

        Handles local paths, HF repo IDs, and ``repo/file`` paths.
        See module docstring for details.

        Parameters
        ----------
        subfolder:
            When set and downloading a full repo, restricts the download to
            files inside this subfolder (plus root JSON configs).  This avoids
            downloading the entire repo when only one component is needed
            (e.g. ``subfolder="vae"`` to skip transformer weights).
        """
        # 1. Local path?
        local = Path(path_or_repo)
        if local.exists():
            return local

        # 2. Parse HF path
        repo_id, filename = _parse_hf_path(path_or_repo)

        if filename is not None:
            # Specific file from a repo
            from huggingface_hub import hf_hub_download

            logger.info(f"Loading file from HuggingFace Hub: {repo_id}/{filename}")
            cached = hf_hub_download(repo_id=repo_id, filename=filename)
            return Path(cached)

        # Full repo download (or subfolder-scoped)
        from huggingface_hub import snapshot_download

        if subfolder:
            # Only download the subfolder contents + root config files
            allow = [f"{subfolder}/**", "*.json"]
            logger.info(f"Loading repo from HuggingFace Hub: {repo_id} (subfolder: {subfolder})")
        else:
            allow = ModelLoader._DEFAULT_ALLOW
            logger.info(f"Loading repo from HuggingFace Hub: {repo_id}")

        cached = snapshot_download(repo_id=repo_id, allow_patterns=allow)
        return Path(cached)

    @staticmethod
    def resolve_single_file(path_or_repo: str) -> Path:
        """Resolve a path that points to a single model file (e.g. a .safetensors).

        If it's a local file, return it. If it's a HF repo, download and return
        the first .safetensors file found.
        """
        local = Path(path_or_repo)
        if local.is_file():
            return local
        if local.is_dir():
            safetensors = list(local.glob("*.safetensors"))
            if safetensors:
                return safetensors[0]
            raise FileNotFoundError(f"No .safetensors file found in {local}")

        resolved = ModelLoader.resolve_path(path_or_repo)
        if resolved.is_file():
            return resolved
        safetensors = list(resolved.glob("*.safetensors"))
        if safetensors:
            return safetensors[0]
        raise FileNotFoundError(f"No .safetensors file found in {resolved}")

    @staticmethod
    def resolve_config(config: str | dict | None, model_type: str) -> dict[str, Any] | None:
        """Resolve a config value to a dict.

        - ``dict`` → returned as-is
        - ``str`` that is valid JSON → parsed
        - ``str`` that is a file path → read and parsed
        - ``None`` → returns ``None`` (caller uses library defaults)
        """
        if config is None:
            return None
        if isinstance(config, dict):
            return config

        # Try JSON string first
        try:
            return json.loads(config)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try file path
        path = Path(config)
        if path.is_file():
            return json.loads(path.read_text())

        raise ValueError(f"Cannot resolve config for {model_type}: {config!r}")
