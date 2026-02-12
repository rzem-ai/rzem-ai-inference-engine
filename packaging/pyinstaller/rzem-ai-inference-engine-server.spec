# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for RZEM AI Inference Engine (Server variant).

Builds a standalone executable that includes:
- All ML dependencies (PyTorch, diffusers, transformers)
- Web stack (FastAPI, uvicorn) for the 'serve' command
- Both 'generate' and 'serve' CLI commands

Build with:
    pyinstaller packaging/pyinstaller/rzem-ai-inference-engine-server.spec --clean --noconfirm

Output:
    dist/rzem-ai-inference-engine-server/
"""

import sys
from pathlib import Path

block_cipher = None

# Project root (two levels up from this spec file)
project_root = Path(__file__).parent.parent.parent.resolve()
src_path = project_root / 'src'

a = Analysis(
    [str(src_path / 'rzem_ai_inference_engine' / '__main__.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Core ML stack
        'torch',
        'torch.distributed',
        'torch.distributed.nn',
        'torch.distributed.distributed_c10d',
        'diffusers',
        'diffusers.models',
        'diffusers.pipelines',
        'transformers',
        'transformers.models',
        'accelerate',
        'safetensors',
        'safetensors.torch',
        'huggingface_hub',
        'huggingface_hub.hf_api',
        'huggingface_hub.utils',

        # Diffusers models (lazy loaded in pipelines)
        'diffusers.models.transformers.flux',
        'diffusers.models.transformers.flux2',
        'diffusers.models.transformers.z_image',
        'diffusers.models.transformers.qwen_image',
        'diffusers.models.vae',
        'diffusers.models.autoencoders',
        'diffusers.models.autoencoders.autoencoder_kl',
        'diffusers.quantizers',
        'diffusers.quantizers.gguf',

        # Transformers models (lazy loaded)
        'transformers.models.clip',
        'transformers.models.clip.modeling_clip',
        'transformers.models.clip.tokenization_clip',
        'transformers.models.t5',
        'transformers.models.t5.modeling_t5',
        'transformers.models.t5.tokenization_t5_fast',
        'transformers.models.qwen2',
        'transformers.models.qwen2.modeling_qwen2',
        'transformers.models.qwen3',
        'transformers.models.qwen3.modeling_qwen3',
        'transformers.models.qwen3.tokenization_qwen3_fast',
        'transformers.models.auto',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.tokenization_auto',

        # Web stack (for serve command)
        'fastapi',
        'fastapi.responses',
        'fastapi.staticfiles',
        'uvicorn',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'websockets',
        'starlette',
        'starlette.middleware',
        'starlette.middleware.cors',

        # Utilities
        'click',
        'loguru',
        'pydantic',
        'pydantic.fields',
        'pydantic.main',
        'PIL',
        'PIL.Image',
        'PIL._tkinter_finder',
        'einops',
        'sentencepiece',
        'python_multipart',

        # Optional GGUF support
        'gguf',

        # All package modules (explicitly listed for clarity)
        'rzem_ai_inference_engine',
        'rzem_ai_inference_engine.engine',
        'rzem_ai_inference_engine.types',
        'rzem_ai_inference_engine.cli',

        # Pipeline modules (all four must be included)
        'rzem_ai_inference_engine.pipeline',
        'rzem_ai_inference_engine.pipeline.base',
        'rzem_ai_inference_engine.pipeline.flux1',
        'rzem_ai_inference_engine.pipeline.flux2',
        'rzem_ai_inference_engine.pipeline.z_image',
        'rzem_ai_inference_engine.pipeline.qwen_image',
        'rzem_ai_inference_engine.pipeline.lora_applicator',

        # API modules
        'rzem_ai_inference_engine.api',
        'rzem_ai_inference_engine.api.app',
        'rzem_ai_inference_engine.api.routes',
        'rzem_ai_inference_engine.api.models',
        'rzem_ai_inference_engine.api.state',
        'rzem_ai_inference_engine.api.ws',

        # Model management modules
        'rzem_ai_inference_engine.models',
        'rzem_ai_inference_engine.models.cache',
        'rzem_ai_inference_engine.models.loader',
        'rzem_ai_inference_engine.models.memory',

        # Queue modules
        'rzem_ai_inference_engine.queue',
        'rzem_ai_inference_engine.queue.manager',
        'rzem_ai_inference_engine.queue.processor',
    ],
    hookspath=[str(Path(__file__).parent / 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='rzem-ai-inference-engine-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='rzem-ai-inference-engine-server',
)
