"""
PyInstaller hook for rzem_ai_inference_engine package.

Ensures all package modules are collected, especially those that might be
imported lazily or dynamically.
"""

from PyInstaller.utils.hooks import collect_submodules

# Collect all package submodules
hiddenimports = collect_submodules('rzem_ai_inference_engine')

# Explicitly list critical modules to ensure they're included
hiddenimports += [
    # Pipeline modules (all four pipelines must be included)
    'rzem_ai_inference_engine.pipeline',
    'rzem_ai_inference_engine.pipeline.base',
    'rzem_ai_inference_engine.pipeline.flux1',
    'rzem_ai_inference_engine.pipeline.flux2',
    'rzem_ai_inference_engine.pipeline.z_image',
    'rzem_ai_inference_engine.pipeline.qwen_image',
    'rzem_ai_inference_engine.pipeline.lora_applicator',

    # API modules (for serve command)
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

    # Core modules
    'rzem_ai_inference_engine.engine',
    'rzem_ai_inference_engine.types',
    'rzem_ai_inference_engine.cli',
]
