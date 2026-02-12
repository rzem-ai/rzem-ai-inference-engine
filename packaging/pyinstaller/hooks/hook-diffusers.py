"""
PyInstaller hook for diffusers package.

Collects all diffusers model classes and submodules that are imported
conditionally in pipeline files. This ensures PyInstaller's static analysis
can discover these dynamic imports.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all diffusers submodules
hiddenimports = collect_submodules('diffusers')

# Add specific model classes that are lazily imported in pipelines
hiddenimports += [
    # Transformer models
    'diffusers.models.transformers.flux',
    'diffusers.models.transformers.flux2',
    'diffusers.models.transformers.z_image',
    'diffusers.models.transformers.qwen_image',

    # VAE models
    'diffusers.models.vae',
    'diffusers.models.autoencoders.autoencoder_kl',

    # Quantization support
    'diffusers.quantizers',
    'diffusers.quantizers.gguf',
]

# Collect configuration files and model configs
datas = collect_data_files('diffusers', include_py_files=True)
