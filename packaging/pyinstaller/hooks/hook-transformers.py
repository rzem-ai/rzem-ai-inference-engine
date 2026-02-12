"""
PyInstaller hook for transformers package.

Collects transformers model classes, tokenizers, and data files that are
imported conditionally in pipeline files.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all transformers submodules
hiddenimports = collect_submodules('transformers')

# Add specific model classes used in pipelines
hiddenimports += [
    # CLIP (FLUX.1)
    'transformers.models.clip',
    'transformers.models.clip.modeling_clip',
    'transformers.models.clip.tokenization_clip',

    # T5 (FLUX.1)
    'transformers.models.t5',
    'transformers.models.t5.modeling_t5',
    'transformers.models.t5.tokenization_t5',
    'transformers.models.t5.tokenization_t5_fast',

    # Qwen2 (legacy)
    'transformers.models.qwen2',
    'transformers.models.qwen2.modeling_qwen2',
    'transformers.models.qwen2.tokenization_qwen2',
    'transformers.models.qwen2.tokenization_qwen2_fast',

    # Qwen3 (Z-Image, FLUX.2, Qwen-Image)
    'transformers.models.qwen3',
    'transformers.models.qwen3.modeling_qwen3',
    'transformers.models.qwen3.tokenization_qwen3',
    'transformers.models.qwen3.tokenization_qwen3_fast',

    # Auto classes
    'transformers.models.auto',
    'transformers.models.auto.modeling_auto',
    'transformers.models.auto.tokenization_auto',
]

# Collect tokenizer data files (vocab, merges, special tokens)
datas = collect_data_files('transformers', include_py_files=True)
