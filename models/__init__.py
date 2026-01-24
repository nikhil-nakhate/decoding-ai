"""
Models module for nanoVLA-RL

Provides model architectures and registration system.

Submodules:
    - vae: Variational Autoencoder
    - dit: Diffusion Transformer
    - paligemma: Vision-Language Model (SigLIP + Gemma)

Usage:
    from models import VAE, DIT, build_model_from_cfg
    from models.vae import VAE, Discriminator, LPIPS
    from models.dit import DIT, PatchEmbedding
    from models.paligemma import PaliGemmaForConditionalGeneration
"""

from .registry import (
    register_model,
    get_model_class,
    list_models,
    is_registered,
    clear_registry,
)
from .build import build_model_from_cfg, register_models

# Import main model classes for convenience
from .vae import VAE
from .dit import DIT
from .paligemma import PaliGemmaForConditionalGeneration

# Auto-register all models
register_models()

__all__ = [
    # Registry functions
    'register_model',
    'get_model_class',
    'list_models',
    'is_registered',
    'clear_registry',
    'build_model_from_cfg',
    'register_models',
    # Model classes
    'VAE',
    'DIT',
    'PaliGemmaForConditionalGeneration',
]
