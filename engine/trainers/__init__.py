"""
Trainers module

Provides training classes for different model types.
"""

from .base import BaseTrainer
from .vae_trainer import VAETrainer

__all__ = [
    'BaseTrainer',
    'VAETrainer',
]
