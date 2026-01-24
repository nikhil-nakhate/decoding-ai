"""
VAE (Variational Autoencoder) module

Exports:
    - VAE: Main VAE model
    - Discriminator: PatchGAN discriminator for adversarial training
    - LPIPS: Perceptual loss model
    - DownBlock, MidBlock, UpBlock: Building blocks
"""

from .vae import VAE
from .discriminator import Discriminator
from .lpips import LPIPS
from .blocks import DownBlock, MidBlock, UpBlock, UpBlockUnet, get_time_embedding

__all__ = [
    'VAE',
    'Discriminator',
    'LPIPS',
    'DownBlock',
    'MidBlock',
    'UpBlock',
    'UpBlockUnet',
    'get_time_embedding',
]
