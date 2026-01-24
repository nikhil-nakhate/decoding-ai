"""
Diffusion Utilities

Provides utility functions for diffusion models.
"""

import pickle
import glob
import os
import torch


def load_latents(latent_path):
    r"""
    Simple utility to load pre-computed latents for faster LDM training.
    
    Args:
        latent_path: Path to directory containing pickled latent files
        
    Returns:
        Dictionary mapping filenames to latent tensors
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps
