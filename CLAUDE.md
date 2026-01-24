# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Critical**: Always use the `llm_agents` conda environment for all Python operations:
```bash
conda activate llm_agents
```

All Python commands, scripts, and terminal operations must run within this environment.

## Project Overview

This is a PyTorch-based project implementing multiple vision and generative models:

1. **VAE (Variational Autoencoder)** - Image compression and latent encoding
2. **DiT (Diffusion Transformer)** - Diffusion-based image generation in latent space
3. **PaliGemma** - Vision-language model combining SigLIP vision encoder with Gemma language model

The project follows a two-stage training approach for generative models:
- Stage 1: Train VAE to compress images into latent representations
- Stage 2: Train DiT in latent space for efficient diffusion-based generation

## Directory Structure

```
nanoVLA-RL/
├── configs/                    # Configuration files
│   ├── _base_/                 # Base configs for inheritance
│   │   └── default.yaml        # Default training parameters
│   ├── datasets.yaml           # Dataset registry
│   └── vae.yaml                # VAE model config
│
├── preprocessors/              # Data loading and processing
│   ├── celeb_dataset.py        # CelebA-HQ dataset
│   ├── registry.py             # Dataset registry
│   └── paligemma.py            # PaliGemma processor
│
├── models/                     # Model architectures (organized by model type)
│   ├── vae/                    # VAE components
│   │   ├── vae.py, blocks.py, discriminator.py, lpips.py
│   ├── dit/                    # Diffusion Transformer
│   │   ├── dit.py, patch_embed.py, transformer_layer.py, attention.py
│   ├── paligemma/              # Vision-Language Model
│   │   ├── gemma.py, siglip.py
│   ├── registry.py             # Model registry
│   └── build.py                # Model builder
│
├── engine/                     # Training infrastructure
│   ├── trainers/               # Trainer classes
│   │   ├── base.py, vae_trainer.py
│   └── schedulers/             # Noise schedulers
│       └── linear_scheduler.py
│
├── utils/                      # Utilities
│   ├── config.py               # Config loading with inheritance
│   ├── diffusion.py            # Diffusion utilities
│   └── io.py                   # Model I/O
│
├── tools/                      # Entry point scripts
│   ├── train.py                # Unified training
│   ├── infer.py                # Unified inference
│   └── sample.py               # Diffusion sampling
│
└── tests/                      # Unit tests
```

## Common Commands

### Training

```bash
# Train VAE
python tools/train.py --config configs/vae.yaml

# Specify device
python tools/train.py --config configs/vae.yaml --device cuda

# Resume from checkpoint
python tools/train.py --config configs/vae.yaml --resume outputs/checkpoint.pt
```

### Inference

```bash
# VAE Reconstruction
python tools/infer.py --model vae --config configs/vae.yaml

# PaliGemma Vision-Language Inference
python tools/infer.py --model paligemma \
  --model_path <path_to_hf_model> \
  --image <path_to_image> \
  --prompt "describe this image" \
  --max_tokens 100
```

### Sampling

```bash
# DiT Image Sampling (requires pre-trained VAE and DiT)
python tools/sample.py --config configs/celebhq.yaml
```

## Configuration

### Config System

Configs support hierarchical inheritance via `extends` keyword:

```yaml
# configs/vae.yaml
extends: _base_/default

dataset: celebhq

model:
  name: vae
  params:
    z_channels: 4
    down_channels: [128, 256, 384]
    # ...
```

### Dataset Registry

`configs/datasets.yaml` maps dataset names to paths:

```yaml
celebhq:
  path: 'data/CelebAMask-HQ'
  im_size: 128
  im_channels: 3
  type: image_folder
```

## Import Patterns

```python
# Models
from models import VAE, DIT, build_model_from_cfg
from models.vae import VAE, Discriminator, LPIPS
from models.dit import DIT, PatchEmbedding
from models.paligemma import PaliGemmaForConditionalGeneration

# Preprocessors
from preprocessors import CelebDataset, get_dataset_config, PaliGemmaProcessor

# Training
from engine.trainers import BaseTrainer, VAETrainer
from engine.schedulers import LinearNoiseScheduler

# Utils
from utils.config import load_config
from utils.io import load_hf_model
```

## Architecture Details

### VAE Architecture (`models/vae/vae.py`)

**Encoder Pipeline:**
- `encoder_conv_in` → DownBlocks → MidBlocks → `encoder_norm_out` → `encoder_conv_out`
- Outputs mean and log-variance for latent distribution

**Decoder Pipeline:**
- `post_quant_conv` → `decoder_conv_in` → MidBlocks → UpBlocks → `decoder_norm_out` → `decoder_conv_out`

**Building Blocks** (`models/vae/blocks.py`):
- `DownBlock`: Convolution layers with optional downsampling and attention
- `MidBlock`: Middle processing layers with attention
- `UpBlock`: Upsampling with transposed convolutions

### DiT Architecture (`models/dit/dit.py`)

1. **Input**: Noisy latent `x` and timestep `t`
2. **Patch Embedding**: Convert latent into patches
3. **Time Embedding**: Sinusoidal positional encoding for timestep
4. **Transformer Layers**: Stack of `TransformerLayer` with AdaLN
5. **Output**: Predicted noise via unpatchify

### PaliGemma Architecture (`models/paligemma/`)

**Vision Tower** (SigLIP in `siglip.py`):
- Patch-based vision transformer

**Language Model** (Gemma in `gemma.py`):
- Decoder-only transformer with GQA and RoPE

## Adding New Models

1. **Create model** in `models/<type>/`:
```python
class MyModel(nn.Module):
    @classmethod
    def from_config(cls, cfg):
        return cls(...)
```

2. **Register** in `models/build.py`:
```python
from .my_model import MyModel
register_model("my_model")(MyModel)
```

3. **Create config** `configs/my_model.yaml`:
```yaml
extends: _base_/default
model:
  name: my_model
  params: {...}
```

4. **Create trainer** (optional) in `engine/trainers/`

## Device Handling

All scripts include device detection with MPS support:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return types
- Write clear docstrings for classes and functions
- Maintain modular, well-organized code structure
