# nanoVLA-RL

A modular, config-driven framework for training vision and generative models with reinforcement learning capabilities.

## Project Overview

nanoVLA-RL is a PyTorch-based framework that combines:
- **Variational Autoencoders (VAE)** for image compression and latent encoding
- **Diffusion Transformers (DiT)** for high-quality image generation in latent space
- **Vision-Language Models (PaliGemma)** for multimodal understanding
- **Config-driven architecture** for easy experimentation and model swapping

The project uses a two-stage generative modeling approach:
1. **Stage 1**: Train VAE to compress images into efficient latent representations
2. **Stage 2**: Train DiT in latent space for diffusion-based generation

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
├── models/                     # Model architectures
│   ├── vae/                    # VAE components
│   │   ├── vae.py              # Main VAE model
│   │   ├── blocks.py           # Building blocks
│   │   ├── discriminator.py    # PatchGAN discriminator
│   │   └── lpips.py            # Perceptual loss
│   ├── dit/                    # Diffusion Transformer
│   │   ├── dit.py              # Main DiT model
│   │   ├── patch_embed.py      # Patch embedding
│   │   ├── transformer_layer.py
│   │   └── attention.py
│   ├── paligemma/              # Vision-Language Model
│   │   ├── gemma.py            # Gemma LLM
│   │   └── siglip.py           # SigLIP vision encoder
│   ├── registry.py             # Model registry
│   └── build.py                # Model builder
│
├── engine/                     # Training infrastructure
│   ├── trainers/               # Trainer classes
│   │   ├── base.py             # Abstract base trainer
│   │   └── vae_trainer.py      # VAE trainer
│   └── schedulers/             # Noise schedulers
│       └── linear_scheduler.py # DDPM scheduler
│
├── utils/                      # Utilities
│   ├── config.py               # Config loading with inheritance
│   ├── diffusion.py            # Diffusion utilities
│   └── io.py                   # Model I/O (HuggingFace loading)
│
├── tools/                      # Entry point scripts
│   ├── train.py                # Unified training
│   ├── infer.py                # Unified inference
│   └── sample.py               # Diffusion sampling
│
└── tests/                      # Unit tests
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd nanoVLA-RL
```

2. **Create and activate conda environment:**
```bash
conda create -n llm_agents python=3.10
conda activate llm_agents
```

3. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
```

### Dataset Setup

Download CelebA-HQ dataset and organize as:
```
data/
└── CelebAMask-HQ/
    └── CelebA-HQ-img/
        ├── 0.png
        ├── 1.png
        └── ...
```

Update `configs/datasets.yaml` with your data path.

## Quick Start

### Training

**Train VAE:**
```bash
python tools/train.py --config configs/vae.yaml --device cuda
```

**Resume from checkpoint:**
```bash
python tools/train.py --config configs/vae.yaml --resume outputs/checkpoint.pt
```

### Inference

**VAE Reconstruction:**
```bash
python tools/infer.py --model vae --config configs/vae.yaml
```

**PaliGemma Vision-Language:**
```bash
python tools/infer.py --model paligemma \
  --model_path <path_to_model> \
  --image <image_path> \
  --prompt "describe this image"
```

### Sampling

**DiT Image Generation:**
```bash
python tools/sample.py --config configs/celebhq.yaml
```

## Configuration

### Config Inheritance

Configs support inheritance via the `extends` keyword:

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

train:
  epochs: 3
  lr: 0.00001
```

### Dataset Registry

Central `configs/datasets.yaml` maps dataset names to paths:

```yaml
celebhq:
  path: 'data/CelebAMask-HQ'
  im_size: 128
  im_channels: 3
  type: image_folder
```

## Import Conventions

```python
# Models
from models import VAE, DIT, build_model_from_cfg
from models.vae import VAE, Discriminator, LPIPS
from models.dit import DIT, PatchEmbedding

# Preprocessors
from preprocessors import CelebDataset, get_dataset_config, PaliGemmaProcessor

# Training
from engine.trainers import BaseTrainer, VAETrainer
from engine.schedulers import LinearNoiseScheduler

# Utils
from utils.config import load_config
from utils.io import load_hf_model
```

## Adding New Models

1. **Create model file** in `models/<model_type>/`:
```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ...

    @classmethod
    def from_config(cls, cfg):
        return cls(...)
```

2. **Register in `models/build.py`:**
```python
from .my_model import MyModel
register_model("my_model")(MyModel)
```

3. **Create config file:**
```yaml
# configs/my_model.yaml
extends: _base_/default
dataset: celebhq

model:
  name: my_model
  params:
    # model-specific parameters
```

4. **Train:**
```bash
python tools/train.py --config configs/my_model.yaml
```

## Model Details

### VAE (Variational Autoencoder)
- **Architecture**: Encoder-decoder with latent sampling
- **Losses**: Reconstruction (MSE), KL divergence, Perceptual (LPIPS), Adversarial

### DiT (Diffusion Transformer)
- **Architecture**: Vision transformer with adaptive layer normalization (AdaLN)
- **Training**: DDPM-style diffusion in latent space

### PaliGemma
- **Vision encoder**: SigLIP (patch-based ViT)
- **Language model**: Gemma (decoder-only transformer with GQA, RoPE)

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write clear docstrings for classes and methods

## License

See LICENSE file for details.
