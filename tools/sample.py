"""
Diffusion Sampling Script

Generate samples using trained DiT model.

Usage:
    python tools/sample.py --config configs/celebhq.yaml
"""

import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import VAE
from models.dit import DIT
from engine.schedulers import LinearNoiseScheduler


def get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample(model, scheduler, train_config, dit_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, device):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.to(device).decode(xt)
        else:
            ims = xt
            ims = xt[:, :-1, :, :]

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2

        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        samples_dir = os.path.join(train_config['task_name'], 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        img.save(os.path.join(samples_dir, 'x0_{}.png'.format(i)))
        img.close()


def main():
    parser = argparse.ArgumentParser(description='Generate samples with DiT')
    parser.add_argument('--config', dest='config_path',
                        default='configs/celebhq.yaml', type=str,
                        help='Path to config file')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Device to use')
    args = parser.parse_args()

    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    print(f"Using device: {device}")

    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    # Get latent image size
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    model = DIT(
        im_size=im_size,
        im_channels=autoencoder_model_config['z_channels'],
        config=dit_model_config
    ).to(device)

    model.eval()

    # Load DiT checkpoint
    dit_ckpt_path = os.path.join(train_config['task_name'], train_config['dit_ckpt_name'])
    assert os.path.exists(dit_ckpt_path), f"Train DiT first. No checkpoint at {dit_ckpt_path}"

    model.load_state_dict(torch.load(dit_ckpt_path, map_location=device))
    print('Loaded DiT checkpoint')

    # Create output directories
    os.makedirs(train_config['task_name'], exist_ok=True)

    # Load VAE
    vae = VAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_model_config
    )
    vae.eval()

    vae_ckpt_path = os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])
    assert os.path.exists(vae_ckpt_path), f"VAE checkpoint not present at {vae_ckpt_path}. Train VAE first."

    vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device), strict=True)
    print('Loaded VAE checkpoint')

    # Generate samples
    with torch.no_grad():
        sample(model, scheduler, train_config, dit_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, device)

    print(f"\nSamples saved to {os.path.join(train_config['task_name'], 'samples')}")


if __name__ == '__main__':
    main()
