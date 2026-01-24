"""
Unified Inference Script

Run inference with nanoVLA-RL models.

Usage:
    # VAE reconstruction
    python tools/infer.py --config configs/vae.yaml --model vae
    
    # PaliGemma vision-language
    python tools/infer.py --model paligemma --model_path <path> --image <image> --prompt "describe"
"""

import argparse
import torch
import os
import glob
import pickle

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_device(device_arg: str) -> str:
    """Get inference device."""
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_arg == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif device_arg == "cpu":
        return "cpu"
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


def infer_vae(args):
    """Run VAE inference (reconstruction and latent saving)."""
    import torchvision
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    from utils.config import load_config
    from models.vae import VAE
    from preprocessors import CelebDataset, get_dataset_config
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config, search_dir="configs")
    dataset_cfg = get_dataset_config(config["dataset"])
    model_cfg = config.get("model", {}).get("params", {})
    train_cfg = config.get("train", {})
    
    # Create dataset
    im_dataset = CelebDataset(
        split='train',
        im_path=dataset_cfg['path'],
        im_size=dataset_cfg['im_size'],
        im_channels=dataset_cfg['im_channels']
    )
    
    # Load model
    model = VAE(
        im_channels=dataset_cfg['im_channels'],
        model_config=model_cfg
    ).to(device)
    
    ckpt_path = os.path.join(
        train_cfg.get('output_dir', 'outputs'),
        train_cfg.get('vae_autoencoder_ckpt_name', 'vae_autoencoder.pth')
    )
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: No checkpoint found at {ckpt_path}")
    
    model.eval()
    
    # Sample reconstruction
    num_images = train_cfg.get('num_samples', 4)
    ngrid = train_cfg.get('num_grid_rows', 2)
    
    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float().to(device)
    
    output_dir = train_cfg.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        encoded_output, _ = model.encode(ims)
        decoded_output = model.decode(encoded_output)
        
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims_display = (ims + 1) / 2
        
        decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
        input_grid = make_grid(ims_display.cpu(), nrow=ngrid)
        
        decoder_img = torchvision.transforms.ToPILImage()(decoder_grid)
        input_img = torchvision.transforms.ToPILImage()(input_grid)
        
        input_img.save(os.path.join(output_dir, 'input_samples.png'))
        decoder_img.save(os.path.join(output_dir, 'reconstructed_samples.png'))
        
        print(f"Saved reconstruction samples to {output_dir}")
    
    # Optionally save latents
    if train_cfg.get('save_latents', False):
        print("Saving latents...")
        data_loader = DataLoader(im_dataset, batch_size=1, shuffle=False)
        latent_path = os.path.join(output_dir, train_cfg.get('vae_latent_dir_name', 'vae_latents'))
        os.makedirs(latent_path, exist_ok=True)
        
        fname_latent_map = {}
        part_count = 0
        
        for idx, im in enumerate(tqdm(data_loader)):
            _, encoded_output = model.encode(im.float().to(device))
            fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
            
            if (idx + 1) % 1000 == 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path, f'{part_count}.pkl'), 'wb'))
                part_count += 1
                fname_latent_map = {}
        
        if len(fname_latent_map) > 0:
            pickle.dump(fname_latent_map, open(os.path.join(latent_path, f'{part_count}.pkl'), 'wb'))
        
        print(f"Saved latents to {latent_path}")


def _sample_top_p(probs: torch.Tensor, p: float):
    """Sample from top-p (nucleus) distribution."""
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (Subtracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def infer_paligemma(args):
    """Run PaliGemma vision-language inference."""
    from PIL import Image
    
    from utils.io import load_hf_model
    from preprocessors import PaliGemmaProcessor
    from models.paligemma.gemma import KVCache
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if not args.model_path:
        raise ValueError("--model_path required for PaliGemma inference")
    if not args.image:
        raise ValueError("--image required for PaliGemma inference")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_hf_model(args.model_path, device)
    model = model.to(device).eval()
    
    # Get image size from config
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    
    # Create processor
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    
    # Load and process image
    image = Image.open(args.image)
    prompt = args.prompt or "describe this image"
    
    # Process inputs
    model_inputs = processor(text=[prompt], images=[image])
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    
    # Generation parameters
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    do_sample = args.do_sample or True
    
    print(f"\nRunning inference...")
    
    with torch.no_grad():
        kv_cache = KVCache()
        
        # Generate tokens until stop token
        stop_token = processor.tokenizer.eos_token_id
        generated_tokens = []
        
        for _ in range(max_tokens):
            # Get the model outputs
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Sample the next token
            if do_sample:
                # Apply temperature
                next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = _sample_top_p(next_token_logits, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)  # Remove batch dimension
            generated_tokens.append(next_token)
            
            # Stop if the stop token has been generated
            if next_token.item() == stop_token:
                break
            
            # Append the next token to the input
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )
        
        # Decode the generated tokens
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\n{prompt}: {decoded}")


def main():
    parser = argparse.ArgumentParser(description="Unified inference for nanoVLA-RL models")
    parser.add_argument("--model", required=True, choices=["vae", "paligemma"], help="Model type")
    parser.add_argument("--config", help="Config file path (for VAE)")
    parser.add_argument("--model_path", help="Model path (for PaliGemma)")
    parser.add_argument("--image", help="Image path (for PaliGemma)")
    parser.add_argument("--prompt", help="Prompt (for PaliGemma)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    
    args = parser.parse_args()
    
    if args.model == "vae":
        if not args.config:
            raise ValueError("--config required for VAE inference")
        infer_vae(args)
    elif args.model == "paligemma":
        infer_paligemma(args)


if __name__ == "__main__":
    main()
