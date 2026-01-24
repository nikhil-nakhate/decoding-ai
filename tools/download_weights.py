"""
Download Model Weights from HuggingFace

Downloads pre-trained model weights and saves them locally.

Usage:
    # Download PaliGemma (requires HF token for gated models)
    python tools/download_weights.py --model paligemma --output weights/paligemma
    
    # With specific variant
    python tools/download_weights.py --model paligemma --variant 3b-pt-224 --output weights/paligemma-3b
    
    # List available models
    python tools/download_weights.py --list
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Available models and their HuggingFace repo IDs
AVAILABLE_MODELS = {
    "paligemma": {
        "description": "PaliGemma Vision-Language Model",
        "variants": {
            "3b-pt-224": "google/paligemma-3b-pt-224",
            "3b-pt-448": "google/paligemma-3b-pt-448",
            "3b-pt-896": "google/paligemma-3b-pt-896",
            "3b-mix-224": "google/paligemma-3b-mix-224",
            "3b-mix-448": "google/paligemma-3b-mix-448",
        },
        "default_variant": "3b-pt-224",
        "gated": True,  # Requires HF token
    },
    "paligemma2": {
        "description": "PaliGemma 2 Vision-Language Model",
        "variants": {
            "3b-pt-224": "google/paligemma2-3b-pt-224",
            "3b-pt-448": "google/paligemma2-3b-pt-448",
            "3b-pt-896": "google/paligemma2-3b-pt-896",
        },
        "default_variant": "3b-pt-224",
        "gated": True,
    },
}


def list_models():
    """List all available models and variants."""
    print("\nAvailable Models:")
    print("=" * 60)
    
    for model_name, model_info in AVAILABLE_MODELS.items():
        print(f"\n{model_name}")
        print(f"  Description: {model_info['description']}")
        print(f"  Gated (requires HF token): {model_info['gated']}")
        print(f"  Default variant: {model_info['default_variant']}")
        print(f"  Available variants:")
        for variant, repo_id in model_info['variants'].items():
            default_marker = " (default)" if variant == model_info['default_variant'] else ""
            print(f"    - {variant}: {repo_id}{default_marker}")
    
    print("\n" + "=" * 60)
    print("\nUsage examples:")
    print("  python tools/download_weights.py --model paligemma")
    print("  python tools/download_weights.py --model paligemma --variant 3b-pt-448")
    print("  python tools/download_weights.py --model paligemma --output weights/my-paligemma")


def download_model(model_name: str, variant: str = None, output_dir: str = None, token: str = None):
    """Download model from HuggingFace.
    
    Args:
        model_name: Name of the model (e.g., 'paligemma')
        variant: Specific variant (e.g., '3b-pt-224')
        output_dir: Output directory for weights
        token: HuggingFace token for gated models
    """
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)
    
    # Validate model
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        sys.exit(1)
    
    model_info = AVAILABLE_MODELS[model_name]
    
    # Get variant
    if variant is None:
        variant = model_info['default_variant']
    
    if variant not in model_info['variants']:
        print(f"Error: Unknown variant '{variant}' for model '{model_name}'")
        print(f"Available variants: {list(model_info['variants'].keys())}")
        sys.exit(1)
    
    repo_id = model_info['variants'][variant]
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join("weights", f"{model_name}-{variant}")
    
    print(f"\nDownloading {model_name} ({variant})")
    print(f"  Repository: {repo_id}")
    print(f"  Output: {output_dir}")
    print()
    
    # Handle gated models
    if model_info['gated']:
        if token:
            login(token=token)
            print("Logged in with provided token")
        else:
            # Try to use cached token
            try:
                from huggingface_hub import HfFolder
                cached_token = HfFolder.get_token()
                if cached_token:
                    print("Using cached HuggingFace token")
                else:
                    print("\n" + "=" * 60)
                    print("WARNING: This is a gated model that requires authentication.")
                    print("\nTo access PaliGemma models:")
                    print("1. Go to https://huggingface.co/google/paligemma-3b-pt-224")
                    print("2. Accept the license agreement")
                    print("3. Create a HuggingFace token at https://huggingface.co/settings/tokens")
                    print("4. Run: huggingface-cli login")
                    print("   OR pass --token YOUR_TOKEN to this script")
                    print("=" * 60 + "\n")
            except:
                pass
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download
    try:
        print("Starting download... (this may take a while)")
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"\nDownload complete!")
        print(f"Weights saved to: {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                rel_path = os.path.relpath(filepath, local_dir)
                print(f"  {rel_path} ({size_mb:.1f} MB)")
        
        print(f"\n" + "=" * 60)
        print("To run inference:")
        print(f"  python tools/infer.py --model paligemma \\")
        print(f"    --model_path {output_dir} \\")
        print(f"    --image <path_to_image> \\")
        print(f"    --prompt 'describe this image'")
        print("=" * 60)
        
        return local_dir
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("\nThis appears to be an authentication error.")
            print("Make sure you have:")
            print("1. Accepted the model license on HuggingFace")
            print("2. Logged in with: huggingface-cli login")
            print("   OR provided a token with --token")
        
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # List available models
  %(prog)s --model paligemma                # Download default PaliGemma
  %(prog)s --model paligemma --variant 3b-pt-448  # Specific variant
  %(prog)s --model paligemma --token hf_xxx # With HF token
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model to download (e.g., paligemma)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Model variant (e.g., 3b-pt-224, 3b-pt-448)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: weights/<model>-<variant>)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token for gated models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and variants"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if not args.model:
        parser.print_help()
        print("\nError: --model is required (or use --list to see available models)")
        sys.exit(1)
    
    download_model(
        model_name=args.model,
        variant=args.variant,
        output_dir=args.output,
        token=args.token,
    )


if __name__ == "__main__":
    main()
