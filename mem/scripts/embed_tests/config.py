"""
Shared configuration for embedding test scripts.

Supports multiple embedding models for comparison.
"""

import sys
import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Available models: name -> (filename, dimensions, size_mb, notes)
AVAILABLE_MODELS = {
    "nomic": (
        "nomic-embed-text-v1.5.Q8_0.gguf",
        768,
        137,
        "General purpose, fast"
    ),
    "mxbai": (
        "mxbai-embed-large-v1-f16.gguf",
        1024,
        670,
        "Better quality, slower"
    ),
}

DEFAULT_MODEL = "nomic"


def get_model_path(model_name: str) -> Path:
    """
    Get the path for a model, with download instructions if missing.

    Args:
        model_name: One of the keys in AVAILABLE_MODELS

    Returns:
        Path to the model file

    Exits:
        If model not found or unknown model name
    """
    if model_name not in AVAILABLE_MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models:")
        for name, (_, dim, size, notes) in AVAILABLE_MODELS.items():
            print(f"  {name}: {dim} dimensions, ~{size}MB - {notes}")
        sys.exit(1)

    model_file, _, _, _ = AVAILABLE_MODELS[model_name]
    model_path = MODELS_DIR / model_file

    if not model_path.exists():
        print(f"\nModel '{model_name}' not found at {model_path}")
        print_download_instructions(model_name)
        sys.exit(1)

    return model_path


def print_download_instructions(model_name: str):
    """Print download instructions for a specific model"""
    if model_name == "nomic":
        print("\nTo download nomic-embed-text (~137MB):")
        print("  python scripts/setup_embedding_model.py")

    elif model_name == "mxbai":
        print("\nTo download mxbai-embed-large (~670MB, better quality):")
        print("  pip install huggingface_hub")
        print(f"  huggingface-cli download mixedbread-ai/mxbai-embed-large-v1 \\")
        print(f"    --include 'gguf/mxbai-embed-large-v1-f16.gguf' \\")
        print(f"    --local-dir {MODELS_DIR}")
        print(f"\nThen move the file:")
        print(f"  mv {MODELS_DIR}/gguf/mxbai-embed-large-v1-f16.gguf {MODELS_DIR}/")


def add_model_argument(parser: argparse.ArgumentParser):
    """Add the --model argument to an argument parser"""
    parser.add_argument(
        "--model", "-m",
        choices=list(AVAILABLE_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_MODEL})"
    )


def print_model_info(model_name: str):
    """Print information about the selected model"""
    model_file, dim, size, notes = AVAILABLE_MODELS[model_name]
    print(f"Model: {model_name}")
    print(f"  File: {model_file}")
    print(f"  Dimensions: {dim}")
    print(f"  Size: ~{size}MB")
    print(f"  Notes: {notes}")
