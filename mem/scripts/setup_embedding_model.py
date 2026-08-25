#!/usr/bin/env python3
"""
Download and set up a small embedding model for testing.
Uses nomic-embed-text-v1.5 (Q8_0 quantization, ~137MB)
"""

import os
import sys
from pathlib import Path

# Model details
MODEL_NAME = "nomic-embed-text-v1.5.Q8_0.gguf"
MODEL_URL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
MODEL_DIR = Path(__file__).parent.parent / "models"


def download_model():
    """Download the embedding model using huggingface_hub or requests"""
    model_path = MODEL_DIR / MODEL_NAME

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return model_path

    MODEL_DIR.mkdir(exist_ok=True)

    print(f"Downloading {MODEL_NAME}...")
    print(f"URL: {MODEL_URL}")
    print(f"Destination: {model_path}")
    print("This may take a moment (~137MB)...")

    try:
        # Try huggingface_hub first (cleaner progress bar)
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
            filename="nomic-embed-text-v1.5.Q8_0.gguf",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to {downloaded_path}")
        return Path(downloaded_path)
    except ImportError:
        pass

    # Fallback to requests
    try:
        import requests
        from tqdm import tqdm

        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(model_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=MODEL_NAME
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        print(f"Downloaded to {model_path}")
        return model_path
    except ImportError:
        print("Please install requests and tqdm: pip install requests tqdm")
        print(f"Or manually download from: {MODEL_URL}")
        print(f"And place it in: {MODEL_DIR}")
        sys.exit(1)


def test_model(model_path: Path):
    """Quick test to verify the model works"""
    print("\n--- Testing model ---")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from embeddings import EmbeddingGenerator

    print("Loading model...")
    generator = EmbeddingGenerator(str(model_path))

    # Test embedding generation
    test_texts = [
        "Debugging Python TypeError in function call",
        "Fixing type error in Python code",
        "Deploying Docker container to Kubernetes"
    ]

    print("\nGenerating embeddings for test texts...")
    embeddings = generator.generate_embeddings_batch(test_texts)

    print(f"\nEmbedding dimension: {len(embeddings[0])}")

    # Test similarity
    print("\nSimilarity tests:")
    sim_1_2 = generator.cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = generator.cosine_similarity(embeddings[0], embeddings[2])

    print(f"  'Debugging Python TypeError' vs 'Fixing type error in Python': {sim_1_2:.4f}")
    print(f"  'Debugging Python TypeError' vs 'Deploying Docker to K8s': {sim_1_3:.4f}")

    if sim_1_2 > sim_1_3:
        print("\n  ✓ Similar texts have higher similarity (as expected)")
    else:
        print("\n  ✗ Unexpected: unrelated text has higher similarity")

    return generator


def main():
    print("=== Embedding Model Setup ===\n")

    model_path = download_model()
    generator = test_model(model_path)

    print("\n" + "=" * 50)
    print("Setup complete!")
    print(f"\nTo use the model, set the environment variable:")
    print(f"  export EMBEDDING_MODEL_PATH={model_path}")
    print("\nOr use directly in Python:")
    print(f"  from agent_memory.embeddings import EmbeddingGenerator")
    print(f"  generator = EmbeddingGenerator('{model_path}')")

    return model_path


if __name__ == "__main__":
    main()
