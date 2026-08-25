#!/usr/bin/env python3
"""
01_basic_embeddings.py - Introduction to Text Embeddings

This script demonstrates the fundamentals of text embeddings:
1. What embeddings are (dense vector representations of text)
2. How to generate them using llama.cpp
3. Basic properties of embedding vectors

Run: python scripts/embed_tests/01_basic_embeddings.py
     python scripts/embed_tests/01_basic_embeddings.py --model mxbai
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_memory.embeddings import EmbeddingGenerator
from config import get_model_path, add_model_argument, print_model_info, AVAILABLE_MODELS


def main():
    parser = argparse.ArgumentParser(description="Introduction to text embeddings")
    add_model_argument(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("Basic Text Embeddings Demo")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load the embedding model
    # -------------------------------------------------------------------------
    print("\n1. Loading embedding model...")
    model_path = get_model_path(args.model)
    print_model_info(args.model)

    generator = EmbeddingGenerator(str(model_path))

    # -------------------------------------------------------------------------
    # 2. Generate a single embedding
    # -------------------------------------------------------------------------
    print("\n2. Generating embedding for a single text...")

    text = "The quick brown fox jumps over the lazy dog."
    embedding = generator.generate_embedding(text)

    print(f"   Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dtype: {embedding.dtype}")
    print(f"   First 10 values: {embedding[:10]}")

    # -------------------------------------------------------------------------
    # 3. Understand embedding properties
    # -------------------------------------------------------------------------
    print("\n3. Embedding properties...")

    # Embeddings are typically normalized (unit length)
    magnitude = np.linalg.norm(embedding)
    print(f"   Magnitude (L2 norm): {magnitude:.4f}")
    print(f"   (Note: ~1.0 means the embedding is normalized)")

    # Statistics
    print(f"   Min value: {embedding.min():.4f}")
    print(f"   Max value: {embedding.max():.4f}")
    print(f"   Mean value: {embedding.mean():.4f}")
    print(f"   Std deviation: {embedding.std():.4f}")

    # -------------------------------------------------------------------------
    # 4. Embedding caching
    # -------------------------------------------------------------------------
    print("\n4. Embedding caching...")

    # Generate same text again - should use cache
    import time

    start = time.time()
    embedding1 = generator.generate_embedding(text)
    time1 = time.time() - start

    start = time.time()
    embedding2 = generator.generate_embedding(text)  # Cached
    time2 = time.time() - start

    print(f"   First generation: {time1*1000:.2f}ms")
    print(f"   Cached retrieval: {time2*1000:.2f}ms")
    print(f"   Speedup: {time1/time2:.1f}x")
    print(f"   Same embedding? {np.allclose(embedding1, embedding2)}")

    # -------------------------------------------------------------------------
    # 5. Different texts produce different embeddings
    # -------------------------------------------------------------------------
    print("\n5. Different texts produce different embeddings...")

    texts = [
        "Python is a programming language",
        "Java is a programming language",
        "I love pizza",
    ]

    embeddings = [generator.generate_embedding(t) for t in texts]

    print("   Texts:")
    for i, t in enumerate(texts):
        print(f"     [{i}] {t}")

    print("\n   Pairwise differences (L2 distance):")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            print(f"     [{i}] vs [{j}]: {dist:.4f}")

    # -------------------------------------------------------------------------
    # 6. Key takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
    1. Embeddings are dense vectors (768 dimensions for this model)
    2. Similar texts have similar embeddings (low distance)
    3. Different texts have different embeddings (high distance)
    4. Embeddings are typically normalized (magnitude ~1.0)
    5. Caching speeds up repeated embedding generation
    """)


if __name__ == "__main__":
    main()
