#!/usr/bin/env python3
"""
02_similarity_exploration.py - Understanding Semantic Similarity

This script explores how cosine similarity works with embeddings:
1. What cosine similarity measures
2. How semantically similar texts score
3. Edge cases and interesting behaviors

Run: python scripts/embed_tests/02_similarity_exploration.py
     python scripts/embed_tests/02_similarity_exploration.py --model mxbai
"""

import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_memory.embeddings import EmbeddingGenerator
from config import get_model_path, add_model_argument, print_model_info


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity = dot(a, b) / (||a|| * ||b||)

    Range: -1 to 1
      1.0 = identical direction (most similar)
      0.0 = orthogonal (unrelated)
     -1.0 = opposite direction (most dissimilar)

    For normalized vectors (||a|| = ||b|| = 1), this simplifies to just dot(a, b)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare_texts(generator, text1: str, text2: str, label: str = ""):
    """Helper to compare two texts and show similarity"""
    emb1 = generator.generate_embedding(text1)
    emb2 = generator.generate_embedding(text2)
    sim = cosine_similarity(emb1, emb2)

    if label:
        print(f"\n   {label}")
    print(f"   Text 1: '{text1}'")
    print(f"   Text 2: '{text2}'")
    print(f"   Similarity: {sim:.4f}")
    return sim


def main():
    parser = argparse.ArgumentParser(description="Explore semantic similarity with embeddings")
    add_model_argument(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("Semantic Similarity Exploration")
    print("=" * 60)

    model_path = get_model_path(args.model)
    print(f"\nUsing model: {args.model}")
    print_model_info(args.model)

    generator = EmbeddingGenerator(str(model_path))

    # -------------------------------------------------------------------------
    # 1. Identical texts
    # -------------------------------------------------------------------------
    print("\n1. IDENTICAL TEXTS (should be ~1.0)")
    compare_texts(
        generator,
        "The cat sat on the mat",
        "The cat sat on the mat"
    )

    # -------------------------------------------------------------------------
    # 2. Semantically similar texts
    # -------------------------------------------------------------------------
    print("\n2. SEMANTICALLY SIMILAR TEXTS (should be high, 0.7-0.9)")

    compare_texts(
        generator,
        "The cat sat on the mat",
        "A cat is sitting on a rug",
        "Same meaning, different words:"
    )

    compare_texts(
        generator,
        "Python is a programming language",
        "Python is used for software development",
        "Related concepts:"
    )

    compare_texts(
        generator,
        "I need to fix a bug in my code",
        "There's an error in my program that needs debugging",
        "Same intent, different phrasing:"
    )

    # -------------------------------------------------------------------------
    # 3. Related but different topics
    # -------------------------------------------------------------------------
    print("\n3. RELATED BUT DIFFERENT (should be moderate, 0.4-0.7)")

    compare_texts(
        generator,
        "Python is a programming language",
        "JavaScript is a programming language",
        "Same category, different items:"
    )

    compare_texts(
        generator,
        "I need to fix a bug in my code",
        "I need to write unit tests",
        "Same domain (coding), different tasks:"
    )

    # -------------------------------------------------------------------------
    # 4. Unrelated texts
    # -------------------------------------------------------------------------
    print("\n4. UNRELATED TEXTS (should be low, 0.0-0.4)")

    compare_texts(
        generator,
        "Python is a programming language",
        "The weather is sunny today",
        "Completely different topics:"
    )

    compare_texts(
        generator,
        "Debug the TypeError in the function",
        "I love eating pizza with friends",
        "Technical vs casual:"
    )

    # -------------------------------------------------------------------------
    # 5. Interesting edge cases
    # -------------------------------------------------------------------------
    print("\n5. INTERESTING EDGE CASES")

    compare_texts(
        generator,
        "The bank is by the river",
        "I need to go to the bank",
        "Polysemy (same word, different meaning):"
    )

    compare_texts(
        generator,
        "not good",
        "bad",
        "Negation and synonyms:"
    )

    compare_texts(
        generator,
        "This is great!",
        "This is terrible!",
        "Opposite sentiments:"
    )

    compare_texts(
        generator,
        "Fix Python TypeError NoneType",
        "Python NoneType TypeError fix",
        "Word order variation:"
    )

    # -------------------------------------------------------------------------
    # 6. Practical application: Finding similar problems
    # -------------------------------------------------------------------------
    print("\n6. PRACTICAL: Finding Similar Problems")
    print("   Given a new problem, find the most similar past experience")

    # Past experiences (our "memory")
    past_experiences = [
        "Fixed Python TypeError by adding null check",
        "Resolved Docker container port conflict",
        "Debugged Git merge conflict in config file",
        "Optimized slow SQL query with indexes",
        "Fixed CORS error in REST API",
    ]

    # New problem
    new_problem = "Getting a NoneType error in my Python script"

    print(f"\n   New problem: '{new_problem}'")
    print("\n   Similarity to past experiences:")

    new_embedding = generator.generate_embedding(new_problem)
    similarities = []

    for exp in past_experiences:
        exp_embedding = generator.generate_embedding(exp)
        sim = cosine_similarity(new_embedding, exp_embedding)
        similarities.append((sim, exp))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True)

    for sim, exp in similarities:
        indicator = "<<<" if sim > 0.6 else ""
        print(f"     {sim:.4f}: {exp} {indicator}")

    # -------------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Similarity Score Guidelines:")
    print("=" * 60)
    print("""
    0.9 - 1.0 : Nearly identical / paraphrases
    0.7 - 0.9 : Very similar / same topic & intent
    0.5 - 0.7 : Related / same domain
    0.3 - 0.5 : Weakly related / some shared concepts
    0.0 - 0.3 : Unrelated / different topics

    Note: These ranges are approximate and model-dependent.
    """)


if __name__ == "__main__":
    main()
