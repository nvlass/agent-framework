#!/usr/bin/env python3
"""
03_clustering_demo.py - Clustering Embeddings with HDBSCAN

This script demonstrates how to cluster text embeddings:
1. Why clustering is useful for memory systems
2. How HDBSCAN works (vs K-means)
3. Interpreting cluster results

Run: python scripts/embed_tests/03_clustering_demo.py
     python scripts/embed_tests/03_clustering_demo.py --model mxbai
"""

import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_memory.embeddings import EmbeddingGenerator
from config import get_model_path, add_model_argument, print_model_info


def main():
    parser = argparse.ArgumentParser(description="Clustering embeddings with HDBSCAN")
    add_model_argument(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("Clustering Text Embeddings with HDBSCAN")
    print("=" * 60)

    # Check for hdbscan
    try:
        import hdbscan
    except ImportError:
        print("\nHDBSCAN not installed. Run: pip install hdbscan")
        return

    model_path = get_model_path(args.model)
    print(f"\nUsing model: {args.model}")
    print_model_info(args.model)

    generator = EmbeddingGenerator(str(model_path))

    # -------------------------------------------------------------------------
    # 1. Create sample data with clear clusters
    # -------------------------------------------------------------------------
    print("\n1. Creating sample data with natural clusters...")

    # Three distinct topics
    python_texts = [
        "Debugging Python TypeError in function",
        "Python script crashes with NoneType error",
        "Fix Python exception handling",
        "Python null pointer equivalent error",
        "TypeError when calling Python method",
    ]

    docker_texts = [
        "Docker container won't start",
        "Fix Docker networking issues",
        "Docker image build fails",
        "Container port mapping problems",
        "Docker compose service errors",
    ]

    git_texts = [
        "Git merge conflict resolution",
        "Undo last git commit",
        "Git branch diverged from main",
        "Resolve git rebase conflicts",
        "Git history cleanup with squash",
    ]

    # Add some noise (outliers)
    noise_texts = [
        "Weather forecast for tomorrow",
        "Best pizza recipes",
    ]

    all_texts = python_texts + docker_texts + git_texts + noise_texts
    true_labels = (
        ["python"] * len(python_texts) +
        ["docker"] * len(docker_texts) +
        ["git"] * len(git_texts) +
        ["noise"] * len(noise_texts)
    )

    print(f"   Total texts: {len(all_texts)}")
    print(f"   Expected clusters: 3 (python, docker, git)")
    print(f"   Noise points: {len(noise_texts)}")

    # -------------------------------------------------------------------------
    # 2. Generate embeddings
    # -------------------------------------------------------------------------
    print("\n2. Generating embeddings...")

    embeddings = np.array([
        generator.generate_embedding(text)
        for text in all_texts
    ])

    print(f"   Embedding matrix shape: {embeddings.shape}")

    # -------------------------------------------------------------------------
    # 3. Run HDBSCAN clustering
    # -------------------------------------------------------------------------
    print("\n3. Running HDBSCAN clustering...")

    """
    HDBSCAN Parameters:
    - min_cluster_size: Minimum points to form a cluster (default: 5)
      Smaller = more clusters, larger = fewer, bigger clusters
    - min_samples: How conservative clustering is (default: None = min_cluster_size)
      Larger = more points marked as noise
    - metric: Distance metric ('euclidean', 'cosine', etc.)
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,  # At least 3 points per cluster
        min_samples=2,       # Conservative noise detection
        metric='euclidean',  # Works well for normalized embeddings
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    # -------------------------------------------------------------------------
    # 4. Analyze results
    # -------------------------------------------------------------------------
    print("\n4. Clustering results:")

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"   Clusters found: {n_clusters}")
    print(f"   Noise points: {n_noise}")

    # Show what's in each cluster
    print("\n   Cluster contents:")
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            label = "NOISE"
        else:
            label = f"Cluster {cluster_id}"

        members = [
            (all_texts[i], true_labels[i])
            for i, c in enumerate(cluster_labels)
            if c == cluster_id
        ]

        print(f"\n   {label} ({len(members)} members):")
        for text, true_label in members:
            print(f"     [{true_label:6}] {text[:50]}...")

    # -------------------------------------------------------------------------
    # 5. Evaluate clustering quality
    # -------------------------------------------------------------------------
    print("\n5. Clustering quality:")

    # Check if clusters match true labels
    cluster_purity = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue

        members = [true_labels[i] for i, c in enumerate(cluster_labels) if c == cluster_id]
        most_common = max(set(members), key=members.count)
        purity = members.count(most_common) / len(members)
        cluster_purity[cluster_id] = (most_common, purity)

    print("   Cluster purity (how homogeneous each cluster is):")
    for cluster_id, (dominant, purity) in cluster_purity.items():
        print(f"     Cluster {cluster_id}: {purity:.0%} {dominant}")

    # -------------------------------------------------------------------------
    # 6. Why HDBSCAN vs K-means?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Why HDBSCAN for Memory Systems?")
    print("=" * 60)
    print("""
    HDBSCAN advantages over K-means:

    1. NO need to specify number of clusters (k)
       - K-means: Must guess k beforehand
       - HDBSCAN: Discovers natural cluster count

    2. Handles NOISE as a first-class concept
       - K-means: Forces every point into a cluster
       - HDBSCAN: Labels outliers as noise (-1)

    3. Finds clusters of VARYING DENSITY
       - K-means: Assumes spherical, equal-size clusters
       - HDBSCAN: Handles irregular cluster shapes

    4. Better for EMBEDDINGS
       - K-means: Sensitive to outliers
       - HDBSCAN: Robust to noise in embedding space

    Key parameters:
    - min_cluster_size: Minimum points for a cluster
    - min_samples: Controls noise sensitivity
    - metric: 'euclidean' or 'cosine' for embeddings
    """)

    # -------------------------------------------------------------------------
    # 7. Accessing cluster information
    # -------------------------------------------------------------------------
    print("\n7. Useful HDBSCAN attributes:")

    print(f"   cluster_labels_: {clusterer.labels_[:5]}... (cluster assignment)")
    print(f"   probabilities_: {clusterer.probabilities_[:5]}... (membership strength)")

    # Find the most "representative" point in each cluster
    print("\n   Most representative text per cluster:")
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue

        # Get indices and probabilities for this cluster
        indices = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        probs = [clusterer.probabilities_[i] for i in indices]

        # Find the one with highest probability
        best_idx = indices[np.argmax(probs)]
        best_text = all_texts[best_idx]

        print(f"     Cluster {cluster_id}: '{best_text[:50]}...'")


if __name__ == "__main__":
    main()
