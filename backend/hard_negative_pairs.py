"""
Report most similar PLU pairs using embedding centroids.

Usage:
  python hard_negative_pairs.py --store-dir backend\\embedding_store_280 --top-k 20
"""

import argparse
from pathlib import Path

import numpy as np

from embedding_store import EmbeddingStore


def parse_args():
    parser = argparse.ArgumentParser(description="Find hard negative PLU pairs.")
    parser.add_argument("--store-dir", required=True, help="Embedding store dir.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of pairs to output.")
    parser.add_argument("--min-count", type=int, default=5, help="Skip PLUs with fewer samples.")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path.")
    return parser.parse_args()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def main():
    args = parse_args()
    store = EmbeddingStore.load(Path(args.store_dir))

    plu_codes = []
    centroids = []
    counts = []

    for plu_code, indices in store.plu_index.items():
        idx_arr = np.array(indices, dtype=int)
        if idx_arr.size < args.min_count:
            continue
        emb = store.embeddings[idx_arr]
        centroid = _normalize(emb.mean(axis=0))
        plu_codes.append(str(plu_code))
        centroids.append(centroid)
        counts.append(idx_arr.size)

    if len(centroids) < 2:
        print("Not enough PLUs to compare.")
        return

    centroids = np.stack(centroids, axis=0)
    sims = centroids @ centroids.T
    np.fill_diagonal(sims, -1.0)

    pairs = []
    n = sims.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((plu_codes[i], plu_codes[j], float(sims[i, j]), counts[i], counts[j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[: args.top_k]

    lines = ["plu_a,plu_b,similarity,count_a,count_b"]
    for a, b, sim, ca, cb in top_pairs:
        lines.append(f"{a},{b},{sim:.4f},{ca},{cb}")

    output = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Saved report to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
