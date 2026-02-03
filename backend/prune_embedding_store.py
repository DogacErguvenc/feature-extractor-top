"""
Prune an embedding store by selecting representative embeddings per PLU.

Usage:
  python prune_embedding_store.py --store-dir backend\\embedding_store_280 --out-dir backend\\embedding_store_280_pruned --budget-per-plu 200
"""

import argparse
from pathlib import Path

import numpy as np

from embedding_store import EmbeddingStore, EmbeddingStoreConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Prune embedding store per PLU.")
    parser.add_argument("--store-dir", required=True, help="Input embedding store dir.")
    parser.add_argument("--out-dir", required=True, help="Output pruned store dir.")
    parser.add_argument("--budget-per-plu", type=int, default=200, help="Max embeddings per PLU.")
    parser.add_argument("--min-keep", type=int, default=20, help="Minimum embeddings to keep per PLU.")
    parser.add_argument("--strategy", choices=["fps", "random"], default="fps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def farthest_point_sampling(embeddings: np.ndarray, k: int) -> list[int]:
    n = embeddings.shape[0]
    if k >= n:
        return list(range(n))
    centroid = _normalize(embeddings.mean(axis=0))
    sims = embeddings @ centroid
    first = int(np.argmax(sims))
    selected = [first]
    distances = 1.0 - (embeddings @ embeddings[first])
    for _ in range(1, k):
        idx = int(np.argmax(distances))
        selected.append(idx)
        new_dist = 1.0 - (embeddings @ embeddings[idx])
        distances = np.minimum(distances, new_dist)
    return selected


def main():
    args = parse_args()
    store = EmbeddingStore.load(Path(args.store_dir))
    rng = np.random.default_rng(args.seed)

    keep_indices = []
    new_meta = []
    new_plu_index = {}
    new_embeddings = []

    for plu_code, indices in store.plu_index.items():
        idx_arr = np.array(indices, dtype=int)
        count = idx_arr.size
        if count == 0:
            continue
        budget = max(args.min_keep, min(args.budget_per_plu, count))
        plu_embeddings = store.embeddings[idx_arr]
        if args.strategy == "random":
            chosen_local = rng.choice(count, size=budget, replace=False)
            chosen = idx_arr[chosen_local]
        else:
            chosen_local = farthest_point_sampling(plu_embeddings, budget)
            chosen = idx_arr[np.array(chosen_local, dtype=int)]
        for i in chosen.tolist():
            keep_indices.append(i)

    keep_indices = sorted(set(keep_indices))
    for new_idx, old_idx in enumerate(keep_indices):
        new_embeddings.append(store.embeddings[old_idx])
        meta = store.meta[old_idx]
        new_meta.append(meta)
        plu_code = str(meta.get("plu_code"))
        new_plu_index.setdefault(plu_code, []).append(new_idx)

    new_embeddings = np.stack(new_embeddings, axis=0).astype("float32")
    cfg = store.config
    new_cfg = EmbeddingStoreConfig(
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        scale_size=cfg.scale_size,
        crop_mode=cfg.crop_mode,
        created_at=cfg.created_at,
    )
    EmbeddingStore.save(Path(args.out_dir), new_embeddings, new_meta, new_plu_index, new_cfg)
    print(f"Saved pruned store to {args.out_dir} ({new_embeddings.shape[0]} embeddings).")


if __name__ == "__main__":
    main()
