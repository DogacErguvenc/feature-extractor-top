import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from embedding_engine import EmbeddingConfig, EmbeddingEngine
from embedding_store import EmbeddingStore


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embedding accuracy on a val set.")
    parser.add_argument("--val-dir", required=True, help="Validation dataset root.")
    parser.add_argument("--store-dir", default="backend/embedding_store", help="Embedding store dir.")
    parser.add_argument("--model-name", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--scale-size", type=int, default=576)
    parser.add_argument("--crop-mode", default="edge5")
    parser.add_argument("--device", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-sim-start", type=float, default=0.2)
    parser.add_argument("--min-sim-end", type=float, default=0.8)
    parser.add_argument("--min-sim-step", type=float, default=0.02)
    parser.add_argument("--margin-start", type=float, default=0.0)
    parser.add_argument("--margin-end", type=float, default=0.2)
    parser.add_argument("--margin-step", type=float, default=0.02)
    return parser.parse_args()


def iter_images(root: Path):
    for plu_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        plu_code = plu_dir.name
        for path in sorted(plu_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                yield plu_code, path


def topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return -1.0
    k = min(k, values.size)
    if k == 1:
        return float(values.max())
    idx = np.argpartition(values, -k)[-k:]
    return float(values[idx].mean())


def compute_scores(
    query_embeddings: np.ndarray,
    store: EmbeddingStore,
    top_k: int,
):
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings[None, :]
    sims = query_embeddings @ store.embeddings.T
    sims = sims.max(axis=0)
    scores = {}
    for plu_code, indices in store.plu_index.items():
        vals = sims[np.array(indices, dtype=int)]
        score = topk_mean(vals, top_k)
        scores[plu_code] = score
    return scores


def is_match(selected_plu: str, scores: dict, min_sim: float, margin: float) -> bool:
    selected_score = scores[selected_plu]
    best_other = max((v for k, v in scores.items() if k != selected_plu), default=-1.0)
    return selected_score >= min_sim and selected_score >= best_other + margin


def main() -> int:
    args = parse_args()
    val_dir = Path(args.val_dir).resolve()
    store_dir = Path(args.store_dir).resolve()
    if not store_dir.exists():
        print(f"Embedding store not found: {store_dir}")
        print("Run build_embedding_store.py first.")
        return 1

    store = EmbeddingStore.load(store_dir)
    weights_path = Path(args.weights_path).resolve() if args.weights_path else None
    config = EmbeddingConfig(
        model_name=args.model_name,
        weights_path=weights_path,
        image_size=args.image_size,
        scale_size=args.scale_size,
        crop_mode=args.crop_mode,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
    )
    engine = EmbeddingEngine(config)

    samples = []
    for plu_code, path in iter_images(val_dir):
        samples.append((plu_code, path))
    if not samples:
        print("No validation images found.")
        return 1

    classes = sorted(store.plu_index.keys())
    num_classes = len(classes)

    top1_correct = 0
    per_sample_scores = []

    for true_plu, img_path in samples:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        emb = engine.extract_embeddings(img, multi_crop=True)
        scores = compute_scores(emb, store, args.top_k)
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        if pred == true_plu:
            top1_correct += 1
        per_sample_scores.append((true_plu, scores))

    top1_acc = top1_correct / len(samples)
    print(f"top1_accuracy={top1_acc:.4f} samples={len(samples)} classes={num_classes}")

    best = {"acc": -1.0, "min_sim": None, "margin": None, "tar": None, "far": None}
    min_sim_values = np.arange(
        args.min_sim_start, args.min_sim_end + 1e-9, args.min_sim_step
    )
    margin_values = np.arange(
        args.margin_start, args.margin_end + 1e-9, args.margin_step
    )

    total_pos = len(samples)
    total_neg = len(samples) * max(1, (num_classes - 1))

    for min_sim in min_sim_values:
        for margin in margin_values:
            ta = 0
            fa = 0
            for true_plu, scores in per_sample_scores:
                if is_match(true_plu, scores, min_sim, margin):
                    ta += 1
                for wrong_plu in classes:
                    if wrong_plu == true_plu:
                        continue
                    if is_match(wrong_plu, scores, min_sim, margin):
                        fa += 1

            tar = ta / total_pos
            far = fa / total_neg
            acc = (ta + (total_neg - fa)) / (total_pos + total_neg)

            if acc > best["acc"]:
                best = {
                    "acc": acc,
                    "min_sim": float(min_sim),
                    "margin": float(margin),
                    "tar": tar,
                    "far": far,
                }

    print(
        "best_verification: "
        f"acc={best['acc']:.4f} tar={best['tar']:.4f} far={best['far']:.4f} "
        f"min_sim={best['min_sim']:.2f} margin={best['margin']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
