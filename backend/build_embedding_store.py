"""
Build a local embedding store from labeled images.

Folder layout example:
  datasets/embedding/201/*.jpg
  datasets/embedding/202/*.jpg

Usage:
  python build_embedding_store.py --data-dir ..\\datasets\\train --out-dir .\\embedding_store
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np

from embedding_engine import EmbeddingConfig, EmbeddingEngine
from embedding_store import EmbeddingStore, EmbeddingStoreConfig

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build embedding store from labeled images.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder with PLU subfolders.")
    parser.add_argument("--out-dir", type=str, default="embedding_store", help="Output store directory.")
    parser.add_argument("--model-name", type=str, default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--weights-path", type=str, default="")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--scale-size", type=int, default=576)
    parser.add_argument("--crop-mode", type=str, default="edge5")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--plu-map", type=str, default="", help="Optional JSON map of PLU to name.")
    return parser.parse_args()


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def iter_images(data_dir: Path):
    for plu_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        plu_code = plu_dir.name
        files = []
        for ext in SUPPORTED_EXTS:
            files.extend(plu_dir.glob(f"*{ext}"))
            files.extend(plu_dir.glob(f"*{ext.upper()}"))
        for img_path in sorted(files):
            yield plu_code, img_path


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    plu_map = {}
    if args.plu_map:
        plu_map = json.loads(Path(args.plu_map).read_text(encoding="utf-8"))

    device = args.device or ("cuda" if _has_cuda() else "cpu")
    config = EmbeddingConfig(
        model_name=args.model_name,
        weights_path=Path(args.weights_path) if args.weights_path else None,
        image_size=args.image_size,
        scale_size=args.scale_size,
        crop_mode=args.crop_mode,
        device=device,
    )
    engine = EmbeddingEngine(config)

    embeddings = []
    meta = []
    plu_index = {}
    count = 0

    for plu_code, img_path in iter_images(data_dir):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                vec = engine.embed_image(img, multi_crop=True)
            embeddings.append(vec)
            meta.append(
                {
                    "plu_code": plu_code,
                    "plu_name": plu_map.get(plu_code, ""),
                    "filename": img_path.name,
                    "source_path": str(img_path),
                }
            )
            plu_index.setdefault(plu_code, []).append(count)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} images...")
        except Exception as e:
            print(f"Skip {img_path}: {e}")

    if not embeddings:
        raise RuntimeError("No embeddings created; check input data.")

    embeddings_arr = np.stack(embeddings, axis=0).astype("float32")
    store_cfg = EmbeddingStoreConfig(
        model_name=args.model_name,
        image_size=args.image_size,
        scale_size=args.scale_size,
        crop_mode=args.crop_mode,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    EmbeddingStore.save(out_dir, embeddings_arr, meta, plu_index, store_cfg)
    print(f"Saved embedding store to {out_dir}")


if __name__ == "__main__":
    main()
