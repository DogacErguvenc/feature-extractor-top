"""
Build a cosine-similarity embedding store from the existing Butcher ResNet18 checkpoint.

This does not train or modify the ResNet model. It loads the checkpoint configured by
butcher_config.yaml and saves 512-d ResNet feature vectors for labeled reference images.

Usage:
  py -3 backend/build_butcher_embedding_store.py --config backend/butcher_config.yaml
  py -3 backend/build_butcher_embedding_store.py --data-dir C:\datasets_roi\train --out-dir backend\butcher_embedding_store
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np
import yaml

from butcher_runtime import embed_pil_image, load_runtime_config
from embedding_store import EmbeddingStore, EmbeddingStoreConfig


DEFAULT_CONFIG_PATH = (Path(__file__).parent / "butcher_config.yaml").resolve()
DEFAULT_OUT_DIR = (Path(__file__).parent / "butcher_embedding_store").resolve()
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build ResNet18 feature embedding store.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir", type=str, default="", help="Root folder with PLU subfolders.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--plu-map", type=str, default="", help="Optional JSON map of PLU to name.")
    return parser.parse_args()


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _default_data_dir(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {config_path}")

    data_cfg = cfg.get("data", {}) or {}
    root_raw = data_cfg.get("root_dir")
    if not root_raw:
        raise ValueError("Config data.root_dir is required when --data-dir is omitted.")

    root = _resolve_path(root_raw, config_path.parent)
    train_dir = root / "train"
    return train_dir if train_dir.exists() else root


def iter_images(data_dir: Path):
    for plu_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        plu_code = plu_dir.name
        files = []
        for ext in SUPPORTED_EXTS:
            files.extend(plu_dir.glob(f"*{ext}"))
            files.extend(plu_dir.glob(f"*{ext.upper()}"))
        for image_path in sorted(files):
            yield plu_code, image_path


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else _default_data_dir(config_path)
    out_dir = Path(args.out_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    plu_map = {}
    if args.plu_map:
        plu_map = json.loads(Path(args.plu_map).read_text(encoding="utf-8"))

    runtime_cfg = load_runtime_config(config_path)
    if not runtime_cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Butcher checkpoint not found: {runtime_cfg.checkpoint_path}")
    if not runtime_cfg.class_map_path.exists():
        raise FileNotFoundError(f"Butcher class map not found: {runtime_cfg.class_map_path}")

    embeddings = []
    meta = []
    plu_index = {}
    count = 0

    for plu_code, image_path in iter_images(data_dir):
        try:
            with Image.open(image_path) as img:
                vec = embed_pil_image(img, config_path)
            embeddings.append(vec)
            meta.append(
                {
                    "plu_code": str(plu_code),
                    "plu_name": plu_map.get(str(plu_code), ""),
                    "filename": image_path.name,
                    "source_path": str(image_path),
                }
            )
            plu_index.setdefault(str(plu_code), []).append(count)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} images...")
        except Exception as exc:
            print(f"Skip {image_path}: {exc}")

    if not embeddings:
        raise RuntimeError("No embeddings created; check input data.")

    embeddings_arr = np.stack(embeddings, axis=0).astype("float32")
    store_cfg = EmbeddingStoreConfig(
        model_name=f"butcher_{runtime_cfg.backbone_name}_features",
        image_size=runtime_cfg.img_size,
        scale_size=runtime_cfg.img_size,
        crop_mode="resize",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    EmbeddingStore.save(out_dir, embeddings_arr, meta, plu_index, store_cfg)
    print(f"Saved ResNet embedding store to {out_dir} ({embeddings_arr.shape[0]} embeddings).")


if __name__ == "__main__":
    main()
