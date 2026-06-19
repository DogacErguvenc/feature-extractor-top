"""
Rank mixed photos by distance to the dominant ResNet feature centroid.

This is a dataset-cleaning helper, not a production PLU predictor. It does not
train or modify the ResNet model and it never deletes source photos.

Usage:
  python .\rank_butcher_centroid.py --config .\butcher_config.yaml --input-dir "D:\mixed_photos" --out-dir "D:\ranked_output"

Output:
  centroid_ranking.csv
  summary.json
  bands\01_nearest\...
  bands\05_farthest\...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np

from butcher_runtime import embed_pil_image, load_runtime_config


DEFAULT_CONFIG_PATH = (Path(__file__).parent / "butcher_config.yaml").resolve()
DEFAULT_OUT_DIR = (Path(__file__).parent / "centroid_ranked").resolve()
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_BAND_NAMES = [
    "01_nearest",
    "02_near",
    "03_middle",
    "04_far",
    "05_farthest",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank mixed images by similarity to their ResNet centroid."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with mixed images.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan image files in subfolders.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["bands", "ranked", "none"],
        default="bands",
        help="Copy reviewed images into band folders, one ranked folder, or do not copy images.",
    )
    parser.add_argument("--bands", type=int, default=5, help="Number of similarity bands.")
    return parser.parse_args()


def iter_images(input_dir: Path, recursive: bool):
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return (vec / norm).astype("float32")


def band_name(index: int, band_count: int) -> str:
    if band_count == 5:
        return DEFAULT_BAND_NAMES[index]
    return f"{index + 1:02d}_band"


def safe_copy_name(rank: int, similarity: float, source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
    return f"{rank:06d}_sim_{similarity:.4f}_{digest}{source_path.suffix.lower()}"


def copy_ranked_images(records: list[dict], out_dir: Path, copy_mode: str, band_count: int) -> None:
    if copy_mode == "none":
        return

    if copy_mode == "ranked":
        target_root = out_dir / "ranked"
        target_root.mkdir(parents=True, exist_ok=True)
        for rec in records:
            dst = target_root / safe_copy_name(rec["rank"], rec["similarity"], Path(rec["source_path"]))
            shutil.copy2(rec["source_path"], dst)
        return

    target_root = out_dir / "bands"
    for rec in records:
        band = int(rec["band_index"])
        target_dir = target_root / band_name(band, band_count)
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / safe_copy_name(rec["rank"], rec["similarity"], Path(rec["source_path"]))
        shutil.copy2(rec["source_path"], dst)


def write_csv(records: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "rank",
        "similarity",
        "distance",
        "band",
        "filename",
        "source_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec[key] for key in fieldnames})


def run_centroid_ranking(
    config_path: Path,
    input_dir: Path,
    out_dir: Path,
    recursive: bool = False,
    copy_mode: str = "bands",
    bands: int = 5,
    progress_callback=None,
) -> dict:
    config_path = Path(config_path).resolve()
    input_dir = Path(input_dir).resolve()
    out_dir = Path(out_dir).resolve()
    band_count = max(1, int(bands))

    if copy_mode not in {"bands", "ranked", "none"}:
        raise ValueError("copy_mode must be one of: bands, ranked, none")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    runtime_cfg = load_runtime_config(config_path)
    if not runtime_cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Butcher checkpoint not found: {runtime_cfg.checkpoint_path}")
    if not runtime_cfg.class_map_path.exists():
        raise FileNotFoundError(f"Butcher class map not found: {runtime_cfg.class_map_path}")

    image_paths = list(iter_images(input_dir, recursive))
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[np.ndarray] = []
    valid_paths: list[Path] = []
    skipped: list[dict] = []

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            with Image.open(image_path) as img:
                vec = embed_pil_image(img, config_path)
            embeddings.append(normalize(vec))
            valid_paths.append(image_path)
            if progress_callback is not None:
                progress_callback(idx, len(image_paths))
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(image_paths)} images...")
        except Exception as exc:
            skipped.append({"path": str(image_path), "error": str(exc)})
            print(f"Skip {image_path}: {exc}")
            if progress_callback is not None:
                progress_callback(idx, len(image_paths))

    if not embeddings:
        raise RuntimeError("No embeddings created; check input images.")

    matrix = np.stack(embeddings, axis=0).astype("float32")
    centroid = normalize(matrix.mean(axis=0))
    similarities = matrix @ centroid

    order = np.argsort(-similarities)
    records: list[dict] = []
    total = int(order.size)
    for sorted_pos, old_idx in enumerate(order.tolist(), start=1):
        sim = float(similarities[old_idx])
        band_index = min(band_count - 1, int((sorted_pos - 1) * band_count / total))
        records.append(
            {
                "rank": sorted_pos,
                "similarity": round(sim, 6),
                "distance": round(1.0 - sim, 6),
                "band_index": band_index,
                "band": band_name(band_index, band_count),
                "filename": valid_paths[old_idx].name,
                "source_path": str(valid_paths[old_idx]),
            }
        )

    write_csv(records, out_dir / "centroid_ranking.csv")
    copy_ranked_images(records, out_dir, copy_mode, band_count)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(runtime_cfg.checkpoint_path),
        "image_count": len(image_paths),
        "processed_count": len(records),
        "skipped_count": len(skipped),
        "copy_mode": copy_mode,
        "bands": band_count,
        "similarity_min": round(float(similarities.min()), 6),
        "similarity_max": round(float(similarities.max()), 6),
        "similarity_mean": round(float(similarities.mean()), 6),
        "skipped": skipped,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved ranking CSV: {out_dir / 'centroid_ranking.csv'}")
    print(f"Saved summary: {out_dir / 'summary.json'}")
    if copy_mode != "none":
        print(f"Copied review images under: {out_dir}")
    summary["output_dir"] = str(out_dir)
    summary["ranking_csv"] = str(out_dir / "centroid_ranking.csv")
    summary["summary_path"] = str(out_dir / "summary.json")
    return summary


def main():
    args = parse_args()
    run_centroid_ranking(
        config_path=Path(args.config),
        input_dir=Path(args.input_dir),
        out_dir=Path(args.out_dir),
        recursive=args.recursive,
        copy_mode=args.copy_mode,
        bands=args.bands,
    )


if __name__ == "__main__":
    main()
