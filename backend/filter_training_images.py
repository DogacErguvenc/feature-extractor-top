"""
Quick quality filter for PLU image folders.

Usage:
  python filter_training_images.py --data-dir C:\feature-extractor\datasets\train --report quality_report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter low-quality images for training.")
    parser.add_argument("--data-dir", required=True, help="Root folder with PLU subfolders.")
    parser.add_argument("--report", default="quality_report.csv", help="CSV report path.")
    parser.add_argument("--review-dir", default="review_bad", help="Where to move bad files.")
    parser.add_argument("--move-bad", action="store_true", help="Move bad files to review-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Do not move files, only report.")
    parser.add_argument("--min-dim", type=int, default=256, help="Min width/height in pixels.")
    parser.add_argument("--min-area", type=int, default=0, help="Min area (w*h) in pixels.")
    parser.add_argument("--min-filesize", type=int, default=0, help="Min file size in bytes.")
    parser.add_argument("--min-blur", type=float, default=80.0, help="Min Laplacian variance.")
    parser.add_argument("--min-brightness", type=float, default=35.0, help="Min mean gray value.")
    parser.add_argument("--max-brightness", type=float, default=220.0, help="Max mean gray value.")
    parser.add_argument("--min-contrast", type=float, default=10.0, help="Min gray std-dev.")
    parser.add_argument("--dedupe", action="store_true", help="Flag duplicate images per PLU.")
    parser.add_argument("--dedupe-hamming", type=int, default=0, help="Max dHash distance.")
    parser.add_argument("--max-per-plu", type=int, default=0, help="Keep only top-N per PLU.")
    return parser.parse_args()


def iter_images(root: Path):
    for plu_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        plu_code = plu_dir.name
        for path in sorted(plu_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                yield plu_code, path


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized)
    diff = arr[:, 1:] > arr[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def analyze_image(
    path: Path,
    dedupe: bool,
) -> dict:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        arr = np.asarray(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        img_hash = dhash(img) if dedupe else None
    return {
        "width": width,
        "height": height,
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "hash": img_hash,
    }


def safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst
    stem = dst.stem
    suffix = dst.suffix
    for i in range(1, 1000):
        candidate = dst.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
    raise RuntimeError(f"Could not find free name for {dst}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return 1

    records: List[dict] = []
    by_plu: Dict[str, List[dict]] = {}
    errors = 0

    for plu_code, path in iter_images(data_dir):
        rec = {
            "plu_code": plu_code,
            "path": str(path),
            "filename": path.name,
            "filesize": path.stat().st_size,
            "width": 0,
            "height": 0,
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "hash": None,
            "reasons": [],
        }
        try:
            metrics = analyze_image(path, args.dedupe)
            rec.update(metrics)
        except Exception as exc:
            rec["reasons"].append(f"read_error:{exc.__class__.__name__}")
            errors += 1
        records.append(rec)
        by_plu.setdefault(plu_code, []).append(rec)

    for rec in records:
        if rec["reasons"]:
            continue
        width = rec["width"]
        height = rec["height"]
        if args.min_dim and min(width, height) < args.min_dim:
            rec["reasons"].append("small_dim")
        if args.min_area and width * height < args.min_area:
            rec["reasons"].append("small_area")
        if args.min_filesize and rec["filesize"] < args.min_filesize:
            rec["reasons"].append("small_file")
        if args.min_blur and rec["blur"] < args.min_blur:
            rec["reasons"].append("blurry")
        if rec["brightness"] < args.min_brightness:
            rec["reasons"].append("too_dark")
        if rec["brightness"] > args.max_brightness:
            rec["reasons"].append("too_bright")
        if rec["contrast"] < args.min_contrast:
            rec["reasons"].append("low_contrast")

    if args.dedupe:
        for plu_code, items in by_plu.items():
            seen: List[int] = []
            for rec in items:
                if rec["reasons"]:
                    continue
                img_hash = rec["hash"]
                if img_hash is None:
                    continue
                is_dup = False
                for prev in seen:
                    if hamming_distance(img_hash, prev) <= args.dedupe_hamming:
                        is_dup = True
                        break
                if is_dup:
                    rec["reasons"].append("duplicate")
                else:
                    seen.append(img_hash)

    if args.max_per_plu > 0:
        for plu_code, items in by_plu.items():
            good = [r for r in items if not r["reasons"]]
            if len(good) <= args.max_per_plu:
                continue
            good.sort(key=lambda r: (r["blur"], r["contrast"]), reverse=True)
            for rec in good[args.max_per_plu:]:
                rec["reasons"].append("over_limit")

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "plu_code",
                "path",
                "filename",
                "filesize",
                "width",
                "height",
                "blur",
                "brightness",
                "contrast",
                "reasons",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec["plu_code"],
                    rec["path"],
                    rec["filename"],
                    rec["filesize"],
                    rec["width"],
                    rec["height"],
                    f"{rec['blur']:.2f}",
                    f"{rec['brightness']:.2f}",
                    f"{rec['contrast']:.2f}",
                    "|".join(rec["reasons"]),
                ]
            )

    bad = [r for r in records if r["reasons"]]
    print(f"Total images: {len(records)}")
    print(f"Bad images: {len(bad)}")
    if errors:
        print(f"Read errors: {errors}")
    print(f"Report: {report_path}")

    if args.move_bad and not args.dry_run:
        review_dir = Path(args.review_dir).resolve()
        moved = 0
        for rec in bad:
            src = Path(rec["path"])
            dst = review_dir / rec["plu_code"] / src.name
            try:
                safe_move(src, dst)
                moved += 1
            except Exception as exc:
                print(f"Move failed: {src} -> {dst} ({exc})")
        print(f"Moved {moved} files to {review_dir}")
    elif args.move_bad and args.dry_run:
        print("Dry-run enabled: no files moved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
