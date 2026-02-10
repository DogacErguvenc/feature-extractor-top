import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply tray ROI crop to dataset splits (e.g., train/val) without touching originals."
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="Source dataset root containing split folders (example: C:\\feature-extractor\\datasets).",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Destination dataset root for cropped outputs.",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated split names under src-root.",
    )
    parser.add_argument(
        "--roi-config",
        default=str(Path(__file__).parent / "tray_roi.json"),
        help="Path to tray ROI json file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination images if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing files.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when output extension is .jpg/.jpeg.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at first failed file.",
    )
    return parser.parse_args()


def _validate_roi_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("ROI config must be a JSON object")

    points = raw.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("ROI config requires exactly 4 points")

    normalized_points = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each point must be [x, y]")
        normalized_points.append([float(point[0]), float(point[1])])

    source_size = raw.get("source_size")
    if source_size is not None:
        if not isinstance(source_size, list) or len(source_size) != 2:
            raise ValueError("source_size must be [width, height]")
        source_size = [int(source_size[0]), int(source_size[1])]
        if source_size[0] <= 0 or source_size[1] <= 0:
            raise ValueError("source_size must be positive")

    output_size = raw.get("output_size")
    if output_size is not None:
        if not isinstance(output_size, list) or len(output_size) != 2:
            raise ValueError("output_size must be [width, height]")
        output_size = [int(output_size[0]), int(output_size[1])]
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output_size must be positive")

    return {
        "points": normalized_points,
        "source_size": source_size,
        "output_size": output_size,
    }


def _order_points(points: np.ndarray) -> np.ndarray:
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _roi_points_to_pixels(
    points: List[List[float]], width: int, height: int, source_size: Optional[List[int]]
) -> np.ndarray:
    arr = np.array(points, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("ROI points must be shape (4,2)")

    if np.max(arr) <= 1.5 and np.min(arr) >= -0.1:
        arr[:, 0] = arr[:, 0] * float(width)
        arr[:, 1] = arr[:, 1] * float(height)
    elif source_size and len(source_size) == 2 and source_size[0] > 0 and source_size[1] > 0:
        sx = float(width) / float(source_size[0])
        sy = float(height) / float(source_size[1])
        arr[:, 0] = arr[:, 0] * sx
        arr[:, 1] = arr[:, 1] * sy

    arr[:, 0] = np.clip(arr[:, 0], 0, max(0, width - 1))
    arr[:, 1] = np.clip(arr[:, 1], 0, max(0, height - 1))
    return _order_points(arr.astype(np.float32))


def _compute_output_size(points_px: np.ndarray, output_size: Optional[List[int]]) -> Tuple[int, int]:
    if output_size and len(output_size) == 2:
        return int(output_size[0]), int(output_size[1])

    tl, tr, br, bl = points_px
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    out_w = int(max(width_a, width_b))
    out_h = int(max(height_a, height_b))
    return max(1, out_w), max(1, out_h)


def _apply_roi_to_image_array(arr: np.ndarray, cfg: dict) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB image array")

    h, w = arr.shape[:2]
    pts = _roi_points_to_pixels(cfg["points"], w, h, cfg.get("source_size"))
    area = abs(float(cv2.contourArea(pts)))
    if area < 10.0:
        raise ValueError("ROI polygon area is too small")

    out_w, out_h = _compute_output_size(pts, cfg.get("output_size"))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        arr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )


def _iter_images(split_dir: Path) -> Iterable[Path]:
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _save_image(dst: Path, arr: np.ndarray, jpeg_quality: int) -> None:
    img = Image.fromarray(arr)
    suffix = dst.suffix.lower()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".jpg", ".jpeg"}:
        img.save(dst, format="JPEG", quality=int(jpeg_quality))
    elif suffix == ".png":
        img.save(dst, format="PNG")
    elif suffix == ".webp":
        img.save(dst, format="WEBP", quality=int(jpeg_quality))
    elif suffix == ".bmp":
        img.save(dst, format="BMP")
    else:
        img.save(dst)


def _load_roi_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"ROI config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _validate_roi_config(raw)


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    roi_path = Path(args.roi_config).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not src_root.exists():
        print(f"Source root not found: {src_root}")
        return 1
    if not splits:
        print("No splits provided")
        return 1

    try:
        roi_cfg = _load_roi_config(roi_path)
    except Exception as exc:
        print(f"Failed to load ROI config: {exc}")
        return 1

    total = 0
    converted = 0
    skipped = 0
    failed = 0
    by_split: Dict[str, Dict[str, int]] = {}

    for split in splits:
        split_src = src_root / split
        if not split_src.exists():
            print(f"Skip split (not found): {split_src}")
            continue

        by_split.setdefault(split, {"total": 0, "converted": 0, "skipped": 0, "failed": 0})
        for src_img in _iter_images(split_src):
            total += 1
            by_split[split]["total"] += 1

            rel = src_img.relative_to(src_root)
            dst_img = dst_root / rel

            if dst_img.exists() and not args.overwrite:
                skipped += 1
                by_split[split]["skipped"] += 1
                continue

            if args.dry_run:
                converted += 1
                by_split[split]["converted"] += 1
                continue

            try:
                with Image.open(src_img) as img:
                    arr = np.array(img.convert("RGB"), dtype=np.uint8)
                warped = _apply_roi_to_image_array(arr, roi_cfg)
                _save_image(dst_img, warped, args.jpeg_quality)
                converted += 1
                by_split[split]["converted"] += 1
            except Exception as exc:
                failed += 1
                by_split[split]["failed"] += 1
                print(f"Failed: {src_img} -> {exc}")
                if args.strict:
                    return 1

    print("Done.")
    print(f"src_root={src_root}")
    print(f"dst_root={dst_root}")
    print(f"roi_config={roi_path}")
    print(f"total={total} converted={converted} skipped={skipped} failed={failed}")
    for split, stats in by_split.items():
        print(
            f"[{split}] total={stats['total']} converted={stats['converted']} "
            f"skipped={stats['skipped']} failed={stats['failed']}"
        )

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

