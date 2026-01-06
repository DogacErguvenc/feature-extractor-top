import argparse
import sys
from pathlib import Path

from PIL import Image


SUPPORTED_EXTS = {".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert PNG images to JPEG with optional resizing."
    )
    parser.add_argument("--src", required=True, help="Source folder (will be scanned recursively).")
    parser.add_argument(
        "--dst",
        default="",
        help="Destination folder (default: <src>_jpg).",
    )
    # No resizing or quality options: only format conversion.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JPEGs.",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete PNG files after successful conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing files.",
    )
    return parser.parse_args()


def to_rgb(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode == "P":
        if "transparency" in img.info:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, background + (255,))
            return Image.alpha_composite(bg, rgba).convert("RGB")
        return img.convert("RGB")
    return img.convert("RGB")


def iter_images(src_dir: Path):
    for path in src_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def main():
    args = parse_args()
    src_dir = Path(args.src).resolve()
    if not src_dir.exists():
        print(f"Source folder not found: {src_dir}")
        return 1

    dst_dir = Path(args.dst).resolve() if args.dst else Path(str(src_dir) + "_jpg")
    total = 0
    converted = 0
    skipped = 0
    failed = 0

    for img_path in iter_images(src_dir):
        total += 1
        rel = img_path.relative_to(src_dir)
        out_path = dst_dir / rel.with_suffix(".jpg")

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            converted += 1
            continue

        try:
            with Image.open(img_path) as img:
                img = to_rgb(img)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path, format="JPEG")
            converted += 1
            if args.delete_originals:
                img_path.unlink(missing_ok=True)
        except Exception as exc:
            failed += 1
            print(f"Failed: {img_path} -> {exc}")

    print(
        f"Done. total={total} converted={converted} skipped={skipped} failed={failed} "
        f"dst={dst_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
