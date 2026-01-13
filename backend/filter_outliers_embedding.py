"""
Flag outlier images using embedding store centroids.

Usage:
  python filter_outliers_embedding.py --store-dir C:\feature-extractor\backend\embedding_store_392 --report outliers.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from embedding_store import EmbeddingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag embedding outliers by PLU.")
    parser.add_argument("--store-dir", default="backend/embedding_store", help="Embedding store dir.")
    parser.add_argument("--report", default="embedding_outliers.csv", help="CSV report path.")
    parser.add_argument("--percentile", type=float, default=5.0, help="Bottom X%% per PLU to flag. 0 disables.")
    parser.add_argument("--min-sim", type=float, default=0.0, help="Absolute min similarity to PLU centroid.")
    parser.add_argument("--mismatch-margin", type=float, default=0.0, help="Flag if best other exceeds own by margin.")
    parser.add_argument("--copy-dir", default="", help="Copy flagged files to this directory.")
    parser.add_argument("--data-root", default="", help="Optional dataset root to keep relative paths.")
    parser.add_argument("--dry-run", action="store_true", help="No file copies.")
    return parser.parse_args()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_dest(
    src_path: str,
    plu_code: str,
    filename: str,
    copy_dir: Path,
    data_root: Path | None,
) -> Path:
    if data_root and src_path:
        try:
            rel = Path(src_path).resolve().relative_to(data_root)
            return copy_dir / rel
        except Exception:
            pass
    if plu_code:
        return copy_dir / plu_code / filename
    return copy_dir / filename


def main() -> int:
    args = parse_args()
    store_dir = Path(args.store_dir).resolve()
    if not store_dir.exists():
        print(f"Embedding store not found: {store_dir}")
        return 1

    store = EmbeddingStore.load(store_dir)
    embeddings = store.embeddings
    meta = store.meta
    plu_index = store.plu_index

    if embeddings.size == 0:
        print("No embeddings found.")
        return 1

    plu_codes = sorted(plu_index.keys())
    plu_to_idx = {plu: i for i, plu in enumerate(plu_codes)}
    num_plu = len(plu_codes)

    centroids = np.zeros((num_plu, embeddings.shape[1]), dtype=np.float32)
    for plu_code, indices in plu_index.items():
        idx = np.array(indices, dtype=int)
        if idx.size == 0:
            continue
        centroids[plu_to_idx[plu_code]] = _normalize(embeddings[idx].mean(axis=0))

    sims = embeddings @ centroids.T
    own_idx = np.array([plu_to_idx[str(m.get("plu_code"))] for m in meta], dtype=int)
    own_sim = sims[np.arange(len(meta)), own_idx]

    if num_plu > 1:
        sims_other = sims.copy()
        sims_other[np.arange(len(meta)), own_idx] = -1.0
        best_other_idx = sims_other.argmax(axis=1)
        best_other_sim = sims_other[np.arange(len(meta)), best_other_idx]
        best_other_plu = [plu_codes[i] for i in best_other_idx]
    else:
        best_other_sim = np.full(len(meta), -1.0, dtype=np.float32)
        best_other_plu = ["" for _ in range(len(meta))]

    thresholds: Dict[str, float] = {}
    if args.percentile > 0:
        for plu_code, indices in plu_index.items():
            idx = np.array(indices, dtype=int)
            if idx.size == 0:
                continue
            thresholds[plu_code] = float(np.percentile(own_sim[idx], args.percentile))

    rows: List[dict] = []
    flagged = 0
    for i, rec in enumerate(meta):
        plu_code = str(rec.get("plu_code", ""))
        source_path = str(rec.get("source_path", "")) if rec.get("source_path") else ""
        filename = rec.get("filename") or Path(source_path).name

        reasons: List[str] = []
        if args.min_sim > 0 and own_sim[i] < args.min_sim:
            reasons.append("low_sim")
        if args.percentile > 0 and own_sim[i] < thresholds.get(plu_code, -1.0):
            reasons.append("low_percentile")
        if num_plu > 1 and best_other_sim[i] >= own_sim[i] + args.mismatch_margin:
            reasons.append("better_other")

        if reasons:
            flagged += 1

        rows.append(
            {
                "plu_code": plu_code,
                "source_path": source_path,
                "filename": filename,
                "own_sim": f"{own_sim[i]:.4f}",
                "best_other_sim": f"{best_other_sim[i]:.4f}",
                "best_other_plu": best_other_plu[i],
                "reasons": "|".join(reasons),
            }
        )

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plu_code",
                "source_path",
                "filename",
                "own_sim",
                "best_other_sim",
                "best_other_plu",
                "reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total images: {len(rows)}")
    print(f"Flagged: {flagged}")
    print(f"Report: {report_path}")

    if args.copy_dir and not args.dry_run:
        copy_dir = Path(args.copy_dir).resolve()
        data_root = Path(args.data_root).resolve() if args.data_root else None
        copied = 0
        for row in rows:
            if not row["reasons"]:
                continue
            src = Path(row["source_path"])
            if not src.exists():
                continue
            dest = _resolve_dest(
                row["source_path"],
                row["plu_code"],
                row["filename"],
                copy_dir,
                data_root,
            )
            _safe_copy(src, dest)
            copied += 1
        print(f"Copied {copied} files to {copy_dir}")
    elif args.copy_dir and args.dry_run:
        print("Dry-run enabled: no files copied.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
