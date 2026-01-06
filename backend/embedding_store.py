from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EmbeddingStoreConfig:
    model_name: str
    image_size: int
    scale_size: int
    crop_mode: str
    created_at: str


class EmbeddingStore:
    def __init__(
        self,
        embeddings: np.ndarray,
        meta: List[dict],
        plu_index: Dict[str, List[int]],
        config: EmbeddingStoreConfig,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2D")
        self.embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = self.embeddings / norms
        self.meta = meta
        self.plu_index = {str(k): v for k, v in plu_index.items()}
        self.config = config

    @classmethod
    def load(cls, store_dir: Path) -> "EmbeddingStore":
        emb_path = store_dir / "embeddings.npy"
        meta_path = store_dir / "meta.json"
        index_path = store_dir / "plu_index.json"
        config_path = store_dir / "config.json"

        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"PLU index file not found: {index_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        embeddings = np.load(emb_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        plu_index = json.loads(index_path.read_text(encoding="utf-8"))
        cfg_raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = EmbeddingStoreConfig(**cfg_raw)
        return cls(embeddings=embeddings, meta=meta, plu_index=plu_index, config=config)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)

    @classmethod
    def save(
        cls,
        store_dir: Path,
        embeddings: np.ndarray,
        meta: List[dict],
        plu_index: Dict[str, List[int]],
        config: EmbeddingStoreConfig,
    ) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        emb_path = store_dir / "embeddings.npy"
        meta_path = store_dir / "meta.json"
        index_path = store_dir / "plu_index.json"
        config_path = store_dir / "config.json"

        tmp_emb = emb_path.with_suffix(emb_path.suffix + ".tmp")
        with open(tmp_emb, "wb") as f:
            np.save(f, embeddings.astype("float32"))
        os.replace(tmp_emb, emb_path)

        cls._atomic_write(meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
        cls._atomic_write(index_path, json.dumps(plu_index, ensure_ascii=False, indent=2).encode("utf-8"))
        cls._atomic_write(
            config_path,
            json.dumps(config.__dict__, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    @staticmethod
    def _topk_mean(values: np.ndarray, k: int) -> Tuple[float, float]:
        if values.size == 0:
            return -1.0, -1.0
        k = min(k, values.size)
        if k == 1:
            best = float(values.max())
            return best, best
        idx = np.argpartition(values, -k)[-k:]
        top = values[idx]
        return float(top.mean()), float(top.max())

    def score_query(
        self,
        query_embeddings: np.ndarray,
        selected_plu: str,
        top_k: int = 3,
        min_sim: float = 0.35,
        margin: float = 0.05,
    ) -> dict:
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]

        sims = query_embeddings @ self.embeddings.T
        sims = sims.max(axis=0)

        scores: Dict[str, dict] = {}
        for plu_code, indices in self.plu_index.items():
            vals = sims[np.array(indices, dtype=int)]
            score, best = self._topk_mean(vals, top_k)
            scores[plu_code] = {"score": score, "best": best}

        selected_plu = str(selected_plu)
        if selected_plu not in scores:
            raise ValueError(f"No embeddings found for selected PLU {selected_plu}")

        selected_score = scores[selected_plu]["score"]
        best_other_score = max(
            (v["score"] for k, v in scores.items() if k != selected_plu),
            default=-1.0,
        )
        predicted_plu = max(scores.items(), key=lambda item: item[1]["score"])[0]
        predicted_score = scores[predicted_plu]["score"]

        is_match = selected_score >= min_sim and selected_score >= best_other_score + margin
        confidence = float(np.clip(selected_score, 0.0, 1.0) * 100.0)

        analysis = (
            f"Embedding match check: selected_plu={selected_plu}, "
            f"selected_score={selected_score:.3f}, best_other={best_other_score:.3f}, "
            f"predicted_plu={predicted_plu}, predicted_score={predicted_score:.3f}."
        )

        return {
            "analysis": analysis,
            "is_match": is_match,
            "confidence": round(confidence, 2),
            "predicted_plu": predicted_plu,
            "predicted_score": round(predicted_score, 4),
            "selected_score": round(selected_score, 4),
            "best_other_score": round(best_other_score, 4),
            "embedding_count": int(self.embeddings.shape[0]),
        }
