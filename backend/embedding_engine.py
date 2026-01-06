from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import timm


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    weights_path: Optional[Path]
    image_size: int
    scale_size: int
    crop_mode: str
    device: str


class EmbeddingEngine:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: Optional[torch.nn.Module] = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        pretrained = self.config.weights_path is None
        model = timm.create_model(
            self.config.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        if self.config.weights_path:
            weights_path = self.config.weights_path
            if not weights_path.exists():
                raise FileNotFoundError(f"Embedding weights not found: {weights_path}")
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        device = torch.device(self.config.device)
        model.eval().to(device)
        self._model = model
        return model

    @staticmethod
    def _resize_with_padding(img: Image.Image, size: int) -> Image.Image:
        width, height = img.size
        scale = min(size / max(width, 1), size / max(height, 1))
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), color=(114, 114, 114))
        left = (size - new_w) // 2
        top = (size - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def _edge_crops(self, img: Image.Image, crop_size: int, scale_size: int) -> List[Image.Image]:
        padded = self._resize_with_padding(img, scale_size)
        w, h = padded.size
        crop_size = min(crop_size, w, h)
        cx = max((w - crop_size) // 2, 0)
        cy = max((h - crop_size) // 2, 0)
        coords = [
            (cx, cy),
            (cx, 0),
            (cx, h - crop_size),
            (0, cy),
            (w - crop_size, cy),
        ]
        crops: List[Image.Image] = []
        for x, y in coords:
            x = max(0, min(x, w - crop_size))
            y = max(0, min(y, h - crop_size))
            crops.append(padded.crop((x, y, x + crop_size, y + crop_size)))
        return crops

    def _prepare_crops(self, img: Image.Image, multi_crop: bool) -> List[Image.Image]:
        crop_size = self.config.image_size
        if not multi_crop or self.config.crop_mode == "center":
            return [self._resize_with_padding(img, crop_size)]
        scale_size = max(self.config.scale_size, crop_size)
        return self._edge_crops(img, crop_size, scale_size)

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        chw = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(chw)

    def extract_embeddings(self, img: Image.Image, multi_crop: bool = True) -> np.ndarray:
        model = self._load_model()
        crops = self._prepare_crops(img, multi_crop=multi_crop)
        batch = torch.stack([self._to_tensor(c) for c in crops], dim=0)
        device = torch.device(self.config.device)
        batch = batch.to(device)
        with torch.inference_mode():
            feats = model(batch)
            if feats.ndim > 2:
                feats = feats.mean(dim=(2, 3))
            feats = F.normalize(feats, dim=1)
        return feats.detach().cpu().numpy().astype("float32")

    def embed_image(self, img: Image.Image, multi_crop: bool = True) -> np.ndarray:
        feats = self.extract_embeddings(img, multi_crop=multi_crop)
        vec = feats.mean(axis=0)
        norm = np.linalg.norm(vec) + 1e-12
        vec = vec / norm
        return vec.astype("float32")
