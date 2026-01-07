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

    @staticmethod
    def _resize_pos_embed(
        state: dict, model: torch.nn.Module
    ) -> dict:
        if "pos_embed" not in state or not hasattr(model, "pos_embed"):
            return state

        pos_embed = state["pos_embed"]
        if pos_embed.shape == model.pos_embed.shape:
            return state

        num_prefix = getattr(model, "num_prefix_tokens", 1)
        if num_prefix < 0:
            num_prefix = 0

        prefix = pos_embed[:, :num_prefix] if num_prefix else pos_embed[:, :0]
        pos_tokens = pos_embed[:, num_prefix:]
        if pos_tokens.ndim != 3:
            return state

        embed_dim = pos_tokens.shape[-1]
        old_num = pos_tokens.shape[1]
        new_num = model.pos_embed.shape[1] - num_prefix
        old_size = int(round(old_num ** 0.5))
        new_size = int(round(new_num ** 0.5))
        if old_size * old_size != old_num or new_size * new_size != new_num:
            return state

        pos_tokens = pos_tokens.reshape(1, old_size, old_size, embed_dim).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, embed_dim)
        state["pos_embed"] = torch.cat([prefix, pos_tokens], dim=1)
        return state

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        pretrained = self.config.weights_path is None
        try:
            model = timm.create_model(
                self.config.model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                img_size=self.config.image_size,
            )
        except TypeError:
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
            if isinstance(state, dict):
                state = self._resize_pos_embed(state, model)
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
