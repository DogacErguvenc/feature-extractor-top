from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms
import yaml

from butcher_classifier import ButcherClassifier


@dataclass(frozen=True)
class ButcherRuntimeConfig:
    experiment_name: str
    backbone_name: str
    img_size: int
    device: str
    checkpoint_path: Path
    class_map_path: Path


_RUNTIME_CACHE: dict = {
    "signature": None,
    "model": None,
    "device": None,
    "transform": None,
    "class_to_idx": None,
    "idx_to_class": None,
}


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_runtime_config(config_path: Path) -> ButcherRuntimeConfig:
    cfg_path = Path(config_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Butcher config bulunamadi: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Geçersiz butcher config: {cfg_path}")

    exp_name = str(cfg.get("experiment_name", "butcher_resnet18"))
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    backbone_name = str(model_cfg.get("backbone_name", "resnet18"))
    img_size = int(data_cfg.get("img_size", 224))
    device = str(train_cfg.get("device", "cpu"))

    model_root_raw = cfg.get("model_root", "models")
    model_root = _resolve_path(model_root_raw, cfg_path.parent)

    ckpt_raw = cfg.get("checkpoint_path")
    class_map_raw = cfg.get("class_map_path")

    if ckpt_raw:
        checkpoint_path = _resolve_path(ckpt_raw, cfg_path.parent)
    else:
        checkpoint_path = model_root / exp_name / "best.pth"

    if class_map_raw:
        class_map_path = _resolve_path(class_map_raw, cfg_path.parent)
    else:
        class_map_path = model_root / exp_name / "class_to_idx.json"

    return ButcherRuntimeConfig(
        experiment_name=exp_name,
        backbone_name=backbone_name,
        img_size=img_size,
        device=device,
        checkpoint_path=checkpoint_path,
        class_map_path=class_map_path,
    )


def _resolve_torch_device(device_str: str) -> torch.device:
    requested = str(device_str).strip().lower()
    if not requested:
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        return torch.device(requested)
    except Exception:
        return torch.device("cpu")


def _build_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _load_runtime(config_path: Path):
    cfg = load_runtime_config(config_path)

    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Butcher checkpoint bulunamadi: {cfg.checkpoint_path}")
    if not cfg.class_map_path.exists():
        raise FileNotFoundError(f"Butcher class map bulunamadi: {cfg.class_map_path}")

    signature = (
        str(Path(config_path).resolve()),
        str(cfg.checkpoint_path.resolve()),
        str(cfg.class_map_path.resolve()),
        cfg.checkpoint_path.stat().st_mtime_ns,
        cfg.class_map_path.stat().st_mtime_ns,
        cfg.backbone_name,
        cfg.img_size,
        cfg.device,
    )

    if _RUNTIME_CACHE["signature"] == signature:
        return _RUNTIME_CACHE

    with cfg.class_map_path.open("r", encoding="utf-8") as f:
        class_to_idx_raw = json.load(f)

    class_to_idx = {str(k): int(v) for k, v in class_to_idx_raw.items()}
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    if num_classes <= 0:
        raise ValueError("class_to_idx bos olamaz")

    device = _resolve_torch_device(cfg.device)
    model = ButcherClassifier(
        num_classes=num_classes,
        backbone_name=cfg.backbone_name,
        pretrained=False,
    ).to(device)

    state_dict = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _RUNTIME_CACHE["signature"] = signature
    _RUNTIME_CACHE["model"] = model
    _RUNTIME_CACHE["device"] = device
    _RUNTIME_CACHE["transform"] = _build_transform(cfg.img_size)
    _RUNTIME_CACHE["class_to_idx"] = class_to_idx
    _RUNTIME_CACHE["idx_to_class"] = idx_to_class
    return _RUNTIME_CACHE


def predict_base64_image(
    image_base64: str,
    config_path: Path,
    topk: int = 3,
) -> dict:
    runtime = _load_runtime(config_path)
    model = runtime["model"]
    device = runtime["device"]
    transform = runtime["transform"]
    idx_to_class: Dict[int, str] = runtime["idx_to_class"]

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(max(1, int(topk)), int(probs.shape[0]))
    top_probs, top_indices = torch.topk(probs, k=topk)

    top_matches = []
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
        cls_name = idx_to_class[int(idx)]
        top_matches.append(
            {
                "rank": rank,
                "plu_code": cls_name,
                "plu_name": cls_name,
                "score": round(float(p) * 100.0, 2),
                "score_type": "probability_pct",
                "prob": float(p),
            }
        )

    probs_np = probs.detach().cpu().numpy()
    class_probs = {idx_to_class[i]: float(probs_np[i]) for i in range(len(idx_to_class))}

    top1 = top_matches[0]
    return {
        "top_matches": top_matches,
        "class_probs": class_probs,
        "predicted_class": top1["plu_code"],
        "predicted_prob": float(top1["prob"]),
    }
