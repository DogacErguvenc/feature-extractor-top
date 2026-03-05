import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml

from butcher_classifier import ButcherClassifier

DEFAULT_CONFIG_PATH = (Path(__file__).parent / "butcher_config.yaml").resolve()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config dosyasi yolu",
    )
    return parser.parse_args()


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_config(path: str):
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {cfg_path}")
    cfg["_config_path"] = cfg_path
    return cfg


def create_dataloaders(cfg):
    cfg_path = cfg["_config_path"]
    cfg_dir = cfg_path.parent

    data_root = _resolve_path(cfg["data"]["root_dir"], cfg_dir)
    img_size = cfg["data"].get("img_size", 224)
    num_workers = cfg["data"].get("num_workers", 2)
    batch_size = cfg["train"]["batch_size"]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def _resolve_device(device_str: str) -> torch.device:
    requested = str(device_str).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except Exception:
        return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg_path = cfg["_config_path"]
    cfg_dir = cfg_path.parent

    device = _resolve_device(cfg["train"].get("device", "cpu"))
    print("Using device:", device)

    train_loader, val_loader, train_dataset = create_dataloaders(cfg)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)
    print("Num classes:", num_classes)

    model = ButcherClassifier(
        num_classes=num_classes,
        backbone_name=cfg["model"].get("backbone_name", "resnet18"),
        pretrained=bool(cfg["model"].get("pretrained", True)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"].get("lr", 1.0e-3),
        weight_decay=cfg["train"].get("weight_decay", 1.0e-4),
    )

    num_epochs = int(cfg["train"].get("num_epochs", 10))

    exp_name = str(cfg.get("experiment_name", "butcher_resnet18"))
    model_root = _resolve_path(cfg.get("model_root", "models"), cfg_dir)
    exp_dir = model_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    class_to_idx = train_dataset.class_to_idx
    with (exp_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    print("Training started.")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[{epoch:02d}/{num_epochs:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = exp_dir / "best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Yeni en iyi model kaydedildi: {ckpt_path} (val_acc={val_acc:.3f})")

    print("Training finished. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
