"""
Local model training (EfficientNet) and ONNX export.
Dataset layout (project root):
  datasets/train/201/
  datasets/train/202/
  datasets/val/201/
  datasets/val/202/
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Dataset paths (project root datasets/)
BASE_DIR = Path(__file__).resolve().parent.parent / "datasets"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Output paths
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 2

MODEL_VARIANTS = {
    "small": {"model_name": "efficientnet_b0", "image_size": 224, "suffix": ""},
    "large": {"model_name": "efficientnet_b3", "image_size": 300, "suffix": "_large"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train local ONNX model.")
    parser.add_argument("--variant", choices=MODEL_VARIANTS.keys(), default="small")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    return parser.parse_args()


def main():
    args = parse_args()
    variant = MODEL_VARIANTS[args.variant]
    image_size = int(variant["image_size"])
    model_name = variant["model_name"]
    suffix = variant["suffix"]

    onnx_path = MODELS_DIR / f"local_model{suffix}.onnx"
    labels_path = MODELS_DIR / f"local_labels{suffix}.json"
    best_pt_path = MODELS_DIR / f"local_model{suffix}_best.pt"

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=val_tf)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    num_classes = len(train_ds.classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / total if total else 0.0

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        acc = evaluate()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_pt_path)
            print(
                f"Epoch {epoch + 1}/{args.epochs} - val_acc={acc:.3f} "
                f"(best so far, saved to {best_pt_path})"
            )
        else:
            print(f"Epoch {epoch + 1}/{args.epochs} - val_acc={acc:.3f}")

    if best_pt_path.exists():
        state = torch.load(best_pt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best weights from {best_pt_path} (val_acc={best_acc:.3f})")

    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Saved ONNX to {onnx_path} (opset 18, static batch)")

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    name_map = {"201": "Limon", "202": "Sogan"}
    labels = []
    for _, cls in sorted(idx_to_class.items()):
        labels.append({"plu_code": cls, "name": name_map.get(cls, cls)})
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
