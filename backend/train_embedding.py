import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import timm


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tune for DINOv2 embeddings.")
    parser.add_argument("--train-dir", required=True, help="Training dataset root.")
    parser.add_argument("--val-dir", required=True, help="Validation dataset root.")
    parser.add_argument("--model-name", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="backend/models/embedding_dino_ft")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf


def build_balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and device == "cuda"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms(args.image_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    if len(train_ds.classes) < 2:
        print("Need at least 2 classes to train.")
        return 1

    sampler = None
    if not args.no_balanced_sampler:
        sampler = build_balanced_sampler(train_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    num_classes = len(train_ds.classes)
    try:
        model = timm.create_model(
            args.model_name,
            pretrained=True,
            num_classes=num_classes,
            global_pool="avg",
            img_size=args.image_size,
        )
    except TypeError:
        model = timm.create_model(
            args.model_name, pretrained=True, num_classes=num_classes, global_pool="avg"
        )
    if args.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_acc = -1.0
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_dl, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = loss_fn(logits, y)
                loss = loss / max(1, args.accumulation_steps)

            scaler.scale(loss).backward()
            running_loss += loss.item() * max(1, args.accumulation_steps)
            running_acc += accuracy(logits.detach(), y)

            if step % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if len(train_dl) % args.accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)

        train_loss_avg = running_loss / max(1, len(train_dl))
        train_acc_avg = running_acc / max(1, len(train_dl))
        val_loss_avg = val_loss / max(1, len(val_dl))
        val_acc_avg = val_acc / max(1, len(val_dl))

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc_avg:.4f} "
            f"val_loss={val_loss_avg:.4f} val_acc={val_acc_avg:.4f}"
        )

        if val_acc_avg > best_acc:
            best_acc = val_acc_avg
            torch.save(model.state_dict(), best_path)
            print(f"saved best -> {best_path}")

        torch.save(model.state_dict(), last_path)

    labels_path = out_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, ensure_ascii=False, indent=2)

    config_path = out_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"done. best_acc={best_acc:.4f}")
    print(f"weights: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
