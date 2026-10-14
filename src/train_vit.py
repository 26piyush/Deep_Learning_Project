"""Stage 2 — Vision Transformer (ViT-B/16) Disease Classification Training.
Uses ImageNet pretrained weights, modifies classification head for disease classes,
and trains with AdamW optimizer and cross-entropy loss.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image

from src.config import (
    DATA_DIR,
    VIT_CLASSES,
    VIT_IMAGE_SIZE,
    VIT_BATCH_SIZE,
    VIT_EPOCHS,
    VIT_LR,
    SEED,
    PLANTVILLAGE_DIR,
    MODELS_DIR,
    RUNS_DIR,
    RESULTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    get_device,
)
from src.dataset_utils import prepare_plantvillage_dataset
from src.preprocessing import get_vit_transforms, validate_and_open_image
from src.utils import (
    seed_everything,
    calculate_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_json,
)


class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for PlantVillage samples using manifest paths."""

    def __init__(self, samples: List[Dict[str, str]], class_to_idx: Dict[str, int], transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.samples[idx]
        img_path = Path(item["path"])
        if not img_path.exists():
            parts = img_path.parts
            if "data" in parts:
                alt = DATA_DIR.joinpath(*parts[parts.index("data") + 1 :])
                if alt.exists():
                    img_path = alt
        label_str = item["class"]
        label_idx = self.class_to_idx[label_str]

        img = validate_and_open_image(img_path)
        if img is None:
            # Fallback to neutral black image if corrupt
            img = Image.new("RGB", (VIT_IMAGE_SIZE, VIT_IMAGE_SIZE), color=(0, 0, 0))

        tensor_img = self.transform(img)
        return tensor_img, label_idx, str(img_path)


def build_vit_model(num_classes: int) -> nn.Module:
    """Instantiate ViT-B/16 with ImageNet pretrained weights and replace head."""
    print(f"Loading pretrained ViT-B/16 (IMAGENET1K_V1)...")
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    # ViT-B/16 classification head is model.heads.head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
) -> Tuple[float, float]:
    """Train for a single epoch with optional AMP mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int], List[float], List[str]]:
    """Evaluate model on dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_confs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
            all_paths.extend(paths)

    eval_loss = running_loss / max(1, total)
    eval_acc = correct / max(1, total)
    return eval_loss, eval_acc, all_preds, all_labels, all_confs, all_paths


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on PlantVillage dataset")
    parser.add_argument("--epochs", type=int, default=VIT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=VIT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VIT_LR)
    parser.add_argument("--image_size", type=int, default=VIT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--classes", default=None, help="Optional class list; defaults to all 38 configured classes")
    parser.add_argument("--samples_per_class", type=int, default=150, help="Max train samples per class (0 for all)")
    parser.add_argument("--manifest", default=str(PLANTVILLAGE_DIR / "plantvillage_splits.json"))
    parser.add_argument("--out_dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Executing ViT training on device: {device}")

    if args.classes is None:
        classes = list(VIT_CLASSES)
    else:
        delimiter = "|" if "|" in args.classes else ","
        classes = [c.strip() for c in args.classes.split(delimiter) if c.strip()]
    manifest_path = Path(args.manifest)

    # Ensure dataset manifest exists
    if not manifest_path.exists():
        print(f"Manifest not found. Generating splits from raw PlantVillage...")
        prepare_plantvillage_dataset(classes=classes, seed=args.seed)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    raw_train = [s for s in manifest["train"] if s.get("class") in class_to_idx]
    raw_val = [s for s in manifest["val"] if s.get("class") in class_to_idx]

    if args.samples_per_class > 0:
        by_class_train = {}
        for s in raw_train:
            by_class_train.setdefault(s["class"], []).append(s)
        train_samples = []
        for c, s_list in by_class_train.items():
            train_samples.extend(s_list[:args.samples_per_class])

        by_class_val = {}
        val_limit = max(15, args.samples_per_class // 4)
        for s in raw_val:
            by_class_val.setdefault(s["class"], []).append(s)
        val_samples = []
        for c, s_list in by_class_val.items():
            val_samples.extend(s_list[:val_limit])
    else:
        train_samples = raw_train
        val_samples = raw_val

    train_tfm, val_tfm = get_vit_transforms(image_size=args.image_size)
    train_ds = PlantDiseaseDataset(train_samples, class_to_idx, train_tfm)
    val_ds = PlantDiseaseDataset(val_samples, class_to_idx, val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )

    print(f"ViT Training set: {len(train_ds)} samples | Validation set: {len(val_ds)} samples")
    print(f"Classes ({len(classes)}): {classes}")

    model = build_vit_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc, val_preds, val_labels, _, _ = evaluate_model(model, val_loader, criterion, device)
        dur = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch}/{args.epochs} [{dur:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "class_names": classes,
            "image_size": args.image_size,
            "epoch": epoch,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "seed": args.seed,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint_data, MODELS_DIR / "vit_best.pt")
            torch.save(checkpoint_data, RUNS_DIR / "vit" / "best.pt" if (RUNS_DIR / "vit").exists() else MODELS_DIR / "vit_best.pt")
            print(f"  --> Saved new best ViT model (Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nViT training completed in {total_time:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}%")

    # Plot training curves
    plot_training_curves(history, FIGURES_DIR / "vit_training_curve.png", title="ViT-B/16 Disease Classification")

    # Save training history
    save_json(history, METRICS_DIR / "vit_training_history.json")


if __name__ == "__main__":
    main()
