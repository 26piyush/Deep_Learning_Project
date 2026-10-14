"""Stage 5 — DiaMOS Plant Severity Classification Training.
Trains a lightweight pretrained ResNet-18 on discrete severity classes:
No Risk, Very Low, Low, Medium, High.
"""

import argparse
import csv
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
    SEVERITY_CLASSES,
    SEVERITY_IMAGE_SIZE,
    SEVERITY_BATCH_SIZE,
    SEVERITY_EPOCHS,
    SEVERITY_LR,
    SEVERITY_DIR,
    SEVERITY_MODEL_PATH,
    MODELS_DIR,
    RUNS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    SEED,
    get_device,
)
from src.dataset_utils import prepare_diamos_severity
from src.preprocessing import get_severity_transforms, validate_and_open_image
from src.utils import (
    seed_everything,
    plot_training_curves,
    save_json,
)


class SeverityDataset(Dataset):
    """PyTorch Dataset for DiaMOS severity images."""

    def __init__(self, records: List[Dict[str, str]], class_to_idx: Dict[str, int], transform):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.records[idx]
        img_p = Path(item["image_path"])
        if not img_p.exists():
            parts = img_p.parts
            if "data" in parts:
                alt = DATA_DIR.joinpath(*parts[parts.index("data") + 1 :])
                if alt.exists():
                    img_p = alt
        label_idx = self.class_to_idx[item["severity"]]

        img = validate_and_open_image(img_p)
        if img is None:
            img = Image.new("RGB", (SEVERITY_IMAGE_SIZE, SEVERITY_IMAGE_SIZE), color=(0, 0, 0))

        tensor_img = self.transform(img)
        return tensor_img, label_idx, str(img_p)


def build_severity_model(num_classes: int) -> nn.Module:
    """Pretrained ResNet-18 model for discrete severity classification."""
    print(f"Loading pretrained ResNet-18 for severity classification ({num_classes} classes)...")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train discrete severity classifier on DiaMOS Plant")
    parser.add_argument("--csv", default=str(SEVERITY_DIR / "severity_labels.csv"))
    parser.add_argument("--epochs", type=int, default=SEVERITY_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=SEVERITY_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=SEVERITY_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Training severity model on device: {device}")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("Severity CSV not found. Preparing DiaMOS severity dataset...")
        prepare_diamos_severity()

    # Load annotations
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if Path(r["image_path"]).exists():
                records.append(r)

    if not records:
        raise RuntimeError(f"No valid records found in {csv_path}")

    # Determine classes present
    available_classes = sorted(list({r["severity"] for r in records}))
    class_names = [c for c in SEVERITY_CLASSES if c in available_classes]
    if not class_names:
        class_names = available_classes
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Filter records to recognized classes
    records = [r for r in records if r["severity"] in class_to_idx]

    # Stratified split
    import random
    rng = random.Random(args.seed)
    by_class: Dict[str, List[Dict[str, str]]] = {c: [] for c in class_names}
    for r in records:
        by_class[r["severity"]].append(r)

    train_records = []
    val_records = []
    for c, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_split))
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    train_tfm, val_tfm = get_severity_transforms(image_size=SEVERITY_IMAGE_SIZE)
    train_ds = SeverityDataset(train_records, class_to_idx, train_tfm)
    val_ds = SeverityDataset(val_records, class_to_idx, val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )

    print(f"DiaMOS Severity: {len(train_ds)} train samples, {len(val_ds)} val samples.")
    print(f"Classes: {class_names}")

    model = build_severity_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        ep_train_loss = train_loss / max(1, train_total)
        ep_train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        ep_val_loss = val_loss / max(1, val_total)
        ep_val_acc = val_correct / max(1, val_total)
        dur = time.time() - t0

        history["train_loss"].append(float(ep_train_loss))
        history["train_acc"].append(float(ep_train_acc))
        history["val_loss"].append(float(ep_val_loss))
        history["val_acc"].append(float(ep_val_acc))

        print(
            f"Epoch {epoch}/{args.epochs} [{dur:.1f}s] - "
            f"Train Loss: {ep_train_loss:.4f}, Train Acc: {ep_train_acc*100:.2f}% | "
            f"Val Loss: {ep_val_loss:.4f}, Val Acc: {ep_val_acc*100:.2f}%"
        )

        checkpoint_data = {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": SEVERITY_IMAGE_SIZE,
            "epoch": epoch,
            "val_acc": ep_val_acc,
            "val_loss": ep_val_loss,
            "seed": args.seed,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            torch.save(checkpoint_data, SEVERITY_MODEL_PATH)
            runs_sev = RUNS_DIR / "severity"
            runs_sev.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, runs_sev / "best.pt")
            print(f"  --> Saved new best severity model (Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nSeverity model training completed in {total_time:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}%")

    # Plot training curves
    plot_training_curves(history, FIGURES_DIR / "severity_training_curve.png", title="ResNet-18 Severity Classification")
    save_json(history, METRICS_DIR / "severity_training_history.json")


if __name__ == "__main__":
    main()
