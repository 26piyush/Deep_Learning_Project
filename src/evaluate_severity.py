"""Stage 5 — DiaMOS Plant Severity Classification Evaluation.
Computes accuracy, macro precision, recall, F1, and generates confusion matrix.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    SEVERITY_MODEL_PATH,
    SEVERITY_DIR,
    SEVERITY_IMAGE_SIZE,
    SEVERITY_BATCH_SIZE,
    METRICS_DIR,
    FIGURES_DIR,
    SEED,
    get_device,
)
from src.preprocessing import get_severity_transforms
from src.train_severity import SeverityDataset, build_severity_model
from src.utils import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    save_json,
    seed_everything,
)


def evaluate_severity(
    model_path: Path = SEVERITY_MODEL_PATH,
    csv_path: Path = SEVERITY_DIR / "severity_labels.csv",
    batch_size: int = SEVERITY_BATCH_SIZE,
    val_split: float = 0.2,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Evaluate severity model on held-out validation split."""
    seed_everything(seed)
    device = get_device()
    print(f"Evaluating severity model ({model_path}) on device: {device}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Severity model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Read records
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if Path(r["image_path"]).exists() and r["severity"] in class_to_idx:
                records.append(r)

    # Reconstruct exact validation split
    import random
    rng = random.Random(seed)
    by_class = {c: [] for c in class_names}
    for r in records:
        by_class[r["severity"]].append(r)

    train_records, val_records = [], []
    for c, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_split))
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    _, val_tfm = get_severity_transforms(image_size=checkpoint.get("image_size", SEVERITY_IMAGE_SIZE))
    val_ds = SeverityDataset(val_records, class_to_idx, val_tfm)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )

    model = build_severity_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = calculate_classification_metrics(all_labels, all_preds, class_names)
    metrics["training_images"] = len(train_records)
    metrics["validation_images"] = len(val_records)
    metrics["model_path"] = str(model_path)
    metrics["device"] = str(device)

    print("\n" + "=" * 50)
    print("SEVERITY CLASSIFICATION METRICS:")
    print(f"Severity classes:   {len(class_names)}")
    print(f"Training images:    {len(train_records)}")
    print(f"Validation images:  {len(val_records)}")
    print(f"Severity accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:           {metrics['macro_f1'] * 100:.2f}%")
    print(f"Macro Precision:    {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:       {metrics['macro_recall'] * 100:.2f}%")
    print("=" * 50)

    # Save metrics JSON
    metrics_path = METRICS_DIR / "severity_metrics.json"
    save_json(metrics, metrics_path)
    print(f"Saved metrics to: {metrics_path}")

    # Plot Confusion Matrix
    cm_arr = np.array(metrics["confusion_matrix"])
    cm_path = FIGURES_DIR / "severity_confusion_matrix.png"
    plot_confusion_matrix(cm_arr, class_names, cm_path, title="ResNet-18 Severity Classification - Confusion Matrix")
    print(f"Saved confusion matrix figure to: {cm_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Severity Model")
    parser.add_argument("--model", default=str(SEVERITY_MODEL_PATH))
    parser.add_argument("--csv", default=str(SEVERITY_DIR / "severity_labels.csv"))
    parser.add_argument("--batch_size", type=int, default=SEVERITY_BATCH_SIZE)
    args = parser.parse_args()

    evaluate_severity(
        model_path=Path(args.model),
        csv_path=Path(args.csv),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
