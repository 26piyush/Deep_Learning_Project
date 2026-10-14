"""Utility functions for seeding, metrics calculation, plotting, and file handling."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from PIL import Image, ImageDraw, ImageFont


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect CUDA if available, otherwise fallback to CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def save_json(data: Any, filepath: Path) -> None:
    """Save dictionary/list as formatted JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def calculate_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict[str, Any]:
    """Calculate accuracy, macro precision, recall, F1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_names": class_names,
        "total_samples": len(y_true),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names) * 1.0)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Training Progress",
) -> None:
    """Plot training and validation loss and accuracy curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
    ax1.set_title(f"{title} - Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", color="#2ca02c")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", color="#d62728")
    ax2.set_title(f"{title} - Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def draw_bounding_boxes(
    image: Image.Image,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    ious: Optional[List[float]] = None,
) -> Image.Image:
    """Draw Ground Truth (Green) and Predicted (Red) bounding boxes onto an image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Ground truth: Green
    for box in gt_boxes:
        draw.rectangle(box, outline="#00FF00", width=3)
        draw.text((box[0] + 4, box[1] + 4), "Ground Truth", fill="#00FF00")

    # Predicted: Red/Orange
    for i, box in enumerate(pred_boxes):
        iou_text = f"Pred (IoU: {ious[i]:.2f})" if (ious and i < len(ious)) else "Predicted"
        draw.rectangle(box, outline="#FF3300", width=3)
        draw.text((box[0] + 4, max(0, box[1] - 14)), iou_text, fill="#FF3300")

    return img
