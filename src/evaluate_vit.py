"""Stage 2 — Vision Transformer (ViT-B/16) Disease Classification Evaluation.
Generates accuracy, macro precision, recall, F1, confusion matrix, and sample prediction records.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

from src.config import (
    VIT_IMAGE_SIZE,
    VIT_BATCH_SIZE,
    VIT_MODEL_PATH,
    PLANTVILLAGE_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    get_device,
)
from src.preprocessing import get_vit_transforms
from src.train_vit import PlantDiseaseDataset, build_vit_model
from src.utils import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    save_json,
)


def evaluate_vit(
    model_path: Path = VIT_MODEL_PATH,
    manifest_path: Path = PLANTVILLAGE_DIR / "plantvillage_splits.json",
    batch_size: int = VIT_BATCH_SIZE,
    split_name: str = "test",
) -> Dict[str, Any]:
    """Run comprehensive evaluation of ViT model on test split."""
    device = get_device()
    print(f"Evaluating ViT model ({model_path}) on {split_name} split using device: {device}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"ViT model checkpoint not found at: {model_path}")
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"PlantVillage split manifest not found at: {manifest_path}")

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_samples = [s for s in manifest.get(split_name, []) if s.get("class") in class_to_idx]
    if not test_samples:
        raise ValueError(f"No samples found in split: {split_name}")

    _, test_tfm = get_vit_transforms(image_size=checkpoint.get("image_size", VIT_IMAGE_SIZE))
    test_ds = PlantDiseaseDataset(test_samples, class_to_idx, test_tfm)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )

    # Load model
    model = build_vit_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_confs = []
    all_paths = []

    print(f"Running inference over {len(test_ds)} test images...")
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
            all_paths.extend(paths)

    # Compute metrics
    metrics = calculate_classification_metrics(all_labels, all_preds, class_names)
    metrics["model_path"] = str(model_path)
    metrics["evaluated_split"] = split_name
    metrics["device"] = str(device)

    print("\n" + "=" * 50)
    print("VISION TRANSFORMER EVALUATION METRICS:")
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:   {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:      {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics['macro_f1'] * 100:.2f}%")
    print("=" * 50)

    # Save metrics JSON
    metrics_path = METRICS_DIR / "vit_metrics.json"
    save_json(metrics, metrics_path)
    print(f"Saved metrics to: {metrics_path}")

    # Plot Confusion Matrix
    cm_arr = np.array(metrics["confusion_matrix"])
    cm_path = FIGURES_DIR / "vit_confusion_matrix.png"
    plot_confusion_matrix(cm_arr, class_names, cm_path, title="ViT Disease Classification - Confusion Matrix")
    print(f"Saved confusion matrix figure to: {cm_path}")

    # Save sample test prediction records
    prediction_records = []
    for path, pred_idx, label_idx, conf in zip(all_paths, all_preds, all_labels, all_confs):
        prediction_records.append({
            "image_path": path,
            "predicted_class": class_names[pred_idx],
            "ground_truth": class_names[label_idx],
            "confidence": float(conf),
            "is_correct": bool(pred_idx == label_idx),
        })

    preds_json_path = PREDICTIONS_DIR / "vit_test_predictions.json"
    save_json(prediction_records, preds_json_path)

    # Save 5 correct and 3 incorrect visual prediction examples
    correct_samples = [r for r in prediction_records if r["is_correct"]]
    incorrect_samples = [r for r in prediction_records if not r["is_correct"]]

    saved_examples = []
    for i, item in enumerate(correct_samples[:5]):
        saved_examples.append((item, f"vit_success_example_{i+1}.png"))
    for i, item in enumerate(incorrect_samples[:3]):
        saved_examples.append((item, f"vit_difficult_example_{i+1}.png"))

    for item, filename in saved_examples:
        try:
            with Image.open(item["image_path"]) as img:
                vis_img = img.convert("RGB").resize((400, 400))
                draw = ImageDraw.Draw(vis_img)
                color = "#00FF00" if item["is_correct"] else "#FF3333"
                txt = f"Pred: {item['predicted_class']}\nConf: {item['confidence']*100:.1f}%\nGT: {item['ground_truth']}"
                draw.rectangle([(5, 5), (395, 75)], fill=(0, 0, 0, 180))
                draw.text((10, 10), txt, fill=color)
                vis_img.save(PREDICTIONS_DIR / filename)
        except Exception:
            pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Vision Transformer")
    parser.add_argument("--model", default=str(VIT_MODEL_PATH))
    parser.add_argument("--manifest", default=str(PLANTVILLAGE_DIR / "plantvillage_splits.json"))
    parser.add_argument("--batch_size", type=int, default=VIT_BATCH_SIZE)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    evaluate_vit(
        model_path=Path(args.model),
        manifest_path=Path(args.manifest),
        batch_size=args.batch_size,
        split_name=args.split,
    )


if __name__ == "__main__":
    main()
