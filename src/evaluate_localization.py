"""Stage 3 — PlantDoc Localization Evaluation Module.
Computes mAP@50, Precision, Recall, IoU, and generates visual comparison figures
with ground truth and predicted bounding boxes.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.config import (
    PLANTDOC_YOLO_DIR,
    LOCALIZATION_MODEL_PATH,
    METRICS_DIR,
    FIGURES_DIR,
    RUNS_DIR,
    SEED,
)
from src.utils import compute_iou, draw_bounding_boxes, save_json, seed_everything


def evaluate_localization(
    model_path: Path = LOCALIZATION_MODEL_PATH,
    data_yaml: Path = PLANTDOC_YOLO_DIR / "data.yaml",
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    num_visual_examples: int = 5,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Evaluate YOLO detector and generate required metrics and visual evidence."""
    seed_everything(seed)
    model_path = Path(model_path)
    if not model_path.exists():
        fallback_path = RUNS_DIR / "detect" / "plantdoc" / "weights" / "best.pt"
        if fallback_path.exists():
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Localization model not found at {model_path} or {fallback_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model from: {model_path} on device {device}")
    model = YOLO(str(model_path))

    # Run official Ultralytics validation
    print("Running validation via model.val()...")
    val_results = model.val(
        data=str(Path(data_yaml).resolve().as_posix()),
        split="val",
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    metrics_dict = {
        "mAP50": float(val_results.box.map50),
        "mAP50_95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "f1": float(2 * val_results.box.mp * val_results.box.mr / max(1e-6, val_results.box.mp + val_results.box.mr)),
    }

    # Now calculate empirical IoU on sample validation images and generate visualizations
    val_images_dir = Path(data_yaml).parent / "images" / "val"
    val_labels_dir = Path(data_yaml).parent / "labels" / "val"
    vis_dir = FIGURES_DIR / "localization_examples"
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
    visual_records = []
    all_ious = []

    for img_p in image_files[: max(num_visual_examples, 10)]:
        lbl_p = val_labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        try:
            with Image.open(img_p) as pil_img:
                w, h = pil_img.size
                rgb_img = pil_img.convert("RGB")
        except Exception:
            continue

        # Parse ground-truth boxes in pixels [x1, y1, x2, y2]
        gt_boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, cx, cy, bw, bh = map(float, parts[:5])
                    x1 = max(0.0, (cx - bw / 2.0) * w)
                    y1 = max(0.0, (cy - bh / 2.0) * h)
                    x2 = min(float(w), (cx + bw / 2.0) * w)
                    y2 = min(float(h), (cy + bh / 2.0) * h)
                    gt_boxes.append([x1, y1, x2, y2])

        # Run model prediction
        pred_res = model.predict(source=np.array(rgb_img), conf=conf_thresh, imgsz=imgsz, verbose=False)[0]
        pred_boxes_np = pred_res.boxes.xyxy.cpu().numpy() if pred_res.boxes is not None else np.empty((0, 4))
        pred_boxes = pred_boxes_np.tolist()

        # Compute IoUs between GT and predicted
        image_ious = []
        for p_box in pred_boxes:
            best_iou = 0.0
            for g_box in gt_boxes:
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
            image_ious.append(best_iou)
            all_ious.append(best_iou)

        # Draw side-by-side or combined visualization
        annotated_img = draw_bounding_boxes(rgb_img, gt_boxes, pred_boxes, image_ious)
        save_file = vis_dir / f"loc_comparison_{img_p.stem}.png"
        annotated_img.save(save_file)

        visual_records.append({
            "image": img_p.name,
            "gt_boxes_count": len(gt_boxes),
            "pred_boxes_count": len(pred_boxes),
            "mean_iou": float(np.mean(image_ious)) if image_ious else 0.0,
            "saved_figure": str(save_file),
        })

    metrics_dict["mean_iou"] = float(np.mean(all_ious)) if all_ious else 0.0
    metrics_dict["evaluated_samples"] = len(visual_records)
    metrics_dict["model_path"] = str(model_path)

    print("\n" + "=" * 50)
    print("PLANTDOC LOCALIZATION EVALUATION METRICS:")
    print(f"mAP@50:       {metrics_dict['mAP50'] * 100:.2f}%")
    print(f"mAP@50-95:    {metrics_dict['mAP50_95'] * 100:.2f}%")
    print(f"Precision:    {metrics_dict['precision'] * 100:.2f}%")
    print(f"Recall:       {metrics_dict['recall'] * 100:.2f}%")
    print(f"F1-Score:     {metrics_dict['f1'] * 100:.2f}%")
    print(f"Mean IoU:     {metrics_dict['mean_iou']:.4f}")
    print(f"Visual examples saved in: {vis_dir}")
    print("=" * 50)

    save_json(metrics_dict, METRICS_DIR / "localization_metrics.json")
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO localization on PlantDoc")
    parser.add_argument("--model", default=str(LOCALIZATION_MODEL_PATH))
    parser.add_argument("--data", default=str(PLANTDOC_YOLO_DIR / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    evaluate_localization(
        model_path=Path(args.model),
        data_yaml=Path(args.data),
        imgsz=args.imgsz,
        conf_thresh=args.conf,
    )


if __name__ == "__main__":
    main()
