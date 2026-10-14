"""Generate Review 2 end-to-end prediction evidence.
Runs full pipeline on real images: Localization -> Crop -> ViT -> Severity
Saves at least 5 successful examples and 2-3 difficult/fallback examples.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO

from src.config import (
    VIT_MODEL_PATH,
    LOCALIZATION_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    PLANTVILLAGE_DIR,
    PLANTDOC_YOLO_DIR,
    PREDICTIONS_DIR,
    get_device,
)
from src.inference import (
    load_vit_model,
    load_severity_model,
    run_pipeline,
    visualize_result,
)
from src.utils import save_json


def main():
    device = get_device()
    print(f"Generating Review 2 end-to-end prediction evidence using device: {device}")

    # Load models
    yolo = YOLO(str(LOCALIZATION_MODEL_PATH))
    vit_model, vit_tfm, vit_classes = load_vit_model(VIT_MODEL_PATH, device)
    sev_model, sev_tfm, sev_classes = load_severity_model(SEVERITY_MODEL_PATH, device)

    # Gather test images from PlantVillage test split and PlantDoc val split
    with open(PLANTVILLAGE_DIR / "plantvillage_splits.json", "r", encoding="utf-8") as f:
        pv_manifest = json.load(f)

    test_samples = pv_manifest.get("test", [])
    plantdoc_val_dir = PLANTDOC_YOLO_DIR / "images" / "val"
    plantdoc_imgs = sorted(list(plantdoc_val_dir.glob("*.jpg")) + list(plantdoc_val_dir.glob("*.png")))

    candidate_images = []
    # Pick diverse samples from PlantVillage
    for item in test_samples[::60]:
        candidate_images.append({
            "path": Path(item["path"]),
            "source": "PlantVillage",
            "ground_truth_disease": item["class"],
        })

    # Add real-world field images from PlantDoc
    for p in plantdoc_imgs[:10]:
        candidate_images.append({
            "path": p,
            "source": "PlantDoc (Field)",
            "ground_truth_disease": "Field Leaf (Multi-pathogen/Unlabeled)",
        })

    predictions_log = []
    success_count = 0
    difficult_count = 0

    out_evidence_dir = PREDICTIONS_DIR / "e2e_examples"
    out_evidence_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing {len(candidate_images)} images through full pipeline...")

    for i, item in enumerate(candidate_images):
        img_p = item["path"]
        if not img_p.exists():
            continue

        try:
            with Image.open(img_p) as pil_img:
                img_rgb = pil_img.convert("RGB")
        except Exception:
            continue

        res = run_pipeline(
            img_rgb, yolo, vit_model, vit_tfm, vit_classes, sev_model, sev_tfm, sev_classes, device, conf_thresh=0.25
        )

        is_pv = item["source"] == "PlantVillage"
        gt_cls = item["ground_truth_disease"]
        pred_cls = res["disease_raw"]

        # Classification is correct if class matches (for PlantVillage)
        is_match = (gt_cls == pred_cls) if is_pv else True

        # Categorize
        if is_match and not res["used_fallback"] and res["disease_confidence"] > 0.80:
            category = "Successful Detection & Classification"
            prefix = "success"
            success_count += 1
        else:
            category = "Difficult / Boundary / Fallback Case"
            prefix = "difficult"
            difficult_count += 1

        record_filename = f"{prefix}_e2e_example_{len(predictions_log)+1}.jpg"
        save_file = out_evidence_dir / record_filename

        visualize_result(img_rgb, res, save_path=save_file)

        pred_entry = {
            "example_id": len(predictions_log) + 1,
            "source": item["source"],
            "image_path": str(img_p.resolve()),
            "ground_truth": gt_cls,
            "predicted_crop": res["crop"],
            "predicted_disease": res["disease"],
            "predicted_disease_raw": res["disease_raw"],
            "disease_confidence": float(res["disease_confidence"]),
            "predicted_severity": res["severity"],
            "severity_confidence": float(res["severity_confidence"]),
            "bounding_box": res["bounding_box"],
            "localization_status": res["localization_status"],
            "used_fallback": bool(res["used_fallback"]),
            "category": category,
            "saved_visualization": str(save_file),
        }
        predictions_log.append(pred_entry)

    save_json(predictions_log, PREDICTIONS_DIR / "e2e_inference_predictions.json")
    print(f"Generated {len(predictions_log)} total end-to-end predictions.")
    print(f"Saved evidence JSON to {PREDICTIONS_DIR / 'e2e_inference_predictions.json'}")
    print(f"Saved visual prediction images in {out_evidence_dir}")


if __name__ == "__main__":
    main()
