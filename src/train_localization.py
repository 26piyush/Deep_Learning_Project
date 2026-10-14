"""Stage 3 — PlantDoc Disease/Leaf Localization Training using Ultralytics YOLOv8.
Trains YOLOv8n detector with single localization class 0 = leaf_or_disease_region.
"""

import argparse
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

from src.config import (
    PLANTDOC_YOLO_DIR,
    LOCALIZATION_EPOCHS,
    LOCALIZATION_BATCH_SIZE,
    LOCALIZATION_IMGSZ,
    LOCALIZATION_MODEL_PATH,
    RUNS_DIR,
    MODELS_DIR,
    SEED,
)
from src.dataset_utils import prepare_plantdoc_yolo


def train_localization(
    data_yaml: Path = PLANTDOC_YOLO_DIR / "data.yaml",
    epochs: int = LOCALIZATION_EPOCHS,
    batch_size: int = LOCALIZATION_BATCH_SIZE,
    imgsz: int = LOCALIZATION_IMGSZ,
    seed: int = SEED,
) -> Path:
    """Train YOLOv8n on PlantDoc localization dataset."""
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        print("PlantDoc data.yaml not found. Running dataset preparation...")
        prepare_plantdoc_yolo(out_dir=data_yaml.parent, seed=seed)

    # Automatically detect device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training YOLOv8n on device: {device} ({'GPU' if device == 0 else 'CPU'})")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml.resolve().as_posix()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(RUNS_DIR / "detect"),
        name="plantdoc",
        exist_ok=True,
        seed=seed,
        workers=2,
        pretrained=True,
        save=True,
        plots=True,
    )

    best_pt = RUNS_DIR / "detect" / "plantdoc" / "weights" / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, LOCALIZATION_MODEL_PATH)
        print(f"Copied best localization weights to: {LOCALIZATION_MODEL_PATH}")

    return LOCALIZATION_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on PlantDoc dataset")
    parser.add_argument("--data", default=str(PLANTDOC_YOLO_DIR / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=LOCALIZATION_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=LOCALIZATION_BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=LOCALIZATION_IMGSZ)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_localization(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
