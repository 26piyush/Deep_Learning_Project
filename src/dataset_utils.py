"""Dataset preparation, verification, and splitting utilities for Review 2.
Ensures zero data leakage, strict reproducible splits, and dataset documentation.
"""

import argparse
import csv
import io
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import collections
from PIL import Image
import numpy as np

from src.config import (
    PLANTVILLAGE_RAW,
    PLANTVILLAGE_DIR,
    PLANTDOC_RAW_DIR,
    PLANTDOC_YOLO_DIR,
    SEVERITY_ZIP,
    SEVERITY_DIR,
    VIT_CLASSES,
    CANONICAL_TO_RAW_CLASSES,
    PLANTDOC_CROPS_DIR,
    PEAR_LEAVES_DIR,
    UNIFIED_DIR,
    UNIFIED_SPLITS_PATH,
    UNIFIED_42_CLASSES,
    SEED,
)
from src.preprocessing import validate_and_open_image


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# ==============================================================================
# DATASET A: PlantVillage (Vision Transformer)
# ==============================================================================

def prepare_plantvillage_dataset(
    src_dir: Path = PLANTVILLAGE_RAW,
    out_dir: Path = PLANTVILLAGE_DIR,
    classes: List[str] = VIT_CLASSES,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples_per_class: Optional[int] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Prepare PlantVillage subset with stratified train/val/test splits."""
    seed_everything(seed)
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    print(f"\n[PlantVillage] Scanning raw directory: {src_dir}")
    if not src_dir.exists():
        raise FileNotFoundError(f"PlantVillage raw directory does not exist: {src_dir}")

    split_manifest = {
        "classes": classes,
        "train": [],
        "val": [],
        "test": [],
        "stats": {},
    }

    total_images = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for cname in classes:
        raw_name = CANONICAL_TO_RAW_CLASSES.get(cname, cname)
        cdir = src_dir / raw_name
        if not cdir.exists():
            cdir = src_dir / cname
        if not cdir.exists():
            raise FileNotFoundError(f"Class folder not found: {cdir}")

        images = [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        images.sort(key=lambda x: x.name)  # Deterministic order before shuffle
        random.Random(seed).shuffle(images)

        if max_samples_per_class:
            images = images[:max_samples_per_class]

        # Validate images
        valid_images = []
        for img_p in images:
            if validate_and_open_image(img_p) is not None:
                valid_images.append(img_p)

        n = len(valid_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_imgs = valid_images[:n_train]
        val_imgs = valid_images[n_train:n_train + n_val]
        test_imgs = valid_images[n_train + n_val:]

        split_manifest["stats"][cname] = {
            "total": n,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

        # Store relative paths and labels
        for p in train_imgs:
            split_manifest["train"].append({"path": str(p.resolve()), "class": cname})
        for p in val_imgs:
            split_manifest["val"].append({"path": str(p.resolve()), "class": cname})
        for p in test_imgs:
            split_manifest["test"].append({"path": str(p.resolve()), "class": cname})

        # Provide a small preview set in test directory for web app
        test_preview_dir = out_dir / "test" / cname
        test_preview_dir.mkdir(parents=True, exist_ok=True)
        for p in test_imgs[:3]:
            dst = test_preview_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

        total_images += n

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "plantvillage_splits.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    summary = {
        "dataset": "PlantVillage",
        "total_images": total_images,
        "classes": len(classes),
        "class_names": classes,
        "train_count": len(split_manifest["train"]),
        "val_count": len(split_manifest["val"]),
        "test_count": len(split_manifest["test"]),
        "manifest_path": str(manifest_path),
    }

    print("\n" + "=" * 45)
    print("Dataset: PlantVillage")
    print(f"Number of images actually used: {total_images}")
    print(f"Number of classes: {len(classes)}")
    print(f"Training images: {summary['train_count']}")
    print(f"Validation images: {summary['val_count']}")
    print(f"Test images: {summary['test_count']}")
    print("=" * 45)

    return summary


# ==============================================================================
# DATASET B: PlantDoc (YOLO Localization)
# ==============================================================================

def prepare_plantdoc_yolo(
    raw_dir: Path = PLANTDOC_RAW_DIR,
    out_dir: Path = PLANTDOC_YOLO_DIR,
    max_train_samples: int = 1500,
    max_val_samples: int = 350,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Convert PlantDoc annotations (COCO/XML) into YOLO format with single class 0 (leaf_or_disease_region)."""
    seed_everything(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    print(f"\n[PlantDoc] Preparing YOLO localization dataset from: {raw_dir}")

    # Set up YOLO directory structure
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Check for COCO json annotations in plant_doc_detection
    train_json = raw_dir / "annotations" / "plant_diseases_instances_train.json"
    val_json = raw_dir / "annotations" / "plant_diseases_instances_val.json"
    img_base = raw_dir / "plant_diseases" / "images"

    converted_train = 0
    converted_val = 0

    if train_json.exists() and val_json.exists() and img_base.exists():
        print("Found COCO annotations in annotations/ directory.")

        for json_path, split, max_s in [(train_json, "train", max_train_samples), (val_json, "val", max_val_samples)]:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Map image_id -> list of bbox [x, y, w, h]
            boxes_per_image: Dict[int, List[List[float]]] = {}
            for ann in data.get("annotations", []):
                img_id = ann["image_id"]
                bbox = ann["bbox"]  # [x, y, w, h] in pixels
                boxes_per_image.setdefault(img_id, []).append(bbox)

            images = data.get("images", [])
            random.Random(seed).shuffle(images)

            count = 0
            for img_info in images:
                if max_s and count >= max_s:
                    break

                img_id = img_info["id"]
                fname = Path(img_info["file_name"]).name
                img_path = img_base / fname
                if not img_path.exists():
                    continue

                boxes = boxes_per_image.get(img_id, [])
                if not boxes:
                    continue

                w, h = float(img_info.get("width", 0)), float(img_info.get("height", 0))
                if w <= 0 or h <= 0:
                    try:
                        with Image.open(img_path) as im:
                            w, h = float(im.width), float(im.height)
                    except Exception:
                        continue

                # Convert boxes to YOLO format (0, cx, cy, bw, bh) normalized
                yolo_boxes = []
                for bx, by, bw, bh in boxes:
                    if bw <= 1 or bh <= 1:
                        continue
                    cx = (bx + bw / 2.0) / w
                    cy = (by + bh / 2.0) / h
                    nw = bw / w
                    nh = bh / h
                    # Clip coordinates to [0, 1]
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    nw = min(max(nw, 0.0), 1.0)
                    nh = min(max(nh, 0.0), 1.0)
                    yolo_boxes.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                if not yolo_boxes:
                    continue

                # Copy image and write label file
                dst_img = out_dir / "images" / split / fname
                dst_lbl = out_dir / "labels" / split / f"{dst_img.stem}.txt"

                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(yolo_boxes) + "\n")

                count += 1

            if split == "train":
                converted_train = count
            else:
                converted_val = count

    # Generate data.yaml with relative path
    data_yaml = out_dir / "data.yaml"
    # Note: Ultralytics resolves relative paths relative to dataset folder or CWD
    data_yaml_content = (
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: leaf_or_disease_region\n"
    )
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    print(f"PlantDoc YOLO prepared: {converted_train} train images, {converted_val} val images.")
    print(f"Config generated at: {data_yaml}")

    return {
        "dataset": "PlantDoc",
        "train_images": converted_train,
        "val_images": converted_val,
        "classes": 1,
        "data_yaml": str(data_yaml),
    }


# ==============================================================================
# DATASET C: DiaMOS Plant (Severity Classification)
# ==============================================================================

def prepare_diamos_severity(
    zip_path: Path = SEVERITY_ZIP,
    out_dir: Path = SEVERITY_DIR,
    max_images_per_class: int = 300,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Extract and prepare DiaMOS Plant severity dataset from Pear.zip."""
    seed_everything(seed)
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)

    print(f"\n[DiaMOS] Extracting and preparing severity dataset from: {zip_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Severity archive not found: {zip_path}")

    img_out_dir = out_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    severity_mapping = {
        "severity_0": "No Risk",
        "severity_1": "Very Low",
        "severity_2": "Low",
        "severity_3": "Medium",
        "severity_4": "High",
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_data = zf.read("Pear/annotation/csv/diaMOSPlant.csv").decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(csv_data), delimiter=";")
        all_rows = list(reader)

        # Build list of valid leaf image files in zip
        leaf_file_map = {}
        for name in zf.namelist():
            if name.startswith("Pear/leaves/") and not "/._" in name and name.lower().endswith((".jpg", ".png")):
                basename = Path(name).name
                leaf_file_map[basename] = name

        # Group rows by severity class
        class_samples: Dict[str, List[Dict[str, str]]] = {cls: [] for cls in severity_mapping.values()}

        for row in all_rows:
            fname = row.get("filename", "")
            if fname not in leaf_file_map:
                continue

            sev_label = None
            for flag_col, label_name in severity_mapping.items():
                if row.get(flag_col) == "1":
                    sev_label = label_name
                    break

            if sev_label:
                class_samples[sev_label].append({
                    "filename": fname,
                    "zip_entry": leaf_file_map[fname],
                    "severity": sev_label,
                })

        # Extract balanced sample and resize to 256x256 for fast I/O
        csv_records = []
        for sev_label, samples in class_samples.items():
            random.Random(seed).shuffle(samples)
            selected = samples[:max_images_per_class] if max_images_per_class else samples
            print(f"DiaMOS {sev_label}: {len(selected)} samples (from {len(samples)} available)")

            for item in selected:
                dst_path = img_out_dir / item["filename"]
                if not dst_path.exists():
                    try:
                        raw_bytes = zf.read(item["zip_entry"])
                        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                            # Optimize resolution to 256x256 to save disk space and accelerate DataLoader
                            pil_img.thumbnail((300, 300))
                            pil_img.convert("RGB").save(dst_path, "JPEG", quality=90)
                    except Exception as e:
                        continue

                csv_records.append({
                    "image_path": str(dst_path.resolve()),
                    "severity": sev_label,
                })

    csv_path = out_dir / "severity_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "severity"])
        writer.writeheader()
        writer.writerows(csv_records)

    print(f"\nSaved {len(csv_records)} severity annotations to: {csv_path}")

    return {
        "dataset": "DiaMOS Plant",
        "total_images": len(csv_records),
        "classes": len(severity_mapping),
        "class_names": list(severity_mapping.values()),
        "csv_path": str(csv_path),
    }


def prepare_unified_dataset(
    plantvillage_dir: Path = PLANTVILLAGE_RAW,
    plantdoc_crops_dir: Path = PLANTDOC_CROPS_DIR,
    pear_leaves_dir: Path = PEAR_LEAVES_DIR,
    out_dir: Path = UNIFIED_DIR,
    classes: List[str] = UNIFIED_42_CLASSES,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Prepare Unified multi-dataset splits combining PlantVillage, PlantDoc (field crops), and DiaMOS Pear."""
    seed_everything(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("Preparing Unified Multi-Dataset for Vision Transformer")
    print(f"Taxonomy: {len(classes)} classes across 15 crop species")
    print("Datasets: PlantVillage (Lab) + PlantDoc (Field) + DiaMOS (Pear)")
    print("=" * 65)

    split_manifest = {
        "classes": classes,
        "train": [],
        "val": [],
        "test": [],
        "stats": {},
        "source_counts": {"PlantVillage": 0, "PlantDoc": 0, "DiaMOS_Pear": 0},
    }

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total_images = 0

    for cname in classes:
        class_samples = []  # list of (path, source)

        # 1. Check PlantVillage
        raw_name = CANONICAL_TO_RAW_CLASSES.get(cname, cname)
        pv_class_dir = plantvillage_dir / raw_name
        if pv_class_dir.exists():
            for p in pv_class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    class_samples.append((p, "PlantVillage"))

        # 2. Check PlantDoc in-the-wild leaf crops
        pd_class_dir = plantdoc_crops_dir / cname
        if pd_class_dir.exists():
            for p in pd_class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    class_samples.append((p, "PlantDoc"))

        # 3. Check DiaMOS Pear leaves
        if cname.startswith("Pear___"):
            pear_cond = cname.replace("Pear___", "")
            pear_cond_dir = pear_leaves_dir / pear_cond
            if pear_cond_dir.exists():
                for p in pear_cond_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        class_samples.append((p, "DiaMOS_Pear"))

        if not class_samples:
            print(f"Warning: No samples found for class: {cname}")
            continue

        # Deterministic shuffle
        class_samples.sort(key=lambda x: x[0].name)
        random.Random(seed).shuffle(class_samples)

        n = len(class_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_items = class_samples[:n_train]
        val_items = class_samples[n_train:n_train + n_val]
        test_items = class_samples[n_train + n_val:]

        split_manifest["stats"][cname] = {
            "total": n,
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
            "sources": dict(collections.Counter(src for _, src in class_samples)),
        }

        for p, src in train_items:
            split_manifest["train"].append({"path": str(p.resolve()), "class": cname, "source": src})
            split_manifest["source_counts"][src] = split_manifest["source_counts"].get(src, 0) + 1
        for p, src in val_items:
            split_manifest["val"].append({"path": str(p.resolve()), "class": cname, "source": src})
            split_manifest["source_counts"][src] = split_manifest["source_counts"].get(src, 0) + 1
        for p, src in test_items:
            split_manifest["test"].append({"path": str(p.resolve()), "class": cname, "source": src})
            split_manifest["source_counts"][src] = split_manifest["source_counts"].get(src, 0) + 1

        # Save preview test sample in out_dir/test/<cname>
        preview_dir = out_dir / "test" / cname
        preview_dir.mkdir(parents=True, exist_ok=True)
        for p, _ in test_items[:3]:
            dst = preview_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

        total_images += n

    manifest_path = out_dir / "unified_splits.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    # Also keep plantvillage_splits.json synced for transparent drop-in compatibility
    with open(PLANTVILLAGE_DIR / "plantvillage_splits.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    # Sync preview samples to plantvillage/test as well so app.py picks them up
    pv_test = PLANTVILLAGE_DIR / "test"
    for cname in classes:
        c_prev = out_dir / "test" / cname
        if c_prev.exists():
            (pv_test / cname).mkdir(parents=True, exist_ok=True)
            for f in c_prev.glob("*"):
                dst = pv_test / cname / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)

    summary = {
        "dataset": "Unified (PlantVillage + PlantDoc + DiaMOS Pear)",
        "total_images": total_images,
        "classes": len(classes),
        "train_count": len(split_manifest["train"]),
        "val_count": len(split_manifest["val"]),
        "test_count": len(split_manifest["test"]),
        "source_counts": split_manifest["source_counts"],
        "manifest_path": str(manifest_path),
    }

    print("\n" + "=" * 65)
    print(f"Unified Dataset Preparation Complete:")
    print(f"Total Images Used:   {total_images}")
    print(f"Classes:             {len(classes)}")
    print(f"Train Samples:       {summary['train_count']}")
    print(f"Validation Samples:  {summary['val_count']}")
    print(f"Test Samples:        {summary['test_count']}")
    print(f"Source Breakdown:    {summary['source_counts']}")
    print(f"Manifest Path:       {manifest_path}")
    print("=" * 65)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare all datasets for Crop Disease Pipeline")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets including unified")
    parser.add_argument("--unified", action="store_true", help="Prepare Unified multi-dataset")
    parser.add_argument("--plantvillage", action="store_true", help="Prepare PlantVillage")
    parser.add_argument("--plantdoc", action="store_true", help="Prepare PlantDoc YOLO")
    parser.add_argument("--severity", action="store_true", help="Prepare DiaMOS severity")
    args = parser.parse_args()

    if args.all or args.unified:
        prepare_unified_dataset()
    if args.all or args.plantvillage:
        prepare_plantvillage_dataset()
    if args.all or args.plantdoc:
        prepare_plantdoc_yolo()
    if args.all or args.severity:
        prepare_diamos_severity()


if __name__ == "__main__":
    main()
