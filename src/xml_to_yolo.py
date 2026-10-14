"""PlantDoc XML and COCO annotation converter to YOLO format.
Single localization class: 0 = leaf_or_disease_region.
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_image_index(root: Path) -> Dict[str, Path]:
    index = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            index[p.stem] = p
            index[p.name] = p
    return index


def parse_xml_boxes(xml_path: Path) -> Tuple[float, float, List[Tuple[int, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> found in {xml_path}")

    width = float(size.findtext("width", 0))
    height = float(size.findtext("height", 0))
    objects = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", 0))
        ymin = float(box.findtext("ymin", 0))
        xmax = float(box.findtext("xmax", 0))
        ymax = float(box.findtext("ymax", 0))

        if width <= 0 or height <= 0:
            continue

        xmin = max(0.0, min(xmin, width - 1))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height - 1))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height

        objects.append((0, cx, cy, bw, bh))

    return width, height, objects


def main():
    parser = argparse.ArgumentParser(description="Convert PlantDoc annotations to YOLO format")
    parser.add_argument("--src_root", default="data/plant_doc_detection")
    parser.add_argument("--out_root", default="data/plantdoc_yolo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src_root)
    out = Path(args.out_root)
    rng = random.Random(args.seed)

    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Check for COCO json annotations first
    train_json = src / "annotations" / "plant_diseases_instances_train.json"
    val_json = src / "annotations" / "plant_diseases_instances_val.json"
    img_dir = src / "plant_diseases" / "images"

    converted = 0
    if train_json.exists() and val_json.exists():
        from src.dataset_utils import prepare_plantdoc_yolo
        res = prepare_plantdoc_yolo(raw_dir=src, out_dir=out, seed=args.seed)
        converted = res["train_images"] + res["val_images"]
    else:
        # Fall back to XML files
        image_index = build_image_index(src)
        xml_files = list(src.rglob("*.xml"))
        if not xml_files:
            raise RuntimeError(f"No XML or COCO annotations found under {src}")

        for xml_p in xml_files:
            img_p = image_index.get(xml_p.stem)
            if not img_p:
                continue

            try:
                _, _, objs = parse_xml_boxes(xml_p)
            except Exception:
                continue

            if not objs:
                continue

            split = "val" if "test" in str(xml_p).lower() else ("train" if rng.random() < 0.8 else "val")
            dst_img = out / "images" / split / img_p.name
            dst_lbl = out / "labels" / split / f"{img_p.stem}.txt"

            shutil.copy2(img_p, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                for row in objs:
                    f.write(f"0 {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")
            converted += 1

    # Write data.yaml
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: leaf_or_disease_region\n",
        encoding="utf-8",
    )

    print(f"Successfully converted {converted} annotated images to YOLO format at {out}")


if __name__ == "__main__":
    main()
