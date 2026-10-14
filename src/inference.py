"""Stage 4 — End-to-End Integrated Inference Pipeline.
Chains:
  Input Image -> Image Preprocessing -> YOLO Localization -> Smart Context Leaf Crop
  (or Controlled Fallback to Full Image if no box detected) ->
  ViT Disease Classification + Confidence (Top-3) ->
  Severity Classification + Confidence (Top-3) ->
  Visualized and Formatted Result.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

from src.config import (
    VIT_MODEL_PATH,
    LOCALIZATION_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    CROP_HEALTH_WEIGHTS_PATH,
    CROP_HEALTH_META_PATH,
    PREDICTIONS_DIR,
    get_device,
)
from src.preprocessing import validate_and_open_image
from src.utils import save_json
from src.explainability import (
    compute_vit_attention,
    generate_attention_heatmap,
    overlay_heatmap_on_image,
    extract_disease_bounding_boxes,
    draw_hierarchical_annotations,
)
from src.segmentation import (
    extract_leaf_mask,
    extract_lesion_mask,
    calculate_mathematical_severity,
    visualize_dual_masks,
)
from src.prescription import get_agronomic_prescription
from src.knowledge_base import get_disease_knowledge

try:
    from crop_health_model.predictor import CropHealthPredictor
except (ImportError, ValueError):
    from crop_health_model import CropHealthPredictor


def load_vit_model(model_path: Path, device: torch.device) -> Tuple[nn.Module, Any, List[str]]:
    """Load trained ViT model and preprocessing transform."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ViT model not found at {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    image_size = ckpt.get("image_size", 224)
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, tfm, class_names


def load_severity_model(model_path: Path, device: torch.device) -> Tuple[nn.Module, Any, List[str]]:
    """Load trained severity model and preprocessing transform."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Severity model not found at {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    image_size = ckpt.get("image_size", 224)
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, tfm, class_names


def load_crop_health_model(
    weights_path: Path = CROP_HEALTH_WEIGHTS_PATH,
    meta_path: Path = CROP_HEALTH_META_PATH,
    device: Optional[torch.device] = None,
) -> CropHealthPredictor:
    """Load trained Crop Health ResNet-18 model (38 classes across 14 crops)."""
    weights_path = Path(weights_path)
    meta_path = Path(meta_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Crop Health weights not found at {weights_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Crop Health metadata not found at {meta_path}")
    dev_str = device.type if device else None
    return CropHealthPredictor(weights_path=weights_path, meta_path=meta_path, device=dev_str)


def extract_leaf_crop(
    image: Image.Image,
    boxes: np.ndarray,
    crop_mode: str = "smart",
    padding_ratio: float = 0.15,
) -> Tuple[Image.Image, List[int], str, bool]:
    """Intelligently crops the localized leaf region with context padding to prevent symptom truncation."""
    w, h = image.size
    
    if len(boxes) == 0 or crop_mode == "full":
        return image, [0, 0, w, h], "Full Image (Direct Leaf)" if crop_mode == "full" else "Fallback: Full Image (No bounding box detected)", (len(boxes) == 0)
    
    if crop_mode == "raw":
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_b = boxes[int(np.argmax(areas))]
        x1, y1, x2, y2 = max(0, int(best_b[0])), max(0, int(best_b[1])), min(w, int(best_b[2])), min(h, int(best_b[3]))
        return image.crop((x1, y1, x2, y2)), [x1, y1, x2, y2], f"Raw Largest Box ({len(boxes)} detected)", False
        
    # Smart Mode: Union of detected regions with adaptive context padding
    ux1 = float(min(b[0] for b in boxes))
    uy1 = float(min(b[1] for b in boxes))
    ux2 = float(max(b[2] for b in boxes))
    uy2 = float(max(b[3] for b in boxes))
    
    bw = max(1.0, ux2 - ux1)
    bh = max(1.0, uy2 - uy1)
    
    # Expand box with context padding so surrounding leaf margins and lesion patterns are preserved
    pad_x = max(bw * padding_ratio, 12.0)
    pad_y = max(bh * padding_ratio, 12.0)
    
    x1 = max(0, int(ux1 - pad_x))
    y1 = max(0, int(uy1 - pad_y))
    x2 = min(w, int(ux2 + pad_x))
    y2 = min(h, int(uy2 + pad_y))
    
    # If the padded box occupies >88% of the frame, preserve the detected union box while using the frame safely
    box_area = (x2 - x1) * (y2 - y1)
    if box_area >= 0.88 * (w * h):
        return image, [int(ux1), int(uy1), int(ux2), int(uy2)], f"Smart Leaf Focus ({len(boxes)} box(es) detected, covers {int(box_area/(w*h)*100)}% of frame)", False
        
    cropped = image.crop((x1, y1, x2, y2))
    return cropped, [x1, y1, x2, y2], f"Smart Localized Region ({len(boxes)} box(es) merged + {int(padding_ratio*100)}% padding)", False


def annotate_localization(
    image: Image.Image,
    detected_boxes: List[Dict[str, Any]],
    selected_box: List[int],
    used_fallback: bool,
    crop_mode: str = "smart",
) -> Image.Image:
    """Draw crisp bounding boxes, detection confidences, and smart crop regions."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    try:
        font_size = max(11, int(min(w, h) * 0.038))
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    if used_fallback or len(detected_boxes) == 0:
        # Fallback frame
        draw.rectangle([2, 2, w - 3, h - 3], outline="#FF9800", width=4)
        tag = "Controlled Fallback (0 boxes >= conf)"
        draw.rectangle([2, 2, min(w, 240), 24], fill="#FF9800")
        draw.text((6, 4), tag, fill="#FFFFFF", font=font)
        return annotated

    # Draw individual YOLO detected boxes (Vibrant Green)
    for i, item in enumerate(detected_boxes):
        box = item["box"]
        conf = item.get("confidence", 0.0)
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        draw.rectangle([bx1, by1, bx2, by2], outline="#00E676", width=3)
        tag = f"Leaf {i+1}: {conf*100:.1f}%"
        
        try:
            bbox = draw.textbbox((bx1, by1), tag, font=font)
            tw = bbox[2] - bbox[0] + 8
            th = bbox[3] - bbox[1] + 4
        except Exception:
            tw, th = len(tag) * 8 + 8, 18
            
        tag_y = max(0, by1 - th)
        draw.rectangle([bx1, tag_y, min(w, bx1 + tw), tag_y + th], fill="#00E676")
        draw.text((bx1 + 4, tag_y + 2), tag, fill="#003300", font=font)

    # In Smart Crop mode, draw the merged padded ROI (Golden Amber)
    if crop_mode == "smart" and selected_box and selected_box != [0, 0, w, h]:
        sx1, sy1, sx2, sy2 = int(selected_box[0]), int(selected_box[1]), int(selected_box[2]), int(selected_box[3])
        # Draw padded boundary
        draw.rectangle([sx1, sy1, sx2, sy2], outline="#FFB300", width=2)
        pad_tag = "Smart ROI (+15%)"
        try:
            pbox = draw.textbbox((sx1, sy2), pad_tag, font=font)
            pw = pbox[2] - pbox[0] + 8
            ph = pbox[3] - pbox[1] + 4
        except Exception:
            pw, ph = len(pad_tag) * 8 + 8, 18
        pad_y = max(0, min(h - ph, sy2 - ph))
        draw.rectangle([sx1, pad_y, min(w, sx1 + pw), pad_y + ph], fill="#FFB300")
        draw.text((sx1 + 4, pad_y + 2), pad_tag, fill="#3E2723", font=font)

    return annotated


_MODEL_CACHE: Dict[Any, Any] = {}


def load_pipeline_models(
    yolo_path: Path = LOCALIZATION_MODEL_PATH,
    vit_path: Path = VIT_MODEL_PATH,
    severity_path: Path = SEVERITY_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[YOLO, nn.Module, Any, List[str], nn.Module, Any, List[str], torch.device]:
    """Load and cache standard pipeline models."""
    if device is None:
        device = get_device()
    cache_key = (str(yolo_path), str(vit_path), str(severity_path), device.type)
    if cache_key not in _MODEL_CACHE:
        yolo_model = YOLO(str(yolo_path))
        vit_model, vit_tfm, vit_classes = load_vit_model(Path(vit_path), device)
        sev_model, sev_tfm, sev_classes = load_severity_model(Path(severity_path), device)
        _MODEL_CACHE[cache_key] = (
            yolo_model, vit_model, vit_tfm, vit_classes, sev_model, sev_tfm, sev_classes, device
        )
    return _MODEL_CACHE[cache_key]


def get_crop_health_model(
    weights_path: Path = CROP_HEALTH_WEIGHTS_PATH,
    meta_path: Path = CROP_HEALTH_META_PATH,
    device: Optional[torch.device] = None,
) -> CropHealthPredictor:
    """Cached loader for CropHealthPredictor."""
    if device is None:
        device = get_device()
    cache_key = ("crop_health", str(weights_path), str(meta_path), device.type)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_crop_health_model(weights_path, meta_path, device)
    return _MODEL_CACHE[cache_key]


def run_pipeline(
    image: Any,
    yolo_model: Optional[YOLO] = None,
    vit_model: Optional[nn.Module] = None,
    vit_tfm: Any = None,
    vit_classes: Optional[List[str]] = None,
    sev_model: Optional[nn.Module] = None,
    sev_tfm: Any = None,
    sev_classes: Optional[List[str]] = None,
    crop_health_model: Optional[Any] = None,
    device: Optional[torch.device] = None,
    conf_thresh: float = 0.25,
    crop_mode: str = "smart",
    classifier_mode: str = "vit",
    crop_filter: Optional[str] = None,
    save_visualization: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute end-to-end inference on a PIL image or image file path.
    Supports multi-engine classification:
      - 'vit': Specialized 5-Class ViT-B/16 (Solanaceae)
      - 'crop_health': 38-Class ResNet-18 across 14 crops
      - 'hybrid': Smart cascade (ViT first; auto-escalates to 38-Class model if low confidence)
      - 'compare': Side-by-side execution and benchmark telemetry
    """
    if device is None:
        device = get_device()

    # Auto-load pipeline models if not explicitly provided
    need_vit = classifier_mode in ("vit", "hybrid", "compare")
    need_ch = classifier_mode in ("crop_health", "hybrid", "compare")

    if yolo_model is None or sev_model is None or (need_vit and vit_model is None):
        yolo_p = kwargs.get("yolo_path", LOCALIZATION_MODEL_PATH)
        vit_p = kwargs.get("vit_path", VIT_MODEL_PATH)
        sev_p = kwargs.get("severity_path", SEVERITY_MODEL_PATH)
        (
            yolo_model,
            vit_model,
            vit_tfm,
            vit_classes,
            sev_model,
            sev_tfm,
            sev_classes,
            device,
        ) = load_pipeline_models(yolo_p, vit_p, sev_p, device)

    if need_ch and crop_health_model is None:
        ch_w = kwargs.get("crop_health_weights", CROP_HEALTH_WEIGHTS_PATH)
        ch_m = kwargs.get("crop_health_meta", CROP_HEALTH_META_PATH)
        crop_health_model = get_crop_health_model(ch_w, ch_m, device)

    # Load image if file path or numpy array
    if isinstance(image, (str, Path)):
        img_p = Path(image)
        img_rgb = validate_and_open_image(img_p)
        if img_rgb is None:
            img_rgb = Image.open(img_p).convert("RGB")
    elif isinstance(image, Image.Image):
        img_rgb = image.convert("RGB")
    elif hasattr(image, "__array__"):
        img_rgb = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Step 1: Disease/Leaf Localization via YOLO
    yolo_res = yolo_model.predict(
        source=img_rgb, conf=conf_thresh, verbose=False, device=(0 if device.type == "cuda" else "cpu")
    )[0]

    boxes = yolo_res.boxes.xyxy.cpu().numpy() if yolo_res.boxes is not None else np.empty((0, 4))
    confs = yolo_res.boxes.conf.cpu().numpy() if yolo_res.boxes is not None else np.empty((0,))

    raw_boxes = []
    if yolo_res.boxes is not None and len(boxes) > 0:
        for b, c in zip(boxes, confs):
            raw_boxes.append({
                "box": [int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))],
                "confidence": float(c),
                "label": "Leaf / Disease Region",
            })

    # Step 2: Intelligent Bounding Box Crop with Context Padding
    cropped_region, selected_box, localization_status, used_fallback = extract_leaf_crop(
        img_rgb, boxes, crop_mode=crop_mode
    )

    annotated_img = annotate_localization(
        img_rgb, raw_boxes, selected_box, used_fallback, crop_mode=crop_mode
    )

    # Step 3: Disease Classification Function Helpers
    def _evaluate_vit() -> Dict[str, Any]:
        vit_tensor = vit_tfm(cropped_region).unsqueeze(0).to(device)
        with torch.no_grad():
            vit_logits = vit_model(vit_tensor)
            vit_probs = torch.softmax(vit_logits, dim=1)[0]

        filtered_to_crop = None
        if crop_filter:
            crop_clean = crop_filter.lower().replace(" ", "").replace("_", "").replace("(bell)", "")
            matching_indices = [
                i for i, c in enumerate(vit_classes)
                if crop_clean in c.split("___")[0].lower().replace(" ", "").replace("_", "")
            ]
            if matching_indices:
                filtered_to_crop = crop_filter
                mask = torch.zeros_like(vit_probs)
                mask[matching_indices] = vit_probs[matching_indices]
                if mask.sum() > 0:
                    vit_probs = mask / mask.sum()

        vit_idx = int(vit_probs.argmax())
        raw_cls = vit_classes[vit_idx]
        conf = float(vit_probs[vit_idx])

        top3_k = min(3, len(vit_classes))
        top3_vals, top3_idxs = vit_probs.topk(top3_k)
        top3_diseases = [
            {"class": vit_classes[int(idx)], "confidence": float(val)}
            for val, idx in zip(top3_vals, top3_idxs)
        ]

        parts = raw_cls.split("___")
        c_name = parts[0].replace("_", " ") if len(parts) > 1 else "Plant"
        d_name = parts[1].replace("_", " ") if len(parts) > 1 else raw_cls

        scope_str = f"{len(vit_classes)}-Class PlantVillage"
        return {
            "model_type": "ViT-B/16",
            "scope": scope_str,
            "crop": c_name,
            "disease_raw": raw_cls,
            "disease": d_name,
            "confidence": conf,
            "top_predictions": top3_diseases,
            "filtered_to_crop": filtered_to_crop,
            "is_low_confidence": bool(conf < 0.65),
        }

    def _evaluate_crop_health() -> Dict[str, Any]:
        ch_out = crop_health_model.predict(cropped_region, crop_name=crop_filter, top_k=5)
        best = ch_out["prediction"]
        top_preds = ch_out["top_predictions"]
        conf = float(best["confidence"])

        return {
            "model_type": "ResNet-18",
            "scope": "38-Class Full Spectrum (14 Crops)",
            "crop": best["crop"],
            "disease_raw": best["class"],
            "disease": best["disease"].replace("_", " "),
            "confidence": conf,
            "top_predictions": top_preds,
            "filtered_to_crop": ch_out.get("filtered_to_crop"),
            "is_low_confidence": bool(conf < 0.50),
        }

    # Execute classification based on selected engine mode
    vit_result: Optional[Dict[str, Any]] = None
    crop_health_result: Optional[Dict[str, Any]] = None
    comparison_meta: Optional[Dict[str, Any]] = None
    hybrid_cascade_triggered = False

    if classifier_mode == "crop_health":
        crop_health_result = _evaluate_crop_health()
        primary_diag = crop_health_result
        classifier_name = "Crop Health ResNet-18 (38 Classes)"

    elif classifier_mode == "hybrid":
        vit_result = _evaluate_vit()
        if vit_result["is_low_confidence"]:
            crop_health_result = _evaluate_crop_health()
            hybrid_cascade_triggered = True
            primary_diag = crop_health_result
            classifier_name = "Smart Hybrid Cascade (Escalated to 38-Class Model)"
        else:
            primary_diag = vit_result
            classifier_name = "Smart Hybrid Cascade (ViT Confirmed)"

    elif classifier_mode == "compare":
        vit_result = _evaluate_vit()
        crop_health_result = _evaluate_crop_health()
        agreement = bool(vit_result["disease_raw"] == crop_health_result["disease_raw"])
        comparison_meta = {
            "agreement": agreement,
            "vit_prediction": vit_result["disease_raw"],
            "vit_confidence": vit_result["confidence"],
            "crop_health_prediction": crop_health_result["disease_raw"],
            "crop_health_confidence": crop_health_result["confidence"],
        }
        # Select highest confidence as primary
        if vit_result["confidence"] >= crop_health_result["confidence"]:
            primary_diag = vit_result
        else:
            primary_diag = crop_health_result
        classifier_name = "Side-by-Side Comparison (ViT vs ResNet-18)"

    else:  # default 'vit'
        vit_result = _evaluate_vit()
        primary_diag = vit_result
        classifier_name = "Vision Transformer (ViT-B/16)"

    predicted_disease = primary_diag["disease_raw"]
    disease_detail = primary_diag["disease"]
    crop_name = primary_diag["crop"]
    disease_conf = primary_diag["confidence"]
    top3_diseases = primary_diag["top_predictions"][:3]
    is_low_confidence = primary_diag["is_low_confidence"]

    # Step 4: Severity Classification
    sev_tensor = sev_tfm(cropped_region).unsqueeze(0).to(device)
    with torch.no_grad():
        sev_logits = sev_model(sev_tensor)
        sev_probs = torch.softmax(sev_logits, dim=1)[0]
        sev_idx = int(sev_probs.argmax())

    predicted_severity = sev_classes[sev_idx]
    severity_conf = float(sev_probs[sev_idx])

    top3_sev_k = min(3, len(sev_classes))
    top3_sev_vals, top3_sev_idxs = sev_probs.topk(top3_sev_k)
    top3_severities = [
        {"severity": sev_classes[int(idx)], "confidence": float(val)}
        for val, idx in zip(top3_sev_vals, top3_sev_idxs)
    ]

    # Warning handling
    confidence_warning = None
    if is_low_confidence:
        if classifier_mode == "vit":
            confidence_warning = (
                f"Low prediction confidence ({disease_conf*100:.1f}%). "
                f"The image may contain a crop/disease outside the 5 supported Solanaceae classes. "
                f"Tip: Switch to the 38-class Crop Health engine for broader diagnosis."
            )
        else:
            confidence_warning = (
                f"Low prediction confidence ({disease_conf*100:.1f}%). "
                f"Image may have severe background clutter or an unlisted symptom."
            )

    # Step 5: Final Phase — Attention Rollout, Lesion Extraction, Dual-Mask Segmentation & Prescription
    is_healthy = "healthy" in predicted_disease.lower()

    # Compute ViT self-attention map
    try:
        if vit_model is not None and vit_tfm is not None:
            vit_t = vit_tfm(cropped_region).unsqueeze(0).to(device)
            attention_map = compute_vit_attention(vit_model, vit_t, device)
        else:
            attention_map = np.ones((14, 14), dtype=np.float32) / 196.0
    except Exception:
        attention_map = np.ones((14, 14), dtype=np.float32) / 196.0

    # Pixel-Level Dual-Mask Segmentation (Leaf Mask L_mask & Lesion Mask D_mask)
    leaf_mask = extract_leaf_mask(cropped_region)
    lesion_mask = extract_lesion_mask(cropped_region, leaf_mask, attention_map, is_healthy=is_healthy)
    math_severity = calculate_mathematical_severity(leaf_mask, lesion_mask, is_healthy=is_healthy)
    dual_mask_image = visualize_dual_masks(cropped_region, leaf_mask, lesion_mask)

    # Extract grounded discrete disease symptom bounding boxes
    disease_boxes = extract_disease_bounding_boxes(
        cropped_region, leaf_mask, lesion_mask, attention_map=attention_map, is_healthy=is_healthy, disease_name=disease_detail
    )

    # Generate ViT Attention Rollout Heatmap
    heatmap_rgb = generate_attention_heatmap(attention_map, target_size=cropped_region.size)
    heatmap_image = overlay_heatmap_on_image(cropped_region, heatmap_rgb, alpha=0.52)

    # Agronomic Chemical Treatment Prescription & Comprehensive Disease Knowledge Base
    prescription = get_agronomic_prescription(
        crop_name, disease_detail, math_severity["severity_percentage"], predicted_disease
    )
    disease_knowledge = get_disease_knowledge(
        predicted_disease, crop=crop_name, disease=disease_detail
    )

    # Hierarchical Annotations (Leaf Box + Disease Symptom Boxes) on Cropped ROI
    disease_annotated_crop = draw_hierarchical_annotations(
        cropped_region, [0, 0, cropped_region.width, cropped_region.height], disease_boxes, is_healthy=is_healthy, disease_name=disease_detail
    )

    # Map disease lesion boxes back to original input image coordinate space
    mapped_disease_boxes = []
    ox1, oy1 = selected_box[0], selected_box[1]
    for d_item in disease_boxes:
        dx1, dy1, dx2, dy2 = d_item["box"]
        mapped_disease_boxes.append({
            "box": [dx1 + ox1, dy1 + oy1, dx2 + ox1, dy2 + oy1],
            "confidence": d_item["confidence"],
            "area": d_item["area"],
            "label": d_item.get("label", disease_detail),
        })

    # Full Input Image with both Outer Leaf Box + Inner Disease Symptom Boxes
    full_hierarchical_image = draw_hierarchical_annotations(
        img_rgb, selected_box, mapped_disease_boxes, is_healthy=is_healthy, disease_name=disease_detail
    )

    out_result = {
        "crop": crop_name,
        "disease_raw": predicted_disease,
        "disease_raw_class": predicted_disease,
        "disease": disease_detail,
        "disease_confidence": disease_conf,
        "top3_diseases": top3_diseases,
        "severity": predicted_severity,
        "severity_confidence": severity_conf,
        "top3_severities": top3_severities,
        "bounding_box": selected_box,
        "detected_boxes": raw_boxes,
        "max_detection_confidence": float(max(confs)) if len(confs) > 0 else 0.0,
        "total_boxes_found": len(boxes),
        "localization_status": localization_status,
        "localization": localization_status,
        "used_fallback": used_fallback,
        "cropped_image": cropped_region,
        "annotated_image": annotated_img,
        "is_low_confidence": is_low_confidence,
        "confidence_warning": confidence_warning,
        "classifier_mode": classifier_mode,
        "classifier_name": classifier_name,
        "crop_filter": crop_filter,
        "vit_result": vit_result,
        "crop_health_result": crop_health_result,
        "comparison": comparison_meta,
        "hybrid_cascade_triggered": hybrid_cascade_triggered,
        # Final Review 100% Milestone Deliverables
        "disease_boxes": disease_boxes,
        "mapped_disease_boxes": mapped_disease_boxes,
        "disease_annotated_crop": disease_annotated_crop,
        "full_hierarchical_image": full_hierarchical_image,
        "attention_map": attention_map,
        "heatmap_image": heatmap_image,
        "dual_mask_image": dual_mask_image,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
        "mathematical_severity": math_severity,
        "continuous_severity_pct": math_severity["severity_percentage"],
        "agronomic_tier": math_severity["agronomic_tier"],
        "prescription": prescription,
        "disease_knowledge": disease_knowledge,
    }

    if save_visualization or kwargs.get("visualize", False):
        out_result["annotated_image"] = visualize_result(img_rgb, out_result)

    return out_result


def visualize_result(
    image: Image.Image,
    result: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> Image.Image:
    """Create an annotated visualization with bounding box, crop, and prediction metadata."""
    if "annotated_image" in result and result["annotated_image"] is not None:
        img = result["annotated_image"].copy().convert("RGB")
    else:
        img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Banner header
    engine_label = result.get("classifier_name", "ViT-B/16")
    text_lines = [
        f"Crop: {result['crop']}",
        f"Disease: {result['disease']} ({result['disease_confidence']*100:.2f}%)",
        f"Severity: {result['severity']} ({result['severity_confidence']*100:.2f}%)",
        f"Localization: {result['localization_status']}",
        f"Engine: {engine_label}",
    ]
    if result.get("hybrid_cascade_triggered"):
        text_lines.append("Notice: ViT low confidence -> Escalated to 38-class Crop Health engine")

    header_height = len(text_lines) * 22 + 16
    draw.rectangle([(0, 0), (img.width, header_height)], fill=(0, 0, 0, 200))
    y_text = 8
    for line in text_lines:
        draw.text((12, y_text), line, fill="#FFFFFF")
        y_text += 22

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return img


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end crop disease detection inference")
    parser.add_argument("--image", required=True, help="Path to input plant/leaf image")
    parser.add_argument("--localization_model", default=str(LOCALIZATION_MODEL_PATH))
    parser.add_argument("--vit_model", default=str(VIT_MODEL_PATH))
    parser.add_argument("--severity_model", default=str(SEVERITY_MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--crop_mode", choices=["smart", "full", "raw"], default="smart",
                        help="Cropping strategy: smart (padded union), full (whole leaf), raw (tight YOLO box)")
    parser.add_argument(
        "--classifier",
        choices=["vit", "crop_health", "hybrid", "compare"],
        default="vit",
        help="Classification engine: vit (5-class Solanaceae), crop_health (38-class ResNet-18), hybrid (auto-cascade), compare (side-by-side benchmark)",
    )
    parser.add_argument(
        "--crop_filter",
        default=None,
        help="Optional crop name filter (e.g. Tomato, Apple, Corn, Potato) for Crop Health engine",
    )
    parser.add_argument("--save", default=str(PREDICTIONS_DIR / "inference_result.jpg"))
    args = parser.parse_args()

    device = get_device()
    print(f"Executing End-to-End Inference on device: {device} | Engine: {args.classifier}")

    # Load models
    yolo = YOLO(args.localization_model)
    vit_model, vit_tfm, vit_classes = load_vit_model(Path(args.vit_model), device)
    sev_model, sev_tfm, sev_classes = load_severity_model(Path(args.severity_model), device)
    ch_model = get_crop_health_model(device=device)

    # Load and validate image
    img = validate_and_open_image(Path(args.image))
    if img is None:
        raise ValueError(f"Could not open or decode valid image from: {args.image}")

    res = run_pipeline(
        img,
        yolo_model=yolo,
        vit_model=vit_model,
        vit_tfm=vit_tfm,
        vit_classes=vit_classes,
        sev_model=sev_model,
        sev_tfm=sev_tfm,
        sev_classes=sev_classes,
        crop_health_model=ch_model,
        device=device,
        conf_thresh=args.conf,
        crop_mode=args.crop_mode,
        classifier_mode=args.classifier,
        crop_filter=args.crop_filter,
    )

    save_path = Path(args.save)
    visualize_result(img, res, save_path=save_path)

    print("\n" + "=" * 60)
    print(f"END-TO-END INFERENCE RESULT [{res['classifier_name']}]")
    print("=" * 60)
    print(f"Crop:               {res['crop']}")
    print(f"Primary Disease:    {res['disease']}")
    print(f"Disease Confidence: {res['disease_confidence'] * 100:.2f}%")
    print(f"Target Class:       {res['disease_raw']}")
    print("\nTop Disease Candidates:")
    for c in res["top3_diseases"]:
        cls_name = c.get("class", c.get("disease", "Unknown"))
        print(f"  • {cls_name}: {c['confidence']*100:.2f}%")

    if res.get("comparison"):
        comp = res["comparison"]
        print("\n--- Model Benchmark Comparison ---")
        print(f"Agreement:          {'YES (Both models agree)' if comp['agreement'] else 'NO (Divergent diagnosis)'}")
        print(f"ViT-B/16 (5-class): {comp['vit_prediction']} ({comp['vit_confidence']*100:.2f}%)")
        print(f"ResNet-18 (38-cls): {comp['crop_health_prediction']} ({comp['crop_health_confidence']*100:.2f}%)")

    print(f"\nLocalization:       {res['localization_status']}")
    print(f"Region Box:         {res['bounding_box']}")
    print(f"\nSeverity Level:     {res['severity']} ({res['severity_confidence'] * 100:.2f}%)")
    print("Top-3 Severity Estimates:")
    for s in res["top3_severities"]:
        print(f"  • {s['severity']}: {s['confidence']*100:.2f}%")

    if res["is_low_confidence"]:
        print(f"\n⚠️ WARNING: {res['confidence_warning']}")

    print(f"\nSaved Annotation:   {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

