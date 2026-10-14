"""Vision Transformer Explainability & Hierarchical Symptom Localization Engine.
Final Review (100% Phase) Deliverables:
  1. ViT-B/16 Multi-Head Self-Attention Rollout (Exact QKV Dot-Product Calculation)
  2. Attention Overlay Heatmap (Token Attribution)
  3. Grounded Disease Symptom Bounding Boxes (Attention-Gated + Shadow-Filtered)
  4. Hierarchical Two-Tier Annotations (Outer Leaf Box + Inner Disease Spot Boxes)
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


def compute_vit_attention(
    vit_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Extract true attention map from the Vision Transformer's final encoder self-attention block
    by computing the exact scaled dot-product attention matrix across the 12 attention heads.
    
    Args:
        vit_model: Vision Transformer model (ViT-B/16)
        input_tensor: Preprocessed tensor of shape (1, 3, 224, 224)
        device: Active torch device
        
    Returns:
        14x14 numpy array of normalized spatial self-attention scores.
    """
    vit_model.eval()
    saved_inputs: List[torch.Tensor] = []

    def hook_fn(module, inputs, outputs):
        if len(inputs) > 0 and inputs[0] is not None:
            saved_inputs.append(inputs[0].detach())

    hook_handle = None
    target_layer = None
    try:
        if hasattr(vit_model, "encoder") and hasattr(vit_model.encoder, "layers"):
            target_layer = vit_model.encoder.layers[-1].self_attention
            hook_handle = target_layer.register_forward_hook(hook_fn)
    except Exception:
        hook_handle = None

    with torch.no_grad():
        _ = vit_model(input_tensor.to(device))

    if hook_handle is not None:
        hook_handle.remove()

    if len(saved_inputs) > 0 and target_layer is not None:
        x = saved_inputs[0]  # shape (1, 197, 768)
        B, N, C = x.shape
        num_heads = target_layer.num_heads
        head_dim = C // num_heads

        with torch.no_grad():
            qkv = F.linear(x, target_layer.in_proj_weight, target_layer.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1).mean(dim=1)
            cls_attn = attn[0, 0, 1:].detach().cpu().numpy()
            if cls_attn.shape[0] == 196:
                spatial_map = cls_attn.reshape(14, 14)
            else:
                grid_dim = int(np.sqrt(cls_attn.shape[0]))
                spatial_map = cls_attn[: grid_dim * grid_dim].reshape(grid_dim, grid_dim)
    else:
        spatial_map = np.ones((14, 14), dtype=np.float32) / 196.0

    # Min-max normalization
    s_min, s_max = spatial_map.min(), spatial_map.max()
    if s_max > s_min:
        spatial_map = (spatial_map - s_min) / (s_max - s_min)
    else:
        spatial_map = np.zeros_like(spatial_map)

    return spatial_map.astype(np.float32)


def generate_attention_heatmap(
    attention_map: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Upsample normalized 14x14 attention array into full-resolution RGB heatmap."""
    w, h = target_size
    resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)
    u8_map = (resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8_map, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.50,
) -> Image.Image:
    """Alpha-blend colorized attention heatmap over the foliar image."""
    img_np = np.array(image.convert("RGB"))
    if heatmap_rgb.shape[:2] != img_np.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (img_np.shape[1], img_np.shape[0]))
    blended = cv2.addWeighted(img_np, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended)


def extract_disease_bounding_boxes(
    image: Image.Image,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    attention_map: Optional[np.ndarray] = None,
    is_healthy: bool = False,
    disease_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Derive discrete disease symptom bounding boxes tightly encompassing real disease symptoms.
    Uses attention gating to ensure boxes are only placed where the model actively lights up,
    preventing spurious detections on unlit areas, shadows, or camera border artifacts.
    
    Args:
        image: PIL foliar image (cropped region)
        leaf_mask: Binary mask of leaf surface (L_mask)
        lesion_mask: Binary mask of segmented lesion tissue (D_mask)
        attention_map: Optional 14x14 ViT attention array for attribution scoring
        is_healthy: If True, skips lesion extraction
        disease_name: Diagnosed disease name (e.g. 'Early Blight', 'Black Rot')
        
    Returns:
        List of dicts: {"box": [x1, y1, x2, y2], "confidence": float, "area": int, "label": str}
    """
    if is_healthy:
        return []

    w, h = image.size
    lesion_u8 = (lesion_mask > 0).astype(np.uint8) * 255
    if np.sum(lesion_u8 > 0) < 20:
        return []

    # Prepare attention lookup if available
    attn_resized = None
    mean_attn = 0.05
    if attention_map is not None:
        attn_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
        attn_resized = np.clip(attn_resized, 0.0, 1.0)
        mean_attn = max(0.01, float(np.mean(attn_resized)))

    contours, _ = cv2.findContours(lesion_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20, int(0.0003 * (w * h)))
    max_w = int(0.60 * w)
    max_h = int(0.60 * h)

    candidate_boxes: List[Dict[str, Any]] = []
    box_label = disease_name if disease_name and disease_name.lower() != "unknown" else "Disease Spot"

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # For large confluent blights (covering > 60% of ROI dimension),
        # use distance transform to separate into discrete lesion foci
        if bw > max_w or bh > max_h:
            sub_mask = np.zeros_like(lesion_u8)
            cv2.drawContours(sub_mask, [cnt], -1, 255, -1)
            dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
            dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
            peaks = (dist_norm > 0.45).astype(np.uint8) * 255
            sub_cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for sc in sub_cnts:
                if cv2.contourArea(sc) < 15:
                    continue
                sbx, sby, sbw, sbh = cv2.boundingRect(sc)
                pad_w = max(8, int(sbw * 1.1))
                pad_h = max(8, int(sbh * 1.1))
                px1 = max(0, sbx - pad_w)
                py1 = max(0, sby - pad_h)
                px2 = min(w, sbx + sbw + pad_w)
                py2 = min(h, sby + sbh + pad_h)

                sub_roi = lesion_u8[py1:py2, px1:px2]
                density = float(np.mean(sub_roi > 0)) if sub_roi.size > 0 else 0.5
                attn_score = float(np.mean(attn_resized[py1:py2, px1:px2])) if attn_resized is not None else 0.5

                # ATTENTION GATING: Must be an area with active attention (not unlit shadow)
                if attn_score < 0.35 * mean_attn:
                    continue

                conf = round(float(np.clip(0.60 + 0.25 * density + 0.15 * min(1.0, attn_score / max(1e-4, mean_attn)), 0.68, 0.98)), 2)

                candidate_boxes.append({
                    "box": [px1, py1, px2, py2],
                    "confidence": conf,
                    "area": int((px2 - px1) * (py2 - py1)),
                    "density": round(density, 2),
                    "attn": round(attn_score, 3),
                    "label": box_label,
                })
            continue

        # Standard discrete lesion contour
        box_roi = lesion_u8[by : by + bh, bx : bx + bw]
        density = float(np.mean(box_roi > 0)) if box_roi.size > 0 else 0.0
        if density < 0.20:
            continue

        attn_score = float(np.mean(attn_resized[by : by + bh, bx : bx + bw])) if attn_resized is not None else 0.5

        # ATTENTION GATING: Discard regions that are cold/dark in attention map (unlit shadows, noise)
        if attn_score < 0.35 * mean_attn:
            continue

        # Add 3px context padding around lesion spot
        pad = 3
        px1 = max(0, bx - pad)
        py1 = max(0, by - pad)
        px2 = min(w, bx + bw + pad)
        py2 = min(h, by + bh + pad)

        conf = round(float(np.clip(0.60 + 0.25 * density + 0.15 * min(1.0, attn_score / max(1e-4, mean_attn)), 0.68, 0.98)), 2)

        candidate_boxes.append({
            "box": [px1, py1, px2, py2],
            "confidence": conf,
            "area": int(area),
            "density": round(density, 2),
            "attn": round(attn_score, 3),
            "label": box_label,
        })

    # Non-maximum suppression / overlap merging for highly overlapping lesion boxes
    final_boxes: List[Dict[str, Any]] = []
    # Rank by hybrid metric of area, density, and attention illumination
    candidate_boxes.sort(key=lambda x: (x["area"] * x.get("density", 1.0) * max(0.2, x.get("attn", 0.5))), reverse=True)

    for item in candidate_boxes:
        bx1, by1, bx2, by2 = item["box"]
        overlap = False
        for kept in final_boxes:
            kx1, ky1, kx2, ky2 = kept["box"]
            # Intersection over Union
            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                min_box_area = min((bx2 - bx1) * (by2 - by1), (kx2 - kx1) * (ky2 - ky1))
                if (inter_area / max(1, min_box_area)) > 0.65:
                    overlap = True
                    break
        if not overlap:
            final_boxes.append(item)

    # Return top 6 prominent discrete lesion boxes
    return final_boxes[:6]


def draw_hierarchical_annotations(
    image: Image.Image,
    leaf_box: List[int],
    disease_boxes: List[Dict[str, Any]],
    is_healthy: bool = False,
    disease_name: Optional[str] = None,
) -> Image.Image:
    """Draw dual-tier annotations:
      - Tier 1: Outer Neon Green Box (Leaf Boundary localized by YOLOv8)
      - Tier 2: Inner Crimson Red Boxes (Disease spot localization tagged with disease name & confidence)
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    try:
        font_size = max(11, int(min(w, h) * 0.034))
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    # Tier 1: Outer Leaf Boundary (Neon Green)
    if leaf_box and leaf_box != [0, 0, w, h]:
        lx1 = max(0, min(w - 2, int(leaf_box[0])))
        ly1 = max(0, min(h - 2, int(leaf_box[1])))
        lx2 = max(lx1 + 1, min(w - 1, int(leaf_box[2])))
        ly2 = max(ly1 + 1, min(h - 1, int(leaf_box[3])))
        draw.rectangle([lx1, ly1, lx2, ly2], outline="#00E676", width=3)
        l_tag = "🌿 Leaf Boundary"
        try:
            bbox = draw.textbbox((lx1, ly1), l_tag, font=font)
            tw, th = bbox[2] - bbox[0] + 6, bbox[3] - bbox[1] + 4
        except Exception:
            tw, th = 90, 18
        lt_y = max(0, ly1 - th)
        rx1 = min(w, lx1 + tw)
        ry1 = min(h, lt_y + th)
        if rx1 > lx1 and ry1 > lt_y:
            draw.rectangle([lx1, lt_y, rx1, ry1], fill="#00E676")
            draw.text((lx1 + 3, lt_y + 2), l_tag, fill="#003300", font=font)

    # Tier 2: Inner Disease Spots (Crimson Red)
    if is_healthy or len(disease_boxes) == 0:
        if is_healthy:
            h_tag = "✅ Healthy Foliage (No Disease Detected)"
            draw.rectangle([8, 8, min(w - 8, 270), 32], fill="#1B5E20")
            draw.text((12, 12), h_tag, fill="#FFFFFF", font=font)
        return annotated

    # Prepare tag label
    raw_label = disease_name if disease_name and disease_name.lower() != "unknown" else "Disease"
    # Format a concise badge label, e.g. "Early Blight", "Black Rot"
    short_label = raw_label if len(raw_label) <= 15 else (raw_label[:13] + "..")

    for i, d_item in enumerate(disease_boxes):
        dx1 = max(0, min(w - 2, int(d_item["box"][0])))
        dy1 = max(0, min(h - 2, int(d_item["box"][1])))
        dx2 = max(dx1 + 1, min(w - 1, int(d_item["box"][2])))
        dy2 = max(dy1 + 1, min(h - 1, int(d_item["box"][3])))
        conf = d_item["confidence"]

        # Draw red bounding box
        draw.rectangle([dx1, dy1, dx2, dy2], outline="#FF1744", width=2)
        d_tag = f"🎯 {short_label}: {conf*100:.0f}%"

        try:
            bbox = draw.textbbox((dx1, dy1), d_tag, font=font)
            tw, th = bbox[2] - bbox[0] + 6, bbox[3] - bbox[1] + 4
        except Exception:
            tw, th = 95, 16

        dt_y = max(0, dy1 - th)
        bx1 = min(w, dx1 + tw)
        by1 = min(h, dt_y + th)
        if bx1 > dx1 and by1 > dt_y:
            draw.rectangle([dx1, dt_y, bx1, by1], fill="#FF1744")
            draw.text((dx1 + 3, dt_y + 2), d_tag, fill="#FFFFFF", font=font)

    return annotated
