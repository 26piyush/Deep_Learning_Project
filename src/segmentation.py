"""Final Phase — Pixel-Level Dual-Mask Segmentation & Mathematical Severity Engine.
Isolates exact Leaf Contour Mask (L_mask) and Lesion Mask (D_mask)
to calculate continuous infection severity % = (D_mask / L_mask) * 100.
Includes strict shadow rejection to avoid false positives on unlit/shaded leaf folds.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def extract_leaf_mask(image: Image.Image) -> np.ndarray:
    """Segment leaf tissue from neutral background, lab boards, or ambient field noise.
    
    Returns:
        Binary mask (uint8) with 1 for leaf tissue and 0 for background.
    """
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    r = img_np[:, :, 0].astype(np.float32)
    g = img_np[:, :, 1].astype(np.float32)
    b = img_np[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b

    # Standard non-leaf background signatures (PlantVillage lab boards and ambient noise):
    # 1. Blue / purple mounting boards (Hue 100-160, Sat > 18)
    is_blue_purple_bg = (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 160) & (hsv[:, :, 1] > 18)
    # 2. Neutral gray boards (low saturation < 16, intermediate brightness)
    is_neutral_gray_bg = (hsv[:, :, 1] < 16) & (hsv[:, :, 2] > 38) & (hsv[:, :, 2] < 245)
    # 3. Pure black / dark shadows
    is_black_bg = (hsv[:, :, 2] < 24)
    # 4. Pure white / specular overexposure
    is_white_bg = (hsv[:, :, 2] >= 246) & (hsv[:, :, 1] < 18)

    is_bg = is_blue_purple_bg | is_neutral_gray_bg | is_black_bg | is_white_bg

    # Initial candidate is non-background pixels OR pixels with clear foliage green (exg > 5)
    leaf_candidate = ((~is_bg) | (exg > 5.0)).astype(np.uint8) * 255

    # Morphological closing to fill leaf interior holes and opening to prune background speckles
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_candidate = cv2.morphologyEx(leaf_candidate, cv2.MORPH_CLOSE, kernel_close)
    leaf_candidate = cv2.morphologyEx(leaf_candidate, cv2.MORPH_OPEN, kernel_open)

    # Keep connected components representing the leaf (area >= 3% of total image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_candidate)
    leaf_mask = np.zeros_like(leaf_candidate)
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 0.03 * (w * h):
                leaf_mask[labels == i] = 255

    # If segmentation rejected everything (edge cases), fallback to entire image frame
    if np.sum(leaf_mask > 0) < 0.05 * (w * h):
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255

    return (leaf_mask > 0).astype(np.uint8)


def extract_lesion_mask(
    image: Image.Image,
    leaf_mask: np.ndarray,
    attention_map: Optional[np.ndarray] = None,
    is_healthy: bool = False,
) -> np.ndarray:
    """Identify necrotic, chlorotic, and fungal lesion pixels strictly inside the leaf boundary.
    Rejects shaded green foliage, leaf folds, and poorly lit healthy tissue.
    
    Returns:
        Binary mask (uint8) with 1 for active lesion pixels, 0 elsewhere.
    """
    if is_healthy:
        return np.zeros_like(leaf_mask, dtype=np.uint8)

    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    is_leaf = leaf_mask > 0
    if np.sum(is_leaf) < 40:
        return np.zeros_like(leaf_mask, dtype=np.uint8)

    # Erode leaf boundary slightly (3 pixels) to discard perimeter shadows and frame edge transitions
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_leaf = cv2.erode(leaf_mask.astype(np.uint8) * 255, k_erode) > 0

    # Foliage spectral analysis
    r = img_np[:, :, 0].astype(np.float32)
    g = img_np[:, :, 1].astype(np.float32)
    b = img_np[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b

    # Reference median color of healthy foliar parenchyma
    green_ref = inner_leaf & (exg > 14.0) & (hsv[:, :, 0] >= 34) & (hsv[:, :, 0] <= 86)
    if np.sum(green_ref) > 80:
        med_l = float(np.median(lab[green_ref, 0]))
        med_a = float(np.median(lab[green_ref, 1]))
        med_b = float(np.median(lab[green_ref, 2]))
    else:
        med_l = float(np.median(lab[inner_leaf, 0]))
        med_a = float(np.median(lab[inner_leaf, 1]))
        med_b = float(np.median(lab[inner_leaf, 2]))

    # Delta E in LAB color space from median healthy foliar tone
    delta_e = np.sqrt(
        (lab[:, :, 0].astype(np.float32) - med_l) ** 2
        + (lab[:, :, 1].astype(np.float32) - med_a) ** 2
        + (lab[:, :, 2].astype(np.float32) - med_b) ** 2
    )

    # 1. SHADOW REJECTION FILTER:
    # Shaded healthy green leaf tissue maintains green dominance over red (G > R) and positive ExG.
    # Shadows must NEVER be classified as lesions!
    is_healthy_shade = (g > r + 6.0) & (exg > 4.0) & (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 90)

    # 2. PATHOLOGICAL SYMPTOMS (Requiring genuine loss of green):
    # Necrotic spots / blights / rots (brown, rust, dark orange: Hue <= 26 or Hue >= 165, with R matching or exceeding G)
    is_necrotic_brown = ((hsv[:, :, 0] <= 26) | (hsv[:, :, 0] >= 165)) & (r >= g - 6.0) & (hsv[:, :, 1] > 20)
    # Chlorotic halos / yellowing (Hue between 24 and 34, Sat > 30, and reduced green index)
    is_chlorosis = (hsv[:, :, 0] > 24) & (hsv[:, :, 0] <= 34) & (hsv[:, :, 1] > 30) & (exg < 10.0)
    # Dark fungal rot / anthracnose (Must have lost green color — not just a low-light shadow!)
    is_dark_rot = ((hsv[:, :, 2] < 72) | (lab[:, :, 0] < 70)) & (r >= g - 4.0) & (exg < 2.0)
    # Significant color divergence with low green
    is_divergence = (delta_e > 26.0) & (exg < 5.0) & (r >= g - 8.0)

    # Active lesion pixels must be strictly on inner leaf tissue and NOT healthy shade
    lesion_cand = inner_leaf & (is_necrotic_brown | is_chlorosis | is_dark_rot | is_divergence) & (~is_healthy_shade)

    # Morphological filtering to discard isolated pixel noise and connect lesion bodies
    lesion_u8 = (lesion_cand.astype(np.uint8)) * 255
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_u8 = cv2.morphologyEx(lesion_u8, cv2.MORPH_OPEN, k_clean)
    k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_u8 = cv2.morphologyEx(lesion_u8, cv2.MORPH_CLOSE, k_merge)

    return (lesion_u8 > 0).astype(np.uint8)


def calculate_mathematical_severity(
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    is_healthy: bool = False,
) -> Dict[str, Any]:
    """Compute continuous infection severity % = (Lesion Pixels / Leaf Pixels) * 100
    and map to standard Horsfall-Barratt agronomic rating tier.
    """
    if is_healthy:
        return {
            "severity_percentage": 0.0,
            "agronomic_tier": "No Risk (Healthy)",
            "leaf_pixels": int(np.sum(leaf_mask > 0)),
            "lesion_pixels": 0,
            "healthy_pixels": int(np.sum(leaf_mask > 0)),
            "necrosis_ratio": 0.0,
        }

    leaf_pixels = int(np.sum(leaf_mask > 0))
    lesion_pixels = int(np.sum((lesion_mask > 0) & (leaf_mask > 0)))

    if leaf_pixels <= 0:
        return {
            "severity_percentage": 0.0,
            "agronomic_tier": "No Risk",
            "leaf_pixels": 0,
            "lesion_pixels": 0,
            "healthy_pixels": 0,
            "necrosis_ratio": 0.0,
        }

    severity_ratio = float(lesion_pixels / leaf_pixels)
    severity_pct = round(severity_ratio * 100.0, 2)

    # Agronomic Horsfall-Barratt scale mapping
    if severity_pct < 0.5:
        tier = "No Risk / Latent (<0.5%)"
    elif severity_pct < 5.0:
        tier = "Very Low Risk (<5%)"
    elif severity_pct < 15.0:
        tier = "Low Risk (5%–15%)"
    elif severity_pct < 35.0:
        tier = "Medium Risk (15%–35%)"
    else:
        tier = "High Risk / Critical (>35%)"

    return {
        "severity_percentage": severity_pct,
        "agronomic_tier": tier,
        "leaf_pixels": leaf_pixels,
        "lesion_pixels": lesion_pixels,
        "healthy_pixels": max(0, leaf_pixels - lesion_pixels),
        "necrosis_ratio": round(severity_ratio, 4),
    }


def visualize_dual_masks(
    image: Image.Image,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
) -> Image.Image:
    """Generate high-contrast 3-zone color-coded foliar segmentation map:
      - Emerald Green: Healthy Folium Surface (L_mask \ D_mask)
      - Crimson Red: Necrotic/Chlorotic Lesions (D_mask)
      - Dimmed: Background / Non-foliar space
    """
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    h, w, _ = img_np.shape

    is_leaf = leaf_mask > 0
    is_lesion = (lesion_mask > 0) & is_leaf
    is_healthy_leaf = is_leaf & (~is_lesion)
    is_bg = ~is_leaf

    out = img_np.copy()

    # Dim background
    out[is_bg] = out[is_bg] * 0.35

    # Highlight healthy folium in gentle emerald green tint
    out[is_healthy_leaf, 0] = out[is_healthy_leaf, 0] * 0.65 + 0.0 * 0.35
    out[is_healthy_leaf, 1] = out[is_healthy_leaf, 1] * 0.65 + 230.0 * 0.35
    out[is_healthy_leaf, 2] = out[is_healthy_leaf, 2] * 0.65 + 80.0 * 0.35

    # Highlight lesions in vivid crimson/amber
    out[is_lesion, 0] = out[is_lesion, 0] * 0.40 + 255.0 * 0.60
    out[is_lesion, 1] = out[is_lesion, 1] * 0.40 + 23.0 * 0.60
    out[is_lesion, 2] = out[is_lesion, 2] * 0.40 + 68.0 * 0.60

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
