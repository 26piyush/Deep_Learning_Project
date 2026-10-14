"""
Crop Health Model Predictor
----------------------------
Self-contained PyTorch inference class for crop disease detection.
Uses ResNet18 trained on 38 plant disease classes across 14 crops.
"""

from __future__ import annotations
import io
import json
from pathlib import Path
from difflib import get_close_matches
from typing import Union, List, Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Default relative paths
_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _DIR / "model_state.pt"
DEFAULT_META = _DIR / "model_meta.json"


class CropHealthPredictor:
    def __init__(
        self,
        weights_path: Union[str, Path] = DEFAULT_WEIGHTS,
        meta_path: Union[str, Path] = DEFAULT_META,
        device: Optional[str] = None,
    ):
        """
        Initialize the crop health model predictor.
        
        Args:
            weights_path: Path to PyTorch model weights (.pt)
            meta_path: Path to metadata JSON file (.json)
            device: 'cuda', 'cpu', 'mps', or None (auto-detects)
        """
        self.weights_path = Path(weights_path)
        self.meta_path = Path(meta_path)
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Determine computation device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load metadata
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)

        self.idx_to_label: List[str] = self.meta.get("idx_to_label", [])
        self.idx_to_pair: List[Dict[str, str]] = self.meta.get("idx_to_pair", [])
        self.crop_to_indices: Dict[str, List[int]] = self.meta.get("crop_to_indices", {})
        self.crops: List[str] = self.meta.get("crops", sorted(list(self.crop_to_indices.keys())))

        # Build transforms
        img_size = self.meta.get("img_size", 256)
        mean = self.meta.get("mean", [0.485, 0.456, 0.406])
        std = self.meta.get("std", [0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Build ResNet-18 model architecture
        num_classes = len(self.idx_to_label)
        self.model = models.resnet18(weights=None)
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_feats, num_classes)
        )

        # Load trained weights
        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

    def get_supported_crops(self) -> List[str]:
        """Returns the list of supported crop categories."""
        return list(self.crops)

    def get_classes(self) -> List[str]:
        """Returns the full list of 38 disease class labels."""
        return list(self.idx_to_label)

    def fuzzy_match_crop(self, crop_query: str) -> Optional[str]:
        """Match input crop string (case-insensitive) to recognized crops."""
        if not crop_query:
            return None
        q = crop_query.strip().lower()
        for c in self.crops:
            if c.lower() == q:
                return c
        # Fuzzy match
        lower_crops = {c.lower(): c for c in self.crops}
        matches = get_close_matches(q, list(lower_crops.keys()), n=1, cutoff=0.6)
        if matches:
            return lower_crops[matches[0]]
        return None

    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """Convert diverse image inputs into a RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        crop_name: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run inference on an image and return predictions.

        Args:
            image_input: Path, bytes, PIL Image, or numpy array.
            crop_name: Optional crop filter (e.g., 'Tomato', 'Apple').
                       If provided, probabilities are renormalized over that crop's classes.
            top_k: Number of top results to return.

        Returns:
            Dictionary with primary prediction and top_k ranked results.
        """
        pil_img = self._load_image(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        resolved_crop = self.fuzzy_match_crop(crop_name) if crop_name else None

        if resolved_crop and resolved_crop in self.crop_to_indices:
            idxs = self.crop_to_indices[resolved_crop]
            subset_probs = probs[idxs]
            subset_probs = subset_probs / (subset_probs.sum() + 1e-12)
            top_indices_local = subset_probs.argsort()[::-1][:top_k]

            top_predictions = []
            for local_idx in top_indices_local:
                cls_idx = idxs[int(local_idx)]
                p = float(subset_probs[int(local_idx)])
                pair = self.idx_to_pair[cls_idx]
                top_predictions.append({
                    "class": self.idx_to_label[cls_idx],
                    "crop": pair["crop"],
                    "disease": pair["disease"],
                    "confidence": round(p, 4),
                })
            filtered_crop = resolved_crop
        else:
            top_indices = probs.argsort()[::-1][:top_k]
            top_predictions = []
            for cls_idx in top_indices:
                cls_idx = int(cls_idx)
                p = float(probs[cls_idx])
                pair = self.idx_to_pair[cls_idx]
                top_predictions.append({
                    "class": self.idx_to_label[cls_idx],
                    "crop": pair["crop"],
                    "disease": pair["disease"],
                    "confidence": round(p, 4),
                })
            filtered_crop = None

        best = top_predictions[0] if top_predictions else None

        return {
            "prediction": best,
            "top_predictions": top_predictions,
            "filtered_to_crop": filtered_crop,
            "device_used": str(self.device),
        }
