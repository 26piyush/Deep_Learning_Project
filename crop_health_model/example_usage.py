"""
Example: How to use the Crop Health Model in your project.
"""

from pathlib import Path
from PIL import Image
import numpy as np
try:
    from crop_health_model.predictor import CropHealthPredictor
except (ImportError, ValueError):
    from predictor import CropHealthPredictor

def main():
    print("1. Initializing CropHealthPredictor...")
    predictor = CropHealthPredictor()
    
    print(f"-> Model device: {predictor.device}")
    print(f"-> Total classes supported: {len(predictor.get_classes())}")
    print(f"-> Supported crops: {', '.join(predictor.get_supported_crops()[:6])}...")

    # Create a synthetic sample leaf image (greenish RGB test image) for demonstration
    sample_img = Image.new("RGB", (256, 256), color=(40, 160, 60))
    sample_path = Path("sample_test_leaf.jpg")
    sample_img.save(sample_path)
    print(f"\n2. Saved test image to '{sample_path}'")

    # Example A: Standard inference (over all 38 classes)
    print("\n--- Example A: Predict over all crops ---")
    result_a = predictor.predict(sample_path, top_k=3)
    top_pred = result_a["prediction"]
    print(f"Top prediction: {top_pred['class']} (Confidence: {top_pred['confidence']*100:.1f}%)")
    print(f"Crop: {top_pred['crop']}, Disease: {top_pred['disease']}")
    print("Top 3 ranked:")
    for i, item in enumerate(result_a["top_predictions"], 1):
        print(f"  {i}. {item['class']} -> {item['confidence']*100:.2f}%")

    # Example B: Prioritize / Filter by crop (e.g. Tomato)
    print("\n--- Example B: Predict with crop filter ('Tomato') ---")
    result_b = predictor.predict(sample_path, crop_name="Tomato", top_k=3)
    print(f"Filtered crop: {result_b['filtered_to_crop']}")
    for i, item in enumerate(result_b["top_predictions"], 1):
        print(f"  {i}. {item['disease']} -> {item['confidence']*100:.2f}%")

    # Example C: Predict directly from bytes or PIL image or numpy array
    print("\n--- Example C: Predict from in-memory PIL image ---")
    result_c = predictor.predict(sample_img, crop_name="Apple", top_k=2)
    print(f"Top disease: {result_c['prediction']['disease']} ({result_c['prediction']['confidence']*100:.1f}%)")

    # Clean up test image
    if sample_path.exists():
        sample_path.unlink()
    print("\nSuccess! Crop health model is ready for use in any project.")

if __name__ == "__main__":
    main()
