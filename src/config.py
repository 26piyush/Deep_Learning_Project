"""Configuration for Crop Disease Detection using Vision Transformers.
Review 2 / 70% Implementation Pipeline.
"""

from pathlib import Path
import torch

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Dataset paths
PLANTVILLAGE_RAW = DATA_DIR / "PlantVillage-Dataset" / "raw" / "color"
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage"
PLANTDOC_RAW_DIR = DATA_DIR / "plant_doc_detection"
PLANTDOC_YOLO_DIR = DATA_DIR / "plantdoc_yolo"
SEVERITY_ZIP = DATA_DIR / "Pear.zip"
SEVERITY_DIR = DATA_DIR / "severity"

# Ensure essential output folders exist
for p in [MODELS_DIR, RUNS_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random seed
SEED = 42

# Device selection
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset A: PlantVillage ViT Classification Configuration (All 38 Classes across 14 Crops)
VIT_CLASSES_5 = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
]

PLANTVILLAGE_38_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Raw color directory name mapping to canonical class labels
RAW_TO_CANONICAL_CLASSES = {
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
    "Apple___healthy": "Apple___healthy",
    "Blueberry___healthy": "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry___Powdery_mildew",
    "Cherry_(including_sour)___healthy": "Cherry___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "Corn___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn___Northern_Leaf_Blight",
    "Corn_(maize)___healthy": "Corn___healthy",
    "Grape___Black_rot": "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)": "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy": "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot": "Peach___Bacterial_spot",
    "Peach___healthy": "Peach___healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy": "Pepper,_bell___healthy",
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___healthy",
    "Raspberry___healthy": "Raspberry___healthy",
    "Soybean___healthy": "Soybean___healthy",
    "Squash___Powdery_mildew": "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch": "Strawberry___Leaf_scorch",
    "Strawberry___healthy": "Strawberry___healthy",
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot": "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy": "Tomato___healthy",
}
CANONICAL_TO_RAW_CLASSES = {v: k for k, v in RAW_TO_CANONICAL_CLASSES.items()}

# Dataset C: DiaMOS Pear Classes (15th Crop Category)
PEAR_CLASSES = [
    "Pear___curl",
    "Pear___healthy",
    "Pear___slug",
    "Pear___spot",
]

# Combined 42-Class Taxonomy Across All Datasets (PlantVillage + PlantDoc + DiaMOS Pear)
UNIFIED_42_CLASSES = PLANTVILLAGE_38_CLASSES + PEAR_CLASSES

# PlantDoc category name to canonical class label mapping
PLANTDOC_TO_CANONICAL = {
    "apple_scab_leaf": "Apple___Apple_scab",
    "apple_leaf": "Apple___healthy",
    "apple_rust_leaf": "Apple___Cedar_apple_rust",
    "bell_pepper_leaf": "Pepper,_bell___healthy",
    "bell_pepper_leaf_spot": "Pepper,_bell___Bacterial_spot",
    "blueberry_leaf": "Blueberry___healthy",
    "cherry_leaf": "Cherry___healthy",
    "corn_gray_leaf_spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "corn_leaf_blight": "Corn___Northern_Leaf_Blight",
    "corn_rust_leaf": "Corn___Common_rust",
    "peach_leaf": "Peach___healthy",
    "potato_leaf": "Potato___healthy",
    "potato_leaf_early_blight": "Potato___Early_blight",
    "potato_leaf_late_blight": "Potato___Late_blight",
    "raspberry_leaf": "Raspberry___healthy",
    "soyabean_leaf": "Soybean___healthy",
    "squash_powdery_mildew_leaf": "Squash___Powdery_mildew",
    "strawberry_leaf": "Strawberry___healthy",
    "tomato_early_blight_leaf": "Tomato___Early_blight",
    "tomato_septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "tomato_leaf": "Tomato___healthy",
    "tomato_leaf_bacterial_spot": "Tomato___Bacterial_spot",
    "tomato_leaf_late_blight": "Tomato___Late_blight",
    "tomato_leaf_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "tomato_leaf_yellow_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato_mold_leaf": "Tomato___Leaf_Mold",
    "tomato_two_spotted_spider_mites_leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape_leaf": "Grape___healthy",
    "grape_leaf_black_rot": "Grape___Black_rot",
}

# Unified multi-dataset directories
PLANTDOC_CROPS_DIR = DATA_DIR / "plantdoc_leaf_crops"
PEAR_LEAVES_DIR = DATA_DIR / "Pear" / "leaves"
UNIFIED_DIR = DATA_DIR / "unified"
UNIFIED_SPLITS_PATH = UNIFIED_DIR / "unified_splits.json"

# Primary ViT class list is now all 42 classes across 15 crops
VIT_CLASSES = UNIFIED_42_CLASSES
VIT_IMAGE_SIZE = 224
VIT_BATCH_SIZE = 32
VIT_EPOCHS = 4
VIT_LR = 2e-5
VIT_MODEL_PATH = MODELS_DIR / "vit_best.pt"
VIT_5_MODEL_PATH = MODELS_DIR / "vit_5class_best.pt"
VIT_38_MODEL_PATH = MODELS_DIR / "vit_38class_best.pt"

# Dataset B: PlantDoc Localization Configuration
YOLO_MODEL_NAME = "yolov8n.pt"
LOCALIZATION_CLASSES = {0: "leaf_or_disease_region"}
LOCALIZATION_IMGSZ = 640
LOCALIZATION_EPOCHS = 10
LOCALIZATION_BATCH_SIZE = 8
LOCALIZATION_MODEL_PATH = MODELS_DIR / "localization_best.pt"

# Dataset C: DiaMOS Plant Severity Configuration
SEVERITY_CLASSES = [
    "No Risk",
    "Very Low",
    "Low",
    "Medium",
    "High",
]
SEVERITY_IMAGE_SIZE = 224
SEVERITY_BATCH_SIZE = 32
SEVERITY_EPOCHS = 5
SEVERITY_LR = 1e-4
SEVERITY_MODEL_PATH = MODELS_DIR / "severity_best.pt"

# ImageNet normalization standards
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Backwards compatibility aliases
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = SEED
IMAGE_SIZE = VIT_IMAGE_SIZE
VIT_MODEL_NAME = "vit_b_16"
DISEASE_CLASSES = VIT_CLASSES
NUM_DISEASE_CLASSES = len(VIT_CLASSES)
NUM_SEVERITY_CLASSES = len(SEVERITY_CLASSES)
YOLO_CHECKPOINT_PATH = LOCALIZATION_MODEL_PATH
VIT_CHECKPOINT_PATH = VIT_MODEL_PATH
SEVERITY_CHECKPOINT_PATH = SEVERITY_MODEL_PATH

# Crop Health Model Configuration (38-Class ResNet-18 across 14 crops)
CROP_HEALTH_DIR = BASE_DIR / "crop_health_model"
CROP_HEALTH_WEIGHTS_PATH = CROP_HEALTH_DIR / "model_state.pt"
CROP_HEALTH_META_PATH = CROP_HEALTH_DIR / "model_meta.json"
CROP_HEALTH_SCRIPTED_PATH = CROP_HEALTH_DIR / "crop_health_model_scripted.pt"

