# Intelligent Crop Disease Detection, Localization & Severity Grading

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

A hierarchical deep learning pipeline integrating **YOLOv8n** leaf localization, **Vision Transformer (ViT-B/16)** disease classification, and **ResNet-18** foliar severity grading across **42 disease classes and 15 crop species** (68,131 images).

---

## Key Features

* **3-Stage Hierarchical Pipeline:**
  1. **Leaf Localization (YOLOv8n):** Isolates foliar regions, eliminating field background noise (soil, shadows, weeds).
  2. **Smart Context Cropping:** Expands bounding boxes with **+15% margin padding** to preserve transitional lesion halos, with automatic full-frame fallback.
  3. **Disease Classification (ViT-B/16):** Uses global multi-head self-attention across 196 image patches to diagnose 42 conditions across 15 crops.
  4. **Severity Assessment (ResNet-18):** Categorizes foliar damage into 5 discrete risk tiers (`No Risk` to `High`).
* **Multi-Engine Inference:** Supports ViT standalone, ResNet-18 baseline, confidence-based hybrid cascade ($<65\%$ confidence routes to ResNet), and side-by-side consensus.
* **Interactive UI:** Streamlit dashboard with real-time leaf diagnosis, top-3 confidences, and acreage dosage calculator.

---

## System Architecture

```text
[Input Field Image]
        │
        ▼
[Preprocess & Verify RGB]
        │
        ▼
[Stage 1: YOLOv8n Localization] ──► Found box? ──► Smart Crop (+15% Padding)
        │                                  └──► No box?    ──► Controlled Fallback (Full Frame)
        ▼
[Stage 2: ViT-B/16 Diagnosis] ────► 42 Disease Classes (87.44% Test Accuracy)
        │
        ▼
[Stage 3: ResNet-18 Severity] ───► 5 Severity Tiers (No Risk, Very Low, Low, Med, High)
        │
        ▼
[Interactive Dashboard Output] ──► Top-3 Probabilities, Localization Box & Treatment Action
```

---

## Experimental Benchmark Results

Evaluated on held-out test splits on a single NVIDIA GeForce RTX 4050 Laptop GPU:

| Pipeline Component | Architecture | Test Partition | Primary Metric | Score |
| :--- | :--- | :--- | :--- | :---: |
| **Unified Disease Classifier** | ViT-B/16 (ImageNet-1K) | Unified Test (10,259 images) | **Accuracy / Macro-F1** | **87.44% / 86.13%** |
| **Pure Lab Benchmark** | ViT-B/16 | PlantVillage Test (8,178 images) | **Accuracy / Macro-F1** | **97.75% / 96.99%** |
| **Solanaceae Subset** | ViT-B/16 | 5-Class Test (977 images) | **Accuracy / Macro-F1** | **99.49% / 99.46%** |
| **Leaf Localization** | YOLOv8n | PlantDoc Test Boxes | **mAP@50 / Mean IoU** | **72.45% / 0.4669** |
| **Severity Grading** | ResNet-18 | DiaMOS Severity Test (248 images) | **Accuracy / Macro-F1** | **66.53% / 70.22%** |

*For full hyperparameter details and confusion matrices, see [PAPER_METRICS_AND_FIGURES_GUIDE.md](PAPER_METRICS_AND_FIGURES_GUIDE.md).*

---

## Supported Crops & Diseases (42 Classes, 15 Crops)

* **Apple:** Scab, Black Rot, Cedar Apple Rust, Healthy
* **Blueberry:** Healthy
* **Cherry:** Powdery Mildew, Healthy
* **Corn:** Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
* **Grape:** Black Rot, Esca (Black Measles), Leaf Blight, Healthy
* **Orange:** Huanglongbing (Citrus Greening)
* **Peach:** Bacterial Spot, Healthy
* **Pear:** Leaf Curl, Healthy, Slug Sawfly, Leaf Spot *(DiaMOS Plant)*
* **Pepper (Bell):** Bacterial Spot, Healthy
* **Potato:** Early Blight, Late Blight, Healthy
* **Raspberry / Soybean / Squash:** Healthy / Powdery Mildew
* **Strawberry:** Leaf Scorch, Healthy
* **Tomato (10 classes):** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, Yellow Leaf Curl, Mosaic Virus, Healthy

---

## Quick Start Guide

### Option 1: One-Click Windows Launch (Recommended)
Double-click `run_app.bat`. It will set up the virtual environment, install dependencies, verify model weights, and start the Streamlit app.

### Option 2: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/26piyush/Deep_Learning_Project.git
cd Deep_Learning_Project

# 2. Set up virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Streamlit web dashboard
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Model Weights & Checkpoints

Download checkpoints from [GitHub Releases (v1.0.0)](https://github.com/26piyush/Deep_Learning_Project/releases/tag/v1.0.0) and place them in their respective folders:

| Checkpoint File | Size | Destination Folder | Link |
| :--- | :---: | :--- | :--- |
| **`vit_best.pt`** | 327.5 MB | `models/vit_best.pt` | [Download](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/vit_best.pt) |
| **`localization_best.pt`** | 5.9 MB | `models/localization_best.pt` | [Download](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/localization_best.pt) |
| **`severity_best.pt`** | 42.7 MB | `models/severity_best.pt` | [Download](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/severity_best.pt) |
| **`model_state.pt`** | 42.8 MB | `crop_health_model/model_state.pt` | [Download](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/model_state.pt) |

---

## CLI Inference & Evaluation

```bash
# Run single-image inference
python -m src.inference --image testing_data/Tomato___Early_blight_01.jpg --classifier vit

# Evaluate ViT on test split (generates metrics & confusion matrix)
python -m src.evaluate_vit

# Evaluate YOLOv8 localization
python -m src.evaluate_localization

# Evaluate severity classifier
python -m src.evaluate_severity
```

---

## Dataset Credits

* **PlantVillage:** 54,299 laboratory images across 38 classes (Mohanty et al.).
* **PlantDoc:** 7,906 field-acquired images across 29 classes with bounding boxes (Singh et al.).
* **DiaMOS Plant:** 5,926 pear orchard images with severity annotations (Fenu & Malloci).
