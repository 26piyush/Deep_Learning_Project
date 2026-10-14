# 🌿 Precision Crop Pathology & Agronomic Decision Support System

An end-to-end deep learning and vision transformer pipeline for leaf localization, foliar disease diagnosis, multi-tier severity grading, and agronomic prescription generation across 42 disease categories and 15 crop species.

---

## 📌 Key Architectural Highlights

1. **Stage 1 — Hierarchical Localization (YOLOv8n):** Isolates leaves and disease lesions from cluttered real-world field backgrounds with smart context padding (+15% margin) to preserve chlorotic halos.
2. **Stage 2 — Multi-Dataset Foliar Diagnosis (ViT-B/16):** Employs multi-head self-attention Vision Transformers fine-tuned across 42 classes spanning PlantVillage, PlantDoc, and DiaMOS Pear datasets.
3. **Stage 3 — Severity Assessment & Grading (ResNet-18 & Morphological Metrics):** Evaluates foliar damage across discrete risk levels (*No Risk*, *Very Low*, *Low*, *Medium*, *High*) aligned with Horsfall-Barratt rating standards, paired with dual-mask pixel necropsy calculations.
4. **Stage 4 — Agronomic Prescriptions & Treatment Engine:** Evidence-based chemical (FRAC coded), biological, and cultural management recommendations with precise dosage calculations based on crop acreage.
5. **Interactive 4-Engine Dashboard:** Real-time Streamlit application supporting ViT-B/16, ResNet-18 Baseline, Smart Cascade Hybrid Mode, and Side-by-Side Architectural Comparison.

---

## 📂 Project Structure

```text
crop_disease_review2/
├── app.py                      # Interactive Streamlit Web Application
├── run_app.bat                 # One-click Windows launch script
├── requirements.txt            # Python environment dependencies
├── architecture.md             # Comprehensive architecture & faculty viva defense guide
├── README.md                   # Repository documentation
│
├── src/                        # Core Python implementation package
│   ├── config.py               # Central project paths, taxonomy, and hyperparameters
│   ├── inference.py            # End-to-end 4-engine inference pipeline
│   ├── train_vit.py            # Stage 2: Vision Transformer training with AMP
│   ├── evaluate_vit.py         # ViT evaluation (accuracy, macro F1, confusion matrix)
│   ├── train_localization.py   # Stage 1: YOLOv8 leaf & lesion localization training
│   ├── evaluate_localization.py# YOLO evaluation (mAP@50, mean IoU)
│   ├── train_severity.py       # Stage 3: Severity classification training
│   ├── evaluate_severity.py    # Severity classifier evaluation & error analysis
│   ├── explainability.py       # ViT Attention Rollout & feature visualization
│   ├── segmentation.py         # Dual-mask leaf & necrosis pixel segmentation
│   ├── prescription.py         # Agronomic treatment & chemical dosage engine
│   ├── knowledge_base.py       # Pathological taxonomy for 42 crop-disease classes
│   ├── dataset_utils.py        # Stratified dataset preparation & split verification
│   ├── preprocessing.py        # Image validation, EXIF normalization & transforms
│   └── utils.py                # Reproducibility seeds, metrics & plotting helpers
│
├── models/                     # Production model checkpoints
│   ├── vit_best.pt             # Unified 42-class ViT-B/16 checkpoint
│   ├── vit_38class_best.pt     # PlantVillage 38-class benchmark ViT checkpoint
│   ├── vit_5class_best.pt      # Solanaceae 5-class subset ViT checkpoint
│   ├── localization_best.pt    # YOLOv8n leaf detector checkpoint
│   ├── severity_best.pt        # 5-class Horsfall-Barratt severity classifier
│   └── archive/                # Archived experimental trial checkpoints
│
├── crop_health_model/          # Standalone 38-class ResNet-18 inference module
│   ├── predictor.py            # CropHealthPredictor inference engine
│   ├── model_state.pt          # ResNet-18 weights
│   ├── model_meta.json         # Label mappings and normalization parameters
│   └── api_server.py           # Standalone Flask microservice
│
├── data/                       # Dataset directories and split manifests
│   ├── PlantVillage-Dataset/   # Raw PlantVillage image dataset
│   ├── plantvillage/           # Processed PlantVillage splits & test preview
│   ├── plant_doc_detection/    # Raw PlantDoc detection dataset (COCO annotations)
│   ├── plantdoc_yolo/          # Converted YOLO localization dataset
│   ├── plantdoc_leaf_crops/    # Isolated leaf crops for field training
│   ├── Pear/                   # DiaMOS Pear leaf dataset
│   ├── severity/               # DiaMOS 5-level severity classification dataset
│   └── unified/                # 42-class multi-dataset split manifest
│
├── notebooks/                  # Interactive experimentation & Colab notebooks
│   └── Severity_Training.ipynb # Severity model training notebook
│
├── testing_data/               # Curated 212-image benchmark verification suite
├── results/                    # Experimental figures, confusion matrices & metrics
│   ├── figures/                # Training curves, confusion matrices, Grad-CAM heatmaps
│   ├── metrics/                # Evaluated metrics JSON files
│   └── predictions/            # Visual prediction evidence and validation samples
│
└── archive/                    # Archived early prototypes and legacy scaffolds
    └── legacy_prototype/       # Initial scaffold archive
```

---

## 🚀 Quick Start

### 1. Launch the Interactive Dashboard
Run the provided Windows launcher batch script:
```cmd
run_app.bat
```
Or run directly through your activated Python virtual environment:
```bash
streamlit run app.py
```

### 2. Run Model Evaluation
Evaluate the Vision Transformer model:
```bash
python -m src.evaluate_vit
```

Evaluate the YOLOv8 localization model:
```bash
python -m src.evaluate_localization
```

Evaluate the Severity Assessment model:
```bash
python -m src.evaluate_severity
```

---

## 📊 Faculty Viva & Defense Guide
For exhaustive details regarding:
- Mathematical formulations (Multi-Head Self-Attention, Horsfall-Barratt scale, mAP@50)
- Verified benchmark scores across all datasets
- High-yield Viva Voce defense questions and answers

Refer to the in-depth documentation in [architecture.md](architecture.md).
