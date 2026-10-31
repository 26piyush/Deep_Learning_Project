# Precision Crop Pathology and Agronomic Decision Support System

> An End-to-End Hierarchical Deep Learning and Vision Transformer Pipeline for Real-World Foliar Disease Localization, Diagnosis, Dual-Track Severity Grading, and Actionable Agronomic Prescriptions across 42 Pathological Classes and 15 Crop Species.

## Table of Contents
1. [Project Overview and Problem Formulation](#1-project-overview-and-problem-formulation)
2. [System Architecture](#2-system-architecture)
3. [Component-by-Component Technical Breakdown](#3-component-by-component-technical-breakdown)
   - [Stage 1: Hierarchical Localization and Smart Context Cropping](#stage-1-hierarchical-localization-and-smart-context-cropping)
   - [Stage 2: Multi-Engine Foliar Diagnosis (ViT-B/16 and ResNet-18)](#stage-2-multi-engine-foliar-diagnosis-vit-b16-and-resnet-18)
   - [Stage 3: Dual-Track Severity Assessment](#stage-3-dual-track-severity-assessment)
   - [Stage 4: Attention Rollout Explainability](#stage-4-attention-rollout-explainability)
   - [Stage 5: Agronomic Prescription and Treatment Engine](#stage-5-agronomic-prescription-and-treatment-engine)
4. [Taxonomy of Supported Crops and Pathological Classes](#4-taxonomy-of-supported-crops-and-pathological-classes)
5. [Multi-Dataset Engineering and Zero Data Leakage Protocols](#5-multi-dataset-engineering-and-zero-data-leakage-protocols)
6. [Empirical Experimental Results and Benchmarks](#6-empirical-experimental-results-and-benchmarks)
7. [Installation and Quick Start Guide](#7-installation-and-quick-start-guide)
8. [Model Weights and Checkpoint Downloads](#8-model-weights-and-checkpoint-downloads)
9. [Command-Line Evaluation and Inference](#9-command-line-evaluation-and-inference)

---

## 1. Project Overview and Problem Formulation

### Limitations of Conventional Deep Learning in Agricultural Vision
Most existing automated plant pathology systems are trained on clean, isolated laboratory specimens and fail when applied to practical agronomic field environments due to four architectural limitations:

1. **Background Overfitting (Shortcut Learning):** Standard Convolutional Neural Networks (CNNs) trained on raw uncropped field imagery routinely correlate predictions with non-foliar artifacts such as soil texture, weed foliage, human hands, irrigation pipes, and shadow angles rather than true fungal or bacterial lesions.
2. **Absence of Region-of-Interest Localization:** End-to-end global classifiers indicate disease presence but provide no bounding coordinates for leaf boundaries or symptom clusters, preventing automated precision sprayers and agricultural robots from acting upon the leaf.
3. **Absence of Severity Quantification:** A discrete class prediction (*e.g., "Tomato Early Blight"*) offers zero information regarding disease progression. Effective chemical and biological interventions strictly depend on **pathological severity** (how much leaf surface is damaged versus healthy canopy).
4. **Laboratory-to-Field Domain Shift:** Models evaluated exclusively on benchmark datasets with uniform artificial backgrounds (*e.g., PlantVillage*) experience massive accuracy degradation when tested against complex, occluded field leaves (*e.g., PlantDoc*).

### Proposed Solution Architecture
To resolve these deficiencies, this repository implements a modular, multi-tier system:
- **Hierarchical Region Extraction:** Employs Ultralytics YOLOv8n to isolate foliar tissue, eliminating over 80% of background noise.
- **Smart Context Boundary Padding:** Expands localized bounding boxes by +15% padding to ensure marginal chlorotic rings and fungal haloes are fully preserved.
- **Global Multi-Head Attention:** Vision Transformer (ViT-B/16, 86.6M parameters) captures global, long-range relationships across distant lesion margins that local convolutional kernels miss.
- **Dual-Track Severity Assessment:** Evaluates infection stage through both an agronomic 5-tier Horsfall-Barratt classifier and a continuous mathematical pixel necropsy metric (Lesion Area / Leaf Area).
- **Explainability and Prescription Synthesis:** Visualizes self-attention rollout token heatmaps and translates diagnosis into FRAC-coded chemical fungicide schedules, biological bio-controls, and cultural sanitation measures.

---

## 2. System Architecture

```text
===================================================================================================
                               INPUT LAYER: FIELD SPECIMEN
===================================================================================================
                                  [ Input Field Image ]
                                            |
                                            v
                           +---------------------------------+
                           |      IMAGE PREPROCESSING        |
                           |  - RGB Mode Enforcement (3-Ch)  |
                           |  - File Byte Integrity Check    |
                           +---------------------------------+
                                            |
===================================================================================================
                               STAGE 1: LOCALIZATION LAYER
===================================================================================================
                                            |
                                            v
                           +---------------------------------+
                           |    Ultralytics YOLOv8n Leaf     |
                           |      and Region Detector        |
                           +---------------------------------+
                                            |
                       +--------------------+--------------------+
              [ Bounding Box(es) Found ]                  [ No Detection ]
                       |                                         |
                       v                                         v
         +---------------------------+             +---------------------------+
         |    SMART CONTEXT CROP     |             |    CONTROLLED FALLBACK    |
         | - Merge overlapping boxes |             | - Full frame center-crop  |
         | - Add +15% margin padding |             | - Set fallback flag = True|
         | - Prevent halo clipping   |             | - Log telemetry advisory  |
         +---------------------------+             +---------------------------+
                       |                                         |
                       +--------------------+--------------------+
                                            | (Extracted Foliar ROI: 224x224)
===================================================================================================
                             STAGE 2: MULTI-ENGINE DIAGNOSTIC LAYER
===================================================================================================
                                            |
         +----------------------------------+----------------------------------+
         |                                                                     |
         v                                                                     v
+---------------------------------+                   +---------------------------------+
|     ENGINE 1: ViT-B/16          |                   |   ENGINE 2: ResNet-18 BASELINE  |
| - 196 Image Patches (16x16)     |                   | - 11.2M Parameters              |
| - 12 Multi-Head Attention Blocks|                   | - 38 PlantVillage Classes       |
| - 86.6M Parameters              |                   | - 14 Crop Species               |
| - 42 Unified Classes (15 Crops) |                   | - Fast Edge Execution           |
+----------------+----------------+                   +----------------+----------------+
                 |                                                     |
                 +--------------------------+--------------------------+
                 |                          |                          |
                 v                          v                          v
      [ Engine 1: ViT Standalone ] [ Engine 3: Smart Cascade ] [ Engine 4: Side-by-Side ]
      Direct ViT Probabilities     ViT if conf >= 65%;         Concurrent execution with
                                   else escalate to ResNet-18  consensus agreement metric
                                            |
===================================================================================================
                          STAGE 3: DUAL-TRACK SEVERITY ASSESSMENT
===================================================================================================
                                            |
                 +--------------------------+--------------------------+
                 |                                                     |
                 v                                                     v
+---------------------------------+                   +---------------------------------+
|     TRACK A: DISCRETE MODEL     |                   |   TRACK B: MATHEMATICAL RATIO   |
| - ResNet-18 Severity Classifier |                   | - Dual-Mask Pixel Segmentation  |
| - 5 Horsfall-Barratt Levels:    |                   | - Leaf Surface Mask (L_mask)    |
|   * No Risk   (0%)              |                   | - Lesion Symptom Mask (D_mask)  |
|   * Very Low  (1% - 5%)         |                   | - Physical Area Calculation:    |
|   * Low       (6% - 15%)        |                   |         Infected Lesion Area    |
|   * Medium    (16% - 35%)       |                   |   % = ------------------------- |
|   * High      (> 35%)           |                   |          Total Leaf Area        |
+----------------+----------------+                   +----------------+----------------+
                 |                                                     |
===================================================================================================
                    STAGE 4 & 5: EXPLAINABILITY AND PRESCRIPTION SYNTHESIS
===================================================================================================
                 |                                                     |
                 +--------------------------+--------------------------+
                 |                          |
                 v                          v
+---------------------------------+  +---------------------------------+
|   STAGE 4: ATTENTION ROLLOUT    |  |   STAGE 5: AGRONOMIC ACTION     |
| - Multi-Head Self-Attention Map |  | - FRAC-Coded Chemical Fungicide |
| - Visual Attention Heatmap      |  | - Dilution Dosage per Liter     |
| - Symptom Bounding Box Overlay  |  | - Biological Bio-Controls       |
| - Transparent Diagnostic Audit  |  | - Cultural Field Sanitation     |
+----------------+----------------+  | - Intervention Spray Timelines  |
                 |                   +----------------+----------------+
                 |                                    |
===================================================================================================
                               OUTPUT PRESENTATION LAYER
===================================================================================================
                 +------------------+-----------------+
                                    |
                                    v
                 +-------------------------------------+
                 |  INTERACTIVE STREAMLIT DASHBOARD    |
                 |  - Real-Time Foliar Inspection      |
                 |  - 15-Crop Diagnostic Filter        |
                 |  - Top-3 Probability Distributions  |
                 |  - Side-by-Side Architectural Delta |
                 |  - Acreage Dosage Calculator        |
                 +-------------------------------------+
```

---

## 3. Component-by-Component Technical Breakdown

### Stage 1: Hierarchical Localization and Smart Context Cropping
* **Model:** Ultralytics YOLOv8n (nano variant, ~3.2M parameters) trained on field bounding box labels.
* **Smart Context Padding:** Standard detectors draw tight bounding boxes that clip off the outer perimeter of fungal halos and yellow margins. The pipeline automatically expands detected boxes outward by **15% on all four sides** to keep the entire transitional lesion intact.
* **Controlled Fallback Mechanism:** If an image is heavily occluded or poorly lit and the detector finds zero leaves, the system does not crash. It automatically falls back to analyzing the center of the image, tags `used_fallback = True`, and displays an explanatory note in the output log.

---

### Stage 2: Multi-Engine Foliar Diagnosis (ViT-B/16 and ResNet-18)

#### How the Vision Transformer Works (in Plain English)
Instead of looking at small groups of pixels at a time like traditional CNNs, the Vision Transformer processes the leaf like words in a sentence:

1. **Breaking the Leaf into Patches:** The 224x224 leaf image is sliced into a 14x14 grid of 196 small square patches (each 16x16 pixels).
2. **Global Communication (Self-Attention):** Every single patch compares itself to all other 195 patches simultaneously across 12 different inspection heads.
3. **Connecting Distant Symptoms:** If a patch near the leaf tip contains a brown lesion, the model immediately checks distant patches near the stem or margins for matching yellow halos or leaf discoloration to make a confident diagnosis.
4. **Final Decision:** The combined information from all patches flows into a classification head that calculates confidence scores across all 42 disease classes.

#### The 4 Diagnostic Engine Modes
1. **Engine 1 - Vision Transformer (ViT-B/16 Unified):** The primary diagnostic model (86.6M parameters) fine-tuned across all lab and field datasets (42 classes, 15 crops).
2. **Engine 2 - Crop Health ResNet-18 Baseline:** An 11.2M parameter residual network covering 38 PlantVillage classes, providing a fast baseline for speed and comparison.
3. **Engine 3 - Smart Hybrid Cascade:** Runs the Vision Transformer first. If its confidence is 65% or higher, the diagnosis is accepted. If confidence falls below 65%, the system automatically calls the 38-class ResNet-18 model as a second opinion.
4. **Engine 4 - Side-by-Side Dual Engine Comparison:** Runs both models at the same time on the exact same leaf, displaying both predictions, confidence scores, and whether the two models agree.

---

### Stage 3: Dual-Track Severity Assessment
To give practical value to farmers and agronomists, the system calculates disease severity using two separate methods:

* **Track A - Discrete Severity Classifier (ResNet-18):**
  A dedicated network trained on the DiaMOS Plant dataset that categorizes damage into 5 practical risk levels based on the standard **Horsfall-Barratt plant pathology scale**:
  - `No Risk`: Healthy leaf (0% tissue damage)
  - `Very Low`: Early infection spots (1% to 5% leaf damage)
  - `Low`: Noticeable spots (6% to 15% leaf damage)
  - `Medium`: Significant spreading (16% to 35% leaf damage)
  - `High`: Severe necrosis and defoliation risk (more than 35% leaf damage)

* **Track B - Continuous Mathematical Pixel Ratio:**
  Color-space segmentation isolates the green leaf surface from the brown/yellow diseased spots to calculate the exact percentage of tissue lost:
  ```text
  Severity Percentage = (Infected Lesion Area / Total Leaf Area) * 100%
  ```

---

### Stage 4: Attention Rollout Explainability
To make the AI transparent instead of a "black box", the system implements Multi-Head Attention Rollout:
- As the Vision Transformer passes information through its 12 layers, it assigns attention weights to different patches of the leaf.
- We trace these attention scores backward from the final decision to the original image patches.
- This creates an **explainability heatmap** overlaid on the leaf: red and yellow areas highlight the exact fungal spots and tissue lesions the model used to reach its diagnosis.

---

### Stage 5: Agronomic Prescription and Treatment Engine
Once the disease and its severity are identified, the system looks up verified management recommendations:
- **FRAC-Coded Chemical Fungicides:** Recommends commercial fungicides (*e.g., Chlorothalonil 75% WP, Metalaxyl-M + Mancozeb, Difenoconazole*) along with exact water dilution rates (*e.g., 2.0 to 2.5 grams per liter of water*).
- **Biological Controls:** Preventative bio-agents such as *Bacillus subtilis* or *Trichoderma harzianum*.
- **Cultural Field Practices:** Sanitation, pruning infected branches, canopy aeration, and switching to drip irrigation.
- **Spray Timelines:** Clear application schedules based on whether the severity is mild or emergency-level.
- **Acreage Tank Mix Calculator:** Calculates the exact amount of commercial product needed based on the user's field size.

---

## 4. Taxonomy of Supported Crops and Pathological Classes

The system provides diagnostic coverage across **42 conditions spanning 15 crop species**:

| Crop Species | Supported Conditions |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pear** | Pear Leaf Curl, Pear Healthy, Pear Slug Sawfly, Pear Leaf Spot (DiaMOS Plant) |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy |

---

## 5. Multi-Dataset Engineering and Zero Data Leakage Protocols

To bridge the gap between clean lab photos and real-world field images, three distinct datasets were unified:

```text
                            ALL DATASETS (68,131 Images)
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
        v                                v                                v
[ PlantVillage ]                 [ PlantDoc ]                    [ DiaMOS Plant ]
- Controlled lab setting         - Real-world field images       - Real-world pear orchard
- 54,299 images                  - 7,906 leaf crops              - 5,926 leaf images
- 38 classes across 14 crops     - Cluttered backgrounds         - 4 disease classes + severity
        |                                |                                |
        +--------------------------------+--------------------------------+
                                         |
                                         v
                      [ STRATIFIED UNIFIED PARTITIONING ]
                          (Fixed Random Seed: 42)
                                         |
        +--------------------------------+--------------------------------+
        | (70%)                          | (15%)                          | (15%)
        v                                v                                v
   TRAINING SET                     VALIDATION SET                    TEST SET
  (47,672 Images)                  (10,200 Images)                (10,259 Images)
  Model Learning                   Checkpoint Selection           Unseen Final Test
```

### Zero Data Leakage Protocols
1. **Split First, Augment Second:** Splitting into train, validation, and test sets was performed strictly on original images before any rotations or color adjustments were created.
2. **Deterministic Partitioning:** A fixed random seed (`seed = 42`) and file hash manifests guarantee that no test image or cropped variant was ever seen during training.
3. **Macro F1 Evaluation:** Performance is measured using Macro F1 (the unweighted average of F1 scores across all classes), ensuring that rare diseases are evaluated with the exact same importance as common diseases.

---

## 6. Empirical Experimental Results and Benchmarks

All models were evaluated on dedicated held-out test splits on local hardware (NVIDIA GeForce RTX 4050 Laptop GPU):

| Model Component | Architecture | Test Partition / Split | Accuracy | Macro F1 | Primary Metric |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Unified ViT Classifier** | ViT-B/16 (ImageNet-1K) | Unified Test Split (10,259 images) | **87.44%** | **86.13%** | Top-1 Accuracy |
| **ViT Pure Lab Benchmark** | ViT-B/16 | PlantVillage Held-Out Test (8,178 images) | **97.75%** | **96.99%** | Macro F1 |
| **ViT Solanaceae Subset** | ViT-B/16 | Review 2 Stratified Test Split (977 images) | **99.49%** | **99.46%** | Macro F1 |
| **Leaf Localization** | YOLOv8n | PlantDoc Test Annotations | -- | -- | **72.45% mAP@50** (0.467 IoU) |
| **Severity Assessment** | ResNet-18 | DiaMOS Discrete Severity Test Split | **66.53%** | **70.22%** | Macro F1 |

Training curves, confusion matrices, and verification outputs are archived in `results/`.

---

## 7. Installation and Quick Start Guide

### Option A: One-Click Windows Execution (Recommended)
Double-click `run_app.bat`. The script automatically:
1. Verifies that Python 3.10+ is installed on your computer.
2. Creates the `.venv` virtual environment if it does not already exist.
3. Upgrades `pip` and installs all packages from `requirements.txt`.
4. Checks for model weight files in `models/` and gives friendly guidance.
5. Launches the interactive Streamlit dashboard.

---

### Option B: Manual Terminal Execution

```bash
# 1. Clone the repository
git clone https://github.com/26piyush/Deep_Learning_Project.git
cd Deep_Learning_Project

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

The web application will be accessible in your browser at `http://localhost:8501`.

---

## 8. Model Weights and Checkpoint Downloads

Due to GitHub's 100 MB single-file repository threshold, trained model checkpoints are hosted on [GitHub Releases (Release v1.0.0)](https://github.com/26piyush/Deep_Learning_Project/releases/tag/v1.0.0).

Download the checkpoint files and place them into their designated local directories:

| Model Checkpoint | File Size | Destination Path | Direct Download Link |
| :--- | :---: | :--- | :--- |
| **`vit_best.pt`** | 327.48 MB | `models/vit_best.pt` | [Download vit_best.pt](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/vit_best.pt) |
| **`localization_best.pt`** | 5.93 MB | `models/localization_best.pt` | [Download localization_best.pt](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/localization_best.pt) |
| **`severity_best.pt`** | 42.72 MB | `models/severity_best.pt` | [Download severity_best.pt](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/severity_best.pt) |
| **`model_state.pt`** | 42.78 MB | `crop_health_model/model_state.pt` | [Download model_state.pt](https://github.com/26piyush/Deep_Learning_Project/releases/download/v1.0.0/model_state.pt) |

*(Note: If the application is launched prior to downloading model checkpoints, the UI opens safely with download reminders).*

---

## 9. Command-Line Evaluation and Inference

Execute standalone evaluation scripts from your activated virtual environment:

```bash
# Evaluate Vision Transformer (Accuracy, Macro F1, Confusion Matrix)
python -m src.evaluate_vit

# Evaluate YOLOv8 Leaf Localization (mAP@50, Mean IoU)
python -m src.evaluate_localization

# Evaluate Severity Assessment Classifier (Horsfall-Barratt metrics)
python -m src.evaluate_severity

# Execute end-to-end inference on a single specimen image
python -m src.inference --image testing_data/Tomato___Early_blight_01.jpg --classifier vit
```

---

## License and Attribution
This project is developed for academic research in precision agriculture and computer vision. Datasets utilized are credited to the **PlantVillage Project**, **PlantDoc (IIT Roorkee / CVPRW)**, and **DiaMOS Plant (University of Milan)**.
