# Comprehensive Project Architecture & Faculty Viva Defense Guide
## Crop Disease Detection & Severity Assessment Using Vision Transformers and Deep Learning

**Project Title:** Intelligent Crop Disease Detection, Localization, and Severity Grading using Vision Transformers  
**Current Milestone:** Review 2 (75%–80% Completed Working Prototype)  
**Target Milestone:** Final Phase (100% Comprehensive System)  
**Primary Tech Stack:** PyTorch 2.5 (CUDA 12.1), Ultralytics YOLOv8, Vision Transformer (ViT-B/16), ResNet-18, Streamlit, Albumentations  

---

## Table of Contents
1. [Project Motivation & Problem Statement](#1-project-motivation--problem-statement)
2. [Current Architecture (Review 2 Working Implementation)](#2-current-architecture-review-2-working-implementation)
3. [Final Architecture (100% Phase Target)](#3-final-architecture-100-phase-target)
4. [Comparative Architecture Analysis (Current vs. Final)](#4-comparative-architecture-analysis-current-vs-final)
5. [Multi-Dataset Integration & Zero Data Leakage](#5-multi-dataset-integration--zero-data-leakage)
6. [Component-by-Component Technical Breakdown](#6-component-by-component-technical-breakdown)
7. [Verified Experimental Results & Metrics](#7-verified-experimental-results--metrics)
8. [Professor & Examiner Viva Voce FAQ (High-Yield Questions & Answers)](#8-professor--examiner-viva-voce-faq-high-yield-questions--answers)
9. [Repository Codebase Map for Instant Demo Reference](#9-repository-codebase-map-for-instant-demo-reference)

---

## 1. Project Motivation & Problem Statement

### The Problem with Existing AI Solutions
1. **Background Overfitting (Shortcut Learning):** Most published crop disease classifiers feed raw field images directly into Convolutional Neural Networks (CNNs). In real agricultural conditions, CNNs learn spurious correlations with background soil, irrigation pipes, or sunlight rather than the actual foliar pathology.
2. **Lack of Disease Localization:** Knowing a plant has "Late Blight" without knowing *where* the leaf is infected prevents automated robotic spraying and precise localized treatment.
3. **Absence of Severity Assessment:** A binary or class-only diagnosis ("Early Blight") does not inform the farmer whether the crop requires emergency fungicide or light monitoring. Agronomic decisions depend on **infection severity**.
4. **Laboratory vs. Field Domain Gap:** Models trained purely on lab datasets (like clean PlantVillage leaves on neutral backgrounds) fail catastrophically when deployed on noisy field imagery (PlantDoc).

### The Proposed Hierarchical Solution
This project implements a **3-stage hierarchical pipeline**:
1. **Stage 1 (Localization):** Uses YOLOv8 to isolate leaf and lesion regions, filtering out 80%+ of background field noise.
2. **Stage 2 (Foliar Diagnosis):** Uses Vision Transformers (`ViT-B/16`) with multi-head self-attention to capture long-range token relationships across lesion margins.
3. **Stage 3 (Severity Grading):** Employs a dedicated classifier to grade foliar damage into discrete risk tiers (`No Risk`, `Very Low`, `Low`, `Medium`, `High`).

---

## 2. Current Architecture (Review 2 Working Implementation)

The current system is an **end-to-end operational prototype** running on local hardware (PyTorch + CUDA on NVIDIA RTX GPU) and demonstrated via an interactive Streamlit dashboard.

```
                           [ INPUT FIELD IMAGE ]
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      IMAGE PREPROCESSING        │
                    │  • RGB Format Verification      │
                    │  • Corrupt Byte Checking        │
                    │  • Exif Orientation Normalization│
                    └─────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    STAGE 1: LOCALIZATION        │
                    │      Ultralytics YOLOv8n        │
                    │ (Trained on PlantDoc Bounding   │
                    │          Box Labels)            │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
           [ Bounding Box Found ]            [ No Box Found ]
                    │                                 │
                    ▼                                 ▼
       ┌─────────────────────────┐       ┌─────────────────────────┐
       │   SMART CONTEXT CROP    │       │   CONTROLLED FALLBACK   │
       │ • Merge overlapping box │       │ • Pass Full Image Frame │
       │ • Apply +15% padding    │       │ • Log Telemetry Warning │
       │ • Prevent lesion clipping│       │ • Fallback flag = True  │
       └─────────────────────────┘       └─────────────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     │ (Leaf Region: 224×224)
                                     ▼
       ┌─────────────────────────────────────────────────────────────┐
       │            STAGE 2 & 3: MULTI-ENGINE DIAGNOSIS              │
       │                                                             │
       │   ┌─────────────────────────┐   ┌─────────────────────────┐ │
       │   │  Vision Transformer     │   │  Crop Health Baseline   │ │
       │   │       (ViT-B/16)        │   │       (ResNet-18)       │ │
       │   │ • 86M Parameters        │   │ • 11.2M Parameters      │ │
       │   │ • 42 Unified Classes    │   │ • 38 PlantVillage Classes│ │
       │   │ • 15 Crop Categories    │   │ • 14 Crop Categories    │ │
       │   │ • Self-Attention Foliar │   │ • Fast Baseline Head    │ │
       │   └─────────────────────────┘   └─────────────────────────┘ │
       │                │                             │              │
       └────────────────┼─────────────────────────────┼──────────────┘
                        │                             │
                        ▼                             ▼
             ┌─────────────────────┐       ┌─────────────────────┐
             │  Primary Disease    │       │  Severity Grading   │
             │  Diagnosis + Top-3  │       │     (ResNet-18)     │
             │  Confidence Scores  │       │  (5 Discrete Tiers) │
             └─────────────────────┘       └─────────────────────┘
                        │                             │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────┐
                    │   OUTPUT EXPLAINABLE DOSSIER    │
                    │ • Localized Leaf Coordinate Box │
                    │ • Crop Species & Disease Name   │
                    │ • Top-3 Disease Probabilities   │
                    │ • Agronomic Infection Severity  │
                    │ • Optional Side-by-Side Consens.│
                    └─────────────────────────────────┘
```

### Architectural Modes Available in Current Prototype
1. **Engine 1 — Vision Transformer ViT-B/16:** Primary self-attention engine trained on unified multi-dataset (42 classes across 15 crops).
2. **Engine 2 — Crop Health ResNet-18:** Established 38-class residual CNN baseline for performance and speed benchmarking.
3. **Engine 3 — Smart Hybrid Cascade:** Executes ViT first; if ViT confidence $< 65\%$, the system automatically escalates diagnosis to the 38-class ResNet-18 model.
4. **Engine 4 — Side-by-Side Model Comparison:** Runs both architectures concurrently on the same leaf crop, providing comparative confidence deltas and an automated agreement consensus indicator.

---

## 3. Final Architecture (100% Phase Target)

In the final phase (the remaining 20–25% of the overall project lifecycle), the system evolves from **discrete bounding box detection** to **pixel-level precision segmentation**, **attention rollout explainability**, and **accelerated edge runtime deployment**.

```
                           [ INPUT FIELD IMAGE ]
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   HYBRID ACTIVE PREPROCESSING   │
                    │  • CLAHE Contrast Enhancement   │
                    │  • Spectral Vegetation Indexing │
                    └─────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  STAGE 1: INSTANCE SEGMENTATION │
                    │    YOLOv8-Seg / DeepLabV3+ /    │
                    │     Segment Anything (SAM)      │
                    └─────────────────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   [ EXACT LEAF CONTOUR MASK ]                     [ BACKGROUND PIXELS ]
   (Pixel coordinates: L_mask)                     (Masked to 0 / Null)
             │
             ├───────────────────────────────────────────────┐
             ▼                                               ▼
┌─────────────────────────────┐               ┌─────────────────────────────┐
│ STAGE 2: ADAPTIVE VIT       │               │ STAGE 3: PIXEL SEVERITY     │
│ (Swin Transformer / ViT-B)  │               │      LESION SEGMENTATION    │
│ • Mask-conditioned Tokenizer│               │ • Lesion Pixel Mask (D_mask)│
│ • Multi-Head Self-Attention │               │ • Mathematical Ratio:       │
│ • Disease Classification    │               │   Severity % =              │
│                             │               │   (D_mask / L_mask) × 100%  │
└─────────────────────────────┘               └─────────────────────────────┘
             │                                               │
             ▼                                               ▼
┌─────────────────────────────┐               ┌─────────────────────────────┐
│ STAGE 4: VISUAL ATTENTION   │               │ STAGE 5: AGRONOMIC ACTION   │
│       EXPLAINABILITY        │               │         PRESCRIPTION        │
│ • Attention Rollout Map     │               │ • Exact Leaf Damage %       │
│ • Grad-CAM Layer Overlays   │               │ • Dosage & Spray Schedule   │
└─────────────────────────────┘               └─────────────────────────────┘
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  STAGE 6: EDGE DEPLOYMENT CORE  │
                    │ • ONNX Runtime / TensorRT FP16  │
                    │ • Microservices REST / gRPC API │
                    │ • Drone / Smartphone Edge Ready │
                    └─────────────────────────────────┘
```

---

## 4. Comparative Architecture Analysis (Current vs. Final)

| Dimension | Current Architecture (Review 2 — ~75-80%) | Final Architecture (100% Complete Target) | Academic / Technical Rationale |
|---|---|---|---|
| **Localization Method** | **Bounding Box Detection** (YOLOv8n) | **Pixel Mask Segmentation** (YOLOv8-Seg / SAM) | Bounding boxes include background corners; polygon masks isolate 100% pure leaf tissue. |
| **Lesion Isolation** | Merged bbox with 15% context padding | Dual-mask segmentation (Leaf Mask vs. Lesion Mask) | Enables mathematical ratio calculation of infected surface area. |
| **Severity Estimation** | **Discrete Categorical** (5 levels: ResNet-18) | **Continuous Mathematical %** ($\frac{\text{Lesion Pixels}}{\text{Leaf Pixels}} \times 100$) | Replaces heuristic grading with exact physical percentage of necrosis. |
| **Classification Engine** | Pretrained `ViT-B/16` (ImageNet initialized) | `Swin Transformer` or Multi-Scale Hierarchical ViT | Swin uses shifted windows, achieving $\mathcal{O}(N)$ complexity instead of $\mathcal{O}(N^2)$ global attention. |
| **Explainability** | Localized bounding box & confidence bars | **Attention Rollout & Grad-CAM Heatmaps** | Visualizes which specific pixel tokens drove the attention weights of the transformer heads. |
| **Dataset Scope** | **42 Classes across 15 Crops** (68,131 Images) | Extended field dataset with drone & multi-spectral samples | Increases invariance to ambient field shadows and wet-leaf specular reflections. |
| **Comparative Baselines** | ResNet-18 | ResNet-50, EfficientNet-B4, and Swin-T Benchmarking | Fulfills academic requirements for deep ablation and empirical superiority proofs. |
| **Deployment Target** | Interactive Streamlit Localhost Web Application | **ONNX / TensorRT Edge Runtime** (Mobile / Jetson) | Enables offline edge inference on tractors and agricultural UAVs without internet access. |

---

## 5. Multi-Dataset Integration & Zero Data Leakage

To eliminate the common academic critique that models only work on lab benchmarks, this project combines **three diverse datasets**:

```
                              ALL PROJECT DATASETS (68,131 Images)
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             │                                 │                                 │
             ▼                                 ▼                                 ▼
   [ PlantVillage Dataset ]          [ PlantDoc Dataset ]             [ DiaMOS Plant Dataset ]
   • Controlled Lab Setting          • Real-World Field Images        • Real-World Pear Crop
   • 54,299 images                   • 7,906 leaf crops               • 5,926 leaf images
   • 38 classes across 14 crops      • 29 disease/healthy classes     • 4 classes (curl, slug, spot,
   • Uniform lighting, clean         • Occlusions, shadows, multi-      healthy) + 1,243 severity
     backgrounds                       leaf clusters                    annotations
             │                                 │                                 │
             └─────────────────────────────────┼─────────────────────────────────┘
                                               │
                                               ▼
                             [ UNIFIED STRATIFIED PARTITIONING ]
                                (Fixed Seed: 42 | No Overlap)
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼ (70%)                           ▼ (15%)                           ▼ (15%)
      TRAINING SET                      VALIDATION SET                      TEST SET
     (47,672 Images)                   (10,200 Images)                   (10,259 Images)
  • Used for AdamW back-            • Used for checkpoint             • Held-out strictly
    propagation & loss weights        tuning & early stopping           unseen evaluation
```

### The 15 Supported Crops and 42 Conditions
1. **Apple (4):** Apple scab, Black rot, Cedar apple rust, Healthy
2. **Blueberry (1):** Healthy
3. **Cherry (2):** Powdery mildew, Healthy
4. **Corn / Maize (4):** Gray leaf spot, Common rust, Northern leaf blight, Healthy
5. **Grape (4):** Black rot, Esca (Black Measles), Leaf blight (Isariopsis), Healthy
6. **Orange (1):** Huanglongbing (Citrus greening)
7. **Peach (2):** Bacterial spot, Healthy
8. **Pear (4):** Pear leaf curl, Pear healthy, Pear slug sawfly, Pear leaf spot *(from DiaMOS)*
9. **Pepper Bell (2):** Bacterial spot, Healthy
10. **Potato (3):** Early blight, Late blight, Healthy
11. **Raspberry (1):** Healthy
12. **Soybean (1):** Healthy
13. **Squash (1):** Powdery mildew
14. **Strawberry (2):** Leaf scorch, Healthy
15. **Tomato (10):** Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Tomato yellow leaf curl virus, Mosaic virus, Healthy

---

## 6. Component-by-Component Technical Breakdown

### Component 1: Image Preprocessing Pipeline
- **RGB Enforcement:** Converts all color modes (RGBA, Grayscale, CMYK) to standard 3-channel RGB.
- **Corrupt File Guard:** Validates file byte headers using `PIL.Image.verify()` to prevent silent pipeline crashes.
- **Normalization:** Employs ImageNet normalization standard:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### Component 2: YOLOv8 Leaf & Region Localization
- **Architecture:** Ultralytics YOLOv8n (nano variant, ~3.2M parameters) optimized for edge speed.
- **Role:** Extracts the precise bounding box of foliar tissue.
- **Smart Context Cropping:**
  - Tight bounding boxes often clip outer necrotic rings (e.g., Target Spot or Early Blight halos).
  - The pipeline automatically merges overlapping detection boxes and applies **15% contextual boundary padding**.
- **Controlled Fallback Mechanism:**
  - If field lighting or angle results in zero YOLO detections ($N_{\text{boxes}} = 0$), the pipeline **does not crash**.
  - It safely falls back to a normalized center-weighted frame, sets `used_fallback = True`, and displays an explanatory advisory in the telemetry log.

### Component 3: Vision Transformer (ViT-B/16) Disease Classification
- **Architecture:** Vision Transformer Base with $16 \times 16$ patch resolution (`ViT-B/16`).
  - Total Parameters: $\approx 86.6\text{M}$.
  - Hidden Dimension: $D = 768$, Attention Heads: $h = 12$, Transformer Encoder Layers: $L = 12$.
- **Mathematical Principle:**
  1. Input image $x \in \mathbb{R}^{H \times W \times C}$ ($224 \times 224 \times 3$) is divided into $N = \frac{HW}{P^2} = \frac{224 \times 224}{16 \times 16} = 196$ non-overlapping patches.
  2. Each patch is flattened and linearly projected to dimension $D = 768$.
  3. A learnable class token $[x_{\text{class}}]$ is prepended, yielding $197$ tokens.
  4. Standard 1D learnable position embeddings $E_{\text{pos}} \in \mathbb{R}^{197 \times 768}$ are added.
  5. Tokens pass through 12 Multi-Head Self-Attention (MSA) blocks:
     $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     $$\text{MSA}(X) = [\text{head}_1, \dots, \text{head}_h]W^O$$
  6. The representation of the $[x_{\text{class}}]$ token is fed into a linear classification head:
     $$\hat{y} = \text{Softmax}(W_{\text{head}} \cdot z_L^0)$$

### Component 4: Severity Classifier
- **Architecture:** Pretrained ResNet-18 modified with a 5-class linear output head.
- **Dataset:** DiaMOS Plant leaf images labeled into 5 discrete severity classes (`No Risk`, `Very Low`, `Low`, `Medium`, `High`).
- **Function:** Categorizes foliar damage to inform immediate chemical intervention levels.

---

## 7. Verified Experimental Results & Metrics

All metrics reflect genuine execution on local hardware (NVIDIA GeForce RTX 4050 Laptop GPU):

| Component | Architecture | Evaluation Dataset & Split | Metric 1 | Metric 2 | Primary Checkpoint |
|---|---|---|---|---|---|
| **ViT Multi-Dataset (Current)** | ViT-B/16 | Unified (Lab + Field + Pear) Test Split (10,259 images) | **87.44% Accuracy** | **86.13% Macro F1** | `models/vit_best.pt` |
| **ViT Benchmark (Pure Lab)** | ViT-B/16 | PlantVillage Held-Out Test (8,178 images) | **97.75% Accuracy** | **96.99% Macro F1** | `models/vit_38class_best.pt` |
| **ViT Solanaceae Subset** | ViT-B/16 | Review 2 Stratified Test Split (977 images) | **99.49% Accuracy** | **99.46% Macro F1** | `models/vit_5class_best.pt` |
| **Leaf Localization** | YOLOv8n | PlantDoc Test Bounding Boxes | **72.45% mAP@50** | **0.4669 Mean IoU** | `models/localization_best.pt` |
| **Severity Grading** | ResNet-18 | DiaMOS Discrete Severity Test Split | **66.53% Accuracy** | **70.22% Macro F1** | `models/severity_best.pt` |

---

## 8. Professor & Examiner Viva Voce FAQ

### Q1: Why did you choose Vision Transformers (ViT) over traditional CNNs like ResNet or VGG?
> **Answer:** CNNs operate on localized receptive fields (typically $3 \times 3$ or $7 \times 7$ kernels). While excellent for texture extraction, they struggle with global context and long-range spatial dependencies. A foliar disease often manifests as multiple irregular lesions distributed across an entire leaf blade with varying margins. Vision Transformers use global self-attention across all image patches simultaneously, enabling the model to correlate distant lesion spots and leaf margin discoloration directly. Furthermore, ViT provides superior transfer learning capability when fine-tuned from large-scale pretraining.

### Q2: Vision Transformers typically require millions of images (JFT-300M) to train from scratch. How did you train ViT on your dataset without severe overfitting?
> **Answer:** We utilized **transfer learning** with ImageNet-1K pretrained weights (`ViT_B_16_Weights.IMAGENET1K_V1`). The self-attention projection weights and multi-layer perceptron layers already possessed rich visual feature extractors. We replaced only the final classification head (`model.heads.head = nn.Linear(768, num_classes)`) and fine-tuned the network using AdamW with weight decay ($10^{-4}$) and a conservative learning rate ($2 \times 10^{-5}$ to $3 \times 10^{-5}$) using mixed-precision acceleration.

### Q3: What is the purpose of the YOLOv8 stage? Why not feed the image directly into the Vision Transformer?
> **Answer:** When farmers take photos in the field, more than 70% of the image often contains background clutter—soil, neighboring crops, hands holding the leaf, weeds, and varying sky illumination. Direct end-to-end classifiers frequently suffer from **shortcut learning**, where the model accidentally classifies the soil color or background instead of the pathogen. YOLOv8 isolates only the relevant leaf region, normalizing the input and ensuring the Vision Transformer computes self-attention solely on foliar pathology.

### Q4: What happens if YOLOv8 fails to detect a leaf in an image? Does the pipeline fail?
> **Answer:** No. We engineered a **Controlled Fallback Mechanism**. If YOLOv8 returns zero bounding boxes (which can occur with severe occlusions or extreme camera angles), the pipeline automatically falls back to processing the full image frame resized to $224 \times 224$. The system flags `used_fallback = True` in the output telemetry and displays a warning, ensuring robust fault tolerance during real-world usage.

### Q5: How did you ensure there is zero data leakage between training and testing?
> **Answer:** We enforced three strict criteria:
> 1. Dataset splitting was conducted **at the root image level before any augmentations** were applied.
> 2. Stratified train/val/test splits ($70\% / 15\% / 15\%$) were generated deterministically using a fixed random seed (`seed = 42`).
> 3. Manifest files (`unified_splits.json`, `plantvillage_splits.json`) store unique image hashes and absolute paths, ensuring that no training image or augmented derivative can ever appear in the held-out test split.

### Q6: Why did you use discrete severity classification (5 levels) in Review 2 instead of exact percentage calculation?
> **Answer:** For Review 2 (70–80% milestone), discrete severity classes (`No Risk`, `Very Low`, `Low`, `Medium`, `High`) map directly to established agronomic standards such as the **Horsfall-Barratt rating system**, which agronomists use in visual field surveys. In our Final Phase roadmap, this will be supplemented with dual-mask pixel segmentation to compute exact mathematical necrosis percentages ($\frac{\text{Lesion Pixels}}{\text{Leaf Pixels}} \times 100\%$).

### Q7: Explain the difference between Micro F1 and Macro F1, and why Macro F1 was prioritized.
> **Answer:** 
> - **Micro F1** calculates metrics globally by aggregating all true positives, false positives, and false negatives across all classes. It is heavily dominated by majority classes.
> - **Macro F1** computes the F1 score independently for each class and calculates the unweighted mean across all classes:
>   $$\text{Macro F1} = \frac{1}{C}\sum_{c=1}^C \text{F1}_c$$
> Because agricultural datasets often have class imbalances (e.g., common diseases have thousands of samples while rarer diseases have fewer), **Macro F1** guarantees that performance on minority diseases is evaluated with equal rigor.

### Q8: What is Smart Context Padding, and why is it necessary?
> **Answer:** Standard object detectors produce tight bounding boxes around detected objects. In crop disease diagnosis, cropping strictly to the bounding box often cuts off the outer perimeter of fungal rings, chlorotic halos, and leaf edges, which are vital diagnostic discriminators. Smart Context Padding merges overlapping leaf detections and extends each boundary by **15% padding**, preserving pathological lesion margins without introducing excessive background noise.

### Q9: What is the Smart Cascade Hybrid Mode in your web application?
> **Answer:** High-capacity Vision Transformers (86M parameters) excel at complex multi-head feature extraction, but in edge scenarios or low-confidence boundary predictions, single-model decisions can be risky. Our **Smart Cascade Hybrid Mode** evaluates the ViT output probability. If $\max(P_{\text{ViT}}) \ge 65\%$, the ViT diagnosis is confirmed; if confidence is $< 65\%$, the system automatically routes the crop to the 38-class ResNet-18 baseline model and alerts the user of the cascade escalation.

### Q10: What are the specific deliverables for the Final Phase (the remaining 20–25%)?
> **Answer:** 
> 1. **Pixel-Level Lesion Segmentation:** Replacing bounding boxes with instance/semantic segmentation masks (YOLOv8-Seg or Segment Anything Model) to calculate the continuous infected area percentage.
> 2. **Transformer Explainability:** Implementing **Attention Rollout** and **Grad-CAM** heatmaps to visually project the self-attention heads onto the leaf image for agronomists to inspect.
> 3. **Extended Baseline Study:** Empirical comparison across four architectures: ViT-B/16 vs. Swin Transformer vs. ResNet-50 vs. EfficientNet-B4.
> 4. **Edge Acceleration:** Exporting models to **ONNX Runtime** and **TensorRT FP16** to benchmark inference latency for deployment on mobile phones and agricultural drones.

---

## 9. Repository Codebase Map for Instant Demo Reference

When presenting to faculty, use this reference to quickly show relevant implementations:

| Pipeline Stage | Primary Implementation File | Key Function / Class |
|---|---|---|
| **System Configuration** | [src/config.py](file:///c:/Users/piyus/Documents/crop_disease_review2/src/config.py) | `VIT_CLASSES`, `UNIFIED_42_CLASSES`, hyperparameters, paths |
| **Data Preparation & Splits** | [src/dataset_utils.py](file:///c:/Users/piyus/Documents/crop_disease_review2/src/dataset_utils.py) | `prepare_unified_dataset()`, `prepare_plantvillage_dataset()` |
| **ViT Model Training** | [src/train_vit.py](file:///c:/Users/piyus/Documents/crop_disease_review2/src/train_vit.py) | `build_vit_model()`, `train_one_epoch()` (with PyTorch AMP) |
| **ViT Evaluation & Confusion Matrix** | [src/evaluate_vit.py](file:///c:/Users/piyus/Documents/crop_disease_review2/src/evaluate_vit.py) | `evaluate_vit()` (Accuracy, Macro F1, Confusion Matrix) |
| **End-to-End Inference Pipeline** | [src/inference.py](file:///c:/Users/piyus/Documents/crop_disease_review2/src/inference.py) | `run_pipeline()`, `extract_leaf_crop()`, `_evaluate_vit()` |
| **Streamlit Interactive UI** | [app.py](file:///c:/Users/piyus/Documents/crop_disease_review2/app.py) | 4 engines, 15 crops dropdown, real-time leaf diagnosis |
| **Curated 210-Image Test Suite** | [testing_data/](file:///c:/Users/piyus/Documents/crop_disease_review2/testing_data/) | Ground-truth labeled images across all 42 classes |
| **Evaluation Metrics JSON** | [results/metrics/vit_metrics.json](file:///c:/Users/piyus/Documents/crop_disease_review2/results/metrics/vit_metrics.json) | Real test split accuracy, macro precision/recall/F1 |
| **Trained ViT Checkpoints** | [models/vit_best.pt](file:///c:/Users/piyus/Documents/crop_disease_review2/models/vit_best.pt) | Unified 42-class multi-dataset trained model weights |
