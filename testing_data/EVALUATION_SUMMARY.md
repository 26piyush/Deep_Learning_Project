# Held-Out Testing Dataset: Diseases & Evaluation Metrics Report

**Project:** Intelligent Crop Disease Detection, Localization and Severity Grading Using Vision Transformers  
**Evaluation Scope:** Complete held-out testing collection of **209 images** across **42 classes** and **15 crops**  
**Generated:** 2026-10-31 10:49:51  
**Execution Hardware:** CPU / CUDA | Avg Throughput: **753.6 ms / image**  

---

## 1. Executive Summary & Core Metric KPIs

| Metric Indicator | Held-Out Value | Description & Clinical Utility |
| :--- | :--- | :--- |
| **Top-1 Classification Accuracy** | **89.47%** | Primary diagnostic prediction matches true disease exactly |
| **Top-3 Classification Accuracy** | **97.13%** | True disease present within model's top 3 differential diagnoses |
| **Macro-Averaged F1-Score** | **89.34%** | Unweighted harmonic mean across all 42 balanced disease classes |
| **Macro-Averaged Precision** | **91.51%** | True positive proportion over all predicted positives |
| **Macro-Averaged Recall** | **89.40%** | Sensitivity / detection rate across all disease categories |
| **YOLO Leaf Localization Rate** | **99.04%** | Successful foliar bounding box extraction (vs. fallback full-frame) |
| **Average Inference Latency** | **753.6 ms** | End-to-end pipeline speed per leaf image |

---

## 2. Accuracy Breakdown by Dataset Source

| Dataset Source | Modality / Environment | Total Images | Correct | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **PlantVillage** | Laboratory Controlled Foliage | 163 | 158 | **96.93%** |
| **DiaMOS_Pear** | Specialized Orchard Pear Foliage | 20 | 18 | **90.00%** |
| **PlantDoc** | In-the-Wild Field Foliage | 26 | 11 | **42.31%** |

---

## 3. Crop-by-Crop Performance Breakdown (All 15 Crops)

| Crop Species | Total Test Images | Correct Diagnoses | Accuracy (%) | Avg Diagnostic Confidence (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Grape** | 20 | 20 | **100.00%** | 94.42% |
| **Peach** | 10 | 10 | **100.00%** | 86.07% |
| **Raspberry** | 5 | 5 | **100.00%** | 87.31% |
| **Squash** | 5 | 5 | **100.00%** | 93.31% |
| **Strawberry** | 10 | 10 | **100.00%** | 84.61% |
| **Pear** | 20 | 18 | **90.00%** | 69.51% |
| **Corn** | 20 | 18 | **90.00%** | 85.79% |
| **Pepper, bell** | 10 | 9 | **90.00%** | 74.40% |
| **Tomato** | 50 | 44 | **88.00%** | 76.29% |
| **Apple** | 20 | 17 | **85.00%** | 75.64% |
| **Cherry** | 10 | 8 | **80.00%** | 86.53% |
| **Potato** | 15 | 12 | **80.00%** | 72.83% |
| **Orange** | 5 | 4 | **80.00%** | 86.20% |
| **Soybean** | 5 | 4 | **80.00%** | 79.24% |
| **Blueberry** | 4 | 3 | **75.00%** | 43.97% |

---

## 4. Stored Artifact Locations

All data and metrics have been compiled and persisted in standard spreadsheet and machine-readable formats:

1. **Per-Image Predictions Spreadsheet (CSV):**  
   - `results/testing_data_metrics/testing_data_predictions.csv`  
   - `testing_data/predictions_and_metrics.csv`  
   *(Contains row-by-row columns: filename, ground truth disease, predicted disease, confidence, Top-1 match, Top-3 match, severity tier, continuous lesion %, bounding boxes, latency)*

2. **Detailed Per-Image Metadata (JSON):**  
   - `results/testing_data_metrics/testing_data_predictions.json`

3. **Aggregate Summary Metrics (JSON):**  
   - `results/testing_data_metrics/summary_metrics.json`

4. **Per-Class Metrics Breakdown (CSV):**  
   - `results/testing_data_metrics/class_performance.csv`

5. **High-Resolution Confusion Matrix Plot (PNG & CSV):**  
   - `results/testing_data_metrics/confusion_matrix.png`  
   - `results/testing_data_metrics/confusion_matrix.csv`

6. **Visual Annotated Sample Predictions:**  
   - `results/testing_data_metrics/annotated_samples/`

---

## 5. Misclassification Analysis (22 images out of 209)

| Image Filename | Source | Ground Truth Disease | Predicted Disease | Top-1 Conf | Top-3 Match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Apple___Apple_scab_04.jpg` | PlantDoc | Apple___Apple_scab | Potato___Early_blight | 9.28% | MISS |
| `Apple___Apple_scab_05.jpg` | PlantDoc | Apple___Apple_scab | Corn___Cercospora_leaf_spot Gray_leaf_spot | 8.39% | HIT |
| `Apple___healthy_05.jpg` | PlantDoc | Apple___healthy | Peach___healthy | 51.58% | HIT |
| `Blueberry___healthy_01.jpg` | PlantDoc | Blueberry___healthy | Pepper,_bell___healthy | 18.85% | MISS |
| `Cherry___healthy_02.jpg` | PlantDoc | Cherry___healthy | Apple___healthy | 43.18% | HIT |
| `Cherry___healthy_04.jpg` | PlantDoc | Cherry___healthy | Peach___healthy | 70.01% | HIT |
| `Corn___Common_rust_04.jpg` | PlantDoc | Corn___Common_rust | Corn___Northern_Leaf_Blight | 75.42% | HIT |
| `Corn___Common_rust_05.jpg` | PlantDoc | Corn___Common_rust | Corn___Northern_Leaf_Blight | 58.80% | HIT |
| `Orange___Haunglongbing_(Citrus_greening)_05.jpg` | PlantVillage | Orange___Haunglongbing_(Citrus_greening) | Pepper,_bell___Bacterial_spot | 46.90% | HIT |
| `Pepper,_bell___Bacterial_spot_03.jpg` | PlantDoc | Pepper,_bell___Bacterial_spot | Tomato___Tomato_mosaic_virus | 18.19% | HIT |
| `Potato___Early_blight_03.jpg` | PlantDoc | Potato___Early_blight | Tomato___Late_blight | 76.26% | MISS |
| `Potato___Early_blight_04.jpg` | PlantDoc | Potato___Early_blight | Tomato___Late_blight | 72.62% | MISS |
| `Potato___Late_blight_05.jpg` | PlantVillage | Potato___Late_blight | Tomato___Late_blight | 46.94% | HIT |
| `Soybean___healthy_01.jpg` | PlantDoc | Soybean___healthy | Potato___healthy | 31.40% | HIT |
| `Tomato___Bacterial_spot_03.jpg` | PlantDoc | Tomato___Bacterial_spot | Potato___Early_blight | 33.90% | MISS |
| `Tomato___Early_blight_01.jpg` | PlantVillage | Tomato___Early_blight | Tomato___Target_Spot | 77.47% | HIT |
| `Tomato___Leaf_Mold_05.jpg` | PlantDoc | Tomato___Leaf_Mold | Apple___Apple_scab | 16.13% | MISS |
| `Tomato___Septoria_leaf_spot_01.jpg` | PlantVillage | Tomato___Septoria_leaf_spot | Tomato___Early_blight | 76.84% | HIT |
| `Tomato___Septoria_leaf_spot_05.jpg` | PlantDoc | Tomato___Septoria_leaf_spot | Pepper,_bell___Bacterial_spot | 29.46% | HIT |
| `Tomato___Spider_mites Two-spotted_spider_mite_03.jpg` | PlantVillage | Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | 69.85% | HIT |
| `Pear___spot_04.jpg` | DiaMOS_Pear | Pear___spot | Pear___slug | 57.82% | HIT |
| `Pear___spot_05.jpg` | DiaMOS_Pear | Pear___spot | Pear___slug | 48.10% | HIT |
