"""Evaluation and Telemetry Pipeline for the Held-Out Testing Dataset.
Processes all 210 images in testing_data across 42 classes and 15 crops:
  1. Runs end-to-end inference (Localization -> ViT Classification -> Severity -> Segmentation -> Prescription)
  2. Records per-image predictions, confidences, top-3 candidates, severity metrics, and latency
  3. Computes comprehensive evaluation metrics (Top-1 Accuracy, Top-3 Accuracy, Precision, Recall, F1, Confusion Matrix)
  4. Exports structured results to CSV, JSON, PNG confusion matrix, and publication-ready Markdown reports.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    BASE_DIR,
    LOCALIZATION_MODEL_PATH,
    VIT_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    CROP_HEALTH_WEIGHTS_PATH,
    CROP_HEALTH_META_PATH,
    get_device,
)
from src.inference import load_pipeline_models, get_crop_health_model, run_pipeline
from src.knowledge_base import get_disease_knowledge
from src.prescription import get_agronomic_prescription


def evaluate_testing_dataset(
    testing_data_dir: Path = BASE_DIR / "testing_data",
    labels_csv_path: Path = BASE_DIR / "testing_data" / "labels.csv",
    output_dir: Path = BASE_DIR / "results" / "testing_data_metrics",
    yolo_path: Path = LOCALIZATION_MODEL_PATH,
    vit_path: Path = VIT_MODEL_PATH,
    severity_path: Path = SEVERITY_MODEL_PATH,
    ch_weights_path: Path = CROP_HEALTH_WEIGHTS_PATH,
    ch_meta_path: Path = CROP_HEALTH_META_PATH,
    conf_thresh: float = 0.20,
    save_annotated_samples: bool = True,
) -> Dict[str, Any]:
    """Execute evaluation over all images in testing_data directory."""
    testing_data_dir = Path(testing_data_dir)
    labels_csv_path = Path(labels_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found at: {labels_csv_path}")

    df_meta = pd.read_csv(labels_csv_path)
    print(f"Loaded {len(df_meta)} records from {labels_csv_path}")

    # Set up hardware device and models
    device = get_device()
    print(f"Loading pipeline models on device: {device}...")
    models = load_pipeline_models(yolo_path, vit_path, severity_path, device)
    yolo, vit, vit_tfm, vit_classes, sev, sev_tfm, sev_classes, _ = models

    try:
        ch_model = get_crop_health_model(ch_weights_path, ch_meta_path, device)
    except Exception as e:
        print(f"Note: CropHealth model not loaded ({e}), using ViT primary.")
        ch_model = None

    class_to_idx = {c: i for i, c in enumerate(vit_classes)}
    print(f"Model taxonomy verified: {len(vit_classes)} unified classes across 15 crops.")

    per_image_records = []
    y_true_indices = []
    y_pred_indices = []
    top3_hits = []

    annotated_dir = output_dir / "annotated_samples"
    if save_annotated_samples:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(df_meta)
    print(f"Starting evaluation of {total_images} testing images...")
    start_all = time.time()

    for idx, row in df_meta.iterrows():
        fname = str(row["filename"]).strip()
        img_path = testing_data_dir / fname
        gt_class = str(row["class"]).strip()
        gt_crop = str(row["crop"]).strip()
        gt_disease = str(row["disease"]).strip()
        source_ds = str(row.get("source_dataset", "Unknown")).strip()

        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping.")
            continue

        t0 = time.time()
        try:
            res = run_pipeline(
                image=img_path,
                yolo_model=yolo,
                vit_model=vit,
                vit_tfm=vit_tfm,
                vit_classes=vit_classes,
                sev_model=sev,
                sev_tfm=sev_tfm,
                sev_classes=sev_classes,
                crop_health_model=ch_model,
                device=device,
                conf_thresh=conf_thresh,
                classifier_mode="vit",
            )
        except Exception as exc:
            print(f"Error evaluating {fname}: {exc}")
            continue

        latency_ms = (time.time() - t0) * 1000.0

        pred_class = res["disease_raw"]
        pred_crop = res["crop"]
        pred_disease = res["disease"]
        pred_conf = float(res["disease_confidence"])

        # Check Top-1 and Top-3 accuracy
        is_correct = (pred_class == gt_class)
        top3_list = res.get("top3_diseases", [])
        top3_classes = [t.get("class", "") for t in top3_list]
        is_top3 = gt_class in top3_classes

        top2_class = top3_classes[1] if len(top3_classes) > 1 else ""
        top2_conf = float(top3_list[1].get("confidence", 0.0)) if len(top3_list) > 1 else 0.0
        top3_class = top3_classes[2] if len(top3_classes) > 2 else ""
        top3_conf = float(top3_list[2].get("confidence", 0.0)) if len(top3_list) > 2 else 0.0

        # Severity & Segmentation
        sev_class = res.get("severity", "Unknown")
        sev_conf = float(res.get("severity_confidence", 0.0))
        math_sev = res.get("mathematical_severity", {})
        cont_sev_pct = float(math_sev.get("severity_percentage", 0.0))
        agronomic_tier = math_sev.get("agronomic_tier", "N/A")

        # Localization
        boxes_count = res.get("total_boxes_found", 0)
        selected_box = res.get("bounding_box", [0, 0, 0, 0])
        used_fallback = res.get("used_fallback", False)
        max_box_conf = float(res.get("max_detection_confidence", 0.0))

        # Agronomics & Knowledge
        prescription = res.get("prescription", {})
        if isinstance(prescription, dict):
            primary_chem = str(prescription.get("chemical_treatment", "N/A"))
            chem_dosage = str(prescription.get("dosage", "N/A"))
        else:
            primary_chem = str(prescription)
            chem_dosage = "N/A"

        dk = res.get("disease_knowledge", {})
        if isinstance(dk, dict):
            causal_organism = str(dk.get("causal_organism", "N/A"))
            pathogen_type = str(dk.get("pathogen_type", "Foliar Condition"))
        else:
            causal_organism = "N/A"
            pathogen_type = "Foliar Condition"

        # Record for telemetry
        record = {
            "image_index": idx + 1,
            "filename": fname,
            "source_dataset": source_ds,
            "ground_truth_class": gt_class,
            "ground_truth_crop": gt_crop,
            "ground_truth_disease": gt_disease,
            "predicted_class": pred_class,
            "predicted_crop": pred_crop,
            "predicted_disease": pred_disease,
            "is_correct": bool(is_correct),
            "top1_match": "MATCH" if is_correct else "MISMATCH",
            "is_top3_correct": bool(is_top3),
            "top3_match": "HIT" if is_top3 else "MISS",
            "confidence": pred_conf,
            "confidence_percentage": f"{pred_conf * 100:.2f}%",
            "top2_class": top2_class,
            "top2_confidence": top2_conf,
            "top3_class": top3_class,
            "top3_confidence": top3_conf,
            "predicted_severity": sev_class,
            "severity_confidence": sev_conf,
            "continuous_severity_pct": cont_sev_pct,
            "agronomic_tier": agronomic_tier,
            "localization_boxes_found": boxes_count,
            "used_fallback": bool(used_fallback),
            "max_detection_confidence": max_box_conf,
            "selected_box_xyxy": str(selected_box),
            "pathogen_type": pathogen_type,
            "causal_organism": causal_organism,
            "recommended_treatment": primary_chem,
            "chemical_dosage": chem_dosage,
            "latency_ms": round(latency_ms, 2),
        }
        per_image_records.append(record)

        if gt_class in class_to_idx and pred_class in class_to_idx:
            y_true_indices.append(class_to_idx[gt_class])
            y_pred_indices.append(class_to_idx[pred_class])
        top3_hits.append(is_top3)

        # Periodically log progress
        if (idx + 1) % 25 == 0 or (idx + 1) == total_images:
            curr_acc = np.mean([r["is_correct"] for r in per_image_records]) * 100
            print(f"[{idx + 1}/{total_images}] Done. Running Accuracy: {curr_acc:.2f}%")

        # Optionally save annotated visualization for sample images
        if save_annotated_samples and (idx % 10 == 0 or not is_correct):
            try:
                annotated_img = res.get("full_hierarchical_image") or res.get("annotated_image")
                if annotated_img is not None:
                    status_prefix = "correct" if is_correct else "mismatch"
                    out_img_name = f"{idx+1:03d}_{status_prefix}_{fname}"
                    annotated_img.save(annotated_dir / out_img_name)
            except Exception:
                pass

    total_time = time.time() - start_all
    avg_latency = total_time / max(1, len(per_image_records))

    print(f"\nInference completed in {total_time:.2f}s ({avg_latency:.3f}s / image).")

    # 1. Convert per-image records to DataFrame
    df_results = pd.DataFrame(per_image_records)

    # Save Per-Image CSV to both results and testing_data
    csv_results_path = output_dir / "testing_data_predictions.csv"
    df_results.to_csv(csv_results_path, index=False)
    print(f"Saved per-image predictions CSV: {csv_results_path}")

    # Also save copy right inside testing_data for easy access
    testing_data_csv = testing_data_dir / "predictions_and_metrics.csv"
    df_results.to_csv(testing_data_csv, index=False)
    print(f"Saved copy in testing_data: {testing_data_csv}")

    # Save Per-Image JSON
    json_results_path = output_dir / "testing_data_predictions.json"
    with open(json_results_path, "w", encoding="utf-8") as f:
        json.dump(per_image_records, f, indent=2)
    print(f"Saved per-image predictions JSON: {json_results_path}")

    # 2. Compute Global Classification Metrics
    accuracy = float(accuracy_score(y_true_indices, y_pred_indices))
    top3_acc = float(np.mean(top3_hits))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="weighted", zero_division=0
    )

    # Per-Class Metrics
    target_names = [vit_classes[i] for i in sorted(list(set(y_true_indices + y_pred_indices)))]
    cls_report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    class_perf_rows = []
    for cls_name in target_names:
        if cls_name in cls_report:
            item = cls_report[cls_name]
            class_perf_rows.append({
                "class": cls_name,
                "precision": round(float(item["precision"]), 4),
                "recall": round(float(item["recall"]), 4),
                "f1_score": round(float(item["f1-score"]), 4),
                "support": int(item["support"]),
            })
    df_class_perf = pd.DataFrame(class_perf_rows)
    class_perf_csv = output_dir / "class_performance.csv"
    df_class_perf.to_csv(class_perf_csv, index=False)

    # 3. Compute Crop-Level Metrics
    crop_stats = []
    for crop_name, group in df_results.groupby("ground_truth_crop"):
        total_crop = len(group)
        correct_crop = int(group["is_correct"].sum())
        crop_acc = (correct_crop / total_crop) * 100.0
        avg_conf = float(group["confidence"].mean()) * 100.0
        crop_stats.append({
            "crop": crop_name,
            "total_images": total_crop,
            "correct": correct_crop,
            "accuracy_pct": round(crop_acc, 2),
            "avg_confidence_pct": round(avg_conf, 2),
        })
    df_crop_perf = pd.DataFrame(crop_stats).sort_values(by="accuracy_pct", ascending=False)
    crop_perf_csv = output_dir / "crop_performance.csv"
    df_crop_perf.to_csv(crop_perf_csv, index=False)

    # 4. Compute Dataset Source Metrics (PlantVillage vs PlantDoc vs DiaMOS Pear)
    source_stats = []
    for src_name, group in df_results.groupby("source_dataset"):
        total_src = len(group)
        correct_src = int(group["is_correct"].sum())
        src_acc = (correct_src / total_src) * 100.0
        source_stats.append({
            "source_dataset": src_name,
            "total_images": total_src,
            "correct": correct_src,
            "accuracy_pct": round(src_acc, 2),
        })
    df_source_perf = pd.DataFrame(source_stats).sort_values(by="accuracy_pct", ascending=False)
    source_perf_csv = output_dir / "dataset_source_performance.csv"
    df_source_perf.to_csv(source_perf_csv, index=False)

    # 5. Localization & Severity Telemetry
    total_samples = len(df_results)
    successful_localization = int((~df_results["used_fallback"]).sum())
    loc_success_rate = (successful_localization / max(1, total_samples)) * 100.0
    fallback_rate = 100.0 - loc_success_rate
    avg_boxes = float(df_results["localization_boxes_found"].mean())
    avg_cont_sev = float(df_results["continuous_severity_pct"].mean())

    # 6. Confusion Matrix Computation and Plotting
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(vit_classes)))
    cm_df = pd.DataFrame(cm, index=vit_classes, columns=vit_classes)
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Plot high-res Confusion Matrix
    plt.figure(figsize=(26, 22))
    short_labels = [c.replace("___", " \n ") for c in vit_classes]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar=True,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title(
        f"Held-Out Testing Data Confusion Matrix (42 Unified Classes)\nOverall Top-1 Accuracy: {accuracy*100:.2f}% | Top-3 Accuracy: {top3_acc*100:.2f}%",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    plt.xlabel("Predicted Class", fontsize=13, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    cm_fig_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Saved high-res confusion matrix: {cm_fig_path}")

    # 7. Summary JSON
    summary_metrics = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_testing_images": total_samples,
        "total_classes": len(vit_classes),
        "total_crops": len(df_crop_perf),
        "top1_accuracy": round(accuracy, 4),
        "top1_accuracy_pct": round(accuracy * 100, 2),
        "top3_accuracy": round(top3_acc, 4),
        "top3_accuracy_pct": round(top3_acc * 100, 2),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "localization_leaf_detection_rate_pct": round(loc_success_rate, 2),
        "localization_fallback_rate_pct": round(fallback_rate, 2),
        "average_detected_boxes_per_image": round(avg_boxes, 2),
        "average_continuous_severity_pct": round(avg_cont_sev, 2),
        "total_execution_time_seconds": round(total_time, 2),
        "average_latency_ms_per_image": round(avg_latency * 1000.0, 2),
        "dataset_sources": df_source_perf.to_dict(orient="records"),
        "crop_accuracies": df_crop_perf.to_dict(orient="records"),
    }

    summary_json_path = output_dir / "summary_metrics.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Saved summary metrics JSON: {summary_json_path}")

    # 8. Generate Publication-Grade Markdown Report
    generate_markdown_report(
        summary_metrics=summary_metrics,
        df_crop=df_crop_perf,
        df_source=df_source_perf,
        df_results=df_results,
        output_report_path=BASE_DIR / "TESTING_DATA_EVALUATION_REPORT.md",
        testing_data_summary_path=testing_data_dir / "EVALUATION_SUMMARY.md",
    )

    print("\n" + "=" * 65)
    print("TESTING DATA EVALUATION SUMMARY:")
    print(f"Total Test Images:   {total_samples}")
    print(f"Top-1 Accuracy:      {accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy:      {top3_acc * 100:.2f}%")
    print(f"Macro F1-Score:      {f1_macro * 100:.2f}%")
    print(f"Macro Precision:     {prec_macro * 100:.2f}%")
    print(f"Macro Recall:        {rec_macro * 100:.2f}%")
    print(f"Leaf Localization:   {loc_success_rate:.2f}% detected")
    print(f"Avg Latency/Image:   {avg_latency * 1000.0:.1f} ms")
    print("=" * 65)

    return summary_metrics


def generate_markdown_report(
    summary_metrics: Dict[str, Any],
    df_crop: pd.DataFrame,
    df_source: pd.DataFrame,
    df_results: pd.DataFrame,
    output_report_path: Path,
    testing_data_summary_path: Path,
) -> None:
    """Generate professional Markdown evaluation reports."""
    top1_acc = summary_metrics["top1_accuracy_pct"]
    top3_acc = summary_metrics["top3_accuracy_pct"]
    macro_f1 = summary_metrics["macro_f1"] * 100
    macro_prec = summary_metrics["macro_precision"] * 100
    macro_rec = summary_metrics["macro_recall"] * 100
    loc_rate = summary_metrics["localization_leaf_detection_rate_pct"]
    avg_lat = summary_metrics["average_latency_ms_per_image"]
    total_imgs = summary_metrics["total_testing_images"]

    # Identify any misclassifications for analysis
    misclassified = df_results[~df_results["is_correct"]]
    misclassified_count = len(misclassified)

    report_lines = [
        "# Held-Out Testing Dataset: Diseases & Evaluation Metrics Report",
        "",
        f"**Project:** Intelligent Crop Disease Detection, Localization and Severity Grading Using Vision Transformers  ",
        f"**Evaluation Scope:** Complete held-out testing collection of **{total_imgs} images** across **42 classes** and **15 crops**  ",
        f"**Generated:** {summary_metrics['evaluation_timestamp']}  ",
        f"**Execution Hardware:** {summary_metrics.get('device', 'CPU / CUDA')} | Avg Throughput: **{avg_lat:.1f} ms / image**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Metric KPIs",
        "",
        "| Metric Indicator | Held-Out Value | Description & Clinical Utility |",
        "| :--- | :--- | :--- |",
        f"| **Top-1 Classification Accuracy** | **{top1_acc:.2f}%** | Primary diagnostic prediction matches true disease exactly |",
        f"| **Top-3 Classification Accuracy** | **{top3_acc:.2f}%** | True disease present within model's top 3 differential diagnoses |",
        f"| **Macro-Averaged F1-Score** | **{macro_f1:.2f}%** | Unweighted harmonic mean across all 42 balanced disease classes |",
        f"| **Macro-Averaged Precision** | **{macro_prec:.2f}%** | True positive proportion over all predicted positives |",
        f"| **Macro-Averaged Recall** | **{macro_rec:.2f}%** | Sensitivity / detection rate across all disease categories |",
        f"| **YOLO Leaf Localization Rate** | **{loc_rate:.2f}%** | Successful foliar bounding box extraction (vs. fallback full-frame) |",
        f"| **Average Inference Latency** | **{avg_lat:.1f} ms** | End-to-end pipeline speed per leaf image |",
        "",
        "---",
        "",
        "## 2. Accuracy Breakdown by Dataset Source",
        "",
        "| Dataset Source | Modality / Environment | Total Images | Correct | Accuracy |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for _, r in df_source.iterrows():
        src = r["source_dataset"]
        tot = r["total_images"]
        cor = r["correct"]
        acc = r["accuracy_pct"]
        env = "Laboratory Controlled Foliage" if "Village" in src else ("In-the-Wild Field Foliage" if "Doc" in src else "Specialized Orchard Pear Foliage")
        report_lines.append(f"| **{src}** | {env} | {tot} | {cor} | **{acc:.2f}%** |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Crop-by-Crop Performance Breakdown (All 15 Crops)",
        "",
        "| Crop Species | Total Test Images | Correct Diagnoses | Accuracy (%) | Avg Diagnostic Confidence (%) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, r in df_crop.iterrows():
        c_name = r["crop"]
        tot = r["total_images"]
        cor = r["correct"]
        acc = r["accuracy_pct"]
        conf = r["avg_confidence_pct"]
        report_lines.append(f"| **{c_name}** | {tot} | {cor} | **{acc:.2f}%** | {conf:.2f}% |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Stored Artifact Locations",
        "",
        "All data and metrics have been compiled and persisted in standard spreadsheet and machine-readable formats:",
        "",
        "1. **Per-Image Predictions Spreadsheet (CSV):**  ",
        "   - `results/testing_data_metrics/testing_data_predictions.csv`  ",
        "   - `testing_data/predictions_and_metrics.csv`  ",
        "   *(Contains row-by-row columns: filename, ground truth disease, predicted disease, confidence, Top-1 match, Top-3 match, severity tier, continuous lesion %, bounding boxes, latency)*",
        "",
        "2. **Detailed Per-Image Metadata (JSON):**  ",
        "   - `results/testing_data_metrics/testing_data_predictions.json`",
        "",
        "3. **Aggregate Summary Metrics (JSON):**  ",
        "   - `results/testing_data_metrics/summary_metrics.json`",
        "",
        "4. **Per-Class Metrics Breakdown (CSV):**  ",
        "   - `results/testing_data_metrics/class_performance.csv`",
        "",
        "5. **High-Resolution Confusion Matrix Plot (PNG & CSV):**  ",
        "   - `results/testing_data_metrics/confusion_matrix.png`  ",
        "   - `results/testing_data_metrics/confusion_matrix.csv`",
        "",
        "6. **Visual Annotated Sample Predictions:**  ",
        "   - `results/testing_data_metrics/annotated_samples/`",
        "",
    ])

    if misclassified_count > 0:
        report_lines.extend([
            "---",
            "",
            f"## 5. Misclassification Analysis ({misclassified_count} images out of {total_imgs})",
            "",
            "| Image Filename | Source | Ground Truth Disease | Predicted Disease | Top-1 Conf | Top-3 Match |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for _, mr in misclassified.iterrows():
            report_lines.append(
                f"| `{mr['filename']}` | {mr['source_dataset']} | {mr['ground_truth_class']} | {mr['predicted_class']} | {mr['confidence_percentage']} | {mr['top3_match']} |"
            )
        report_lines.append("")

    report_content = "\n".join(report_lines)

    # Save to main report path
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved primary evaluation report: {output_report_path}")

    # Save to testing_data quick summary path
    with open(testing_data_summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved testing_data summary report: {testing_data_summary_path}")


if __name__ == "__main__":
    evaluate_testing_dataset()
