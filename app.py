"""Crop Disease Detection using Vision Transformers & Crop Health Models — Final Review (100% Phase).
Full Agronomic Decision Support & Folio-Pathology System.
Pipeline:
  1. Hierarchical Localization: YOLOv8 Leaf Boundary + Discrete Disease Lesion Boxes
  2. ViT Attention Rollout: Self-Attention Heatmap Token Explainability
  3. Dual-Mask Segmentation: Leaf Area (L_mask) vs. Necrotic Symptom (D_mask)
  4. Dual-Track Severity: Discrete DL Classifier + Continuous Mathematical Surface Area Formula (D_mask / L_mask * 100%)
  5. Agronomic Prescriptions: Dosage-specific Chemical & Biological Fungicide Schedules
"""

import json
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO

from src.config import (
    BASE_DIR,
    VIT_MODEL_PATH,
    LOCALIZATION_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    CROP_HEALTH_WEIGHTS_PATH,
    CROP_HEALTH_META_PATH,
    PLANTVILLAGE_DIR,
    PREDICTIONS_DIR,
    VIT_CLASSES,
    get_device,
)
from src.inference import (
    load_vit_model,
    load_severity_model,
    get_crop_health_model,
    run_pipeline,
    visualize_result,
)

st.set_page_config(
    page_title="Crop Pathology & Decision Support — Final Review (100% Phase)",
    page_icon="🌿",
    layout="wide",
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #424242;
        margin-bottom: 1.0rem;
    }
    .engine-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 12px;
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .engine-vit {
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #90caf9;
    }
    .engine-ch {
        background-color: #e8f5e9;
        color: #1b5e20;
        border: 1px solid #a5d6a7;
    }
    .engine-hybrid {
        background-color: #f3e5f5;
        color: #4a148c;
        border: 1px solid #ce93d8;
    }
    .engine-compare {
        background-color: #fff8e1;
        color: #f57f17;
        border: 1px solid #ffe082;
    }
    .metric-card {
        background-color: #fbfbfb;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .rx-card {
        background: linear-gradient(135deg, #f1f8e9 0%, #ffffff 100%);
        border: 1.5px solid #81c784;
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
    }
    .rx-urgency-high {
        color: #c62828;
        font-weight: 700;
        background-color: #ffebee;
        padding: 3px 10px;
        border-radius: 8px;
        display: inline-block;
    }
    .rx-urgency-mod {
        color: #ef6c00;
        font-weight: 700;
        background-color: #fff3e0;
        padding: 3px 10px;
        border-radius: 8px;
        display: inline-block;
    }
    .rx-urgency-low {
        color: #2e7d32;
        font-weight: 700;
        background-color: #e8f5e9;
        padding: 3px 10px;
        border-radius: 8px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 Precision Crop Pathology & Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><strong>Final Review (100% Phase)</strong> • Hierarchical Disease Bounding Boxes • ViT-B/16 Self-Attention Rollout • Dual-Mask Segmentation • Continuous Severity • Diagnosis, Symptoms & Precautions</div>', unsafe_allow_html=True)

# Sidebar configurations
with st.sidebar:
    st.header("⚙️ System Configuration")
    device = get_device()
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    st.info(f"**Compute Device:** {device.type.upper()} ({dev_name})")

    st.markdown("### 🧠 Classification Engine")
    engine_selection = st.radio(
        "Diagnostic Model Engine",
        options=[
            "Vision Transformer ViT-B/16 (Unified Multi-Dataset: 42 Classes / 15 Crops)",
            "Crop Health ResNet-18 (38 Classes / 14 Crops)",
            "Smart Hybrid Cascade (ViT with ResNet-18 Fallback)",
            "Side-by-Side Comparison (ViT vs. ResNet-18)",
        ],
        index=0,
        help="ViT is trained across all 3 project datasets (PlantVillage + PlantDoc field leaves + DiaMOS Pear) covering 15 crops and 42 classes."
    )

    engine_map = {
        "Vision Transformer ViT-B/16 (Unified Multi-Dataset: 42 Classes / 15 Crops)": "vit",
        "Crop Health ResNet-18 (38 Classes / 14 Crops)": "crop_health",
        "Smart Hybrid Cascade (ViT with ResNet-18 Fallback)": "hybrid",
        "Side-by-Side Comparison (ViT vs. ResNet-18)": "compare",
    }
    chosen_classifier_mode = engine_map[engine_selection]

    st.markdown("### 🌾 Crop Species Filter (Optional)")
    crop_options = [
        "Auto-Detect / All 15 Crops",
        "Apple",
        "Blueberry",
        "Cherry",
        "Corn",
        "Grape",
        "Orange",
        "Peach",
        "Pear",
        "Pepper (Bell)",
        "Potato",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry",
        "Tomato",
    ]
    selected_crop_choice = st.selectbox(
        "Target Crop Species",
        crop_options,
        index=0,
        help="If the plant type is known, filtering renormalizes predictions specifically over diseases for that crop."
    )
    crop_filter_value = None
    if selected_crop_choice != "Auto-Detect / All 15 Crops":
        crop_filter_value = selected_crop_choice.replace(" (Bell)", "")

    st.markdown("---")
    st.markdown("### 🎯 Cropping & Localization Strategy")
    crop_mode_selection = st.radio(
        "Leaf Region Mode",
        options=["Smart Padded Region (Recommended)", "Full Leaf Image (Direct)", "Raw Tight YOLO Box"],
        index=0,
        help="Smart mode merges leaf detection boxes and applies 15% context padding so lesion rings and leaf margins are not clipped."
    )

    crop_mode_map = {
        "Smart Padded Region (Recommended)": "smart",
        "Full Leaf Image (Direct)": "full",
        "Raw Tight YOLO Box": "raw",
    }
    chosen_crop_mode = crop_mode_map[crop_mode_selection]

    conf_thresh = st.slider(
        "YOLO Detection Confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.20,
        step=0.05,
        help="Detection confidence threshold for leaf localization."
    )

    st.markdown("---")
    st.markdown("### 🔬 Final Review Deliverables")
    st.markdown("""
    - ✅ **Hierarchical Localization**: Green Outer Leaf Box + Red/Orange Inner Disease Bounding Boxes
    - ✅ **ViT Attention Rollout**: Multi-head self-attention token maps explaining diagnostic attribution
    - ✅ **Dual-Mask Segmentation**: Pixel-accurate $L_{\\text{mask}}$ (leaf) & $D_{\\text{mask}}$ (disease area)
    - ✅ **Continuous Severity Formula**: $\\text{Severity \\%} = \\frac{\\sum D_{\\text{mask}}}{\\sum L_{\\text{mask}}} \\times 100\\%$
    - ✅ **Diagnosis, Symptoms & Precautions**: Scientific causal organism, pathogen profile, precautions & cures
    """)

    st.markdown("### Model Checkpoints")
    yolo_input = st.text_input("YOLO Localization Model", str(LOCALIZATION_MODEL_PATH))
    vit_input = st.text_input("ViT Disease Classifier", str(VIT_MODEL_PATH))
    sev_input = st.text_input("Severity Classifier", str(SEVERITY_MODEL_PATH))
    ch_input = st.text_input("Crop Health 38-Class Model", str(CROP_HEALTH_WEIGHTS_PATH))


@st.cache_resource
def get_loaded_models(yolo_p: str, vit_p: str, sev_p: str, ch_p: str):
    dev = get_device()
    yolo = YOLO(yolo_p)
    vit, vit_tfm, vit_classes = load_vit_model(Path(vit_p), dev)
    sev, sev_tfm, sev_classes = load_severity_model(Path(sev_p), dev)
    ch_model = get_crop_health_model(weights_path=Path(ch_p), device=dev)
    return dev, yolo, vit, vit_tfm, vit_classes, sev, sev_tfm, sev_classes, ch_model


# Input Selection: Upload file or pick verified sample
st.subheader("1. Input Foliar Specimen")
col_input1, col_input2 = st.columns([1, 1])

uploaded_file = None
selected_sample = None

with col_input1:
    uploaded_file = st.file_uploader(
        "Upload a plant or field leaf image (JPG/PNG):",
        type=["jpg", "jpeg", "png"],
    )

with col_input2:
    sample_candidates = []
    pv_test_dir = PLANTVILLAGE_DIR / "test"
    if pv_test_dir.exists():
        for cdir in sorted(pv_test_dir.iterdir()):
            if cdir.is_dir():
                imgs = sorted(list(cdir.glob("*.jpg")) + list(cdir.glob("*.JPG")))
                if imgs:
                    sample_candidates.append(imgs[0])
    test_suite_dir = BASE_DIR / "testing_data"
    if test_suite_dir.exists() and not sample_candidates:
        sample_candidates.extend(sorted(list(test_suite_dir.glob("*.jpg")) + list(test_suite_dir.glob("*.JPG")))[:42])
    e2e_dir = PREDICTIONS_DIR / "e2e_examples"
    if e2e_dir.exists():
        sample_candidates.extend(sorted(list(e2e_dir.glob("*.jpg")))[:6])

    if sample_candidates:
        options = ["None (Use upload)"] + [f"{p.parent.name}/{p.name}" for p in sample_candidates]
        chosen_sample_name = st.selectbox("Or choose a verified benchmark sample:", options)
        if chosen_sample_name != "None (Use upload)":
            idx = options.index(chosen_sample_name) - 1
            selected_sample = sample_candidates[idx]

input_image = None
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
        st.stop()
elif selected_sample is not None:
    try:
        input_image = Image.open(selected_sample).convert("RGB")
        st.caption(f"Loaded verified sample: `{selected_sample.name}`")
    except Exception as e:
        st.error(f"Error loading sample image: {e}")
        st.stop()

if input_image is not None:
    yolo_path = Path(yolo_input)
    vit_path = Path(vit_input)
    sev_path = Path(sev_input)
    ch_path = Path(ch_input)

    # Validate model paths exist
    missing_models = []
    if not yolo_path.exists():
        missing_models.append(f"YOLO ({yolo_path})")
    if not vit_path.exists() and chosen_classifier_mode in ("vit", "hybrid", "compare"):
        missing_models.append(f"ViT ({vit_path})")
    if not sev_path.exists():
        missing_models.append(f"Severity ({sev_path})")
    if not ch_path.exists() and chosen_classifier_mode in ("crop_health", "hybrid", "compare"):
        missing_models.append(f"Crop Health ({ch_path})")

    if missing_models:
        st.warning(f"Models not found: {', '.join(missing_models)}. Please verify paths.")
        st.stop()

    with st.spinner("Executing End-to-End Foliar Pathology Pipeline (Localization → ViT Attention → Segmentation → Severity → Prescription)..."):
        dev, yolo, vit, vit_tfm, vit_classes, sev, sev_tfm, sev_classes, ch_model = get_loaded_models(
            str(yolo_path), str(vit_path), str(sev_path), str(ch_path)
        )
        result = run_pipeline(
            input_image,
            yolo_model=yolo,
            vit_model=vit,
            vit_tfm=vit_tfm,
            vit_classes=vit_classes,
            sev_model=sev,
            sev_tfm=sev_tfm,
            sev_classes=sev_classes,
            crop_health_model=ch_model,
            device=dev,
            conf_thresh=conf_thresh,
            crop_mode=chosen_crop_mode,
            classifier_mode=chosen_classifier_mode,
            crop_filter=crop_filter_value,
        )

    # Low-confidence / Out-of-domain warning alert
    if result["is_low_confidence"]:
        st.warning(f"⚠️ **Attention**: {result['confidence_warning']}")

    if result.get("hybrid_cascade_triggered"):
        st.info("🔄 **Smart Cascade Triggered**: Vision Transformer exhibited low confidence (<65%), so diagnosis autonomously escalated to the 38-class Crop Health engine.")

    # Section 2: Visual Inspection & Hierarchical Localization Gallery
    st.markdown("---")
    st.subheader("2. Visual Analytics & Hierarchical Localization Gallery")

    gal_c1, gal_c2, gal_c3, gal_c4 = st.columns(4)

    with gal_c1:
        st.markdown("##### 📷 1. Raw Specimen")
        st.image(input_image, use_container_width=True, caption=f"Resolution: {input_image.width}×{input_image.height} px")

    with gal_c2:
        st.markdown("##### 🎯 2. Hierarchical Localization")
        hier_img = result.get("full_hierarchical_image", result.get("annotated_image", input_image))
        n_spots = len(result.get("disease_boxes", []))
        d_name_display = result["disease"]
        st.image(
            hier_img,
            use_container_width=True,
            caption=f"Green Box: Leaf Boundary | Red Boxes: {n_spots} {d_name_display} Spot(s)"
        )

    with gal_c3:
        st.markdown("##### 🌟 3. ViT Attention Rollout")
        hm_img = result.get("heatmap_image", result.get("cropped_image", input_image))
        st.image(
            hm_img,
            use_container_width=True,
            caption="ViT-B/16 Self-Attention Token Attribution"
        )

    with gal_c4:
        st.markdown("##### 🔬 4. Dual-Mask Segmentation")
        seg_img = result.get("dual_mask_image", result.get("cropped_image", input_image))
        st.image(
            seg_img,
            use_container_width=True,
            caption=f"Green: Healthy Tissue (L_mask) | Red: {d_name_display} Area (D_mask)"
        )

    # Telemetry metrics row
    st.markdown("#### 📊 Localization & Foliar Telemetry")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.metric("Leaf Regions", f"{result['total_boxes_found']} leaf box(es)")
    with t2:
        peak_c = result.get("max_detection_confidence", 0.0)
        st.metric("Leaf Detector Conf", f"{peak_c*100:.1f}%" if peak_c > 0 else "N/A")
    with t3:
        st.metric(f"Disease Spots ({d_name_display})", f"{n_spots} detected")
    with t4:
        bbox = result.get("bounding_box", [0, 0, input_image.width, input_image.height])
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        coverage = (bw * bh) / max(1, input_image.width * input_image.height) * 100
        st.metric("Leaf Frame Coverage", f"{coverage:.1f}%")
    with t5:
        cont_sev = result.get("continuous_severity_pct", 0.0)
        st.metric("Mathematical Necrosis", f"{cont_sev:.2f}%")

    if result.get("disease_boxes"):
        with st.expander(f"🔍 View Detected {d_name_display} Bounding Box Coordinates & Regions", expanded=False):
            st.json(result["disease_boxes"])

    # Section 3: Diagnostic Findings & Severity
    st.markdown("---")
    st.subheader("3. Multi-Engine Disease Diagnosis & Dual-Track Severity")

    badge_cls = "engine-vit" if chosen_classifier_mode == "vit" else (
        "engine-ch" if chosen_classifier_mode == "crop_health" else (
            "engine-hybrid" if chosen_classifier_mode == "hybrid" else "engine-compare"
        )
    )
    st.markdown(f'<span class="engine-badge {badge_cls}">Active Engine: {result["classifier_name"]}</span>', unsafe_allow_html=True)
    if result.get("crop_filter"):
        st.caption(f"🎯 Applied Crop Filter: **{result['crop_filter']}**")

    # Comparative view vs Standard view
    if chosen_classifier_mode == "compare" and result.get("comparison"):
        st.markdown("### ⚖️ Side-by-Side Model Benchmark Comparison")
        comp = result["comparison"]
        if comp["agreement"]:
            st.success("🎯 **Model Consensus**: Both Vision Transformer and ResNet-18 agreed on the exact diagnosis!")
        else:
            st.info("⚖️ **Comparative Divergence**: Models produced different top classifications (analyzed below).")

        col_comp1, col_comp2 = st.columns([1, 1])
        with col_comp1:
            st.markdown("#### 🌟 Vision Transformer (ViT-B/16)")
            vit_r = result["vit_result"]
            st.metric(
                label="ViT Primary Diagnosis",
                value=vit_r["disease"],
                delta=f"{vit_r['confidence']*100:.2f}% Conf",
            )
            st.write(f"**Crop:** {vit_r['crop']} | **Class:** `{vit_r['disease_raw']}`")
            for c in vit_r["top_predictions"][:3]:
                c_lbl = c["class"].replace("___", " - ").replace("_", " ")
                st.write(f"• **{c_lbl}**: {c['confidence']*100:.1f}%")
                st.progress(float(min(1.0, max(0.0, c["confidence"]))))

        with col_comp2:
            st.markdown("#### 🌾 Crop Health Model (ResNet-18)")
            ch_r = result["crop_health_result"]
            st.metric(
                label="ResNet-18 Diagnosis",
                value=ch_r["disease"],
                delta=f"{ch_r['confidence']*100:.2f}% Conf",
            )
            st.write(f"**Crop:** {ch_r['crop']} | **Class:** `{ch_r['disease_raw']}`")
            for c in ch_r["top_predictions"][:3]:
                c_lbl = c["class"].replace("___", " - ").replace("_", " ")
                st.write(f"• **{c_lbl}**: {c['confidence']*100:.1f}%")
                st.progress(float(min(1.0, max(0.0, c["confidence"]))))

    else:
        # Standard Diagnosis Display
        col_diag1, col_diag2 = st.columns([1.2, 1.0])
        with col_diag1:
            st.markdown("#### 🔬 Folio-Pathology Diagnosis")
            st.metric(
                label="Identified Foliar Condition",
                value=result["disease"],
                delta=f"{result['disease_confidence']*100:.2f}% Diagnostic Confidence",
            )
            st.write(f"**Crop Species:** {result['crop']}")
            st.write(f"**Target Class:** `{result['disease_raw']}`")

            st.markdown("##### Ranked Predictions:")
            for cand in result["top3_diseases"]:
                c_name = cand.get("class", cand.get("disease", "Unknown")).replace("___", " - ").replace("_", " ")
                c_conf = cand["confidence"]
                st.write(f"**{c_name}**: {c_conf*100:.1f}%")
                st.progress(float(min(1.0, max(0.0, c_conf))))

        with col_diag2:
            st.markdown("#### 🌡️ Dual-Track Severity Assessment")
            sev_card_c1, sev_card_c2 = st.columns(2)
            with sev_card_c1:
                st.markdown("**Track A: Neural Classifier**")
                st.metric("DL Severity Tier", result["severity"], f"{result['severity_confidence']*100:.1f}% Conf")
            with sev_card_c2:
                st.markdown("**Track B: Continuous Formula**")
                st.metric("Necrotic Surface %", f"{cont_sev:.2f}%", result.get("agronomic_tier", "N/A"))

            m_data = result.get("mathematical_severity", {})
            st.caption(
                f"Formula: `Severity % = (D_mask / L_mask) × 100` | "
                f"Leaf Pixels: {m_data.get('leaf_pixels', 0):,} | Lesion Pixels: {m_data.get('lesion_pixels', 0):,}"
            )

            st.markdown("##### Neural Severity Distribution:")
            for s_cand in result["top3_severities"]:
                s_name = s_cand["severity"]
                s_conf = s_cand["confidence"]
                st.write(f"• **{s_name}**: {s_conf*100:.1f}%")
                st.progress(float(min(1.0, max(0.0, s_conf))))

    # Section 4: Diagnosis, Symptoms and Precautions
    st.markdown("---")
    st.subheader("4. Diagnosis, Symptoms and Precautions")

    dk = result.get("disease_knowledge", {})
    xai_title = dk.get("title", f"{result['crop']} {result['disease']}")
    pathogen_type = dk.get("pathogen_type", "Foliar Condition")
    causal_agent = dk.get("causal_organism", "N/A")
    precautions_list = dk.get("precautions", [])
    curative_list = dk.get("curative_measures", [])
    favorable_env = dk.get("favorable_conditions", "N/A")

    # XAI Diagnostic Card
    st.markdown(f"""
    <div style="background-color: #f1f8e9; border: 1.5px solid #a5d6a7; border-radius: 10px; padding: 16px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #1b5e20;">🔬 {xai_title}</span>
            <span style="background-color: #e8f5e9; color: #2e7d32; font-weight: 600; padding: 3px 10px; border-radius: 8px; border: 1px solid #c8e6c9;">
                Class: {pathogen_type}
            </span>
        </div>
        <p style="margin: 0 0 6px 0; color: #333; font-size: 0.95rem;">
            <strong>Causal Organism:</strong> <em>{causal_agent}</em>
        </p>
        <p style="margin: 0; color: #555; font-size: 0.92rem;">
            <strong>🌡️ Favorable Environmental Conditions:</strong> {favorable_env}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Precautions vs. Cure Side-by-Side Display
    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.markdown("##### 🛡️ Common Precautions (Prevention)")
        st.caption("Foliar hygiene, sanitation, and pre-infection agricultural controls:")
        if precautions_list:
            for p_idx, prec in enumerate(precautions_list, 1):
                st.write(f"**{p_idx}.** {prec}")
        else:
            st.write("• Inspect foliar canopy regularly under adequate lighting.")

    with xai_col2:
        st.markdown("##### 💊 Curative Measures & Remedial Actions (Cure)")
        st.caption("Immediate interventions, chemical cures, and biological applications:")
        if curative_list:
            for c_idx, cure in enumerate(curative_list, 1):
                st.write(f"**{c_idx}.** {cure}")
        else:
            st.write("• Apply recommended broad-spectrum protective fungicides.")

else:
    st.info("👆 Upload an image of a plant leaf or choose a verified benchmark sample above to execute the Final Review 100% Pipeline.")
