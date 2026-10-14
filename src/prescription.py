"""Final Phase — Agronomic Treatment & Chemical Prescription Module.
Provides verified chemical dosages, biological controls, and intervention
schedules based on identified pathogen and exact mathematical severity percentage.
"""

from typing import Dict, Any, Optional


# Agronomic knowledge base indexed by disease pattern
AGRONOMIC_DATABASE = {
    "early_blight": {
        "chemical_treatment": "Chlorothalonil 75% WP or Mancozeb 75% WG",
        "dosage": "2.0 – 2.5 g per liter of water",
        "biological_control": "Trichoderma harzianum or Bacillus subtilis foliar spray",
        "cultural_measures": "Remove lower infected foliage; practice drip irrigation to keep canopy dry.",
        "application_schedule": "Apply immediately; repeat at 7–10 day intervals until blight halts.",
    },
    "late_blight": {
        "chemical_treatment": "Metalaxyl-M + Mancozeb (Ridomil Gold) or Cymoxanil",
        "dosage": "2.5 g per liter of water (Systemic + Contact)",
        "biological_control": "Copper hydroxide (Bordeaux mixture 1%) preventative spray",
        "cultural_measures": "Rogue out infected plant debris immediately; avoid excessive nitrogen fertilization.",
        "application_schedule": "Emergency treatment within 24 hours; repeat every 5–7 days under cool, wet conditions.",
    },
    "leaf_mold": {
        "chemical_treatment": "Difenoconazole 25% EC or Copper Oxychloride 50% WP",
        "dosage": "1.0 ml / 2.5 g per liter of water",
        "biological_control": "Potassium bicarbonate spray (3 g/L)",
        "cultural_measures": "Improve greenhouse ventilation; lower relative humidity below 85%.",
        "application_schedule": "Apply every 10–14 days at early symptom onset.",
    },
    "septoria": {
        "chemical_treatment": "Azoxystrobin 23% SC or Chlorothalonil 75% WP",
        "dosage": "1.0 ml / 2.0 g per liter of water",
        "biological_control": "Neem oil extract (1500 ppm) combined with sulfur dust",
        "cultural_measures": "Mulch around plant base to prevent soil splash onto lower foliage.",
        "application_schedule": "Apply every 7–10 days; ensure thorough coverage of both leaf surfaces.",
    },
    "black_rot": {
        "chemical_treatment": "Myclobutanil 10% WP or Mancozeb 75% WG",
        "dosage": "1.5 – 2.0 g per liter of water",
        "biological_control": "Copper octanoate (Copper soap)",
        "cultural_measures": "Prune mummified fruit and diseased vines during dormancy; burn pruned residue.",
        "application_schedule": "Apply pre-bloom through post-bloom at 10–14 day intervals.",
    },
    "apple_scab": {
        "chemical_treatment": "Captan 50% WP or Difenoconazole 25% EC",
        "dosage": "2.5 g / 0.5 ml per liter of water",
        "biological_control": "Liquid lime sulfur (dormant spray) or Bacillus amyloliquefaciens",
        "cultural_measures": "Rake and destroy fallen leaves in autumn; prune canopies to maximize airflow.",
        "application_schedule": "Spray starting at green tip stage; repeat after prolonged wetting periods.",
    },
    "cedar_apple_rust": {
        "chemical_treatment": "Myclobutanil 10% WP or Propiconazole 25% EC",
        "dosage": "1.5 g / 1.0 ml per liter of water",
        "biological_control": "Sulfur-based protective dust",
        "cultural_measures": "Eradicate red cedar / juniper trees within 1 km radius if feasible.",
        "application_schedule": "Apply 3–4 sprays beginning at pink bud stage through petal fall.",
    },
    "powdery_mildew": {
        "chemical_treatment": "Tebuconazole 25% WG or Hexaconazole 5% SC",
        "dosage": "1.0 g / 2.0 ml per liter of water",
        "biological_control": "Potassium bicarbonate (3 g/L) or Wettable Sulfur 80% WP",
        "cultural_measures": "Avoid overhead watering; prune dense foliage to increase sunlight penetration.",
        "application_schedule": "Spray at first white fungal patch; repeat every 7–10 days.",
    },
    "bacterial_spot": {
        "chemical_treatment": "Copper Oxychloride 50% WP + Streptomycin Sulfate (Plantomycin)",
        "dosage": "2.5 g + 0.1 g per liter of water",
        "biological_control": "Bacillus subtilis strain QST 713 (Serenade ASO)",
        "cultural_measures": "Use certified pathogen-free seed; avoid field operations when leaves are wet.",
        "application_schedule": "Apply preventative bactericide every 5–7 days during warm rainy spells.",
    },
    "spider_mites": {
        "chemical_treatment": "Abamectin 1.9% EC or Spiromesifen 22.9% SC",
        "dosage": "0.5 – 1.0 ml per liter of water",
        "biological_control": "Predatory mites (Phytoseiulus persimilis) or Horticultural oil 1%",
        "cultural_measures": "Maintain adequate field irrigation; wash off webbing with high-pressure water.",
        "application_schedule": "Spray twice at 5-day intervals to target freshly hatched nymphs.",
    },
    "yellow_leaf_curl": {
        "chemical_treatment": "Vector Control: Imidacloprid 17.8% SL or Thiamethoxam 25% WG",
        "dosage": "0.5 ml / 0.3 g per liter of water (targets Whitefly vector)",
        "biological_control": "Yellow sticky traps (20 per acre) + Beauveria bassiana entomopathogen",
        "cultural_measures": "Install 50-mesh insect-proof netting in nursery; eradicate volunteer weed hosts.",
        "application_schedule": "Control whitefly population aggressively; rogue infected plants immediately.",
    },
    "mosaic_virus": {
        "chemical_treatment": "No curative chemical exists (Viral pathogen). Control aphid vectors using Acetamiprid 20% SP.",
        "dosage": "0.4 g per liter of water",
        "biological_control": "Neem seed kernel extract (NSKE 5%)",
        "cultural_measures": "Sanitize hands and pruning tools with 10% trisodium phosphate (TSP); do not smoke near crops.",
        "application_schedule": "Immediate rogueing and destruction of symptomatic plants.",
    },
    "common_rust": {
        "chemical_treatment": "Azoxystrobin + Difenoconazole or Mancozeb 75% WP",
        "dosage": "1.0 ml / 2.5 g per liter of water",
        "biological_control": "Trichoderma viride foliar treatment",
        "cultural_measures": "Plant rust-resistant hybrid seed; avoid late planting dates.",
        "application_schedule": "Apply when pustules first appear on lower canopy leaves.",
    },
    "healthy": {
        "chemical_treatment": "None required (Foliage is clinically healthy).",
        "dosage": "N/A",
        "biological_control": "Preventative bio-stimulant or Seaweed extract (2 ml/L)",
        "cultural_measures": "Continue balanced N-P-K fertigation and scheduled soil moisture checks.",
        "application_schedule": "Standard agronomic monitoring routine.",
    },
}


def get_agronomic_prescription(
    crop: str,
    disease: str,
    severity_pct: float,
    raw_class: str,
) -> Dict[str, Any]:
    """Retrieve actionable chemical prescription and intervention schedule.
    
    Args:
        crop: Crop species name (e.g. Tomato, Apple)
        disease: Human-readable disease condition
        severity_pct: Continuous infection % from segmentation
        raw_class: Raw canonical class string
        
    Returns:
        Prescription metadata dictionary.
    """
    is_healthy = "healthy" in raw_class.lower()

    if is_healthy:
        return {
            "urgency_level": "Routine Monitoring (Healthy)",
            "urgency_color": "#2E7D32",
            "chemical_treatment": "None needed. Crop exhibits vigorous, disease-free foliar tissue.",
            "dosage": "N/A",
            "biological_control": "Preventative seaweed extract / Trichoderma soil drench.",
            "cultural_measures": "Maintain scheduled irrigation and balanced nitrogen-potassium ratio.",
            "application_schedule": "Re-inspect canopy in 14 days.",
            "is_emergency": False,
        }

    # Match key in database
    d_clean = disease.lower().replace(" ", "_")
    matched_entry = None

    for key, val in AGRONOMIC_DATABASE.items():
        if key in d_clean or key in raw_class.lower():
            matched_entry = val
            break

    if matched_entry is None:
        # Default broad-spectrum recommendation
        matched_entry = {
            "chemical_treatment": "Broad-Spectrum Protectant: Mancozeb 75% WG or Copper Oxychloride 50% WP",
            "dosage": "2.0 – 2.5 g per liter of water",
            "biological_control": "Neem oil 1500 ppm foliar spray",
            "cultural_measures": "Prune symptomatic leaves; sanitize field implements.",
            "application_schedule": "Apply promptly; repeat at 10-day intervals if symptoms spread.",
        }

    # Determine urgency based on continuous severity %
    if severity_pct >= 35.0:
        urgency = "🚨 CRITICAL INTERVENTION (>35% Foliar Damage)"
        urgency_color = "#D32F2F"
        is_emergency = True
        schedule_override = "Immediate chemical application required within 12–24 hours to prevent total canopy defoliation."
    elif severity_pct >= 15.0:
        urgency = "⚠️ ACTIVE TREATMENT (15%–35% Moderate Blight)"
        urgency_color = "#F57C00"
        is_emergency = False
        schedule_override = matched_entry["application_schedule"]
    elif severity_pct >= 5.0:
        urgency = "🟡 EARLY OUTBREAK (5%–15% Low Infection)"
        urgency_color = "#FBC02D"
        is_emergency = False
        schedule_override = matched_entry["application_schedule"]
    else:
        urgency = "🟢 INCIPIENT / TRACE (<5% Initial Lesions)"
        urgency_color = "#388E3C"
        is_emergency = False
        schedule_override = "Preventative treatment: apply bio-fungicide or light protective spray."

    return {
        "urgency_level": urgency,
        "urgency_color": urgency_color,
        "chemical_treatment": matched_entry["chemical_treatment"],
        "dosage": matched_entry["dosage"],
        "biological_control": matched_entry["biological_control"],
        "cultural_measures": matched_entry["cultural_measures"],
        "application_schedule": schedule_override if severity_pct >= 35.0 else matched_entry["application_schedule"],
        "is_emergency": is_emergency,
    }
