"""Hard-coded Agronomic Knowledge Base & Explainable AI (XAI) Pathology Descriptions.
Comprehensive pathological profiles for all 42 crop-disease classes across 15 crop species:
  - Scientific Causal Organism & Pathogen Classification
  - Visual Symptom Rationale (Explaining ViT Attention & Feature Localization)
  - Precautionary Measures (Foliar Sanitation, Irrigation, Cultural Prevention)
  - Curative Interventions (Chemical Fungicides/Bactericides, Bio-controls, Cultural Cures)
  - Environmental Disease Triggers (Temperature & Humidity Ranges)
"""

from typing import Dict, Any, Optional

DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    # ------------------ APPLE ------------------
    "Apple___Apple_scab": {
        "title": "Apple Scab",
        "causal_organism": "Venturia inaequalis (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The Vision Transformer self-attention heads localized olive-green to dull olive-brown velvety, "
            "circular lesions with indistinct feathery margins on the leaf cuticle. As the infection progresses, "
            "lesions become raised, dark olive-black, and corky, causing leaf distortion and premature foliar drop."
        ),
        "precautions": [
            "Select scab-resistant apple cultivars (e.g., Liberty, Prima, Freedom, Enterprise).",
            "Prune canopy aggressively during winter dormancy to promote rapid leaf drying after rain.",
            "Rake, chop, or burn fallen leaves in autumn to eliminate overwintering fungal pseudothecia in leaf litter.",
            "Avoid overhead irrigation; ensure foliage remains dry during early morning and evening hours.",
        ],
        "curative_measures": [
            "Apply Captan 50% WP (2.5 g/L) or Difenoconazole 25% EC (0.5 ml/L) at green tip through petal fall.",
            "Use protective liquid lime sulfur (2–3 ml/L) or copper soap during dormant and delayed-dormant stages.",
            "Alternate fungicide biochemical modes of action (FRAC 3 and FRAC 9) to prevent strain resistance.",
            "Prune heavily infected spurs immediately to arrest secondary conidial spread.",
        ],
        "favorable_conditions": "Cool, moist spring weather with temperatures between 13°C and 24°C and leaf wetness > 6 hours.",
    },
    "Apple___Black_rot": {
        "title": "Apple Black Rot (Frogeye Leaf Spot)",
        "causal_organism": "Botryosphaeria obtusa (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The self-attention layer concentrated on characteristic 'frogeye' foliar spots: circular lesions with "
            "light tan or grey centers bordered by distinct dark purple-brown concentric rings. Minute black fungal "
            "pycnidia fruiting bodies are visible clustered within the necrotic centers."
        ),
        "precautions": [
            "Prune out all dead wood, fire-blight strikes, and cankered branches during winter pruning.",
            "Meticulously remove all mummified fruit hanging in trees or fallen on the orchard floor.",
            "Maintain tree vigor with balanced fertilization; avoid mechanical bark wounding and pruning gouges.",
            "Disinfect pruning shears in 10% bleach or 70% isopropyl alcohol between cuts.",
        ],
        "curative_measures": [
            "Spray Myclobutanil 10% WP (1.5 g/L) or Mancozeb 75% WG (2.5 g/L) from pink bud through early summer.",
            "Apply bio-fungicidal Bacillus subtilis or Copper octanoate protective sprays.",
            "Excise diseased branch cankers 15 cm below visible infection margin and treat cut with pruning sealant.",
        ],
        "favorable_conditions": "Warm, humid conditions (20°C to 27°C) with frequent showers or heavy morning fog.",
    },
    "Apple___Cedar_apple_rust": {
        "title": "Cedar Apple Rust",
        "causal_organism": "Gymnosporangium juniperi-virginianae (Heteroecious Rust Fungus)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "Attention strongly attributes to vivid bright orange-yellow circular spots (aecia) on the adaxial "
            "(upper) leaf surface. Small black fungal pycnia exude sticky nectar droplets at spot centers, followed "
            "by cylindrical tube-like aecial spore cups erupting through the abaxial leaf underside."
        ),
        "precautions": [
            "Eradicate eastern red cedar (Juniperus virginiana) trees within a 1 km radius of the orchard where feasible.",
            "Manually remove and destroy gelatinous brown-orange galls on nearby ornamental junipers in early spring.",
            "Plant rust-immune or resistant varieties (e.g., Redfree, Liberty, Freedom, Priscilla).",
        ],
        "curative_measures": [
            "Apply Myclobutanil 10% WP (1.5 g/L) or Propiconazole 25% EC (1.0 ml/L) at pink bud through petal fall.",
            "Apply protective wettable sulfur (3.0 g/L) or Mancozeb 75% WP before forecasted spring rains.",
            "Maintain scheduled protective coverage until cedar galls complete basidiospore release (mid-June).",
        ],
        "favorable_conditions": "Wet spring rains at 10°C to 24°C while cedar galls are gelatinized and releasing spores.",
    },
    "Apple___healthy": {
        "title": "Clinically Healthy Apple Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Uniform self-attention distribution across vibrant green parenchyma with high chlorophyll reflectance (ExG > 25), "
            "smooth leaf margins, intact cuticle wax, and total absence of chlorotic yellowing or necrotic lesions."
        ),
        "precautions": [
            "Continue seasonal foliar monitoring and balanced N-P-K fertigation.",
            "Maintain dormant winter sanitation sprays (horticultural mineral oil + copper).",
            "Monitor soil pH (optimal 6.0–6.8) and ensure adequate calcium-boron nutrition for cell wall resilience.",
        ],
        "curative_measures": ["No therapeutic chemical treatment needed."],
        "favorable_conditions": "Optimal orchard growing conditions.",
    },

    # ------------------ BLUEBERRY ------------------
    "Blueberry___healthy": {
        "title": "Clinically Healthy Blueberry Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Attention heads show balanced weighting across deep green, waxy, elliptical leaves with clean venation, "
            "intact cuticle integrity, and zero signs of rust, anthracnose, or Phomopsis twig blight."
        ),
        "precautions": [
            "Maintain acidic soil pH (4.5–5.2) using elemental sulfur and organic pine bark mulch.",
            "Apply drip irrigation at soil level; avoid wetting leaves during warm afternoons.",
            "Prune weak, low-lying canes annually to maximize air circulation through the bush center.",
        ],
        "curative_measures": ["No chemical treatment required."],
        "favorable_conditions": "Optimal acidic growing conditions.",
    },

    # ------------------ CHERRY ------------------
    "Cherry___Powdery_mildew": {
        "title": "Cherry Powdery Mildew",
        "causal_organism": "Podosphaera clandestina (Fungus: Erysiphales)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model localized patches of white, powdery, superficial mycelial and conidial mats across the "
            "adaxial leaf surface. Symptomatic leaves exhibit upward margin curling, puckering, chlorotic mottling, "
            "and late-season leathery necrosis."
        ),
        "precautions": [
            "Prune dense canopy shoots in late spring to increase sunlight penetration and wind movement.",
            "Avoid excessive late-season nitrogen fertilization which stimulates succulent, susceptible shoot flushes.",
            "Maintain drip irrigation rather than overhead sprinklers to prevent microclimate humidity spikes.",
        ],
        "curative_measures": [
            "Apply Tebuconazole 25% WG (1.0 g/L), Hexaconazole 5% SC (2.0 ml/L), or Trifloxystrobin 50% WG.",
            "Spray Potassium bicarbonate (3.0 g/L) or Wettable Sulfur 80% WP (2.5 g/L) as eco-friendly contact cures.",
            "Begin treatment at petal fall and repeat at 10–14 day intervals until terminal shoot growth ceases.",
        ],
        "favorable_conditions": "Warm days (18°C to 28°C) with high ambient relative humidity and shaded, stagnant airflow.",
    },
    "Cherry___healthy": {
        "title": "Clinically Healthy Cherry Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Uniform token attention across glossy, dark-green ovate leaves with serrated margins, unblemished venation, "
            "and absence of fungal mycelia or shothole lesions."
        ),
        "precautions": [
            "Apply dormant copper sprays before bud swell to suppress bacterial canker and shothole.",
            "Maintain weed-free root zones and mulch with composted organic matter.",
        ],
        "curative_measures": ["No therapeutic intervention required."],
        "favorable_conditions": "Optimal temperate orchard conditions.",
    },

    # ------------------ CORN (MAIZE) ------------------
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "title": "Corn Gray Leaf Spot",
        "causal_organism": "Cercospora zeae-maydis (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The self-attention layer isolated distinct narrow, rectangular, tan-to-gray necrotic lesions bounded "
            "strictly by parallel leaf veins. Lesions expand longitudinally into block-shaped blights that coalesce "
            "to destroy large areas of photosynthesizing corn canopy."
        ),
        "precautions": [
            "Plant corn hybrids with high genetic resistance scores against Cercospora zeae-maydis.",
            "Rotate crops with non-host species (soybeans, alfalfa, or small grains) for at least 2 consecutive seasons.",
            "Till under surface crop residue in high-risk continuous-corn systems to accelerate fungal decay.",
        ],
        "curative_measures": [
            "Apply Azoxystrobin + Propiconazole (Quilt Xcel: 1.0 ml/L) or Pyraclostrobin (Headline: 0.8 ml/L).",
            "Spray between VT (tasseling) and R2 (blister stage) when disease reaches the third leaf below the ear.",
            "Ensure full foliar coverage using high-clearance spray rigs with minimum 150 L/ha water volume.",
        ],
        "favorable_conditions": "Extended periods of high relative humidity (> 90%) and warm temperatures (25°C to 32°C).",
    },
    "Corn___Common_rust": {
        "title": "Corn Common Rust",
        "causal_organism": "Puccinia sorghi (Fungus: Pucciniales)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "Attention highlighted small, oval-to-elongate golden-brown to cinnamon-brown powdery pustules (uredinia) "
            "erupting prominently on both the upper and lower surfaces of the corn leaf blade. Pustules rupture the "
            "epidermis and release powdery cinnamon-red spores."
        ),
        "precautions": [
            "Select corn hybrids containing the Rp1-D specific resistance gene or general adult-plant resistance.",
            "Avoid late planting dates that expose young vulnerable corn to peak mid-summer airborne spore showers.",
        ],
        "curative_measures": [
            "Apply Azoxystrobin 23% SC (1.0 ml/L) or Mancozeb 75% WP (2.5 g/L) at first pustule detection.",
            "Spray economically if rust pustules cover > 5% of leaves on > 50% of plants before silking.",
        ],
        "favorable_conditions": "Moderate temperatures (16°C to 25°C) combined with high humidity (> 95%) and heavy nighttime dews.",
    },
    "Corn___Northern_Leaf_Blight": {
        "title": "Northern Corn Leaf Blight (NCLB)",
        "causal_organism": "Exserohilum turcicum / Setosphaeria turcica (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The Vision Transformer localized massive, long, elliptical 'cigar-shaped' or spindle-shaped grayish-green "
            "to tan necrotic lesions (3 to 15 cm in length). Dark olive-black fungal spores form velvety bands across "
            "lesion surfaces during humid periods."
        ),
        "precautions": [
            "Plant resistant corn hybrids carrying Ht1, Ht2, Ht3, or HtN multi-genic resistance.",
            "Rotate fields away from corn for at least 1–2 years to deplete overwintering chlamydospores in residue.",
            "Deep plow crop stubble into the soil profile where no-till is not strictly required.",
        ],
        "curative_measures": [
            "Spray Pyraclostrobin + Fluxapyroxad (Priaxor: 0.6 ml/L) or Propiconazole 25% EC (1.0 ml/L).",
            "Apply between V12 and R1 growth stages if lesions appear on the ear leaf or higher.",
        ],
        "favorable_conditions": "Moderate temperatures (18°C to 27°C) accompanied by extended dew periods of 6–18 hours.",
    },
    "Corn___healthy": {
        "title": "Clinically Healthy Corn Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Equally balanced attention across broad, deep-green, upright maize leaves with strong central midribs, "
            "parallel venation, and absence of pustules, rectangular lesions, or necrotic blights."
        ),
        "precautions": [
            "Maintain soil nutrient testing with side-dressed nitrogen during V4–V6 rapid growth stage.",
            "Monitor fields weekly for scouting economic thresholds of foliar rust and blights.",
        ],
        "curative_measures": ["No therapeutic chemical application required."],
        "favorable_conditions": "Optimal agronomic field conditions.",
    },

    # ------------------ GRAPE ------------------
    "Grape___Black_rot": {
        "title": "Grapevine Black Rot",
        "causal_organism": "Guignardia bidwellii (Fungus: Botryosphaeriaceae)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "Attention isolated small circular-to-oval reddish-brown necrotic spots with dark brown margins. "
            "Tiny, elevated black pimple-like pycnidia fruiting bodies are visible arranged in characteristic "
            "concentric circles just inside the lesion perimeter."
        ),
        "precautions": [
            "Prune out mummified berry clusters, dead canes, and vine debris during dormant pruning.",
            "Train vines on trellises to create an open canopy allowing sunlight and airflow to reach all leaves.",
            "Mow vineyard ground cover closely to reduce relative humidity beneath the fruiting zone.",
        ],
        "curative_measures": [
            "Spray Myclobutanil 10% WP (1.5 g/L), Mancozeb 75% WG (2.5 g/L), or Azoxystrobin 23% SC (1.0 ml/L).",
            "Apply preventative sprays starting at 2–3 inch shoot growth, pre-bloom, bloom, and post-bloom.",
            "Apply biological Copper octanoate or Serenade ASO (Bacillus subtilis) under organic regimens.",
        ],
        "favorable_conditions": "Warm temperatures (21°C to 30°C) with leaf wetness periods lasting 7 to 24 hours.",
    },
    "Grape___Esca_(Black_Measles)": {
        "title": "Grapevine Esca (Black Measles / Trunk Disease)",
        "causal_organism": "Fomitiporia mediterranea / Phaeomoniella chlamydospora (Fungal Complex)",
        "pathogen_type": "Fungal Complex",
        "visual_symptom_rationale": (
            "The model highlighted striking 'tiger-stripe' foliar symptoms: interveinal chlorotic yellow patches "
            "that rapidly turn necrotic brown, framed by narrow bands of green tissue remaining along the primary veins."
        ),
        "precautions": [
            "Delay annual pruning until late winter when wound healing is rapid and spore release is lowest.",
            "Coat large pruning wounds (> 2 cm) immediately with protective acrylic paint or Trichoderma wound paste.",
            "Disinfect pruning shears between individual vines to prevent vascular spore inoculation.",
        ],
        "curative_measures": [
            "No curative systemic chemical cures internal wood decay once established in the trunk.",
            "Remedial surgery: cut trunk back to healthy white vascular tissue below visible necrotic wood.",
            "Apply trunk renewal technique using vigorous suckers emerging from below cankered zones.",
            "Rogue out severely collapsed vines and replant with certified disease-free nursery stock.",
        ],
        "favorable_conditions": "Summer heat stress and high evapotranspiration following moist spring infections.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "title": "Grape Leaf Blight (Isariopsis Clavispora Spot)",
        "causal_organism": "Pseudocercospora clavispora / Isariopsis clavispora (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model focused on large, irregular, angular dark-brown to black foliar blights, often bounded by "
            "minor leaf veins. Symptomatic leaves show severe marginal necrosis, curling, and premature defoliation."
        ),
        "precautions": [
            "Remove and compost or bury fallen autumn vineyard leaf litter.",
            "Ensure proper vine canopy management with timely shoot tucking and leaf thinning around grape clusters.",
        ],
        "curative_measures": [
            "Spray Copper Oxychloride 50% WP (2.5 g/L) or Mancozeb 75% WP (2.5 g/L) post-harvest.",
            "Apply Difenoconazole 25% EC (0.5 ml/L) if disease appears prior to harvest veraison.",
        ],
        "favorable_conditions": "High humidity, prolonged leaf wetness, and warm post-monsoon temperatures (22°C to 28°C).",
    },
    "Grape___healthy": {
        "title": "Clinically Healthy Grapevine Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Uniform token attention across vibrant green, palmate grape leaves with clear, crisp margins, "
            "intact cuticle surface, healthy vascular veins, and zero necrotic spots or tiger-striping."
        ),
        "precautions": [
            "Maintain canopy shoot positioning, suckering, and balanced potassium-magnesium vine fertigation.",
            "Apply routine preventative sulfur sprays during early shoot development.",
        ],
        "curative_measures": ["No therapeutic chemical treatment needed."],
        "favorable_conditions": "Optimal vineyard terroir conditions.",
    },

    # ------------------ ORANGE (CITRUS) ------------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "title": "Citrus Greening (Huanglongbing - HLB)",
        "causal_organism": "Candidatus Liberibacter asiaticus (Fastidious Phloem-Limited Bacterium)",
        "pathogen_type": "Bacterial (Vector-Borne)",
        "visual_symptom_rationale": (
            "The model identified asymmetrical 'blotchy mottle' across the citrus leaf: irregular patches of "
            "interveinal chlorosis that cross secondary leaf veins without symmetry between left and right leaf halves. "
            "Veins appear swollen, corky, and yellowed."
        ),
        "precautions": [
            "Plant exclusively certified pathogen-free nursery trees propagated in insect-proof screened facilities.",
            "Scout orchards continuously for Asian Citrus Psyllid (Diaphorina citri) vectors using yellow sticky cards.",
            "Eradicate non-commercial citrus and ornamental host plants (such as Murraya paniculata / Orange Jasmine).",
        ],
        "curative_measures": [
            "No chemical cure exists for infected citrus trees; bacterium resides deep within the vascular phloem.",
            "Aggressive Vector Control: Spray Imidacloprid 17.8% SL (0.5 ml/L) or Thiamethoxam 25% WG (0.3 g/L).",
            "Release biological parasitic wasps (Tamarixia radiata) for eco-friendly psyllid nymph suppression.",
            "Immediately push, shred, and burn HLB-positive trees to halt transmission to adjacent healthy groves.",
        ],
        "favorable_conditions": "Presence of active Asian Citrus Psyllid vector populations and warm subtropical climate.",
    },

    # ------------------ PEACH ------------------
    "Peach___Bacterial_spot": {
        "title": "Peach Bacterial Spot (Bacteriosis)",
        "causal_organism": "Xanthomonas arboricola pv. pruni (Gram-Negative Bacterium)",
        "pathogen_type": "Bacterial",
        "visual_symptom_rationale": (
            "Attention isolated small (1–3 mm), angular, water-soaked dark-purple to black spots clustered along "
            "the leaf midrib and tip. Diseased necrotic tissue frequently drops out of the leaf blade, creating a "
            "characteristic ragged 'shot-hole' appearance."
        ),
        "precautions": [
            "Select bacterial-spot resistant peach cultivars (e.g., Candor, Dixired, Harbrite, Redhaven).",
            "Avoid planting orchards in loose, sandy soils with high wind exposure without windbreaks.",
            "Do not over-fertilize with nitrogen, which creates excessively succulent and vulnerable foliage.",
        ],
        "curative_measures": [
            "Apply preventative Copper Hydroxide (2.0 g/L) or Copper Oxychloride 50% WP during late dormancy.",
            "Apply low-dose Oxytetracycline (Mycoshield: 1.5 g/L) or Copper soap during bloom and petal fall.",
            "Spray biological Bacillus subtilis (Serenade ASO) during warm, humid post-bloom periods.",
        ],
        "favorable_conditions": "Warm (20°C to 28°C), windy weather accompanied by frequent rain and blowing sand/dew.",
    },
    "Peach___healthy": {
        "title": "Clinically Healthy Peach Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Even attention distribution over glossy, lanceolate, emerald-green peach leaves with smooth margins, "
            "clean central midribs, and absence of shot-hole punctures or leaf curling."
        ),
        "precautions": [
            "Apply dormant winter lime-sulfur or copper sprays to preemptively prevent peach leaf curl (Taphrina).",
            "Maintain soil drainage and mulch to preserve consistent root hydration.",
        ],
        "curative_measures": ["No therapeutic application required."],
        "favorable_conditions": "Optimal temperate orchard conditions.",
    },

    # ------------------ PEPPER (BELL) ------------------
    "Pepper,_bell___Bacterial_spot": {
        "title": "Pepper Bacterial Leaf Spot",
        "causal_organism": "Xanthomonas campestris pv. vesicatoria (Gram-Negative Bacterium)",
        "pathogen_type": "Bacterial",
        "visual_symptom_rationale": (
            "The model localized small, water-soaked, circular to irregularly angular dark-brown spots on the leaves. "
            "Lesions develop pale brownish-white centers surrounded by greasy yellow chlorotic halos, causing severe "
            "foliar scorch and leaf drop."
        ),
        "precautions": [
            "Use certified pathogen-free, hot-water-treated pepper seeds and clean transplant flats.",
            "Select bacterial spot-resistant pepper hybrids carrying Bs1, Bs2, Bs3, or Bs4 resistance genes.",
            "Avoid field cultivation and harvesting while foliage is wet from dew or rain.",
            "Rotate crops with non-solanaceous plants (corn, beans, brassicas) for a minimum of 2 years.",
        ],
        "curative_measures": [
            "Apply Copper Hydroxide 77% WP (2.0 g/L) tank-mixed with Mancozeb 75% WP (2.0 g/L) for synergy.",
            "Spray Streptomycin Sulfate (Plantomycin: 0.1 g/L) during severe early vegetative outbreaks.",
            "Apply systemic acquired resistance (SAR) inducers such as Acibenzolar-S-methyl (Actigard).",
        ],
        "favorable_conditions": "High temperatures (24°C to 30°C) with frequent rainstorms, splashing irrigation, and high humidity.",
    },
    "Pepper,_bell___healthy": {
        "title": "Clinically Healthy Bell Pepper Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Homogeneous token attention across smooth, glossy, ovate dark-green pepper leaves with crisp margins, "
            "healthy petiole attachment, and total absence of bacterial lesions or mosaic distortion."
        ),
        "precautions": [
            "Maintain uniform drip fertigation with adequate calcium to prevent blossom end rot in developing fruit.",
            "Scout weekly for aphids, thrips, and broad mites beneath the foliar canopy.",
        ],
        "curative_measures": ["No therapeutic intervention required."],
        "favorable_conditions": "Optimal warm solanaceous field conditions.",
    },

    # ------------------ POTATO ------------------
    "Potato___Early_blight": {
        "title": "Potato Early Blight (Target Spot)",
        "causal_organism": "Alternaria solani (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "Attention strongly isolates characteristic 'target-board' lesions: circular to irregular dark brown-black "
            "spots exhibiting distinct concentric rings surrounded by a chlorotic yellow halo. Older lower canopy "
            "leaves senesce and turn papery-brown."
        ),
        "precautions": [
            "Rotate potato fields with non-solanaceous crops (cereals, legumes) for 3 years.",
            "Provide balanced nitrogen and potassium fertilization; plants under nutrient stress are highly susceptible.",
            "Use drip irrigation; avoid sprinkler irrigation late in the afternoon or evening.",
            "Destroy potato cull piles and volunteer potato sprouts before spring planting.",
        ],
        "curative_measures": [
            "Spray Chlorothalonil 75% WP (2.5 g/L), Mancozeb 75% WP (2.5 g/L), or Azoxystrobin 23% SC (1.0 ml/L).",
            "Apply bio-fungicides: Bacillus subtilis or Trichoderma harzianum foliar formulations.",
            "Begin protective sprays when lower canopy leaves touch each other across the furrow row.",
        ],
        "favorable_conditions": "Warm temperatures (24°C to 29°C) alternating with wet weather or heavy morning dews.",
    },
    "Potato___Late_blight": {
        "title": "Potato Late Blight",
        "causal_organism": "Phytophthora infestans (Oomycete: Water Mold)",
        "pathogen_type": "Oomycete",
        "visual_symptom_rationale": (
            "Attention highlights rapidly expanding, water-soaked, dark-brown to purplish-black necrotic lesions "
            "with pale yellowish-green borders. In moist air, delicate white, cottony downy fungal growth (sporangia) "
            "is prominently visible on the lesion undersides."
        ),
        "precautions": [
            "Plant strictly certified disease-free seed tubers with zero late blight tolerance.",
            "Eliminate all potato cull piles and volunteer plants within 5 km of production fields.",
            "Hill soil deeply around rows to prevent spores washing from foliage down into potato tubers.",
            "Kill vines completely with a desiccant 2–3 weeks prior to harvest to prevent tuber infection.",
        ],
        "curative_measures": [
            "Emergency Treatment: Metalaxyl-M + Mancozeb (Ridomil Gold: 2.5 g/L) or Cymoxanil (Curzate: 2.0 g/L).",
            "Spray Fluopicolide (Infinito: 1.5 ml/L) or Mandipropamid (Revus: 0.8 ml/L) under high disease pressure.",
            "Apply immediately within 24 hours of symptom detection; repeat every 5–7 days in cool wet weather.",
        ],
        "favorable_conditions": "Cool (10°C to 20°C), overcast, rainy weather with prolonged relative humidity > 90%.",
    },
    "Potato___healthy": {
        "title": "Clinically Healthy Potato Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Even attention distribution over compound, pinnate, rich-green potato leaves with crisp leaflet margins, "
            "clean petioles, and absence of water-soaked blotches or target rings."
        ),
        "precautions": [
            "Maintain preventative scouting programs and monitor regional Late Blight Decision Support Systems (DSS).",
            "Keep soil hilled and maintain consistent moisture for optimum tuber bulking.",
        ],
        "curative_measures": ["No chemical treatment required."],
        "favorable_conditions": "Optimal potato growing conditions.",
    },

    # ------------------ RASPBERRY & SOYBEAN ------------------
    "Raspberry___healthy": {
        "title": "Clinically Healthy Raspberry Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Clean self-attention profile over serrated, compound, rich-green cane leaves with silver-white undersides, "
            "intact veins, and zero spur blight, anthracnose, or rust lesions."
        ),
        "precautions": [
            "Prune out fruited floricanes immediately after summer harvest to allow fresh primocanes to flourish.",
            "Trellis canes to maintain open airflow and rapid canopy drying after rainfall.",
        ],
        "curative_measures": ["No therapeutic intervention needed."],
        "favorable_conditions": "Optimal temperate caneberry conditions.",
    },
    "Soybean___healthy": {
        "title": "Clinically Healthy Soybean Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Uniform token attention across trifoliate, lush-green soybean leaflets with clean pubescence, "
            "unblemished midribs, and absence of frogeye leaf spots, brown spot, or rust pustules."
        ),
        "precautions": [
            "Maintain scheduled field scouting through R1 (flowering) and R3 (pod set) developmental stages.",
            "Rotate crops annually with corn or wheat to disrupt soil-borne pathogens.",
        ],
        "curative_measures": ["No therapeutic application required."],
        "favorable_conditions": "Optimal field agronomic conditions.",
    },

    # ------------------ SQUASH & STRAWBERRY ------------------
    "Squash___Powdery_mildew": {
        "title": "Squash Powdery Mildew",
        "causal_organism": "Podosphaera xanthii / Erysiphe cichoracearum (Fungus: Erysiphales)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model isolated conspicuous white, circular, powdery talcum-powder-like fungal colonies spreading "
            "across both leaf surfaces and petioles. Heavily infected leaves turn chlorotic, senesce prematurely, "
            "and expose developing squash fruit to sunscald."
        ),
        "precautions": [
            "Plant powdery mildew-resistant squash varieties (look for PMR labeling on seed packets).",
            "Space plants generously (0.9 to 1.2 m apart) to maximize sunlight penetration and foliar aeration.",
            "Plant in full sun; avoid shaded field margins or dense windbreaks.",
        ],
        "curative_measures": [
            "Spray Potassium bicarbonate (3.0 g/L) or Horticultural mineral oil (1%) at the very first white patch.",
            "Apply Tebuconazole 25% WG (1.0 g/L), Triflumizole, or Pyraclostrobin in commercial rotations.",
            "Spray Wettable Sulfur 80% WP (2.5 g/L); do not apply sulfur when temperatures exceed 32°C.",
        ],
        "favorable_conditions": "Warm (20°C to 28°C), dry weather with high relative humidity and dense, shaded foliage.",
    },
    "Strawberry___Leaf_scorch": {
        "title": "Strawberry Leaf Scorch",
        "causal_organism": "Diplocarpon earlianum (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The self-attention heads isolated numerous small, irregular, dark-purple to reddish-purple circular spots. "
            "Unlike leaf spot, scorch spots never develop pale white centers; lesions coalesce rapidly until entire leaf "
            "margins curl upward and appear scorched or burned by fire."
        ),
        "precautions": [
            "Plant certified disease-free strawberry crowns from reputable nurseries.",
            "Avoid overhead irrigation; use drip tape placed under plastic mulch.",
            "Renovate strawberry beds immediately after harvest by mowing old foliage and incorporating fertilizer.",
            "Ensure wide plant spacing on raised beds to encourage fast morning drying.",
        ],
        "curative_measures": [
            "Spray Captan 50% WP (2.5 g/L) or Thiophanate-methyl 70% WP (1.0 g/L) at early leaf emergence.",
            "Apply Azoxystrobin 23% SC (1.0 ml/L) or Copper Hydroxide (2.0 g/L) in rotational programs.",
            "Rake and destroy heavily scorched foliage during post-harvest bed renovation.",
        ],
        "favorable_conditions": "Frequent rainfall, warm temperatures (20°C to 25°C), and extended leaf wetness > 12 hours.",
    },
    "Strawberry___healthy": {
        "title": "Clinically Healthy Strawberry Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Balanced token attention across deep-green trifoliate strawberry leaves with distinct serrations, "
            "glossy cuticle surfaces, robust petioles, and absence of purple scorch or angular leaf spots."
        ),
        "precautions": [
            "Maintain clean straw or plastic mulch to keep fruit and leaves from contacting wet bare soil.",
            "Provide drip fertigation with balanced nitrogen-potassium for sustained crown development.",
        ],
        "curative_measures": ["No therapeutic intervention required."],
        "favorable_conditions": "Optimal raised-bed horticultural conditions.",
    },

    # ------------------ TOMATO ------------------
    "Tomato___Bacterial_spot": {
        "title": "Tomato Bacterial Spot",
        "causal_organism": "Xanthomonas vesicatoria / perforans (Gram-Negative Bacterium)",
        "pathogen_type": "Bacterial",
        "visual_symptom_rationale": (
            "The self-attention heads isolated small (under 3 mm), dark-brown to black angular water-soaked lesions "
            "surrounded by a distinctive greasy, translucent margin and yellow halo. As spots age, the dead centers "
            "dry up and drop out, giving leaflets a ragged perforated look."
        ),
        "precautions": [
            "Use certified pathogen-free seeds treated with hot water (50°C for 25 min) or hydrochloric acid.",
            "Avoid sprinkler irrigation; use drip irrigation to prevent bacterial water-splash dissemination.",
            "Do not work in, prune, or harvest tomato fields while plants are wet with morning dew.",
            "Rotate out of solanaceous crops for at least 2–3 seasons.",
        ],
        "curative_measures": [
            "Apply Copper Oxychloride 50% WP (2.5 g/L) tank-mixed with Mancozeb 75% WP (2.0 g/L) for bactericidal synergy.",
            "Add Streptomycin Sulfate (Plantomycin: 0.1 g/L) during severe early vegetative outbreaks.",
            "Apply biological Bacillus subtilis (Serenade ASO) or bacteriophage bio-products (AgriPhage).",
        ],
        "favorable_conditions": "Warm temperatures (24°C to 30°C) combined with high humidity, driving rains, and splashing water.",
    },
    "Tomato___Early_blight": {
        "title": "Tomato Early Blight",
        "causal_organism": "Alternaria solani / linariae (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model localized brown-to-black necrotic lesions displaying distinct concentric rings (resembling "
            "a target board) surrounded by a prominent chlorotic yellow halo. Older bottom leaves develop extensive "
            "blighting and yellowing, leading to severe defoliation from the base upward."
        ),
        "precautions": [
            "Prune off the lowest 30 cm of tomato foliage once plants establish to eliminate soil-contact splash.",
            "Mulch heavily around the plant base with clean straw or plastic mulch.",
            "Practice drip irrigation exclusively; ensure the foliar canopy stays completely dry.",
            "Stake and trellis tomato vines to maximize airflow throughout the canopy.",
        ],
        "curative_measures": [
            "Apply Chlorothalonil 75% WP (2.5 g/L), Mancozeb 75% WP (2.5 g/L), or Difenoconazole 25% EC (0.5 ml/L).",
            "Spray bio-fungicides: Trichoderma harzianum or Bacillus amyloliquefaciens foliar formulations.",
            "Prune and dispose of heavily blighted lower leaves immediately; do not compost diseased leaves.",
        ],
        "favorable_conditions": "Warm temperatures (24°C to 29°C) with alternating wet-dry cycles, heavy dew, or rainy spells.",
    },
    "Tomato___Late_blight": {
        "title": "Tomato Late Blight",
        "causal_organism": "Phytophthora infestans (Oomycete: Water Mold)",
        "pathogen_type": "Oomycete",
        "visual_symptom_rationale": (
            "Attention isolated rapidly expanding, dark, irregular water-soaked necrotic blotches on leaflets and "
            "stems. Lesions appear greasy and dark greenish-brown, accompanied by white, cottony fungal-like sporulation "
            "visible on the leaf underside under high humidity."
        ),
        "precautions": [
            "Plant late blight-resistant tomato varieties (e.g., Defiant, Mountain Merit, Plum Regal, Iron Lady).",
            "Destroy any volunteer tomato or potato plants and weed nightshades (Solanum nigrum) near the field.",
            "Ensure wide spacing (60–90 cm between plants) for rapid canopy drying.",
        ],
        "curative_measures": [
            "Emergency Spray: Metalaxyl-M + Mancozeb (Ridomil Gold: 2.5 g/L) or Cymoxanil (Curzate: 2.0 g/L).",
            "Apply Mandipropamid (Revus: 0.8 ml/L) or Dimethomorph 50% WP (1.0 g/L) to arrest advancing blight.",
            "Act within 24 hours of first detection; spray thoroughly on both upper and lower leaf surfaces.",
        ],
        "favorable_conditions": "Cool (15°C to 22°C), overcast, foggy or rainy weather with relative humidity above 90%.",
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato Leaf Mold",
        "causal_organism": "Passalora fulva / Cladosporium fulvum (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model isolated pale green to distinct yellow chlorotic blotches on the adaxial (upper) leaf surface, "
            "which correspond directly to olive-green to velvety brown moldy sporulation patches on the abaxial (lower) surface."
        ),
        "precautions": [
            "Ventilate greenhouses and high tunnels aggressively to keep relative humidity below 85%.",
            "Increase plant spacing and prune suckers to improve air circulation within the canopy.",
            "Water early in the morning using drip irrigation to ensure leaves dry rapidly.",
        ],
        "curative_measures": [
            "Spray Difenoconazole 25% EC (0.5 ml/L) or Copper Oxychloride 50% WP (2.5 g/L).",
            "Apply Potassium bicarbonate (3.0 g/L) as an eco-friendly contact cure for mold patches.",
            "Rogue out and destroy severely mold-covered leaves from the lower canopy.",
        ],
        "favorable_conditions": "High relative humidity (> 85%) and moderate temperatures (21°C to 24°C) inside greenhouses/tunnels.",
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Tomato Septoria Leaf Spot",
        "causal_organism": "Septoria lycopersici (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model isolated dense clusters of small (2–3 mm), circular lesions featuring distinct dark brown "
            "borders and prominent light gray or tan centers. Minute black speck-like pycnidia fruiting bodies are "
            "clearly visible peppered across the center of each spot."
        ),
        "precautions": [
            "Mulch heavily beneath tomato plants to create a barrier against soil splash harboring overwintering spores.",
            "Avoid overhead sprinkler irrigation; always irrigate directly at the base via drip emitters.",
            "Practice a minimum 2-year crop rotation away from all solanaceous species.",
            "Sterilize stakes and cages with a 10% bleach solution before seasonal reuse.",
        ],
        "curative_measures": [
            "Apply Chlorothalonil 75% WP (2.0 g/L) or Azoxystrobin 23% SC (1.0 ml/L) at first symptom appearance.",
            "Spray Copper Hydroxide (2.0 g/L) combined with Mancozeb 75% WP (2.0 g/L) every 7–10 days.",
            "Prune off infected lower leaves as soon as spots appear to prevent upward spore splash.",
        ],
        "favorable_conditions": "Warm (20°C to 25°C), wet weather with frequent rainfall or heavy morning dews.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Two-Spotted Spider Mites",
        "causal_organism": "Tetranychus urticae (Acarid Foliar Pest)",
        "pathogen_type": "Acarid Pest",
        "visual_symptom_rationale": (
            "The model detected fine, dense chlorotic stippling (pale yellow pinpoint flecks) across the adaxial leaf "
            "surface caused by microscopic mites piercing plant epidermal cells. In advanced infestations, delicate "
            "silken webbing and bronze foliar drying become visible."
        ),
        "precautions": [
            "Keep fields and greenhouse perimeters free of broadleaf weeds that harbor overwintering mite populations.",
            "Avoid water-stress; well-hydrated plants with adequate overhead humidity are far less hospitable to mites.",
            "Avoid broad-spectrum synthetic pyrethroid insecticides that wipe out natural predatory mite populations.",
        ],
        "curative_measures": [
            "Apply Abamectin 1.9% EC (0.5 ml/L), Spiromesifen 22.9% SC (1.0 ml/L), or Bifenazate 43.2% SC.",
            "Release biological predatory mites: Phytoseiulus persimilis or Neoseiulus californicus (1–2 per plant).",
            "Spray Horticultural Neem Oil (3.0 ml/L) or insecticidal potassium soap on leaf undersides.",
        ],
        "favorable_conditions": "Hot (> 30°C), dry, dusty field conditions with low relative humidity.",
    },
    "Tomato___Target_Spot": {
        "title": "Tomato Target Spot",
        "causal_organism": "Corynespora cassiicola (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "Attention isolated pinpoint brown spots that expand into circular brown lesions with faint concentric "
            "zoning, yellow chlorotic margins, and sunken centers. Unlike early blight, target spots are smaller, "
            "more numerous, and affect both upper and lower canopy leaves simultaneously."
        ),
        "precautions": [
            "Maintain generous row spacing and stake vines to promote air movement and rapid drying.",
            "Avoid overhead irrigation; use drip irrigation exclusively.",
            "Rotate crops with non-hosts (corn, sorghum) for at least 2 seasons.",
        ],
        "curative_measures": [
            "Apply Azoxystrobin + Difenoconazole (1.0 ml/L) or Boscalid (1.0 g/L) at early onset.",
            "Spray protective Chlorothalonil 75% WP (2.5 g/L) on a 7–10 day protective schedule.",
        ],
        "favorable_conditions": "Warm temperatures (20°C to 28°C) and extended high humidity (> 80%).",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "causal_organism": "Tomato Yellow Leaf Curl Geminivirus (Transmitted by Whitefly Bemisia tabaci)",
        "pathogen_type": "Viral (Vector-Borne)",
        "visual_symptom_rationale": (
            "The model highlighted severe foliar distortion: upward cupping and curling of leaflet margins, "
            "interveinal chlorosis, reduction in leaf blade size, and bushy, stunted, erect terminal growth."
        ),
        "precautions": [
            "Plant TYLCV-resistant tomato hybrids (carrying Ty-1, Ty-2, or Ty-3 resistance genes).",
            "Cover nurseries and seedling beds with 50-mesh insect-proof fine netting to block whitefly vectors.",
            "Hang yellow sticky cards (20–30 per acre) to monitor and mass-trap adult whiteflies.",
            "Eradicate alternative weed hosts (nightshades, mallow, sowthistle) around the field perimeter.",
        ],
        "curative_measures": [
            "No chemical cure exists for viral plant infections; immediately rogue and bury symptomatic plants.",
            "Vector Suppression: Spray Imidacloprid 17.8% SL (0.5 ml/L) or Spiromesifen 22.9% SC (1.0 ml/L).",
            "Apply biological Beauveria bassiana entomopathogenic fungus against whitefly nymphs.",
        ],
        "favorable_conditions": "Hot, dry conditions favoring explosive whitefly (Bemisia tabaci) vector reproduction.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus (ToMV)",
        "causal_organism": "Tomato Mosaic Tobamovirus (Mechanically Transmitted)",
        "pathogen_type": "Viral (Contact-Transmitted)",
        "visual_symptom_rationale": (
            "The model detected mottled foliar discoloration: irregular patches of alternating light-green and "
            "dark-green mosaic patterns, leaflet puckering/blistering, and severe 'fern-leaf' or shoestring leaflet distortion."
        ),
        "precautions": [
            "Select ToMV-resistant tomato hybrids (carrying the Tm-2 or Tm-2^2 resistance gene).",
            "Enforce strict sanitation: wash hands with soap and dip pruning shears in 10% trisodium phosphate (TSP) or skim milk.",
            "Prohibit smoking or tobacco handling near tomato crops (tobacco can carry infectious tobamoviruses).",
            "Purchase certified virus-free seed from reliable seed houses.",
        ],
        "curative_measures": [
            "No chemical viricide exists; rogue and destroy symptomatic plants immediately.",
            "Do not touch adjacent healthy plants after handling infected vines.",
            "Sterilize all stakes, ties, and trellis wires with 20% household bleach before the next planting season.",
        ],
        "favorable_conditions": "Mechanical plant handling, pruning, and touching during warm vegetative growth.",
    },
    "Tomato___healthy": {
        "title": "Clinically Healthy Tomato Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Uniform token attention across rich dark-green compound tomato foliage with crisp dentate leaflet margins, "
            "robust petioles, healthy glandular trichomes, and zero bacterial spots, blight lesions, or viral mottling."
        ),
        "precautions": [
            "Maintain consistent drip fertigation and calcium levels to ensure vigorous vegetative and fruit growth.",
            "Continue preventative bi-weekly foliar scouting across all crop stages.",
        ],
        "curative_measures": ["No therapeutic intervention required."],
        "favorable_conditions": "Optimal solanaceous growing conditions.",
    },

    # ------------------ PEAR (DiaMOS Plant) ------------------
    "Pear___curl": {
        "title": "Pear Leaf Curl (Pear Leaf Blister / Mite Injury)",
        "causal_organism": "Epitrimerus pyri (Pear Rust Mite) / Taphrina bullata",
        "pathogen_type": "Fungal / Acarid",
        "visual_symptom_rationale": (
            "The model highlighted severe abnormal foliar curling: upward margin rolling, blistered leaf puckering, "
            "and reddish-brown discoloration along the leaf margins."
        ),
        "precautions": [
            "Apply dormant horticultural mineral oil + liquid lime sulfur before bud break to suffocate overwintering mites.",
            "Prune out heavily curled terminal twigs during spring inspection.",
        ],
        "curative_measures": [
            "Apply Abamectin 1.9% EC (0.5 ml/L) or Spiromesifen (1.0 ml/L) if mites are actively feeding.",
            "Spray Copper Hydroxide (2.0 g/L) if fungal blister (Taphrina) is present.",
        ],
        "favorable_conditions": "Cool spring bud swell followed by warm dry vegetative growth.",
    },
    "Pear___healthy": {
        "title": "Clinically Healthy Pear Foliage",
        "causal_organism": "None (Healthy Tissue)",
        "pathogen_type": "Healthy",
        "visual_symptom_rationale": (
            "Even attention distribution across oval, finely serrated, glossy green pear leaves with clean venation, "
            "firm petioles, and absence of leaf curl, slug skeletonization, or fungal scab spots."
        ),
        "precautions": [
            "Maintain dormant winter sanitation sprays (horticultural oil + copper).",
            "Ensure balanced fertilization and proper summer canopy pruning.",
        ],
        "curative_measures": ["No therapeutic intervention required."],
        "favorable_conditions": "Optimal pome fruit growing conditions.",
    },
    "Pear___slug": {
        "title": "Pear Slug (Pear Sawfly Skeletonization)",
        "causal_organism": "Caliroa cerasi (Pear Sawfly Larva / Slug)",
        "pathogen_type": "Insect Pest",
        "visual_symptom_rationale": (
            "Attention isolated severe 'window-pane' foliar skeletonization: larvae feed on the upper leaf epidermis "
            "and mesophyll tissue, leaving only the transparent lower cuticle and a skeleton of brown veins."
        ),
        "precautions": [
            "Scout leaf undersides in late spring and mid-summer for shiny, olive-green, slug-like sawfly larvae.",
            "Cultivate soil beneath trees in late autumn to expose overwintering sawfly pupae to frost and predators.",
        ],
        "curative_measures": [
            "Apply Spinosad 45% SC (0.5 ml/L) or Lambda-cyhalothrin 5% EC (0.5 ml/L) for rapid larval knockdown.",
            "Dust foliage with Horticultural Diatomaceous Earth or spray insecticidal potassium soap.",
            "Wash small trees with a high-pressure water stream to dislodge soft-bodied slug larvae.",
        ],
        "favorable_conditions": "Warm early summer and mid-summer generations when adult sawflies emerge to lay eggs.",
    },
    "Pear___spot": {
        "title": "Pear Leaf Spot (Fabraea / Entomosporium Spot)",
        "causal_organism": "Diplocarpon mespili / Entomosporium mespili (Fungus: Ascomycete)",
        "pathogen_type": "Fungal",
        "visual_symptom_rationale": (
            "The model isolated small, discrete circular reddish-purple spots that enlarge into brown necrotic lesions "
            "with tiny raised black spore blisters at the center. Severe spot clusters cause premature foliar yellowing and drop."
        ),
        "precautions": [
            "Rake and destroy fallen pear leaves in autumn to eliminate overwintering ascocarps.",
            "Prune dense interior canopy shoots during winter to maximize sunlight and airflow.",
            "Avoid overhead irrigation; keep pear foliage dry.",
        ],
        "curative_measures": [
            "Apply Mancozeb 75% WP (2.5 g/L) or Myclobutanil 10% WP (1.5 g/L) starting at green tip stage.",
            "Apply protective Copper Oxychloride 50% WP (2.5 g/L) every 10–14 days during wet spring spells.",
        ],
        "favorable_conditions": "Mild, wet spring and summer weather with frequent rain showers and temperatures between 15°C and 25°C.",
    },
}


def get_disease_knowledge(
    raw_class: str,
    crop: str = "",
    disease: str = "",
) -> Dict[str, Any]:
    """Retrieve comprehensive pathological description, explainability rationale,
    precautions, and curative measures for any detected condition.
    
    Args:
        raw_class: Exact dataset class name (e.g. 'Tomato___Early_blight')
        crop: Crop name fallback
        disease: Disease detail fallback
        
    Returns:
        Dictionary with title, causal_organism, pathogen_type, visual_symptom_rationale,
        precautions, curative_measures, and favorable_conditions.
    """
    if raw_class in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[raw_class]

    # Flexible matching if exact key is slightly altered
    clean_target = raw_class.lower().replace(" ", "").replace("_", "")
    for k, v in DISEASE_KNOWLEDGE_BASE.items():
        clean_k = k.lower().replace(" ", "").replace("_", "")
        if clean_k in clean_target or clean_target in clean_k:
            return v

    # Fallback template
    is_healthy = "healthy" in raw_class.lower() or "healthy" in disease.lower()
    c_name = crop if crop else "Plant"
    d_name = disease if disease else ("Healthy" if is_healthy else "Pathological Condition")

    if is_healthy:
        return {
            "title": f"Clinically Healthy {c_name} Foliage",
            "causal_organism": "None (Healthy Tissue)",
            "pathogen_type": "Healthy",
            "visual_symptom_rationale": (
                f"The Vision Transformer exhibited uniform self-attention distribution across vibrant green {c_name} "
                "parenchyma with high chlorophyll reflectance, intact cuticle integrity, and zero necrotic lesions."
            ),
            "precautions": [
                "Continue standard seasonal foliar monitoring and balanced N-P-K fertigation.",
                "Maintain weed-free root zones and scheduled soil moisture checks.",
            ],
            "curative_measures": ["No therapeutic intervention required."],
            "favorable_conditions": "Optimal agronomic growing conditions.",
        }

    return {
        "title": f"{c_name} {d_name}",
        "causal_organism": f"Pathogenic Foliar Agent ({d_name})",
        "pathogen_type": "Foliar Pathogen",
        "visual_symptom_rationale": (
            f"The Vision Transformer self-attention heads isolated distinctive necrotic foliar lesions and "
            f"chlorotic discoloration patterns characteristic of {d_name} on {c_name} leaves."
        ),
        "precautions": [
            "Inspect foliar canopy regularly under adequate lighting.",
            "Avoid overhead irrigation; practice drip irrigation to keep foliage dry.",
            "Ensure proper plant spacing to maximize airflow.",
            "Remove and destroy infected plant debris promptly.",
        ],
        "curative_measures": [
            "Apply registered broad-spectrum protective fungicide (e.g. Mancozeb 75% WP: 2.5 g/L or Copper Oxychloride).",
            "Apply bio-fungicides: Bacillus subtilis or Trichoderma foliar spray.",
            "Prune infected foliage to prevent pathogen spread to adjacent healthy tissues.",
        ],
        "favorable_conditions": "Warm, humid conditions with extended foliar wetness.",
    }
