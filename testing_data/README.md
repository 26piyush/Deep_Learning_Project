# Labelled Testing Dataset (210 Images across 42 Classes)

This directory contains a balanced held-out testing collection with the **exact plant disease or health status directly labeled in each file name**.

---

## 1. File Naming Convention

Every image is formatted as:
```text
<Crop>___<Condition>_<Index:02d>.jpg
```

**Examples:**
- `Apple___Apple_scab_01.jpg`
- `Corn___Common_rust_03.jpg`
- `Grape___Esca_(Black_Measles)_02.jpg`
- `Pear___spot_04.jpg`
- `Tomato___Early_blight_05.jpg`

---

## 2. Dataset Composition

- **Total Images:** 210 images (exactly 5 balanced samples per class)
- **Total Classes:** 42 conditions across 15 crop species
- **Sources Included:**
  - 🌿 **PlantVillage**: Laboratory benchmark leaves
  - 🌾 **PlantDoc**: In-the-wild field disease leaves
  - 🍐 **DiaMOS Plant**: Pear crop leaves (*curl, healthy, slug, spot*)

---

## 3. Metadata Files

- **`labels.csv`**: A CSV table mapping every file to its `crop`, `disease`, `class`, `source_dataset`, and `original_file`.
- **`manifest.json`**: Machine-readable JSON metadata for automated test scripts.

---

## 4. How to Use

### In the Web Application (Streamlit)
1. Open `http://localhost:8501`.
2. Under **"1. Provide Input Leaf Image"**, click **Browse files** (or drag & drop).
3. Select any image from this `testing_data/` folder. The file name tells you the ground-truth disease immediately!

### In the Python CLI
```powershell
python -m src.inference --image testing_data/Tomato___Early_blight_01.jpg --classifier vit
python -m src.inference --image testing_data/Pear___spot_01.jpg --classifier vit
```
