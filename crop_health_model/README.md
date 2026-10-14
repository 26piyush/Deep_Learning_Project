# Crop Health Disease Prediction Model

A self-contained, production-ready crop disease classification module extracted from the SIH project.

This package contains the trained **ResNet-18** model capable of diagnosing **38 plant health conditions** across **14 agricultural crops**.

---

## 📁 Package Contents

```text
crop_health_model/
├── model_state.pt                  # Trained PyTorch model weights (~42.8 MB)
├── model_meta.json                 # Label mappings, crop-to-index dict, mean/std
├── labels.txt                      # Plain text list of 38 disease classes
├── predictor.py                    # Core standalone inference class (CropHealthPredictor)
├── api_server.py                   # Ready-to-run Flask REST API microservice
├── example_usage.py                # Demonstration & test script
├── export_model.py                 # Export script for TorchScript and ONNX
├── crop_health_model_scripted.pt   # Standalone TorchScript traced model
├── requirements.txt                # Minimal Python dependencies
└── README.md                       # This guide
```

---

## 🌾 Supported Crops & Diseases (38 Classes)

The model supports disease diagnosis across 14 crops:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn**: Cercospora leaf spot (Gray leaf spot), Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca (Black Measles), Leaf blight (Isariopsis Leaf Spot), Healthy
- **Orange**: Haunglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper (Bell)**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy

---

## 🚀 How to Use in Another Project

### Option 1: Direct Python Import (Recommended)

1. **Copy the `crop_health_model` folder** into your new project directory:
   ```text
   my_new_project/
   ├── crop_health_model/
   │   ├── predictor.py
   │   ├── model_state.pt
   │   ├── model_meta.json
   │   └── ...
   └── main.py
   ```

2. **Install dependencies**:
   ```bash
   pip install -r crop_health_model/requirements.txt
   ```

3. **In your code (`main.py`)**:
   ```python
   from crop_health_model.predictor import CropHealthPredictor

   # Initialize predictor (auto-detects CUDA / Apple MPS / CPU)
   predictor = CropHealthPredictor()

   # Predict on an image (supports file path, PIL Image, bytes, or numpy array)
   result = predictor.predict("path/to/leaf.jpg", top_k=3)

   print("Top Diagnosis:", result["prediction"]["disease"])
   print("Confidence:", result["prediction"]["confidence"])
   print("Full Class:", result["prediction"]["class"])

   # Optional: Filter specifically to a known crop (e.g., Tomato)
   tomato_result = predictor.predict("path/to/tomato_leaf.jpg", crop_name="Tomato")
   print(tomato_result["top_predictions"])
   ```

---

### Option 2: Run as a Standalone REST API Microservice

If your other project is written in Node.js, Next.js, Go, PHP, Flutter, or React Native, you can run this model as a dedicated microservice.

1. **Start the API service**:
   ```bash
   python api_server.py
   ```
   *(By default runs on `http://0.0.0.0:5000`)*

2. **Endpoints**:
   - `GET /health` — Check server status & supported crops.
   - `POST /predict` — Upload an image:
     ```bash
     curl -X POST http://localhost:5000/predict \
       -F "image=@leaf.jpg" \
       -F "crop=Tomato"
     ```

3. **Sample JSON Response**:
   ```json
   {
     "success": true,
     "device_used": "cpu",
     "filtered_to_crop": "Tomato",
     "prediction": {
       "class": "Tomato___Early_blight",
       "crop": "Tomato",
       "disease": "Early_blight",
       "confidence": 0.9412
     },
     "top_predictions": [
       {
         "class": "Tomato___Early_blight",
         "crop": "Tomato",
         "disease": "Early_blight",
         "confidence": 0.9412
       },
       {
         "class": "Tomato___Late_blight",
         "crop": "Tomato",
         "disease": "Late_blight",
         "confidence": 0.0421
       }
     ]
   }
   ```

---

### Option 3: Export to ONNX / TorchScript

If you need to deploy without a full PyTorch installation (e.g., using `onnxruntime` or mobile/embedded):
- **TorchScript** (already included as `crop_health_model_scripted.pt`)
- **ONNX**:
  ```bash
  python export_model.py --format onnx
  ```
  *(Requires `pip install onnx`)*

---

## 🛠️ Verification Test

To verify that the model loads and predicts properly on your machine:
```bash
python example_usage.py
```
