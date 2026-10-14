"""
Standalone REST API for Crop Health Model
-----------------------------------------
Run this file to serve predictions over HTTP:
    python api_server.py

Endpoints:
    GET  /health   -> Status and model metadata
    GET  /crops    -> List supported crops
    POST /predict  -> Predict disease from uploaded image file
"""

import os
try:
    from crop_health_model.predictor import CropHealthPredictor
except (ImportError, ValueError):
    from predictor import CropHealthPredictor

app = Flask(__name__)
predictor = CropHealthPredictor()

@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "device": str(predictor.device),
        "total_classes": len(predictor.get_classes()),
        "supported_crops": predictor.get_supported_crops(),
    })

@app.get("/crops")
def get_crops():
    return jsonify({
        "crops": predictor.get_supported_crops(),
    })

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file in request"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file provided"}), 400

    crop = request.args.get("crop") or request.form.get("crop")
    top_k = request.args.get("top_k") or request.form.get("top_k") or 5
    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 5

    try:
        image_bytes = file.read()
        result = predictor.predict(image_bytes, crop_name=crop, top_k=top_k)
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting Crop Health API server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
