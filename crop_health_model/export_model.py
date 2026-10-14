"""
Export Crop Health Model to TorchScript and ONNX
------------------------------------------------
Usage:
    python export_model.py --format torchscript
    python export_model.py --format onnx
"""

import argparse
from pathlib import Path
import torch
try:
    from crop_health_model.predictor import CropHealthPredictor
except (ImportError, ValueError):
    from predictor import CropHealthPredictor

def export(format_type: str = "torchscript"):
    print(f"Loading model for {format_type} export...")
    predictor = CropHealthPredictor(device="cpu")
    model = predictor.model
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    out_dir = Path(__file__).resolve().parent

    if format_type.lower() == "torchscript":
        output_file = out_dir / "crop_health_model_scripted.pt"
        print(f"Tracing model to TorchScript...")
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(output_file))
        print(f"Saved TorchScript model to: {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")

    elif format_type.lower() == "onnx":
        output_file = out_dir / "crop_health_model.onnx"
        print(f"Exporting model to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_file),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"Saved ONNX model to: {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    args = parser.parse_args()
    export(args.format)
