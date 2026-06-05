 
import torch
import torch.onnx as torch_onnx
import onnx
import subprocess
import sys
import os
 
from train import FERModel
 
MODEL_PATH = "outputs/fer_run/best_emotion_model.pt"
ONNX_PATH = "emotion_model.onnx"
TFLITE_PATH = "emotion_model.tflite"
NUM_CLASSES = 7
IMG_SIZE = 48
 
 
def export_to_onnx(model):
    dummy_input = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE))
    torch_onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"Exported ONNX model to {ONNX_PATH}", flush=True)
 
 
def onnx_to_tflite():
    # onnx2tf converts directly from ONNX to TFLite without the onnx_tf mess
    result = subprocess.run(
        [
            sys.executable, "-m", "onnx2tf",
            "-i", ONNX_PATH,
            "-o", "onnx2tf_output",
            "--non_verbose",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, flush=True)
        raise RuntimeError(f"onnx2tf failed with exit code {result.returncode}")
 
    print(result.stdout, flush=True)
 
    # onnx2tf outputs to a folder, find the tflite file and move it
    tflite_candidates = [
        f for f in os.listdir("onnx2tf_output") if f.endswith(".tflite")
    ]
    if not tflite_candidates:
        raise FileNotFoundError("No .tflite file found in onnx2tf_output/")
 
    # prefer INT8 if available
    int8 = [f for f in tflite_candidates if "int8" in f.lower()]
    chosen = int8[0] if int8 else tflite_candidates[0]
    os.rename(os.path.join("onnx2tf_output", chosen), TFLITE_PATH)
 
    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"TFLite model saved to {TFLITE_PATH} ({size_kb:.1f} KB)", flush=True)
 
 
if __name__ == "__main__":
    model = FERModel(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
 
    export_to_onnx(model)
    onnx_to_tflite()