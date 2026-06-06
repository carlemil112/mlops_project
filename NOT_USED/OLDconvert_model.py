import torch
import torch.onnx as torch_onnx
import subprocess
import sys
import os
import shutil
import tensorflow as tf

from model.fer_model import FERModel

MODEL_PATH = "outputs/fer_run/best_emotion_model.pt"
ONNX_PATH = "emotion_model.onnx"
ONNX2TF_OUTPUT_DIR = "onnx2tf_output"
TFLITE_PATH = "emotion_model_quantized.tflite"
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


def onnx_to_saved_model():
    # onnx2tf converts ONNX to a TensorFlow SavedModel directory.
    if os.path.exists(ONNX2TF_OUTPUT_DIR):
        shutil.rmtree(ONNX2TF_OUTPUT_DIR)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            ONNX_PATH,
            "-o",
            ONNX2TF_OUTPUT_DIR,
            "--non_verbose",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, flush=True)
        raise RuntimeError(f"onnx2tf failed with exit code {result.returncode}")

    print(result.stdout, flush=True)

    saved_model_path = os.path.join(ONNX2TF_OUTPUT_DIR, "saved_model.pb")
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"Expected TensorFlow SavedModel at {saved_model_path}")


def quantize_saved_model_to_tflite():
    # Dynamic-range post-training quantization. This does not need calibration data.
    converter = tf.lite.TFLiteConverter.from_saved_model(ONNX2TF_OUTPUT_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as file:
        file.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(
        f"Quantized TFLite model saved to {TFLITE_PATH} ({size_kb:.1f} KB)", flush=True
    )


if __name__ == "__main__":
    model = FERModel(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    export_to_onnx(model)
    onnx_to_saved_model()
    quantize_saved_model_to_tflite()
