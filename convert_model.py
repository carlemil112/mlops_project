import json
import os

import torch
from torch.ao.quantization import quantize_dynamic

from model.fer_model import FERModel


INPUT_MODEL_PATH = "outputs/fer_run/best_emotion_model.pt"
OUTPUT_DIR = "outputs/fer_run"
QUANTIZED_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_emotion_model_dynamic_quantized.pt"
)
METADATA_PATH = os.path.join(OUTPUT_DIR, "quantization_metadata.json")
IMG_SIZE = 48


def infer_num_classes(state_dict):
    classifier_weight = state_dict["classifier.5.weight"]
    return classifier_weight.shape[0]


def file_size_kb(path):
    return os.path.getsize(path) / 1024


def load_model():
    state_dict = torch.load(INPUT_MODEL_PATH, map_location="cpu")
    num_classes = infer_num_classes(state_dict)

    model = FERModel(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes


def quantize_model(model):
    return quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def smoke_test(model):
    dummy_input = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE))
    with torch.no_grad():
        output = model(dummy_input)
    return list(output.shape)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, num_classes = load_model()
    quantized_model = quantize_model(model)
    output_shape = smoke_test(quantized_model)

    torch.save(quantized_model, QUANTIZED_MODEL_PATH)

    original_size_kb = file_size_kb(INPUT_MODEL_PATH)
    quantized_size_kb = file_size_kb(QUANTIZED_MODEL_PATH)
    size_reduction_percent = (
        (original_size_kb - quantized_size_kb) / original_size_kb * 100
        if original_size_kb > 0
        else 0
    )

    metadata = {
        "quantization_type": "dynamic",
        "framework": "pytorch",
        "quantized_layers": ["torch.nn.Linear"],
        "dtype": "qint8",
        "input_model_path": INPUT_MODEL_PATH,
        "quantized_model_path": QUANTIZED_MODEL_PATH,
        "num_classes": num_classes,
        "smoke_test_output_shape": output_shape,
        "original_size_kb": round(original_size_kb, 2),
        "quantized_size_kb": round(quantized_size_kb, 2),
        "size_reduction_percent": round(size_reduction_percent, 2),
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=2)

    print(f"Loaded model: {INPUT_MODEL_PATH}", flush=True)
    print(f"Quantized model saved to: {QUANTIZED_MODEL_PATH}", flush=True)
    print(f"Metadata saved to: {METADATA_PATH}", flush=True)
    print(
        "Model size: "
        f"{original_size_kb:.1f} KB -> {quantized_size_kb:.1f} KB "
        f"({size_reduction_percent:.1f}% smaller)",
        flush=True,
    )
    print(f"Smoke test output shape: {output_shape}", flush=True)


if __name__ == "__main__":
    main()
