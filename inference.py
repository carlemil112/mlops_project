import os
import time

import torch

from model.fer_model import FERModel


ORIGINAL_MODEL_PATH = "outputs/fer_run/best_emotion_model.pt"
QUANTIZED_MODEL_PATH = "outputs/fer_run/best_emotion_model_dynamic_quantized.pt"
IMG_SIZE = 48
WARMUP_RUNS = 20
BENCHMARK_RUNS = 200


def infer_num_classes(state_dict):
    return state_dict["classifier.5.weight"].shape[0]


def load_original_model():
    state_dict = torch.load(ORIGINAL_MODEL_PATH, map_location="cpu")
    model = FERModel(num_classes=infer_num_classes(state_dict))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_quantized_model():
    model = torch.load(QUANTIZED_MODEL_PATH, map_location="cpu")
    model.eval()
    return model


def benchmark_model(model, sample_input):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(sample_input)

        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            model(sample_input)
        end = time.perf_counter()

    return (end - start) / BENCHMARK_RUNS * 1000


def file_size_kb(path):
    return os.path.getsize(path) / 1024


def main():
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        raise FileNotFoundError(f"Missing original model: {ORIGINAL_MODEL_PATH}")
    if not os.path.exists(QUANTIZED_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing quantized model: {QUANTIZED_MODEL_PATH}. "
            "Run python convert_model.py first."
        )

    original_model = load_original_model()
    quantized_model = load_quantized_model()
    sample_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    original_latency_ms = benchmark_model(original_model, sample_input)
    quantized_latency_ms = benchmark_model(quantized_model, sample_input)
    speedup = original_latency_ms / quantized_latency_ms

    print("Inference speed comparison")
    print(f"Original model:  {original_latency_ms:.4f} ms/inference")
    print(f"Quantized model: {quantized_latency_ms:.4f} ms/inference")
    print(f"Speedup:         {speedup:.2f}x")
    print()
    print(f"Original size:   {file_size_kb(ORIGINAL_MODEL_PATH):.1f} KB")
    print(f"Quantized size:  {file_size_kb(QUANTIZED_MODEL_PATH):.1f} KB")


if __name__ == "__main__":
    main()
