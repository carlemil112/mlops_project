# evaluate.py - runs after training in the Jenkins pipeline:
#   1. evaluates best_emotion_model.pt on the val set and logs accuracy to MLflow
#   2. converts the model: PyTorch -> ONNX -> TFLite via convert_model.py
#   3. benchmarks FP32 PyTorch vs TFLite (inference time + accuracy)
#   4. logs everything back to the existing MLflow run via run_id

import os
import sys
import time
import subprocess
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import mlflow
import mlflow.pytorch
from hydra import compose, initialize_config_dir

from train import FERDataset
from model.fer_model import FERModel

MODEL_PATH = "outputs/fer_run/best_emotion_model.pt"
ONNX_PATH = "emotion_model.onnx"
TFLITE_PATH = "emotion_model_quantized.tflite"


def get_val_loader(cfg):
    # Mirrors the split in train_pytorch.py - same seed gives the same val set
    data_path = cfg.paths.data_path
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.script.img_size, cfg.script.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[cfg.script.dataset_mean],
                std=[cfg.script.dataset_std],
            ),
        ]
    )

    full_dataset = FERDataset(data_path, transform=transform)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    _, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=False,
    )
    return val_loader, full_dataset.num_classes


def evaluate_pytorch(cfg, device):
    print("Evaluating PyTorch model on val set...", flush=True)

    val_loader, num_classes = get_val_loader(cfg)

    model = FERModel(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    latencies = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            t0 = time.perf_counter()
            logits = model(images)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) / images.size(0) * 1000)  # ms per sample
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_latency = np.mean(latencies)

    print(f"  val accuracy : {accuracy:.4f} ({correct}/{total})", flush=True)
    print(f"  avg latency  : {avg_latency:.3f} ms/sample", flush=True)
    return accuracy, avg_latency


def run_conversion():
    print("Running PyTorch -> ONNX -> TFLite conversion...", flush=True)
    result = subprocess.run(
        [sys.executable, "convert_model.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, flush=True)
        raise RuntimeError(
            f"convert_model.py failed with exit code {result.returncode}"
        )
    print(result.stdout, flush=True)


def benchmark_tflite(cfg):
    print("Benchmarking TFLite model...", flush=True)

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow not installed - cannot benchmark TFLite")

    val_loader, _ = get_val_loader(cfg)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []

    for images, labels in val_loader:
        # TFLite expects NHWC - PyTorch gives NCHW so we transpose
        images_np = images.numpy().transpose(0, 2, 3, 1)

        for i in range(images_np.shape[0]):
            img = np.expand_dims(images_np[i], axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], img)

            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)

            output = interpreter.get_tensor(output_details[0]["index"])
            pred = np.argmax(output, axis=1)[0]
            if pred == labels[i].item():
                correct += 1
            total += 1

    accuracy = correct / total
    avg_latency = np.mean(latencies)
    model_size_kb = os.path.getsize(TFLITE_PATH) / 1024

    print(f"  val accuracy : {accuracy:.4f} ({correct}/{total})", flush=True)
    print(f"  avg latency  : {avg_latency:.3f} ms/sample", flush=True)
    print(f"  model size   : {model_size_kb:.1f} KB", flush=True)
    return accuracy, avg_latency, model_size_kb


def log_to_mlflow(
    run_id, pytorch_acc, pytorch_latency, tflite_acc, tflite_latency, tflite_size_kb
):
    mlflow.set_tracking_uri("http://172.24.198.42:5050")

    accuracy_drop = pytorch_acc - tflite_acc
    speedup = pytorch_latency / tflite_latency if tflite_latency > 0 else 0

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "eval_pytorch_val_accuracy": pytorch_acc,
                "eval_pytorch_latency_ms": pytorch_latency,
                "eval_tflite_val_accuracy": tflite_acc,
                "eval_tflite_latency_ms": tflite_latency,
                "eval_tflite_size_kb": tflite_size_kb,
                "eval_accuracy_drop": accuracy_drop,
                "eval_inference_speedup": speedup,
            }
        )

        mlflow.log_artifact(TFLITE_PATH, artifact_path="quantized_model")
        mlflow.log_artifact(ONNX_PATH, artifact_path="quantized_model")

    print(f"Logged metrics to MLflow run {run_id}", flush=True)
    print(f"  accuracy drop : {accuracy_drop:.4f}", flush=True)
    print(f"  speedup       : {speedup:.2f}x", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", required=True, help="MLflow run ID from the training stage"
    )
    parser.add_argument(
        "--config-dir",
        default=os.path.join(os.getcwd(), "configs"),
        help="Path to Hydra config directory",
    )
    args = parser.parse_args()

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(config_name="config")

    device = torch.device("cpu")

    pytorch_acc, pytorch_latency = evaluate_pytorch(cfg, device)
    run_conversion()
    tflite_acc, tflite_latency, tflite_size_kb = benchmark_tflite(cfg)

    log_to_mlflow(
        run_id=args.run_id,
        pytorch_acc=pytorch_acc,
        pytorch_latency=pytorch_latency,
        tflite_acc=tflite_acc,
        tflite_latency=tflite_latency,
        tflite_size_kb=tflite_size_kb,
    )

    print("\nDone.", flush=True)
    print(
        f"  PyTorch  acc={pytorch_acc:.4f}  latency={pytorch_latency:.2f} ms",
        flush=True,
    )
    print(
        f"  TFLite   acc={tflite_acc:.4f}  latency={tflite_latency:.2f} ms", flush=True
    )


if __name__ == "__main__":
    main()
