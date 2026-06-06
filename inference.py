import json
import os
import time
from pathlib import Path

import mlflow
import torch
from carbontracker.tracker import CarbonTracker
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.fer_model import FERModel


ORIGINAL_MODEL_PATH = "outputs/fer_run/best_emotion_model.pt"
QUANTIZED_MODEL_PATH = "outputs/fer_run/best_emotion_model_dynamic_quantized.pt"
RESULTS_PATH = "outputs/fer_run/inference_comparison.json"
WARMUP_BATCHES = 20


class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = sorted(
            class_name
            for class_name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, class_name))
        )
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (
                            os.path.join(class_path, file_name),
                            self.class_to_idx[class_name],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


class NoopTracker:
    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def stop(self):
        pass


def build_tracker():
    try:
        return CarbonTracker(epochs=1, log_to_file=False)
    except Exception as error:
        print(f"CarbonTracker disabled: {error}", flush=True)
        return NoopTracker()


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


def load_config():
    config_dir = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config")


def resolve_test_path(cfg):
    configured_test_path = OmegaConf.select(cfg, "paths.test_data_path")
    if configured_test_path:
        return configured_test_path

    train_path = Path(cfg.paths.data_path)
    dataset_root = train_path.parent
    return str(dataset_root / "test")


def build_test_loader(cfg):
    test_path = resolve_test_path(cfg)
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Missing test data folder: {test_path}")

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

    dataset = FERDataset(test_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader, test_path


def warmup(model, loader):
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= WARMUP_BATCHES:
                break
            model(images)


def evaluate(model, loader):
    correct = 0
    total = 0
    latencies = []

    with torch.inference_mode():
        for images, labels in loader:
            start = time.perf_counter()
            logits = model(images)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0
    latency_ms = sum(latencies) / len(latencies) if latencies else 0
    return accuracy, latency_ms, total


def file_size_kb(path):
    return os.path.getsize(path) / 1024


def check_files():
    for path in [ORIGINAL_MODEL_PATH, QUANTIZED_MODEL_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing file: {path}")


def log_to_mlflow(cfg, results):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.name)

    run_id_path = "outputs/fer_run/mlflow_run_id.txt"
    run_id = None
    if os.path.exists(run_id_path):
        with open(run_id_path) as file:
            run_id = file.read().strip() or None

    with mlflow.start_run(run_id=run_id, run_name="inference_quantization_comparison"):
        mlflow.log_params(
            {
                "test_data_path": results["test_data_path"],
                "samples": results["samples"],
                "original_model_path": ORIGINAL_MODEL_PATH,
                "quantized_model_path": QUANTIZED_MODEL_PATH,
            }
        )
        mlflow.log_metrics(
            {
                "inference_original_accuracy": results["original_accuracy"],
                "inference_quantized_accuracy": results["quantized_accuracy"],
                "inference_accuracy_drop": results["accuracy_drop"],
                "inference_original_latency_ms": results["original_latency_ms"],
                "inference_quantized_latency_ms": results["quantized_latency_ms"],
                "inference_speedup": results["speedup"],
                "inference_original_size_kb": results["original_size_kb"],
                "inference_quantized_size_kb": results["quantized_size_kb"],
            }
        )

        mlflow.log_artifact(RESULTS_PATH, artifact_path="inference")
        mlflow.log_artifact(QUANTIZED_MODEL_PATH, artifact_path="quantized_model")

        metadata_path = "outputs/fer_run/quantization_metadata.json"
        if os.path.exists(metadata_path):
            mlflow.log_artifact(metadata_path, artifact_path="quantized_model")


def main():
    check_files()

    cfg = load_config()
    test_loader, test_path = build_test_loader(cfg)
    original_model = load_original_model()
    quantized_model = load_quantized_model()

    tracker = build_tracker()
    tracker.epoch_start()
    warmup(original_model, test_loader)
    warmup(quantized_model, test_loader)
    original_acc, original_latency, samples = evaluate(original_model, test_loader)
    quantized_acc, quantized_latency, _ = evaluate(quantized_model, test_loader)
    tracker.epoch_end()
    tracker.stop()

    speedup = original_latency / quantized_latency if quantized_latency else 0
    accuracy_drop = original_acc - quantized_acc
    original_size = file_size_kb(ORIGINAL_MODEL_PATH)
    quantized_size = file_size_kb(QUANTIZED_MODEL_PATH)

    results = {
        "test_data_path": test_path,
        "samples": samples,
        "original_accuracy": original_acc,
        "quantized_accuracy": quantized_acc,
        "accuracy_drop": accuracy_drop,
        "original_latency_ms": original_latency,
        "quantized_latency_ms": quantized_latency,
        "speedup": speedup,
        "original_size_kb": original_size,
        "quantized_size_kb": quantized_size,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as file:
        json.dump(results, file, indent=2)

    print("Inference comparison")
    print(f"Test path:        {test_path}")
    print(f"Samples:          {samples}")
    print(f"Original acc:     {original_acc:.4f}")
    print(f"Quantized acc:    {quantized_acc:.4f}")
    print(f"Accuracy drop:    {accuracy_drop:.4f}")
    print(f"Original latency: {original_latency:.4f} ms/sample")
    print(f"Quantized latency:{quantized_latency:.4f} ms/sample")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Original size:    {original_size:.1f} KB")
    print(f"Quantized size:   {quantized_size:.1f} KB")
    print(f"Results file:     {RESULTS_PATH}")

    log_to_mlflow(cfg, results)
    print("Logged inference comparison to MLflow")


if __name__ == "__main__":
    main()
