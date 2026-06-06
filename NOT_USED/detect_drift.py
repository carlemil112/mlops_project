import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchdrift
import mlflow
import hydra
from omegaconf import DictConfig
from model.fer_model import FERModel

import requests


class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="L"):
        self.samples = []
        self.transform = transform
        self.mode = mode

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(cls_path, fname), self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert(self.mode)
        if self.transform:
            img = self.transform(img)
        return img, label


def extract_features_and_score(
    loader, feature_extractor, detector, device, is_color_rgb=False
):
    """
    Hjælpefunktion til at trække features ud fra en loader og beregne
    drift score + p-værdi via det fittede detector-objekt.
    """
    feature_extractor.eval()
    all_features = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)

            # Hvis loaderen spytter RGB ud (Scenario 2), konverterer vi til grayscale her batch-vis
            if is_color_rgb:
                imgs = imgs.mean(dim=1, keepdim=True)

            features = feature_extractor(imgs)
            all_features.append(features.cpu())

    # Saml alle batches til én 2D tensor: (Antal_samples, Feature_dimensioner)
    test_features = torch.cat(all_features, dim=0)

    # Beregn MMD score og P-værdi
    drift_score = detector(test_features)
    p_value = detector.compute_p_value(drift_score)

    return drift_score.item(), p_value.item()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def detect_drift(cfg: DictConfig):
    # 1. Load config parameters
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.name)
    # Config loading with hydra
    TRAIN_DIR = cfg.train_script.data.data_path  # Balanced data
    IMG_SIZE = cfg.train_script.data.image_size  # Standard size image
    BATCH_SIZE = cfg.train_script.training.batch_size  # Amount if images pr. batch
    DATASET_MEAN = cfg.train_script.data.dataset_mean
    DATASET_STD = cfg.train_script.data.dataset_std
    SEED = cfg.train_script.seed

    # 2. Define transforms
    reference_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # target_size=(IMG_SIZE, IMG_SIZE)
            transforms.ToTensor(),  # [0,255] → [0.0, 1.0] + tensor
            transforms.Normalize(
                mean=[DATASET_MEAN],
                std=[DATASET_STD],  # (pixel - mean) / std
            ),
        ]
    )
    drift_transform_scale = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE * 2, IMG_SIZE * 2)
            ),  # Double size to simulate scenario with larger input image
            transforms.ToTensor(),
            transforms.Normalize(mean=[DATASET_MEAN], std=[DATASET_STD]),
        ]
    )
    drift_transform_color = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Same size
            transforms.ToTensor(),  # 3 channels automatically
            transforms.Normalize(
                mean=[DATASET_MEAN, DATASET_MEAN, DATASET_MEAN],
                std=[DATASET_STD, DATASET_STD, DATASET_STD],
            ),
        ]
    )
    # 3. Load dataset + split 80/20

    # Manual loading of seed, to make sure the same images are used as input to functions

    reference_dataset = FERDataset(TRAIN_DIR, transform=reference_transform, mode="L")

    val_size = int(0.2 * len(reference_dataset))
    train_size = len(reference_dataset) - val_size

    _, test_subset = random_split(
        reference_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    reference_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Load model
    local_path = mlflow.artifacts.download_artifacts(
        "mlflow-artifacts:/27/b19b0326c40b4873b288d81a9fc6e28e/artifacts/checkpoints/best_emotion_model.pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FERModel(num_classes=reference_dataset.num_classes)
    model.load_state_dict(torch.load(local_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. Lav driftede loaders
    scale_dataset = FERDataset(TRAIN_DIR, drift_transform_scale, mode="L")
    color_dataset = FERDataset(TRAIN_DIR, drift_transform_color, mode="RGB")

    scale_loader = DataLoader(scale_dataset, batch_size=BATCH_SIZE)
    color_loader = DataLoader(color_dataset, batch_size=BATCH_SIZE)

    # 6. Kør TorchDrift

    reference_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

    # Feature extractor (remove last layer)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    # 6a. TRÆK REFERENCE FEATURES UD MANUELT
    print("Akkumulerer reference features...")
    ref_features_list = []
    with torch.no_grad():
        for imgs, _ in reference_loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)
            ref_features_list.append(feats.cpu())  # Gem på CPU for at spare VRAM

    # Saml til én stor 2D-tensor: (Antal_samples, Feature_dim)
    # reference_features = torch.cat(ref_features_list, dim=0)

    # 6b. KONFIGURÉR OG FIT DETECTOR DIREKTE
    detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(reference_loader, feature_extractor, detector, n_batches=10)

    # Run on drifted data
    scale_score = detector(
        torchdrift.utils.extract_features(scale_loader, feature_extractor)
    )
    color_score = detector(
        torchdrift.utils.extract_features(color_loader, feature_extractor)
    )

    scale_p_val = detector.compute_p_value(scale_score)
    color_p_val = detector.compute_p_value(color_score)

    print(f"Scale drift p-value: {scale_p_val:.4f}")
    print(f"Color drift p-value: {color_p_val:.4f}")

    # 7. Log to MLflow

    # Scenario 2 (Color)
    color_score, color_p_val = extract_features_and_score(
        color_loader, feature_extractor, detector, device, is_color_rgb=True
    )

    # 7. Log til MLflow
    with mlflow.start_run():
        mlflow.log_metric("scale_drift_score", scale_score)
        mlflow.log_metric("scale_drift_p_value", scale_p_val)

        mlflow.log_metric("color_drift_score", color_score)
        mlflow.log_metric("color_drift_p_value", color_p_val)

        mlflow.set_tag("drift_detection", "torchdrift_mmd")

    print(f"Scenario 1 (Scale) - Score: {scale_score:.4f}, P-Value: {scale_p_val:.4f}")
    print(f"Scenario 2 (Color) - Score: {color_score:.4f}, P-Value: {color_p_val:.4f}")

    try:
        requests.post(
            "http://172.24.198.42:8000/metrics/drift",
            json={
                "scale_drift_score": scale_score.item(),
                "color_drift_score": color_score.item(),
                "scale_drift_p_value": scale_p_val.item(),
                "color_drift_p_value": color_p_val.item(),
            },
            timeout=2,
        )
    except requests.RequestException:
        pass

    if __name__ == "__main__":
        detect_drift()
