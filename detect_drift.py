import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# For detecting datadrift!
import torchdrift

import mlflow
from model.fer_model import FERModel
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, random_split


class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="L"):
        self.samples = []
        self.transform = transform
        self.mode = mode

        # Finds all subfolders (happy, sad, etc)
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
    

# DETECT DRIFT FUNCTION
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
        
    # Model path
    out_dir = os.path.join("outputs", "fer_run")
    best_model_path = os.path.join(out_dir, "best_emotion_model.pt")


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
            transforms.Resize((IMG_SIZE * 2, IMG_SIZE * 2)), # Double size to simulate scenario with larger input image
            transforms.ToTensor(),
            transforms.Normalize(transforms.Normalize(mean=[DATASET_MEAN], std=[DATASET_STD])),
               
        ])
    drift_transform_color = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Same size
            transforms.ToTensor(),                    # 3 channels automatically
            transforms.Normalize(mean=[DATASET_MEAN, DATASET_MEAN, DATASET_MEAN], 
                                 std=[DATASET_STD, DATASET_STD, DATASET_STD]),
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
        generator=torch.Generator().manual_seed(SEED)
    )

    # 4. Load model
    torch.load() # VENTER TIL AT JEG ER SIKKER PÅ MODEL PATH FRA MLFLOW


    # 5. Lav driftede billeder

    scale_dataset = FERDataset(TRAIN_DIR, drift_transform_scale, mode="L")
    color_dataset = FERDataset(TRAIN_DIR, drift_transform_color, mode="RGB")
    
    scale_loader = DataLoader(scale_dataset, batch_size=BATCH_SIZE)
    color_loader = DataLoader(color_dataset, batch_size=BATCH_SIZE)

    # 6. Kør TorchDrift



    # 7. Log til MLflow
    