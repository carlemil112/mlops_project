import os
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext

from carbontracker.tracker import CarbonTracker

# model script
from model.fer_model import FERModel

# Config file handling with hydra
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# to start training with deepspeed: deepspeed --num_gpus=2 train_pytorch.py
# Watch VRAM while it runs with: watch -n 1 nvidia-smi
import deepspeed
import mlflow
import mlflow.pytorch
import sys

import requests


def _get_arg_value(args, name):
    prefix = f"{name}="
    for i, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _strip_arg(args, name):
    stripped_args = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == name:
            skip_next = True
            continue
        if arg.startswith(f"{name}="):
            continue
        stripped_args.append(arg)
    return stripped_args


# Filter out --local_rank argument injected by DeepSpeed before Hydra sees it
print(f"sys.argv before filter: {sys.argv}", flush=True)
local_rank_arg = _get_arg_value(sys.argv, "--local_rank")
if local_rank_arg is not None:
    os.environ.setdefault("LOCAL_RANK", local_rank_arg)
sys.argv = _strip_arg(sys.argv, "--local_rank")
print(f"sys.argv after filter: {sys.argv}", flush=True)

print(f"CWD: {os.getcwd()}", flush=True)
print(f"configs exists: {os.path.exists('configs')}", flush=True)
print(f"files in CWD: {os.listdir('.')}", flush=True)

"""
@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig): ...


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    print("=== TRAIN FUNCTION CALLED ===", flush=True)
    print(f"CWD: {os.getcwd()}", flush=True)
    print(f"cfg.mlflow: {cfg.mlflow}", flush=True)

"""
# Dataset


class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

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

        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # MLFlow configuration
    mlflow.set_tracking_uri("http://172.24.198.42:5050")
    # Config loading with hydra

    SEED = cfg.seed
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_rank_zero = rank == 0
    if is_rank_zero:
        mlflow.set_experiment(cfg.mlflow.name)

    DATA_PATH = os.path.join(get_original_cwd(), cfg.paths.data_path)
    # Transforms
    # This function runs on every image hitting the model
    transform = transforms.Compose(
        [
            transforms.Resize(
                (cfg.script.img_size, cfg.script.img_size)
            ),  # target_size=(IMG_SIZE, IMG_SIZE)
            transforms.ToTensor(),  # [0,255] → [0.0, 1.0] + tensor
            transforms.Normalize(
                mean=[cfg.script.dataset_mean],
                std=[cfg.script.dataset_std],  # (pixel - mean) / std
            ),
        ]
    )
    # Data split + DataLoader
    print("Opsætter data generators...")

    full_dataset = FERDataset(DATA_PATH, transform=transform)

    # Svarer til validation_split=0.2
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),  # Reproducible split
    )

    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=SEED,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    # Generator for train data
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.script.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
    )

    # Generator for val data
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
    )

    # Carbontracker implementation before training
    try:

        class DummyTracker:
            def epoch_start(self):
                pass

            def epoch_end(self):
                pass

            def stop(self):
                pass

        try:
            if is_rank_zero:
                tracker = CarbonTracker(cfg.script.epochs, log_to_file=False)
            else:
                tracker = DummyTracker()
        except Exception:
            tracker = DummyTracker()
    except Exception as e:
        print(f"CarbonTracker init fejlede: {e}, fortsætter uden tracking", flush=True)
        tracker = None

    # Training
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    print(f"Bruger device: {device}")

    num_classes = full_dataset.num_classes
    model = FERModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.script.lr)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        optimizer=optimizer,
        model=model,
        model_parameters=model.parameters(),
        config=cfg.deepspeed.config_path,
    )

    # Boiler-plate for setup af model før læring (strategi)
    criterion = nn.CrossEntropyLoss()

    # Kontroller learning_rate dynamisk
    reduce_lr = ReduceLROnPlateau(
        model_engine.optimizer.optimizer,  # ← den indpakkede originale
        mode="min",
        factor=0.2,
        patience=5,
        min_lr=0.00001,
    )

    # Output mappe
    out_dir = os.path.join("outputs", "fer_run")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "best_emotion_model.pt")

    best_val_acc = float("-inf")
    early_stop_count = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    run_context = mlflow.start_run() if is_rank_zero else nullcontext()
    with run_context:
        if is_rank_zero:
            mlflow.set_tags(
                {
                    "git.commit": os.getenv("GIT_COMMIT", ""),
                    "git.branch": os.getenv("GIT_BRANCH", ""),
                    "jenkins.job": os.getenv("JOB_NAME", ""),
                    "jenkins.build_number": os.getenv("BUILD_NUMBER", ""),
                    "jenkins.build_url": os.getenv("BUILD_URL", ""),
                    "data.version": os.getenv("DATA_VERSION", ""),
                    "docker_image": f"{os.getenv('REGISTRY_URL', '')}/rasmil112:{os.getenv('GIT_COMMIT', '')[:7]}",
                    "distributed.world_size": world_size,
                }
            )

            # write run ID to file so Jenkins can pass it to evaluate.py
            run_id = mlflow.active_run().info.run_id
            with open("outputs/fer_run/mlflow_run_id.txt", "w") as f:
                f.write(run_id)

            mlflow.log_params(
                {
                    "train_dir": DATA_PATH,
                    "img_size": cfg.script.img_size,
                    "batch_size_per_gpu": cfg.script.batch_size,
                    "global_batch_size": cfg.script.batch_size * world_size,
                    "epochs_max": cfg.script.epochs,
                    "dataset_mean": cfg.script.dataset_mean,
                    "dataset_std": cfg.script.dataset_std,
                }
            )

        # Fit model to training data
        if is_rank_zero:
            print("Starter træning...")

        for epoch in range(cfg.script.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # initiating carbontracker
            if is_rank_zero:
                tracker.epoch_start()

            # --- Træning ---
            model_engine.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model_engine(images)
                loss = criterion(logits, labels)
                model_engine.backward(loss)
                model_engine.step()

                running_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

            if torch.distributed.is_initialized():
                train_stats = torch.tensor(
                    [running_loss, correct, total], dtype=torch.float64, device=device
                )
                torch.distributed.all_reduce(
                    train_stats, op=torch.distributed.ReduceOp.SUM
                )
                running_loss, correct, total = train_stats.tolist()

            train_loss = running_loss / total
            train_acc = correct / total

            # Validation
            model_engine.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model_engine(images)
                    loss = criterion(logits, labels)

                    val_loss_sum += loss.item() * labels.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            if torch.distributed.is_initialized():
                val_stats = torch.tensor(
                    [val_loss_sum, val_correct, val_total],
                    dtype=torch.float64,
                    device=device,
                )
                torch.distributed.all_reduce(
                    val_stats, op=torch.distributed.ReduceOp.SUM
                )
                val_loss_sum, val_correct, val_total = val_stats.tolist()

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if is_rank_zero:
                print(
                    f"Epoch [{epoch+1}/{cfg.script.epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )

                try:
                    requests.post(
                        "http://172.24.198.42:8000/metrics/training",
                        json={
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        },
                        timeout=2,
                    )
                except requests.RequestException:
                    pass

            # Find and save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if is_rank_zero:
                    torch.save(model_engine.module.state_dict(), best_model_path)
                    print(f"  → Ny bedste model gemt (val_acc={best_val_acc:.4f})")

            # Kontroller learning_rate dynamisk
            reduce_lr.step(val_loss)

            # Early stopping (før 50 epochs)
            # Stop hvis den ikke ser forbedring efter 10 epochs
            if val_loss < min(history["val_loss"][:-1], default=float("inf")):
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= 10:
                    if is_rank_zero:
                        print(f"Early stopping efter epoch {epoch+1}")
                    break

            if is_rank_zero:
                tracker.epoch_end()

        if is_rank_zero:
            tracker.stop()

        # Plots + best model + model registry in MLflow
        plot_path = os.path.join(out_dir, "training_results.png")
        if is_rank_zero:
            plot_training_history(history, plot_path)

            mlflow.log_artifact(plot_path, artifact_path="plots")
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
            fresh_model = FERModel(num_classes)
            fresh_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
            fresh_model.eval()

            mlflow.pytorch.log_model(
                fresh_model,
                artifact_path="model",
                registered_model_name="fer_emotion_model",
            )

    # Resultater visualisering


def plot_training_history(history, plot_path):
    acc = history["train_acc"]
    val_acc = history["val_acc"]
    loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training vs Validation Accuracy")

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training vs Validation Loss")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    try:
        train()
    except SystemExit as e:
        print(f"SystemExit: {e}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
