# magnitude pruning experiment for FER model
# runs locally, produces a plot of pruning degree vs accuracy (D4.3)
# and fine-tunes a strongly pruned model (D4.4)

import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.fer_model import FERModel

MODEL_PATH = "pruning_experiments/model.pth"
DATA_PATH = "data/FER-2013/test"
IMG_SIZE = 48
DATASET_MEAN = 0.5147
DATASET_STD = 0.2536
BATCH_SIZE = 64
FINETUNE_EPOCHS = 5
FINETUNE_LR = 0.0001
PRUNING_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STRONG_PRUNE_LEVEL = 0.7  # level used for fine-tuning experiment


class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = sorted(
            c for c in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, c))
        )
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[DATASET_MEAN], std=[DATASET_STD]),
    ])


def get_loader(path, shuffle=False):
    dataset = FERDataset(path, transform=get_transform())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        num_classes = model["classifier.5.weight"].shape[0]
        fer_model = FERModel(num_classes=num_classes)
        fer_model.load_state_dict(model)
        return fer_model
    model.eval()
    return model


def apply_pruning(model, amount):
    # magnitude pruning on all conv and linear layers
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def remove_pruning_masks(model):
    # make pruning permanent
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0


def finetune(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader)
        print(f"  Fine-tune epoch {epoch+1}/{epochs} - val acc: {val_acc:.4f}", flush=True)

    return model


def main():
    print("Loading model and data...", flush=True)
    test_loader = get_loader(DATA_PATH)

    # use train set for fine-tuning
    train_path = DATA_PATH.replace("test", "train_balanced")
    if not os.path.exists(train_path):
        train_path = DATA_PATH.replace("test", "train")
    train_loader = get_loader(train_path, shuffle=True)

    # D4.3: accuracy at different pruning levels
    print("\nD4.3: Pruning sweep...", flush=True)
    accuracies = []

    for amount in PRUNING_LEVELS:
        model = load_model()
        if amount > 0:
            apply_pruning(model, amount)
        acc = evaluate(model, test_loader)
        accuracies.append(acc)
        print(f"  Pruning {int(amount*100)}%: accuracy = {acc:.4f}", flush=True)

    # plot pruning degree vs accuracy
    plt.figure(figsize=(8, 5))
    plt.plot([int(a * 100) for a in PRUNING_LEVELS], [a * 100 for a in accuracies],
             marker="o", linewidth=2, color="#1f77b4")
    plt.xlabel("Pruning degree (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Pruning degree vs. accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pruning_experiments/pruning_vs_accuracy.png", dpi=150)
    plt.close()
    print("\nPlot saved to pruning_experiments/pruning_vs_accuracy.png", flush=True)

    # D4.4: fine-tune strongly pruned model
    print(f"\nD4.4: Fine-tuning {int(STRONG_PRUNE_LEVEL*100)}% pruned model...", flush=True)
    pruned_model = load_model()
    apply_pruning(pruned_model, STRONG_PRUNE_LEVEL)

    acc_before = evaluate(pruned_model, test_loader)
    print(f"  Accuracy before fine-tuning: {acc_before:.4f}", flush=True)

    remove_pruning_masks(pruned_model)
    finetuned_model = finetune(pruned_model, train_loader, test_loader, FINETUNE_EPOCHS, FINETUNE_LR)

    acc_after = evaluate(finetuned_model, test_loader)
    print(f"  Accuracy after fine-tuning:  {acc_after:.4f}", flush=True)
    print(f"  Recovery: {acc_after - acc_before:.4f}", flush=True)

    torch.save(finetuned_model.state_dict(), "pruning_experiments/finetuned_pruned_model.pt")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()