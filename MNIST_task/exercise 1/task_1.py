"""
Task 1 – Train on digits 0-4 and record baseline accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Config
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

# Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_full = datasets.MNIST("data", train=True, download=True, transform=transform)
test_full = datasets.MNIST("data", train=False, download=True, transform=transform)


def filter_by_labels(dataset, labels):
    indices = [i for i, (_, y) in enumerate(dataset) if y in labels]
    return Subset(dataset, indices)


task1_train = filter_by_labels(train_full, range(5))  # digits 0-4
task1_test = filter_by_labels(test_full, range(5))

train_loader = DataLoader(task1_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(task1_test, batch_size=1000, shuffle=False)


# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 outputs – keeps architecture consistent

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)


# Train/Eval
def train(model, optimizer, loader):
    model.train()
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        correct += model(data).argmax(dim=1).eq(target).sum().item()
        total += len(target)
    return 100.0 * correct / total


# Run Task 1
model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

for epoch in range(1, EPOCHS + 1):
    train(model, optimizer, train_loader)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch}/{EPOCHS}  –  accuracy on digits 0-4: {acc:.2f}%")

task1_accuracy = evaluate(model, test_loader)
print(f"\nTask-1 baseline accuracy (0-4): {task1_accuracy:.2f}%")

# Save model weights – needed as the starting point for Tasks 2 & 3
torch.save(model.state_dict(), "task1_model.pth")
print("Model saved: task1_model.pth")
