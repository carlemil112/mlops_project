"""
Task 1 – Train on full MNIST (0-9) and record baseline accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED       = 42
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 0.01
MOMENTUM   = 0.9
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST("data", train=True,  download=True, transform=transform)
test_full  = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_full, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_full,  batch_size=1000,       shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1    = nn.Conv2d(1, 32, 3, 1)
        self.conv2    = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1      = nn.Linear(9216, 128)
        self.fc2      = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)

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
        total   += len(target)
    return 100.0 * correct / total

@torch.no_grad()
def evaluate_per_class(model, loader):
    model.eval()
    correct = {i: 0 for i in range(10)}
    total   = {i: 0 for i in range(10)}
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        preds = model(data).argmax(dim=1)
        for t, p in zip(target, preds):
            total[t.item()]   += 1
            correct[t.item()] += (t == p).item()
    return {i: 100.0 * correct[i] / total[i] for i in range(10)}

model     = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

for epoch in range(1, EPOCHS + 1):
    train(model, optimizer, train_loader)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch}/{EPOCHS}  –  overall accuracy: {acc:.2f}%")

print("\nPer-class accuracy:")
per_class = evaluate_per_class(model, test_loader)
for digit, acc in per_class.items():
    print(f"  Digit {digit}: {acc:.2f}%")

torch.save(model.state_dict(), "task2_model.pth")
print("\nModel saved: task2_model.pth")