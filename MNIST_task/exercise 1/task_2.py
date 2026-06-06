"""
Task 2 – Naïve sequential learning on 5-9 (catastrophic forgetting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
MOMENTUM = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_full = datasets.MNIST("data", train=True, download=True, transform=transform)
test_full = datasets.MNIST("data", train=False, download=True, transform=transform)


def filter_by_labels(dataset, labels):
    indices = [i for i, (_, y) in enumerate(dataset) if y in labels]
    return Subset(dataset, indices)


task2_train = filter_by_labels(train_full, range(5, 10))
task1_test = filter_by_labels(test_full, range(5))  # to measure forgetting
task2_test = filter_by_labels(test_full, range(5, 10))  # to measure new learning

train_loader = DataLoader(task2_train, batch_size=BATCH_SIZE, shuffle=True)
test1_loader = DataLoader(task1_test, batch_size=1000, shuffle=False)
test2_loader = DataLoader(task2_test, batch_size=1000, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)


# Load task 1 weights, same starting point
model = Net().to(DEVICE)
model.load_state_dict(torch.load("MNIST task/exercise 1/task1_model.pth"))

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        correct += model(data).argmax(dim=1).eq(target).sum().item()
        total += len(target)
    return 100.0 * correct / total


TASK1_BASELINE = 99.69  # recorded value from Task 1

acc_old_history = []
acc_new_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()

    acc_old = evaluate(model, test1_loader)
    acc_new = evaluate(model, test2_loader)
    acc_old_history.append(acc_old)
    acc_new_history.append(acc_new)
    print(
        f"Epoch {epoch}/{EPOCHS}  –  old(0-4): {acc_old:.2f}%  |  new(5-9): {acc_new:.2f}%"
    )

# Plot
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(8, 5))
plt.axhline(
    TASK1_BASELINE,
    color="green",
    linestyle="--",
    label=f"Task-1 baseline ({TASK1_BASELINE}%)",
)
plt.plot(epochs, acc_old_history, "r-o", label="Old digits (0-4)")
plt.plot(epochs, acc_new_history, "b-s", label="New digits (5-9)")
plt.title("Task 2: Naïve Sequential Learning – Catastrophic Forgetting")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task2_forgetting.png", dpi=150)
plt.show()
print("Plot saved - task2_forgetting.png")
