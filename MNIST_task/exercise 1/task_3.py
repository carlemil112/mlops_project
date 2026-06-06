"""
Task 3 – Experience Replay + EWC to mitigate catastrophic forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt

SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
MOMENTUM = 0.9
REPLAY_PER_CLS = 200  # replay samples stored per class
EWC_LAMBDA = 400
FISHER_SAMPLES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_full = datasets.MNIST("data", train=True, download=True, transform=transform)
test_full = datasets.MNIST("data", train=False, download=True, transform=transform)


def filter_by_labels(dataset, labels):
    indices = [i for i, (_, y) in enumerate(dataset) if y in labels]
    return Subset(dataset, indices)


task1_train = filter_by_labels(train_full, range(5))
task2_train = filter_by_labels(train_full, range(5, 10))
task1_test = filter_by_labels(test_full, range(5))
task2_test = filter_by_labels(test_full, range(5, 10))

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


class EWC:
    def __init__(self, model, dataset, device):
        self.device = device
        self.star = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher = self._compute_fisher(model, dataset)

    def _compute_fisher(self, model, dataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.eval()
        for i, (x, y) in enumerate(loader):
            if i >= FISHER_SAMPLES:
                break
            x, y = x.to(self.device), y.to(self.device)
            model.zero_grad()
            F.nll_loss(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        for n in fisher:
            fisher[n] /= FISHER_SAMPLES
        return fisher

    def penalty(self, model):
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.star[n]) ** 2).sum()
        return (EWC_LAMBDA / 2.0) * loss


# ── Memory Buffer ─────────────────────────────────────────────────────────────
class MemoryBuffer:
    def __init__(self, per_class=REPLAY_PER_CLS):
        self.per_class = per_class
        self.store = defaultdict(list)

    def fill(self, dataset):
        for img, lbl in DataLoader(dataset, batch_size=512, shuffle=True):
            for x, y in zip(img, lbl):
                c = y.item()
                if len(self.store[c]) < self.per_class:
                    self.store[c].append(x)
        total = sum(len(v) for v in self.store.values())
        print(
            f"  Memory buffer: {total} samples across classes {sorted(self.store.keys())}"
        )

    def as_loader(self):
        xs = [x for imgs in self.store.values() for x in imgs]
        ys = [c for c, imgs in self.store.items() for _ in imgs]
        ds = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        correct += model(data).argmax(dim=1).eq(target).sum().item()
        total += len(target)
    return 100.0 * correct / total


# ── Setup ─────────────────────────────────────────────────────────────────────
model = Net().to(DEVICE)
model.load_state_dict(torch.load("MNIST task/exercise 1/task1_model.pth"))

print("Filling memory buffer from Task-1 data...")
buf = MemoryBuffer()
buf.fill(task1_train)

print("Computing EWC Fisher information...")
ewc = EWC(model, task1_train, DEVICE)
print("Done.")

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
task2_loader = DataLoader(task2_train, batch_size=BATCH_SIZE, shuffle=True)


# Cyclic replay iterator
class CyclicIter:
    def __init__(self, dl):
        self._loader = dl
        self._it = iter(dl)

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._loader)
            return next(self._it)


replay_iter = CyclicIter(buf.as_loader())

# ── Train ─────────────────────────────────────────────────────────────────────
TASK1_BASELINE = 99.69
acc_old_history = []
acc_new_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    for data, target in task2_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)  # Task-2 loss
        loss += ewc.penalty(model)  # EWC penalty
        rx, ry = next(replay_iter)  # Replay batch
        loss += F.nll_loss(model(rx.to(DEVICE)), ry.to(DEVICE))
        loss.backward()
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
plt.title("Task 3: Experience Replay + EWC")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task3_replay_ewc.png", dpi=150)
plt.show()
print("Plot saved - task3_replay_ewc.png")
