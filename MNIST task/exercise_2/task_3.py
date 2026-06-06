"""
Task 3 – Evaluate forgetting on digit 7 and retention on remaining digits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

FORGET_CLASS = 7
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_full = datasets.MNIST("data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_full, batch_size=1000, shuffle=False)

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

# load both models
baseline = Net().to(DEVICE)
baseline.load_state_dict(torch.load("MNIST task/exercise_2/task2_model.pth", map_location=DEVICE))

unlearned = Net().to(DEVICE)
unlearned.load_state_dict(torch.load("MNIST task/exercise_2/task2_unlearned_model.pth", map_location=DEVICE))

baseline_acc  = evaluate_per_class(baseline,  test_loader)
unlearned_acc = evaluate_per_class(unlearned, test_loader)

# Print comparison 
print(f"\nPer-class accuracy comparison (forget class = digit {FORGET_CLASS}):")
print(f"{'Digit':<8} {'Baseline':>10} {'Unlearned':>10} {'Change':>10}")
print("-" * 42)
for digit in range(10):
    change = unlearned_acc[digit] - baseline_acc[digit]
    marker = "  - forget target" if digit == FORGET_CLASS else ""
    print(f"  {digit:<6} {baseline_acc[digit]:>9.2f}% {unlearned_acc[digit]:>9.2f}% {change:>+9.2f}%{marker}")

#plot
digits = list(range(10))
x      = range(len(digits))

fig, ax = plt.subplots(figsize=(10, 5))
bars_base = ax.bar([i - 0.2 for i in x], [baseline_acc[d]  for d in digits],
                   width=0.4, label="Baseline", color="steelblue")
bars_unl  = ax.bar([i + 0.2 for i in x], [unlearned_acc[d] for d in digits],
                   width=0.4, label="Unlearned", color="tomato")

ax.set_xticks(list(x))
ax.set_xticklabels([f"Digit {d}" + (" ★" if d == FORGET_CLASS else "") for d in digits])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.set_title(f"Exercise 2: Gradient Ascent Unlearning – Digit {FORGET_CLASS} as Forget Target")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("task3_unlearning_eval.png", dpi=150)
plt.show()
print("\nPlot saved - task3_unlearning_eval.png")