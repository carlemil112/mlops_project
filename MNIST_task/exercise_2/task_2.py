"""
Task 2 - Targeted unlearning on digit 7 using gradient ascent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 64
UNLEARN_EPOCHS = 2
LR_UNLEARN = 0.0005
FORGET_CLASS = 7
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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


# Only the forget class for unlearning
forget_train = filter_by_labels(train_full, [FORGET_CLASS])
forget_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True)


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


model = Net().to(DEVICE)
model.load_state_dict(
    torch.load("MNIST task/exercise_2/task2_model.pth", map_location=DEVICE)
)

optimizer = optim.SGD(model.parameters(), lr=LR_UNLEARN, momentum=0.9)

# gradient ascent unlearning
print(f"\nUnlearning digit {FORGET_CLASS} via gradient ascent...")
for epoch in range(1, UNLEARN_EPOCHS + 1):
    model.train()
    for data, target in forget_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        (-loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f"  Unlearn epoch {epoch}/{UNLEARN_EPOCHS} done")

torch.save(model.state_dict(), "task2_unlearned_model.pth")
print("\nUnlearned model saved: task2_unlearned_model.pth")
