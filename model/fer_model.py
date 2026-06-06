import torch.nn as nn
import torch


# Model-arkitektur (CNN)
class FERModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Conv2D: finder simple features (kanter, linjer)
        # BatchNormalization: stabiliserer læringen
        # MaxPooling: gør billedet mindre (halverer størrelsen) for at reducere beregninger
        self.block1 = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=3, padding=1
            ),  # 1 kanal ind (grayscale), 64 filtre ud
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(
                0.4
            ),  # Slukker tilfældige neuroner for at hjælpe med at forhindre overfitting
        )

        # Dybden øges (128 filtre) for at finde mere komplekse mønstre (former, øjne)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
        )

        # Dybden øges igen (128 filtre)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
        )

        # Flatten: laver 2D billedet om til en lang liste af tal. Fully connected lag
        # Efter 3x MaxPool2d: 48 → 24 → 12 → 6, så 128 * 6 * 6 = 4608
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Output Lag: num_classes neuroner – en for hver følelse
            # Softmax er implicit i CrossEntropyLoss, så den tilføjes ikke her
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def main():
    model = FERModel(num_classes=5)
    model.eval()
    input = torch.randn(8, 1, 48, 48)
    with torch.no_grad():
        output = model(input)

    model.forward(x=input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input Sample: \n {output[0]}")


if __name__ == "__main__":
    main()
