import torch.nn as nn
import torch


# PatchGan discriminator
class Discriminator(nn.Module):
    def __init__(
        self,
        input_channels=4,  # 1 for grayscale 3 for RGB
        stride=2,
        padding=1,
        kernel_size=4,
        relu_slope=0.2,
    ):

        super().__init__()

        self.layers = [64, 128, 256, 512]  # layer sizes

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.layers[0], kernel_size, stride, padding),
            nn.LeakyReLU(relu_slope),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layers[0], self.layers[1], kernel_size, stride, padding),
            nn.BatchNorm2d(self.layers[1]),
            nn.LeakyReLU(relu_slope),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layers[1], self.layers[2], kernel_size, stride, padding),
            nn.BatchNorm2d(self.layers[2]),
            nn.LeakyReLU(relu_slope),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                self.layers[2],
                self.layers[3],
                kernel_size,
                1,  # PatchGan changes stride to 1 so spartial map survives
                padding,
            ),
            nn.BatchNorm2d(self.layers[3]),
            nn.LeakyReLU(relu_slope),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.layers[3], 1, kernel_size, 1, padding), nn.Sigmoid()
        )

    # NOT USED IN PATCHGAN!!
    # Get spartial size
    # final_spartial = img_size // (stride**4)
    # 512/(2⁴)=16, which means 16x16 is the final spartial size (height, width)
    # self.flattened = self.layers[3] * final_spartial * final_spartial
    # self.fc = nn.Linear(self.flattened, 1)

    def forward(self, x):
        # concatenated = torch.cat([real_data, fake_data])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        logit_map = self.out_conv(x)
        return logit_map


# Sanity checkkkk

if __name__ == "__main__":

    D = Discriminator(input_channels=4)
    x = torch.randn(8, 4, 256, 256)
    out = D(x)
    print(out.shape)
