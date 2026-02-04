import torch
import torch.nn as nn
import torch.nn.functional as F


# double convolutional layers
def double_conv(channels_in, channels_out):
    convs = nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels_out, channels_out, kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True)
    )
    return convs

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # downsize height and width
        self.pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # down layers
        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        # downsize number of layers
        self.transp1 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.transp2 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.transp3 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.transp4 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        
        # up layers, dobbelt inputchannels because of skip connections
        self.up_conv1 = double_conv(1024, 512)
        self.up_conv2 = double_conv(512, 256)
        self.up_conv3 = double_conv(256, 128)
        self.up_conv4 = double_conv(128, 64)

        # Last layer
        self.out = nn.Conv2d(64, 3, kernel_size = 1)

    def forward(self, input):
        # down
        down1 = self.down_conv1(input)
        down2 = self.pooling(down1)
        down3 = self.down_conv2(down2)
        down4 = self.pooling(down3)
        down5 = self.down_conv3(down4)
        down6 = self.pooling(down5)
        down7 = self.down_conv4(down6)
        down8 = self.pooling(down7)
        down9 = self.down_conv5(down8)

        # up
        up1 = self.transp1(down9)
        up2 = self.up_conv1(torch.cat((down7, up1), 1))
        up3 = self.transp2(up2)
        up4 = self.up_conv2(torch.cat((down5, up3), 1))
        up5 = self.transp3(up4)
        up6 = self.up_conv3(torch.cat((down3, up5), 1))
        up7 = self.transp4(up6)
        up8 = self.up_conv4(torch.cat((down1, up7), 1))

        out = self.out(up8)
        return out

if __name__ == '__main__':
    input_image = torch.rand((1, 1, 32, 32))
    print(input_image.size())
    model = Unet()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)







