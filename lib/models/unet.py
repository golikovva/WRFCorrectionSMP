import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        hidden_channels=(64, 128, 256, 512, 1024),
        bilinear=False,
    ):
        super().__init__()

        if len(hidden_channels) != 5:
            raise ValueError(
                f"hidden_channels must contain 5 values, got {len(hidden_channels)}"
            )

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_channels = tuple(hidden_channels)
        self.bilinear = bilinear

        c1, c2, c3, c4, c5 = hidden_channels

        self.inc = DoubleConv(n_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        factor = 2 if bilinear else 1

        self.down4 = Down(c4, c5 // factor)

        self.up1 = Up(c5, c4 // factor, bilinear)
        self.up2 = Up(c4, c3 // factor, bilinear)
        self.up3 = Up(c3, c2 // factor, bilinear)
        self.up4 = Up(c2, c1, bilinear)

        self.outc = OutConv(c1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)