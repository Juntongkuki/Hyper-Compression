from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, f_channels):
        super(UNet, self).__init__()
        with open(f_channels, 'r') as f:
            channels = f.readlines()
        channels = [int(c.strip()) for c in channels]
        self.inc = inconv(n_channels, channels[0], channels[1])
        self.down1 = down(channels[1], channels[2], channels[3])
        self.down2 = down(channels[3], channels[4], channels[5])
        self.down3 = down(channels[5], channels[6], channels[7])
        self.down4 = down(channels[7], channels[8],  channels[9])
        self.up1 = up(channels[10], channels[11], channels[12])
        self.up2 = up(channels[13], channels[14], channels[15])
        self.up3 = up(channels[16], channels[17], channels[18])
        self.up4 = up(channels[19], channels[20], channels[21])
        self.outc = outconv(channels[21], n_classes)

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
        x = self.outc(x)
        return F.sigmoid(x)
