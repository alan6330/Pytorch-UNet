# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):##（n,3,w,h）
        x1 = self.inc(x)##（n,64,w,h）
        x2 = self.down1(x1)##（n,128,w/2,h/2）
        x3 = self.down2(x2)##（n,256,w/4,h/4）
        x4 = self.down3(x3)##（n,512,w/8,h/8）
        x5 = self.down4(x4)##（n,512,w/16,h/16）
        x = self.up1(x5, x4)##（n,256,w/8,h/8）
        x = self.up2(x, x3)##（n,128,w/4,h/4）
        x = self.up3(x, x2)##（n,64,w/2,h/2）
        x = self.up4(x, x1)##（n,64,w,h）
        x = self.outc(x)##（n,n_classes,w,h）
        return F.sigmoid(x)
