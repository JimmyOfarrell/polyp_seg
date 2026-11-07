import torch.nn as nn
from torch.nn.functional import interpolate

class PointwiseCNN(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, output_size=(512, 512)):
        super().__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.output_size = tuple(output_size)

    def forward(self, x):
        out = self.pointwise_conv(x)
        if out.shape[-2:] != self.output_size: out = interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)
        return out

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        kwargs = {'scale_factor': self.scale_factor, 'mode': self.mode}
        if self.mode in ('bilinear', 'bicubic'): kwargs['align_corners'] = self.align_corners
        x = self.interp(x, **kwargs)
        return x

class ConvUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2,upsample_factors=None):
        super().__init__()
        if upsample_factors is None: upsample_factors = [2, 2]
        layers = []
        current_channels = in_channels

        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=False))
            current_channels = current_channels // 2

        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_net(x)

class TransposeConvUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=None):
        super().__init__()
        if upsample_factors is None: upsample_factors = [2, 2]
        layers = []
        current_channels = in_channels

        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(nn.ConvTranspose2d(current_channels // 2, current_channels // 2, kernel_size=factor, stride=factor))
            current_channels = current_channels // 2

        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_net(x)
    
class ConvTwoUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=None, relu=False):
        super().__init__()
        if upsample_factors is None: upsample_factors = [2, 2]
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            if relu:
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(current_channels // 2, current_channels // 2, kernel_size=1))
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=False))
            current_channels = current_channels // 2
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_net(x)
    
class DeepConvUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=2, upsample_factors=None):
        super().__init__()
        if upsample_factors is None: upsample_factors = [2, 2]
        layers = []
        current_channels = in_channels
        for factor in upsample_factors:
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(current_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(current_channels, current_channels // 2, kernel_size=1))
            layers.append(nn.BatchNorm2d(current_channels//2))
            layers.append(nn.ReLU())
            layers.append(Upsample(scale_factor=factor, mode='bilinear', align_corners=False))
            current_channels = current_channels // 2
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.upsample_net = nn.Sequential(*layers)

    def forward(self,x):
        return self.upsample_net(x)