import torch
import torch.nn as nn
import torch.nn.functional as F
from miniSegmentor.aggregator import PointwiseCNN, ConvUpsample, TransposeConvUpsample, ConvTwoUpsample, DeepConvUpsample

class ModelEmb2DNet(nn.Module):
    is_segmentor = False
    is_model_emb = True
    
    def __init__(self, embedding_dim=256):
        super().__init__()
        if embedding_dim != 256: raise ValueError(f"ModelEmb2DNet requires embedding_dim=256 to match SAM's dense embedding dimension, got out={embedding_dim}")
        self.embedding_dim = embedding_dim

    def preprocess(self, x):
        if x.ndim != 4: raise ValueError(f"ModelEmb2DNet expects 4D input (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != 3: raise ValueError(f"Expected RGB (3 channels), got {x.shape[1]} channels. Polyp pipeline requires RGB input.")
        return x

class Segmentor2DNet(nn.Module):
    is_segmentor = True

    def __init__(self, args, out_channels):
        super().__init__()
        self.args = args
        self.out_channels = out_channels
        self.segmentor = None

    def init_segmentor_head(self, segmentor_type, in_channels=256):
        input_size = int(self.args['input_size'])
        factors = list(self.args['upsample_factor_list'])
        if segmentor_type == 'PointwiseConv': self.segmentor = PointwiseCNN(in_channels=in_channels, out_channels=self.out_channels,output_size=(input_size, input_size))
        elif segmentor_type == 'ConvUpsample': self.segmentor = ConvUpsample(in_channels=in_channels, out_channels=self.out_channels, upsample_factors=factors)
        elif segmentor_type == 'TransposeConvUpsample': self.segmentor = TransposeConvUpsample(in_channels=in_channels, out_channels=self.out_channels, upsample_factors=factors)
        elif segmentor_type == 'ConvTwoUpsample': self.segmentor = ConvTwoUpsample(in_channels=in_channels, out_channels=self.out_channels, upsample_factors=factors, relu=True)
        elif segmentor_type == 'DeepConvUpsample': self.segmentor = DeepConvUpsample(in_channels=in_channels, out_channels=self.out_channels, upsample_factors=factors)
        else: raise ValueError(f"Segmentor '{segmentor_type}' not implemented. Choose from: PointwiseConv, ConvUpsample, TransposeConvUpsample, ConvTwoUpsample, DeepConvUpsample.")

        n_params = sum(p.numel() for p in self.segmentor.parameters())
        print(f"[SEGMENTOR] Aggregator Summary: {segmentor_type}")
        print("-" * 58)
        print(f"{'Property':<22} | {'Value'}")
        print("-" * 58)
        print(f"{'Type':<22} | {segmentor_type}")
        print(f"{'Input channels':<22} | {in_channels}")
        print(f"{'Output channels':<22} | {self.out_channels}")

        if segmentor_type == 'PointwiseConv':
            print(f"{'Structure':<22} | Conv1×1 → interpolate(bilinear)")
            print(f"{'Output size':<22} | ({input_size}, {input_size})")
            print(f"{'Input shape':<22} | (B, {in_channels}, H/4, W/4)")
            print(f"{'Output shape':<22} | (B, {self.out_channels}, {input_size}, {input_size})")

        elif segmentor_type in ('ConvUpsample', 'ConvTwoUpsample', 'DeepConvUpsample', 'TransposeConvUpsample'):    
            scale = 1
            for f in factors: scale *= f
            if segmentor_type == 'ConvUpsample': struct = ' → '.join([f'Conv1×1 + Upsample(×{f})' for f in factors]) + ' → Conv1×1'
            elif segmentor_type == 'ConvTwoUpsample': struct = ' → '.join([f'Conv1×1 + ReLU + Conv1×1 + Upsample(×{f})' for f in factors]) + ' → Conv1×1'
            elif segmentor_type == 'DeepConvUpsample': struct = ' → '.join([f'Conv3×3 + BN + ReLU + Conv1×1 + BN + ReLU + Upsample(×{f})' for f in factors]) + ' → Conv1×1'
            elif segmentor_type == 'TransposeConvUpsample': struct = ' → '.join([f'Conv1×1 + ConvTranspose2d(×{f})' for f in factors]) + ' → Conv1×1'
            print(f"{'Structure':<22} | {struct}")
            print(f"{'Upsample factors':<22} | {factors}  (total ×{scale})")
            print(f"{'Input shape':<22} | (B, {in_channels}, H/4, W/4)")
            print(f"{'Output shape':<22} | (B, {self.out_channels}, {scale}·H/4, {scale}·W/4)")

        print("-" * 58)
        print(f"{'Total parameters':<22} | {n_params:,} ({n_params/1e6:.4f}M)")


    def preprocess(self, x):
        if x.ndim != 4: raise ValueError(f"Segmentor2DNet expects 4D input (B, C, H, W), got shape {tuple(x.shape)} with ndim={x.ndim}.")
        if x.shape[1] != 3: raise ValueError(f"Expected RGB (3 channels), got {x.shape[1]} Polyp pipeline requires RGB.")
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super().__init__()
        P = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out

class UpBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, func=None, drop=0):
        super().__init__()
        P = kernel_size // 2
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.BN = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in, x_up):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), dim=1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh': return torch.tanh(self.BN(x1))
        elif self.func == 'relu': return torch.nn.functional.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid': return torch.sigmoid(self.BN(x1))
        else: return x1