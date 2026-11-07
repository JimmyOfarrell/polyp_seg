import torch
import torch.nn as nn
from models.base import CNNBlock, UpBlockSkip

class SmallDecoder(nn.Module):
    def __init__(self, full_features, embedding_dim, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh

        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2], func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1], func='relu', drop=0)
        self.final = CNNBlock(full_features[1], embedding_dim, kernel_size=3, drop=0)

        self.num_params = sum(p.numel() for p in self.parameters())
        n_params_up1  = sum(p.numel() for p in self.up1.parameters())
        n_params_up2  = sum(p.numel() for p in self.up2.parameters())
        n_params_final = sum(p.numel() for p in self.final.parameters())

        print("[DECODER] SmallDecoder Architecture Summary:")
        print("-" * 58)
        print(f"{'Property':<22} | {'Value'}")
        print("-" * 58)
        print(f"{'Input features':<22} | {full_features}")
        print(f"{'Input channels used':<22} | x[1], x[2], x[3] (skip x[0], x[4])")
        print(f"{'up1 (in → out)':<22} | {full_features[3] + full_features[2]} → {full_features[2]}")
        print(f"{'up2 (in → out)':<22} | {full_features[2] + full_features[1]} → {full_features[1]}")
        print(f"{'final (in → out)':<22} | {full_features[1]} → {embedding_dim}")
        print(f"{'use_tanh':<22} | {use_tanh}")
        print("-" * 58)
        print(f"{'up1 params':<22} | {n_params_up1:,} ({n_params_up1/1e6:.2f}M)")
        print(f"{'up2 params':<22} | {n_params_up2:,} ({n_params_up2/1e6:.2f}M)")
        print(f"{'final params':<22} | {n_params_final:,} ({n_params_final/1e6:.2f}M)")
        print(f"{'Total parameters':<22} | {self.num_params:,} ({self.num_params/1e6:.2f}M)")
        print("-" * 58)
        print(f"{'Output channels':<22} | {embedding_dim} (SAM dense prompt dim)\n")

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = self.final(z)
        if self.use_tanh: out = torch.tanh(out)
        return out
