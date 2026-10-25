from models.hardnet import HarDNet
from models.base import ModelEmb2DNet
from models.model_single import SmallDecoder
import torch.nn.functional as F
import torch.nn as nn

class ModelEmb(ModelEmb2DNet):
    def __init__(self, args, out_ch=256):
        super().__init__(args, out=out_ch)
        self.backbone = HarDNet(depth_wise=bool(args['depth_wise']), arch=int(args['order']), pretrained=args['pretrained'])
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=out_ch)
        for param in self.backbone.parameters(): param.requires_grad = True

        n_bb  = sum(p.numel() for p in self.backbone.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        n_tot = n_bb + n_dec
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('\n' + '='*60)
        print("[MODEL-EMB] ModelEmb Architecture Summary")
        print("-" * 58)
        print(f"{'Property':<22} | {'Value'}")
        print("-" * 58)
        print(f"{'Backbone':<22} | HarDNet-{int(args['order'])}")
        print(f"{'Depth-wise':<22} | {bool(args['depth_wise'])}")
        print(f"{'Pretrained':<22} | {bool(args['pretrained'])}")
        print(f"{'Full features':<22} | {d}")
        print(f"{'Decoder':<22} | SmallDecoder (use_tanh=True)")
        print(f"{'Output channels':<22} | {out_ch}")
        print("-" * 58)
        print(f"{'Backbone params':<22} | {n_bb:,} ({n_bb/1e6:.2f}M)")
        print(f"{'Decoder params':<22} | {n_dec:,} ({n_dec/1e6:.2f}M)")
        print(f"{'Total params':<22} | {n_tot:,} ({n_tot/1e6:.2f}M)")
        print(f"{'Trainable params':<22} | {n_trainable:,} ({n_trainable/1e6:.2f}M)")
        print("-" * 58)
        print(f"{'Input':<22} | (B, 3, H, W) — e.g. (1, 3, 1024, 1024)")
        print(f"{'Output':<22} | (B, {out_ch}, 64, 64)")
        print(f"{'Output range':<22} | [-1, 1] (tanh)")
        print('='*60 + '\n')
        
    def forward(self, img):
        img = self.preprocess(img)
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=False)