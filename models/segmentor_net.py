from models.hardnet import HarDNet
from models.base import Segmentor2DNet
from models.model_single import SmallDecoder

class HardNetSegmentor(Segmentor2DNet):
    def __init__(self, args, segmentor='PointwiseConv', out=256):
        super().__init__(args, out_channels=args['out_channels'])
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), pretrained=args['pretrained'])
        for param in self.backbone.parameters(): param.requires_grad = True
        d = self.backbone.full_features
        self.decoder = SmallDecoder(d, out=out, use_tanh=True)
        self.init_segmentor_head(segmentor_type=segmentor, in_channels=out)

        n_bb  = sum(p.numel() for p in self.backbone.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        n_seg = sum(p.numel() for p in self.segmentor.parameters())
        n_tot = n_bb + n_dec + n_seg
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        Idim = int(args.get('Idim', 512))
        factors = args.get('upsample_factor_list', [2, 2])
        scale = 1
        for f in factors:scale *= f

        print('\n' + '='*60)
        print("[HARDNET-SEG] HardNetSegmentor Architecture Summary")
        print("-" * 58)
        print(f"{'Property':<22} | {'Value'}")
        print("-" * 58)
        print(f"{'Backbone':<22} | HarDNet-{int(args['order'])}")
        print(f"{'Depth-wise':<22} | {bool(int(args['depth_wise']))}")
        print(f"{'Pretrained':<22} | {bool(args['pretrained'])}")
        print(f"{'Full features':<22} | {d}")
        print(f"{'Decoder':<22} | SmallDecoder (use_tanh=True)")
        print(f"{'Aggregator':<22} | {segmentor}")
        print(f"{'Out channels':<22} | {self.out_channels}")
        print("-" * 58)
        print(f"{'Backbone params':<22} | {n_bb:,} ({n_bb/1e6:.2f}M)")
        print(f"{'Decoder params':<22} | {n_dec:,} ({n_dec/1e6:.2f}M)")
        print(f"{'Aggregator params':<22} | {n_seg:,} ({n_seg/1e6:.4f}M)")
        print(f"{'Total params':<22} | {n_tot:,} ({n_tot/1e6:.2f}M)")
        print(f"{'Trainable params':<22} | {n_trainable:,} ({n_trainable/1e6:.2f}M) [before any freeze]")
        print("-" * 58)
        print(f"{'Input':<22} | (B, 3, H, W) — e.g. (1, 3, 1024, 1024)")
        print(f"{'Dense embeddings':<22} | (B, {out}, H/4, W/4) — tanh ∈ [-1, 1]")
        print(f"{'Output':<22} | (B, {self.out_channels}, {Idim}, {Idim}) — raw logits")
        print(f"{'Post-activation':<22} | sigmoid applied by loss/metric")
        print('='*60 + '\n')

            
    def forward(self, img):
        img = self.preprocess(img)
        z = self.backbone(img)
        dec = self.decoder(z)
        out = self.segmentor(dec)
        return out
