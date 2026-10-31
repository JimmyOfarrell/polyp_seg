from models.hardnet import HarDNet
from models.base import Segmentor2DNet
from models.model_single import SmallDecoder

class HardNetSegmentor(Segmentor2DNet):
    def __init__(self, args, segmentor='PointwiseConv',backbone_weight_dir='', embedding_dim=256, use_tanh=True):
        super().__init__(args, out_channels=args['out_channels'])
        self.segmentor_name = segmentor
        self.args = args
        self.depth_wise = bool(self.args['depth_wise'])
        self.arch = int(self.args['order'])
        self.pretrained = bool(self.args['pretrained'])
        self.decoder_embedding_dim = embedding_dim
        self.use_tanh = use_tanh
        self.backbone_weight_dir = backbone_weight_dir
        self.backbone = HarDNet(depth_wise=self.depth_wise, arch=self.arch, pretrained=self.pretrained, weight_dir=self.backbone_weight_dir)
        for param in self.backbone.parameters(): param.requires_grad = True
        self.backbone_full_features = self.backbone.full_features
        self.decoder = SmallDecoder(self.backbone_full_features, embedding_dim=self.decoder_embedding_dim, use_tanh=self.use_tanh)
        self.init_segmentor_head(segmentor_type=self.segmentor_name, in_channels=self.decoder_embedding_dim)

        n_bb  = sum(p.numel() for p in self.backbone.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        n_seg = sum(p.numel() for p in self.segmentor.parameters())
        n_tot = n_bb + n_dec + n_seg
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        input_size = int(self.args['input_size'])
        factors = self.args['upsample_factor_list']
        scale = 1
        for f in factors:scale *= f

        print("[HARDNET-SEG] HardNetSegmentor Architecture Summary:")
        print("-" * 58)
        print(f"{'Property':<22} | {'Value'}")
        print("-" * 58)
        print(f"{'Backbone':<22} | HarDNet-{self.arch}")
        print(f"{'Depth-wise':<22} | {self.depth_wise}")
        print(f"{'Pretrained':<22} | {self.pretrained}")
        print(f"{'Full features':<22} | {self.backbone_full_features}")
        print(f"{'Decoder':<22} | SmallDecoder (use_tanh={self.use_tanh})")
        print(f"{'Aggregator':<22} | {self.segmentor_name}")
        print(f"{'Out channels':<22} | {self.out_channels}")
        print("-" * 58)
        print(f"{'Backbone params':<22} | {n_bb:,} ({n_bb/1e6:.2f}M)")
        print(f"{'Decoder params':<22} | {n_dec:,} ({n_dec/1e6:.2f}M)")
        print(f"{'Aggregator params':<22} | {n_seg:,} ({n_seg/1e6:.4f}M)")
        print(f"{'Total params':<22} | {n_tot:,} ({n_tot/1e6:.2f}M)")
        print(f"{'Trainable params':<22} | {n_trainable:,} ({n_trainable/1e6:.2f}M) [before any freeze]")
        print("-" * 58)
        print(f"{'Input':<22} | (B, 3, H, W) — e.g. (1, 3, 512, 512)")
        print(f"{'Dense embeddings':<22} | (B, {self.decoder_embedding_dim}, H/4, W/4) — tanh ∈ [-1, 1]")
        print(f"{'Output':<22} | (B, {self.out_channels}, {input_size}, {input_size}) — raw logits")
        print(f"{'Post-activation':<22} | sigmoid applied by loss/metric", '\n')

            
    def forward(self, img):
        img = self.preprocess(img)
        z = self.backbone(img)
        dec = self.decoder(z)
        out = self.segmentor(dec)
        return out
