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
        
    def forward(self, img):
        img = self.preprocess(img)
        z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=False)