import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, groups=1, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace=True))

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels=None, stride=1, bias=False):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        if out_channels != in_channels: raise ValueError(f"DWConvLayer requires in_channels == out_channels (got {in_channels} vs {out_channels}).")
        self.add_module('dwconv', nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(in_channels))

class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 1, stride: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel, bias=bias))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride, bias=bias))

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0: return base_ch, 0, []

        out_channels = growth_rate
        link = []

        idx, dv = 0, 1
        while dv <= layer:
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if idx > 0: out_channels *= grmul
            idx += 1
            dv *= 2

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch

        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()

        self.keepBase = keepBase
        self.links = []
        self.out_channels = 0
        layers_ = []

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)

            if dwconv: layers_.append(CombConvLayer(inch, outch))
            else: layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1): self.out_channels += outch

        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer, link in enumerate(self.links):
            inputs = [layers_[i] for i in link]
            x_cat = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            layers_.append(self.layers[layer](x_cat))

        t = len(layers_)
        keep = [layers_[i] for i in range(t) if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1)]
        return torch.cat(keep, dim=1)

class HarDNet(nn.Module):
    _PRETRAINED_URLS = {(68, False): 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth',
                        (85, False): 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth',
                        (68, True):  'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth',
                        (39, True):  'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth'}
    
    def __init__(self, depth_wise=False, arch=68, pretrained=True, weight_dir=''):
        super().__init__()
        if arch not in (39, 68, 85): raise ValueError(f"Unsupported HarDNet arch={arch}. Choose from 39, 68, 85.")

        if arch == 68:
            first_ch = [32, 64]
            ch_list = [128, 256, 320, 640, 1024]
            gr = [14, 16, 20, 40, 160]
            n_layers = [8, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 1, 0]
            grmul = 1.7
            drop_rate = 0.1
            full_features = [64, 128, 320, 640, 1024]
            feature_indices = [1, 4, 9, 12, 15]
            truncate_at     = 16
        elif arch == 85:
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            grmul = 1.7
            drop_rate = 0.2
            full_features = [96, 192, 320, 720, 1280]
            feature_indices = [1, 4, 9, 14, 18]
            truncate_at = 19
        else:
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]
            grmul = 1.6
            drop_rate = 0.1
            full_features = [48, 96, 320, 640, 1024]
            feature_indices = [1, 4, 7, 10, 13]
            truncate_at = 14

        second_kernel = 3
        max_pool = True

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        print(f"[BACKBONE] Initializing HarDNet-{arch}:")
        print(f"[BACKBONE] Pretrained: {pretrained} | Depthwise: {depth_wise}")
        print(f"[BACKBONE] Output channels per stage: {full_features}", '\n')

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel, stride=1))
        if max_pool: self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else: self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85: self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if downSamp[i] == 1:
                if max_pool: self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else: self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.base.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Dropout(drop_rate), nn.Linear(ch, 1000)))

        if pretrained: self._load_pretrained(arch, depth_wise, weight_dir)

        self.features        = full_features[-1]
        self.full_features   = full_features
        self.feature_indices = feature_indices
        self.truncate_at     = truncate_at
        self.base            = self.base[:self.truncate_at] 

        self.num_params = sum(p.numel() for p in self.parameters())
        print("[BACKBONE] HarDNet Architecture Summary:")
        print("-" * 58)
        print(f"{'Property':<22} | {'Value'}")
        print("-" * 58)
        print(f"{'Feature channels':<22} | {self.features}")
        print(f"{'Full features':<22} | {self.full_features}")
        print(f"{'Feature indices':<22} | {self.feature_indices}")
        print(f"{'Base layers':<22} | {len(self.base)}")
        print(f"{'Total parameters':<22} | {self.num_params:,} ({self.num_params / 1e6:.2f}M)")


    def _load_pretrained(self, arch, depth_wise, weight_dir):
        postfix = 'DS' if depth_wise else ''
        postfix_local = 'ds' if depth_wise else ''

        if weight_dir:
            weight_file = os.path.join(weight_dir, f'hardnet{arch}{postfix_local}.pth')
            if os.path.isfile(weight_file):
                print(f"[BACKBONE] Loading local ImageNet weights: {weight_file}")
                try:state = torch.load(weight_file, map_location='cpu', weights_only=True)
                except TypeError: state = torch.load(weight_file, map_location='cpu')
                self.load_state_dict(state)
                print(f"[BACKBONE] Loaded HarDNet-{arch}{postfix} from local file.")
                return
            else: print(f"[BACKBONE] WARNING: weight_dir='{weight_dir}' set but file not found: {weight_file}. Falling back to URL download.")

        key = (arch, depth_wise)
        if key not in self._PRETRAINED_URLS:
            available = list(self._PRETRAINED_URLS.keys())
            raise ValueError(f"No pretrained weights for arch={arch}, depth_wise={depth_wise}. Available: {available}. Options: (1) use one of the available combos, (2) provide local weights via weight_dir, or (3) set pretrained=False.")
        checkpoint = self._PRETRAINED_URLS[key]
        print(f"[BACKBONE] Downloading ImageNet weights: {checkpoint}")
        try:state = torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        except Exception as e: raise RuntimeError(f"Failed to download weights from {checkpoint}: {e}\nIf offline, set pretrained=False or place file 'hardnet{arch}{postfix_local}.pth' in weight_dir.") from e
        self.load_state_dict(state)
        print(f"[BACKBONE] Loaded HarDNet-{arch}{postfix} from URL.", '\n')

    def forward(self, x):
        x2 = x4 = x8 = x16 = x32 = None

        for inx, layer in enumerate(self.base):
            x = layer(x)
            if inx == self.feature_indices[0]: x2 = x
            elif inx == self.feature_indices[1]: x4 = x
            elif inx == self.feature_indices[2]: x8 = x
            elif inx == self.feature_indices[3]: x16 = x
            elif inx == self.feature_indices[4]: x32 = x

        features = (x2, x4, x8, x16, x32)

        if any(f is None for f in features):
            missing = [i for i, f in enumerate(features) if f is None]
            raise RuntimeError(f"HarDNet forward: feature maps {missing} were not captured. Check self.feature_indices = {self.feature_indices} and len(self.base) = {len(self.base)}.")

        return features